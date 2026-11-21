"""通宵连续训练：分轮次持续 LoRA 微调，直到指定时间，自动保留最佳适配器。

用法示例：
    python scripts/overnight_train.py --until 07:00 --serve-after

特点：
- 模型只加载一次，轮次之间不重复加载权重
- 每轮结束做一次验证集困惑度评估，困惑度创新低就备份为 best
- 连续多轮不进步时自动降低学习率（最低 5e-6），避免把权重训废
- 到点（含中途每次 step 检查）自动停止并落盘，可选自动拉起 Web 服务
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import random
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.data import load_jsonl, split  # noqa: E402
from core.runtime import dtype_kwarg, pick_device, resolve_model_source  # noqa: E402
from core.tokens import build_example, collate  # noqa: E402

OUT = ROOT / "outputs" / "overnight"
LOG = OUT / "overnight.log"
STATUS = OUT / "status.json"
METRICS = OUT / "metrics.jsonl"


# ---------------------------------------------------------------- 基础工具

def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    line = f"[{now_text()}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def parse_deadline(text: str) -> datetime:
    """支持 07:00 或 2026-11-15 07:00。"""
    now = datetime.now()
    if " " in text:
        return datetime.strptime(text, "%Y-%m-%d %H:%M")
    hour, minute = (int(part) for part in text.split(":"))
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def humanize(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def write_status(payload: dict) -> None:
    STATUS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def keep_awake(enable: bool) -> None:
    """阻止系统休眠，避免通宵训练被挂起（仅 Windows 有效）。"""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        flag = 0x80000000 | (0x00000001 if enable else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flag)
    except Exception:  # noqa: BLE001
        pass


PARAPHRASES = ("请问{q}", "想确认一下，{q}", "{q}？（简要回答即可）", "麻烦说明：{q}")


def augment_records(records: list[dict], copies: int, seed: int) -> list[dict]:
    """对用户提问做轻量改写，降低小数据集过拟合（答案保持不变）。"""
    if copies <= 0:
        return records
    rng = random.Random(seed)
    expanded = list(records)
    for record in records:
        messages = record["messages"]
        user_index = None
        for index in range(len(messages) - 1, -1, -1):
            if messages[index]["role"] == "user":
                user_index = index
                break
        if user_index is None:
            continue
        question = messages[user_index]["content"]
        if question.startswith("请问") or len(question) > 200:
            continue
        templates = [item for item in PARAPHRASES if item != "{q}？（简要回答即可）" or "？" not in question]
        for variant in rng.sample(templates, k=min(copies, len(templates))):
            new_messages = copy.deepcopy(messages)
            new_messages[user_index]["content"] = variant.format(q=question)
            expanded.append({"messages": new_messages})
    return expanded


class SFTDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_seq_len: int) -> None:
        self.items = [build_example(tokenizer, rec["messages"], max_seq_len) for rec in records]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def training_args(**kwargs) -> TrainingArguments:
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in params and "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy", "no")
    return TrainingArguments(**kwargs)


def trainer_token_kwargs(tokenizer) -> dict:
    params = inspect.signature(Trainer.__init__).parameters
    return {"processing_class": tokenizer} if "processing_class" in params else {"tokenizer": tokenizer}


@torch.inference_mode()
def evaluate(model, tokenizer, records: list[dict], max_seq_len: int, device: str) -> float:
    """验证集困惑度（只统计 assistant 回复部分）。"""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for record in records:
        example = build_example(tokenizer, record["messages"], max_seq_len)
        input_ids = torch.tensor([example["input_ids"]], device=device)
        attention_mask = torch.tensor([example["attention_mask"]], device=device)
        labels = torch.tensor([example["labels"]], device=device)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        valid = int((labels != -100).sum())
        total_loss += float(output.loss) * valid
        total_tokens += valid
    model.train()
    if not total_tokens:
        return float("nan")
    return math.exp(total_loss / total_tokens)


@torch.inference_mode()
def sample_answer(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 60) -> str:
    model.eval()
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    model.train()
    return text.strip()


class RoundTracker(TrainerCallback):
    def __init__(self, deadline: datetime, state: dict) -> None:
        self.deadline = deadline
        self.state = state
        self.stop_reason = None

    def on_step_end(self, args, state, control, **kwargs):
        if datetime.now() >= self.deadline:
            control.should_training_stop = True
            self.stop_reason = "deadline"
        if state.global_step % 5 == 0:
            self.state["round_step"] = state.global_step
            self.state["updated_at"] = now_text()
            write_status(self.state)


# ---------------------------------------------------------------- 主流程

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--until", default="07:00", help="停止时间，如 07:00")
    parser.add_argument("--dataset", default="", help="数据集 jsonl 路径，默认取最新导入的数据集")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--steps-per-round", type=int, default=80)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--augment-copies", type=int, default=2, help="训练集提问改写份数，0 表示不增强")
    parser.add_argument("--threads", type=int, default=max(1, (torch.get_num_threads() or 8) - 2))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--serve-after", action="store_true", help="训练结束后自动启动 Web 服务")
    parser.add_argument("--delete-task", default="", help="训练结束后删除的计划任务名")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    keep_awake(True)
    OUT.mkdir(parents=True, exist_ok=True)

    deadline = parse_deadline(args.until)
    remaining = (deadline - datetime.now()).total_seconds()
    if remaining < 300:
        log(f"距离停止时间不足 5 分钟（{deadline:%Y-%m-%d %H:%M}），不启动训练")
        return 0

    dataset_path = Path(args.dataset) if args.dataset else None
    if dataset_path is None:
        candidates = sorted(
            (ROOT / "data" / "datasets").glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            log("没有找到数据集，先通过 Web 界面导入")
            return 1
        dataset_path = candidates[0]

    records = load_jsonl(dataset_path)
    train_records, eval_records = split(records, args.eval_ratio, args.seed)
    eval_records = eval_records or records[: min(4, len(records))]
    train_records = augment_records(train_records, args.augment_copies, args.seed)
    prompts = []
    for record in eval_records[:2]:
        for message in record["messages"]:
            if message["role"] == "user":
                prompts.append(message["content"])
                break

    status = {
        "started_at": now_text(),
        "deadline": deadline.strftime("%Y-%m-%d %H:%M"),
        "dataset": str(dataset_path),
        "train_samples": len(train_records),
        "eval_samples": len(eval_records),
        "round": 0,
        "total_steps": 0,
        "lr": args.lr,
        "best_perplexity": None,
        "best_round": None,
        "state": "loading",
        "updated_at": now_text(),
    }
    write_status(status)

    log(f"停止时间 {deadline:%Y-%m-%d %H:%M}，剩余 {humanize(remaining)}")
    log(f"数据集 {dataset_path.name}：训练 {len(train_records)} 条 / 验证 {len(eval_records)} 条")
    log(f"设备 {pick_device()}，线程 {args.threads}，每轮 {args.steps_per_round} 步")

    device = pick_device()
    source = resolve_model_source(args.base_model, ROOT / "models")
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("加载基座模型…")
    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        source, **dtype_kwarg(dtype), low_cpu_mem_usage=True, trust_remote_code=True
    )
    model.to(device)
    model.config.use_cache = False

    resumed = False
    latest_adapter = OUT / "latest"
    if (latest_adapter / "adapter_config.json").exists():
        from peft import PeftModel

        log("检测到上次训练的适配器，断点续训")
        model = PeftModel.from_pretrained(model, str(latest_adapter), is_trainable=True)
        resumed = True
    else:
        from peft import LoraConfig, get_peft_model

        lora_kwargs = dict(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        try:
            model = get_peft_model(
                model, LoraConfig(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], **lora_kwargs)
            )
        except ValueError:
            model = get_peft_model(model, LoraConfig(target_modules="all-linear", **lora_kwargs))

    train_dataset = SFTDataset(train_records, tokenizer, args.max_seq_len)
    collator = lambda batch: collate(batch, tokenizer.pad_token_id)  # noqa: E731

    lr = args.lr
    best_ppl = math.inf
    best_round = None
    best_meta_path = OUT / "best_meta.json"
    if best_meta_path.exists() and (OUT / "best" / "adapter" / "adapter_config.json").exists():
        try:
            previous = json.loads(best_meta_path.read_text(encoding="utf-8-sig"))
            best_ppl = float(previous.get("perplexity", math.inf))
            best_round = previous.get("round")
            status["best_perplexity"] = round(best_ppl, 3)
            status["best_round"] = best_round
            log(f"沿用历史最佳：困惑度 {best_ppl:.3f}（第 {best_round} 轮）")
        except Exception:  # noqa: BLE001
            best_ppl = math.inf
    stale_rounds = 0
    total_steps = 0
    round_no = 0
    sec_per_step = 4.5
    started = time.time()

    status["state"] = "training"
    write_status(status)

    try:
        while True:
            remaining = (deadline - datetime.now()).total_seconds()
            if remaining < 120:
                log("已到停止时间")
                break

            budget_steps = int(remaining / max(sec_per_step, 0.5) * 0.75)
            steps = min(args.steps_per_round, max(10, budget_steps))
            round_no += 1
            round_dir = OUT / "tmp_round"

            status.update(
                {
                    "round": round_no,
                    "lr": round(lr, 8),
                    "state": "training",
                    "round_step": 0,
                    "round_target_steps": steps,
                    "remaining": humanize(remaining),
                    "updated_at": now_text(),
                }
            )
            write_status(status)
            log(f"—— 第 {round_no} 轮：{steps} 步，lr={lr:.2e}，剩余 {humanize(remaining)}")

            tracker = RoundTracker(deadline, status)
            round_started = time.time()
            trainer = Trainer(
                model=model,
                args=training_args(
                    output_dir=str(round_dir),
                    per_device_train_batch_size=args.batch,
                    gradient_accumulation_steps=args.accum,
                    learning_rate=lr,
                    max_steps=steps,
                    warmup_steps=min(5, max(1, steps // 20)),
                    logging_steps=10,
                    save_strategy="no",
                    report_to="none",
                    remove_unused_columns=False,
                    dataloader_num_workers=0,
                    seed=args.seed + round_no,
                    lr_scheduler_type="constant",
                    use_cpu=device == "cpu",
                ),
                train_dataset=train_dataset,
                data_collator=collator,
                callbacks=[tracker],
                **trainer_token_kwargs(tokenizer),
            )
            output = trainer.train()
            round_seconds = max(1.0, time.time() - round_started)
            done_steps = int(trainer.state.global_step)
            total_steps += done_steps
            sec_per_step = round_seconds / max(1, done_steps)

            train_loss = (output.metrics or {}).get("train_loss")
            perplexity = evaluate(model, tokenizer, eval_records, args.max_seq_len, device)
            sample = ""
            if prompts:
                try:
                    sample = sample_answer(model, tokenizer, prompts[0], device)
                except Exception as exc:  # noqa: BLE001
                    log(f"抽样生成失败：{exc}")

            improved = perplexity < best_ppl * 0.999
            if improved and math.isfinite(perplexity):
                best_ppl = perplexity
                shutil.rmtree(OUT / "best", ignore_errors=True)
                model.save_pretrained(OUT / "best" / "adapter")
                tokenizer.save_pretrained(OUT / "best" / "adapter")
                best_meta_path.write_text(
                    json.dumps(
                        {
                            "perplexity": round(perplexity, 4),
                            "round": round_no,
                            "total_steps": total_steps,
                            "time": now_text(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                stale_rounds = 0
                status["best_perplexity"] = round(perplexity, 3)
                status["best_round"] = round_no
            else:
                stale_rounds += 1

            # 保存最新权重，供断点续训
            shutil.rmtree(latest_adapter, ignore_errors=True)
            model.save_pretrained(latest_adapter)
            tokenizer.save_pretrained(latest_adapter)

            metric = {
                "round": round_no,
                "time": now_text(),
                "steps": done_steps,
                "total_steps": total_steps,
                "lr": lr,
                "train_loss": train_loss,
                "perplexity": round(perplexity, 3) if math.isfinite(perplexity) else None,
                "best_perplexity": round(best_ppl, 3) if math.isfinite(best_ppl) else None,
                "round_seconds": round(round_seconds, 1),
                "sec_per_step": round(sec_per_step, 2),
                "sample": sample,
            }
            with METRICS.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(metric, ensure_ascii=False) + "\n")

            log(
                f"第 {round_no} 轮完成：{done_steps} 步 / {round_seconds:.0f}s，"
                f"loss={train_loss}, 困惑度={metric['perplexity']}, 最佳={metric['best_perplexity']}"
            )
            if sample:
                log(f"抽样回答（{prompts[0][:16]}）：{sample[:80]}")

            if stale_rounds >= 6:
                new_lr = max(lr * 0.7, 1e-5)
                if new_lr < lr:
                    log(f"连续 {stale_rounds} 轮无进步，学习率 {lr:.2e} → {new_lr:.2e}")
                    lr = new_lr
                stale_rounds = 0
            elif round_no % 10 == 0 and lr > 1e-5:
                lr = max(lr * 0.85, 1e-5)
                log(f"周期性衰减，学习率 → {lr:.2e}（最佳困惑度 {status['best_perplexity']}）")

            status.update(
                {
                    "total_steps": total_steps,
                    "state": "training",
                    "lr": round(lr, 8),
                    "updated_at": now_text(),
                    "last_perplexity": metric["perplexity"],
                }
            )
            write_status(status)

            if tracker.stop_reason == "deadline":
                log("训练在轮内到达停止时间，已停止")
                break
    except KeyboardInterrupt:
        log("收到中断信号，保存当前权重后退出")
    except Exception:  # noqa: BLE001
        log("训练异常：\n" + traceback.format_exc())
    finally:
        try:
            shutil.rmtree(latest_adapter, ignore_errors=True)
            model.save_pretrained(latest_adapter)
            tokenizer.save_pretrained(latest_adapter)
        except Exception:  # noqa: BLE001
            log("保存最新权重失败：\n" + traceback.format_exc())
        keep_awake(False)

    summary = {
        "finished_at": now_text(),
        "duration_minutes": round((time.time() - started) / 60, 1),
        "rounds": round_no,
        "total_steps": total_steps,
        "best_perplexity": round(best_ppl, 3) if math.isfinite(best_ppl) else None,
        "resumed": resumed,
        "dataset": str(dataset_path),
        "best_path": str(OUT / "best" / "adapter"),
        "latest_path": str(latest_adapter),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    status.update({"state": "finished", "finished_at": summary["finished_at"], "updated_at": now_text()})
    write_status(status)
    log(f"全部完成：{summary['rounds']} 轮 / {summary['total_steps']} 步，最佳困惑度 {summary['best_perplexity']}")

    register_adapter(dataset_path, args.base_model, summary, status)

    if args.serve_after:
        try:
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "app.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8765",
                    "--app-dir",
                    str(ROOT),
                ],
                cwd=str(ROOT),
                stdout=(OUT / "server.log").open("a", encoding="utf-8"),
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            log("已自动启动 Web 服务：http://127.0.0.1:8765")
        except Exception:  # noqa: BLE001
            log("启动 Web 服务失败：\n" + traceback.format_exc())

    if args.delete_task:
        try:
            subprocess.run(
                ["schtasks", "/Delete", "/TN", args.delete_task, "/F"],
                capture_output=True,
                timeout=30,
            )
            log(f"已清理计划任务 {args.delete_task}")
        except Exception:  # noqa: BLE001
            log("清理计划任务失败")

    return 0


def register_adapter(dataset_path: Path, base_model: str, summary: dict, status: dict) -> None:
    """把最佳适配器登记到工作室数据库，早上打开界面就能直接用。"""
    best = OUT / "best" / "adapter"
    if not (best / "adapter_config.json").exists():
        log("没有可登记的适配器")
        return
    try:
        from app import db

        db.init()
        dataset_id = None
        for row in db.list_datasets():
            if Path(row["path"]) == dataset_path:
                dataset_id = row["id"]
                break
        import uuid

        adapter_id = uuid.uuid4().hex[:12]
        meta = {
            "base_model": base_model,
            "created_at": now_text(),
            "train_samples": status.get("train_samples"),
            "eval_samples": status.get("eval_samples"),
            "train_loss": None,
            "eval_loss": None,
            "epochs": None,
            "steps": summary.get("total_steps"),
            "lr": status.get("lr"),
            "lora_r": 8,
            "lora_alpha": 16,
            "duration_s": round(summary.get("duration_minutes", 0) * 60),
            "overnight": True,
            "rounds": summary.get("rounds"),
            "perplexity": summary.get("best_perplexity"),
        }
        db.insert(
            "adapters",
            {
                "id": adapter_id,
                "name": f"通宵训练-best-{datetime.now():%m%d-%H%M}",
                "path": str(best),
                "base_model": base_model,
                "dataset_id": dataset_id,
                "job_id": None,
                "meta_json": json.dumps(meta, ensure_ascii=False),
                "created_at": db.now(),
            },
        )
        log(f"最佳适配器已登记到数据库：{adapter_id}")
    except Exception:  # noqa: BLE001
        log("登记适配器失败：\n" + traceback.format_exc())


if __name__ == "__main__":
    raise SystemExit(main())
