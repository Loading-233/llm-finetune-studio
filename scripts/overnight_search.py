"""短跑选优：反复从基座出发做小规模 LoRA 训练，按验证集困惑度挑最优，跑到指定时间。

与「无限续训」的区别：
- 每组试验都从零初始化 LoRA，不会被上一组的过拟合拖累
- 固定验证集打分（默认 25% 数据），分数可比
- 70% 的试验在历史最优参数附近微调，30% 随机探索
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.data import load_jsonl, split  # noqa: E402
from core.runtime import dtype_kwarg, pick_device, resolve_model_source  # noqa: E402
from core.tokens import build_example, collate  # noqa: E402

OUT = ROOT / "outputs" / "search"
LOG = OUT / "search.log"
STATUS = OUT / "status.json"
METRICS = OUT / "metrics.jsonl"
BEST = OUT / "best"


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    line = f"[{now_text()}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def write_status(payload: dict) -> None:
    STATUS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_deadline(text: str) -> datetime:
    now = datetime.now()
    if " " in text:
        return datetime.strptime(text, "%Y-%m-%d %H:%M")
    hour, minute = (int(part) for part in text.split(":"))
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return target if target > now else target + timedelta(days=1)


def keep_awake(enable: bool) -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if enable else 0))
    except Exception:  # noqa: BLE001
        pass


PARAPHRASES = ("请问{q}", "想确认一下，{q}", "麻烦说明：{q}", "{q}？（简要回答即可）")


def augment(records: list[dict], copies: int, seed: int) -> list[dict]:
    import copy

    if copies <= 0:
        return records
    rng = random.Random(seed)
    expanded = list(records)
    for record in records:
        messages = record["messages"]
        index = next((i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "user"), None)
        if index is None:
            continue
        question = messages[index]["content"]
        if question.startswith("请问") or len(question) > 200:
            continue
        for template in rng.sample(list(PARAPHRASES), k=min(copies, len(PARAPHRASES))):
            new_messages = copy.deepcopy(messages)
            new_messages[index]["content"] = template.format(q=question)
            expanded.append({"messages": new_messages})
    return expanded


class SFTDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_seq_len: int) -> None:
        self.items = [build_example(tokenizer, rec["messages"], max_seq_len) for rec in records]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


class LoraState:
    """记录 LoRA 初始权重，每次试验前复位，等价于「从基座重新开始」。"""

    def __init__(self, model) -> None:
        self.model = model
        self.snapshot = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if "lora_" in key and value.dtype.is_floating_point
        }

    def reset(self) -> None:
        state = self.model.state_dict()
        with torch.no_grad():
            for key, value in self.snapshot.items():
                state[key].copy_(value)


@torch.inference_mode()
def evaluate(model, tokenizer, records: list[dict], max_seq_len: int, device: str) -> float:
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
    return math.exp(total_loss / total_tokens) if total_tokens else float("nan")


def sample_config(rng: random.Random, best: dict | None, steps_cap: int = 120) -> dict:
    if best and rng.random() < 0.7:
        lr = best["lr"] * rng.choice([0.5, 0.7, 1.0, 1.4])
        steps = int(best["steps"] * rng.choice([0.7, 1.0, 1.3]))
        accum = best["accum"]
        aug = best["augment"]
    else:
        lr = rng.choice([2e-5, 5e-5, 1e-4, 2e-4])
        steps = rng.choice([30, 50, 80])
        accum = rng.choice([1, 2, 4])
        aug = rng.choice([0, 2])
    return {
        "lr": round(min(max(lr, 1e-5), 3e-4), 8),
        "steps": int(min(max(steps, 10), steps_cap)),
        "accum": accum,
        "augment": aug,
        "batch": 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--until", default="07:00")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--eval-ratio", type=float, default=0.25)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--steps-cap", type=int, default=120, help="单组试验最大步数上限")
    parser.add_argument("--out", default="", help="输出目录，默认 outputs/search")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--serve-after", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    keep_awake(True)

    global OUT, LOG, STATUS, METRICS, BEST
    if args.out:
        OUT = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
        LOG = OUT / "search.log"
        STATUS = OUT / "status.json"
        METRICS = OUT / "metrics.jsonl"
        BEST = OUT / "best"
    OUT.mkdir(parents=True, exist_ok=True)

    deadline = parse_deadline(args.until)
    if (deadline - datetime.now()).total_seconds() < 300:
        log("距离停止时间不足 5 分钟，不启动")
        return 0

    dataset_path = Path(args.dataset) if args.dataset else sorted(
        (ROOT / "data" / "datasets").glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    )[0]
    records = load_jsonl(dataset_path)
    train_base, eval_records = split(records, args.eval_ratio, args.seed)
    log(f"停止时间 {deadline:%Y-%m-%d %H:%M}；训练池 {len(train_base)} 条 / 验证 {len(eval_records)} 条")

    device = pick_device()
    source = resolve_model_source(args.base_model, ROOT / "models")
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    log("加载基座模型…")
    model = AutoModelForCausalLM.from_pretrained(
        source, **dtype_kwarg(dtype), low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)
    model.config.use_cache = False

    from peft import LoraConfig, get_peft_model

    lora_kwargs = dict(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    try:
        model = get_peft_model(
            model, LoraConfig(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], **lora_kwargs)
        )
    except ValueError:
        model = get_peft_model(model, LoraConfig(target_modules="all-linear", **lora_kwargs))
    state = LoraState(model)
    collator = lambda batch: collate(batch, tokenizer.pad_token_id)  # noqa: E731

    dataset_cache: dict[int, SFTDataset] = {}
    rng = random.Random(args.seed)
    best_ppl = math.inf
    best_cfg: dict | None = None
    trials = 0
    started = time.time()

    if (BEST / "adapter" / "adapter_config.json").exists() and (OUT / "best_meta.json").exists():
        try:
            meta = json.loads((OUT / "best_meta.json").read_text(encoding="utf-8-sig"))
            best_ppl = float(meta["perplexity"])
            best_cfg = meta.get("config")
            log(f"已有历史最优：困惑度 {best_ppl:.3f}，配置 {best_cfg}")
        except Exception:  # noqa: BLE001
            pass

    status = {
        "started_at": now_text(),
        "deadline": deadline.strftime("%Y-%m-%d %H:%M"),
        "dataset": str(dataset_path),
        "eval_samples": len(eval_records),
        "trials": 0,
        "best_perplexity": None if math.isinf(best_ppl) else round(best_ppl, 3),
        "best_config": best_cfg,
        "state": "searching",
    }
    write_status(status)

    try:
        while datetime.now() < deadline - timedelta(seconds=120):
            cfg = sample_config(rng, best_cfg, args.steps_cap)
            trials += 1
            if cfg["augment"] not in dataset_cache:
                dataset_cache[cfg["augment"]] = SFTDataset(
                    augment(train_base, cfg["augment"], args.seed), tokenizer, args.max_seq_len
                )
            dataset = dataset_cache[cfg["augment"]]
            state.reset()

            status.update(
                {
                    "trials": trials,
                    "current": cfg,
                    "state": "trial",
                    "updated_at": now_text(),
                }
            )
            write_status(status)
            log(f"—— 试验 #{trials}：lr={cfg['lr']:.2e} steps={cfg['steps']} accum={cfg['accum']} aug={cfg['augment']}")

            round_dir = OUT / "tmp"
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=str(round_dir),
                    per_device_train_batch_size=cfg["batch"],
                    gradient_accumulation_steps=cfg["accum"],
                    learning_rate=cfg["lr"],
                    max_steps=cfg["steps"],
                    warmup_steps=min(5, max(1, cfg["steps"] // 20)),
                    logging_steps=20,
                    save_strategy="no",
                    report_to="none",
                    remove_unused_columns=False,
                    dataloader_num_workers=0,
                    seed=args.seed + trials,
                    lr_scheduler_type="constant",
                    use_cpu=device == "cpu",
                ),
                train_dataset=dataset,
                data_collator=collator,
                **(
                    {"processing_class": tokenizer}
                    if "processing_class" in inspect.signature(Trainer.__init__).parameters
                    else {"tokenizer": tokenizer}
                ),
            )
            trial_started = time.time()
            output = trainer.train()
            perplexity = evaluate(model, tokenizer, eval_records, args.max_seq_len, device)
            train_loss = (output.metrics or {}).get("train_loss")
            seconds = round(time.time() - trial_started, 1)

            improved = math.isfinite(perplexity) and perplexity < best_ppl * 0.995
            if improved:
                best_ppl, best_cfg = perplexity, cfg
                shutil.rmtree(BEST, ignore_errors=True)
                model.save_pretrained(BEST / "adapter")
                tokenizer.save_pretrained(BEST / "adapter")
                (OUT / "best_meta.json").write_text(
                    json.dumps(
                        {
                            "perplexity": round(perplexity, 4),
                            "config": cfg,
                            "trial": trials,
                            "time": now_text(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                status["best_perplexity"] = round(perplexity, 3)
                status["best_config"] = cfg

            metric = {
                "trial": trials,
                "time": now_text(),
                "config": cfg,
                "train_loss": train_loss,
                "perplexity": round(perplexity, 3) if math.isfinite(perplexity) else None,
                "best_perplexity": round(best_ppl, 3) if math.isfinite(best_ppl) else None,
                "seconds": seconds,
            }
            with METRICS.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(metric, ensure_ascii=False) + "\n")
            log(
                f"试验 #{trials} 完成：{seconds}s "
                f"loss={round(train_loss, 3) if isinstance(train_loss, float) else train_loss} "
                f"困惑度={metric['perplexity']} {'← 新最优' if improved else ''}"
            )
            status.update(
                {
                    "updated_at": now_text(),
                    "elapsed_minutes": round((time.time() - started) / 60, 1),
                }
            )
            write_status(status)
    except KeyboardInterrupt:
        log("收到中断，保存当前最优")
    except Exception:  # noqa: BLE001
        log("选优异常：\n" + traceback.format_exc())
    finally:
        keep_awake(False)

    summary = {
        "finished_at": now_text(),
        "duration_minutes": round((time.time() - started) / 60, 1),
        "trials": trials,
        "best_perplexity": round(best_ppl, 3) if math.isfinite(best_ppl) else None,
        "best_config": best_cfg,
        "best_path": str(BEST / "adapter"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    status.update({"state": "finished", "finished_at": summary["finished_at"]})
    write_status(status)
    log(f"选优结束：{trials} 组试验，最优困惑度 {summary['best_perplexity']}")
    standard_eval = run_standard_eval(dataset_path, args.base_model)
    if standard_eval:
        summary["standard_eval"] = standard_eval
        (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    register(summary, dataset_path, args.base_model)

    if args.serve_after:
        try:
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8765",
                 "--app-dir", str(ROOT)],
                cwd=str(ROOT),
                stdout=(OUT / "server.log").open("a", encoding="utf-8"),
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            log("已启动 Web 服务：http://127.0.0.1:8765")
        except Exception:  # noqa: BLE001
            log("启动 Web 服务失败")

    return 0


def run_standard_eval(dataset_path: Path, base_model: str) -> dict | None:
    """用工作室统一的 10% 验证集再评一次，方便和历史适配器比较。"""
    if not (BEST / "adapter" / "adapter_config.json").exists():
        return None
    try:
        from core.evaluate import evaluate_adapter

        report = evaluate_adapter(
            adapter_dir=BEST / "adapter",
            base_model=base_model,
            dataset_path=dataset_path,
            model_dir=ROOT / "models",
            emit=lambda event: None,
            max_new_tokens=96,
            max_prompts=2,
        )
        log(
            f"标准评测（4 条验证集）：基座 {report['perplexity']['base']} → 微调 {report['perplexity']['tuned']}"
        )
        return report["perplexity"]
    except Exception:  # noqa: BLE001
        log("标准评测失败：\n" + traceback.format_exc())
        return None


def register(summary: dict, dataset_path: Path, base_model: str) -> None:
    best = BEST / "adapter"
    if not (best / "adapter_config.json").exists():
        return
    try:
        from app import db

        db.init()
        dataset_id = next((row["id"] for row in db.list_datasets() if Path(row["path"]) == dataset_path), None)
        adapter_id = uuid.uuid4().hex[:12]
        meta = {
            "base_model": base_model,
            "created_at": now_text(),
            "overnight": True,
            "mode": "short-run-search",
            "trials": summary.get("trials"),
            "perplexity": summary.get("best_perplexity"),
            "config": summary.get("best_config"),
            "lora_r": 8,
            "lora_alpha": 16,
            "steps": (summary.get("best_config") or {}).get("steps"),
            "lr": (summary.get("best_config") or {}).get("lr"),
        }
        db.insert(
            "adapters",
            {
                "id": adapter_id,
                "name": f"通宵选优-best-{datetime.now():%m%d-%H%M}",
                "path": str(best),
                "base_model": base_model,
                "dataset_id": dataset_id,
                "job_id": None,
                "meta_json": json.dumps(meta, ensure_ascii=False),
                "created_at": db.now(),
            },
        )
        log(f"最优适配器已登记：{adapter_id}")
    except Exception:  # noqa: BLE001
        log("登记适配器失败：\n" + traceback.format_exc())


if __name__ == "__main__":
    raise SystemExit(main())
