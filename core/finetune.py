"""LoRA 微调核心：把数据集训练成可插拔的适配器。"""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from core.data import load_jsonl, split
from core.runtime import dtype_kwarg, pick_device, resolve_model_source
from core.tokens import build_example, collate

EventFn = Callable[[dict], None]


@dataclass
class TrainConfig:
    base_model: str
    dataset_path: Path
    output_dir: Path
    model_dir: Path = Path("models")
    max_seq_len: int = 1024
    epochs: float = 3.0
    max_steps: int = -1
    batch_size: int = 1
    grad_accum: int = 8
    lr: float = 2e-4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    eval_ratio: float = 0.1
    logging_steps: int = 1
    seed: int = 42
    lr_scheduler: str = "cosine"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["dataset_path"] = str(self.dataset_path)
        payload["output_dir"] = str(self.output_dir)
        payload["model_dir"] = str(self.model_dir)
        return payload


class SFTDataset(Dataset):
    def __init__(self, records: Sequence[dict], tokenizer, max_seq_len: int) -> None:
        self.items = [build_example(tokenizer, rec["messages"], max_seq_len) for rec in records]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


class _ProgressCallback(TrainerCallback):
    def __init__(self, emit: EventFn) -> None:
        self.emit = emit

    def on_train_begin(self, args, state, control, **kwargs):
        self.emit({"type": "stage", "text": f"训练开始，共 {state.max_steps} 步"})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        numeric = {k: round(float(v), 6) for k, v in logs.items() if isinstance(v, (int, float))}
        self.emit({"type": "step", "step": state.global_step, "max_steps": state.max_steps, "logs": numeric})

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        loss = (metrics or {}).get("eval_loss")
        if loss is not None:
            self.emit({"type": "stage", "text": f"第 {state.global_step} 步评估完成，eval_loss={float(loss):.4f}"})

    def on_epoch_end(self, args, state, control, **kwargs):
        self.emit({"type": "stage", "text": f"完成第 {round(state.epoch or 0, 2)} 轮"})


def _trainer_tokens(tokenizer) -> dict:
    params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in params:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def _training_args(**kwargs) -> TrainingArguments:
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in params and "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy", "no")
    if "use_cpu" in params:
        kwargs["use_cpu"] = pick_device() == "cpu"
    # 服务在无窗口/重定向 stdout 下运行时，tqdm 写进度条会抛 OSError(22)，
    # 训练日志已由 _ProgressCallback 输出，这里直接关掉进度条。
    if "disable_tqdm" in params:
        kwargs.setdefault("disable_tqdm", True)
    return TrainingArguments(**kwargs)


def _signature(cfg: TrainConfig) -> dict:
    """训练参数指纹，用来判断历史任务是否可以用作续训来源。"""
    return {
        "base_model": cfg.base_model,
        "dataset": Path(cfg.dataset_path).name,
        "lr": cfg.lr,
        "max_steps": cfg.max_steps,
        "grad_accum": cfg.grad_accum,
        "max_seq_len": cfg.max_seq_len,
    }


def _checkpoints(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.glob("checkpoint-*") if path.name.split("-")[-1].isdigit()]


def _find_resume(cfg: TrainConfig) -> tuple[str | None, str | None]:
    """优先本任务的存档；否则在同样参数的历史任务里找最新的 checkpoint。"""
    local = _checkpoints(cfg.output_dir / "_checkpoints")
    if local:
        best = max(local, key=lambda path: int(path.name.split("-")[-1]))
        return str(best), cfg.output_dir.name

    target = _signature(cfg)
    candidates: list[tuple[Path, str]] = []
    parent = cfg.output_dir.parent
    if parent.exists():
        for sibling in parent.iterdir():
            if sibling == cfg.output_dir or not sibling.is_dir():
                continue
            signature_file = sibling / "signature.json"
            if not signature_file.exists():
                continue
            try:
                if json.loads(signature_file.read_text(encoding="utf-8")) != target:
                    continue
            except json.JSONDecodeError:
                continue
            found = _checkpoints(sibling / "_checkpoints")
            if found:
                candidates.append((max(found, key=lambda path: int(path.name.split("-")[-1])), sibling.name))

    if not candidates:
        return None, None
    candidates.sort(key=lambda item: int(item[0].name.split("-")[-1]), reverse=True)
    return str(candidates[0][0]), candidates[0][1]


def run_finetune(cfg: TrainConfig, emit: EventFn) -> dict:
    started = time.time()
    device = pick_device()
    emit({"type": "stage", "text": f"设备：{device}，基座模型：{cfg.base_model}"})

    records = load_jsonl(cfg.dataset_path)
    train_records, eval_records = split(records, cfg.eval_ratio, cfg.seed)
    if not train_records:
        raise ValueError("数据集为空，无法训练")
    emit({"type": "stage", "text": f"样本切分：训练 {len(train_records)} 条 / 验证 {len(eval_records)} 条"})

    source = resolve_model_source(cfg.base_model, cfg.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device == "mps":
        dtype = torch.float16

    emit({"type": "stage", "text": "加载基座模型权重..."})
    model = AutoModelForCausalLM.from_pretrained(
        source,
        **dtype_kwarg(dtype),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.to(device)
    model.config.use_cache = False

    from peft import LoraConfig, get_peft_model

    lora_kwargs = dict(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    try:
        peft_config = LoraConfig(
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], **lora_kwargs
        )
        model = get_peft_model(model, peft_config)
    except ValueError:
        peft_config = LoraConfig(target_modules="all-linear", **lora_kwargs)
        model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    emit({"type": "stage", "text": f"可训练参数 {trainable:,} / 总参数 {total:,}（{trainable / total:.2%}）"})

    # 梯度检查点：用时间换内存，避免小内存机器训练中途卡死
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    train_dataset = SFTDataset(train_records, tokenizer, cfg.max_seq_len)
    # 验证集抽样上限：CPU 上 1.5B 评估一条要数秒，全量评估会让训练看起来“卡死”
    eval_sample_cap = 40
    eval_subset = eval_records[:eval_sample_cap]
    eval_dataset = SFTDataset(eval_subset, tokenizer, cfg.max_seq_len) if eval_subset else None
    if eval_dataset:
        emit({"type": "stage", "text": f"验证集抽样 {len(eval_subset)} 条（原始 {len(eval_records)} 条）"})

    steps_per_epoch = max(1, len(train_dataset) // max(1, cfg.batch_size * cfg.grad_accum))
    planned_steps = cfg.max_steps if cfg.max_steps > 0 else int(steps_per_epoch * max(1.0, cfg.epochs))
    eval_steps = max(1, planned_steps // 2)

    args_kwargs = dict(
        output_dir=str(cfg.output_dir / "_checkpoints"),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        warmup_ratio=0.03,
        logging_steps=cfg.logging_steps,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=cfg.seed,
        lr_scheduler_type=cfg.lr_scheduler,
        fp16=(dtype == torch.float16 and device == "cuda"),
        bf16=(dtype == torch.bfloat16 and device == "cuda"),
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=eval_steps if eval_dataset else None,
    )
    args_kwargs = {k: v for k, v in args_kwargs.items() if v is not None}

    trainer_kwargs = dict(
        model=model,
        args=_training_args(**args_kwargs),
        train_dataset=train_dataset,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
        callbacks=[_ProgressCallback(emit)],
    )
    if eval_dataset:
        trainer_kwargs["eval_dataset"] = eval_dataset
    trainer = Trainer(**_trainer_tokens(tokenizer), **trainer_kwargs)

    emit({"type": "stage", "text": "开始训练..."})
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "signature.json").write_text(
        json.dumps(_signature(cfg), ensure_ascii=False), encoding="utf-8"
    )
    resume_from, resume_owner = _find_resume(cfg)
    if resume_from:
        step = Path(resume_from).name.split("-")[-1]
        emit({"type": "stage", "text": f"从任务 {resume_owner} 的 checkpoint-{step} 继续训练"})
    train_output = trainer.train(resume_from_checkpoint=resume_from)
    metrics = dict(train_output.metrics or {})

    eval_loss = None
    if eval_dataset:
        emit({"type": "stage", "text": "计算验证集损失..."})
        eval_metrics = trainer.evaluate()
        eval_loss = eval_metrics.get("eval_loss")

    adapter_dir = cfg.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    meta = {
        "base_model": cfg.base_model,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_samples": len(train_records),
        "eval_samples": len(eval_records),
        "trainable_params": trainable,
        "total_params": total,
        "train_loss": metrics.get("train_loss"),
        "eval_loss": eval_loss,
        "epochs": cfg.epochs,
        "steps": trainer.state.global_step,
        "lr": cfg.lr,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "duration_s": round(time.time() - started, 1),
        "config": cfg.to_dict(),
    }
    (adapter_dir / "training_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    trainer.save_state()

    emit({"type": "stage", "text": f"训练完成，适配器已保存到 {adapter_dir}"})
    return meta
