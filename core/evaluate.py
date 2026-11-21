"""适配器评测：验证集困惑度 + 同一提问下的基座/微调对照。"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable, Sequence

from core.data import load_jsonl, split
from core.runtime import runtime

EventFn = Callable[[dict], None]


def _clean_metric(value: float) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), 3)


def build_prompts(records: Sequence[dict], limit: int = 3) -> list[str]:
    prompts: list[str] = []
    for rec in records:
        for message in rec["messages"]:
            if message["role"] == "user":
                prompts.append(message["content"])
                break
        if len(prompts) >= limit:
            break
    return prompts


def evaluate_adapter(
    adapter_dir: str | Path,
    base_model: str,
    dataset_path: str | Path,
    model_dir: Path,
    emit: EventFn | None = None,
    max_new_tokens: int = 160,
    max_prompts: int = 3,
) -> dict:
    log = emit or (lambda event: None)
    adapter_dir = Path(adapter_dir)
    records = load_jsonl(Path(dataset_path))
    _, eval_records = split(records, 0.1, 42)
    eval_records = eval_records or records[: min(4, len(records))]

    started = time.time()
    log({"type": "stage", "text": f"加载基座模型 {base_model} ..."})
    runtime.ensure(base_model, model_dir)

    log({"type": "stage", "text": "计算基座模型困惑度..."})
    base_ppl = _clean_metric(runtime.perplexity(eval_records, use_adapter=False))

    log({"type": "stage", "text": "启用 LoRA 适配器..."})
    runtime.activate_adapter(adapter_dir, base_model, model_dir)

    log({"type": "stage", "text": "计算微调后困惑度..."})
    tuned_ppl = _clean_metric(runtime.perplexity(eval_records, use_adapter=True))

    prompts = build_prompts(eval_records, max_prompts)
    comparisons = []
    for index, prompt in enumerate(prompts, start=1):
        log({"type": "stage", "text": f"生成对照样本 {index}/{len(prompts)}"})
        messages = [
            {"role": "system", "content": "你是一个乐于助人的中文助手，回答准确、简洁。"},
            {"role": "user", "content": prompt},
        ]
        base_answer = runtime.complete(messages, max_new_tokens=max_new_tokens, temperature=0.0, use_adapter=False)
        tuned_answer = runtime.complete(messages, max_new_tokens=max_new_tokens, temperature=0.0, use_adapter=True)
        reference = ""
        for rec in eval_records:
            if any(m["role"] == "user" and m["content"] == prompt for m in rec["messages"]):
                reference = rec["messages"][-1]["content"]
                break
        comparisons.append(
            {"prompt": prompt, "base": base_answer, "tuned": tuned_answer, "reference": reference}
        )

    report = {
        "adapter": adapter_dir.name,
        "base_model": base_model,
        "eval_samples": len(eval_records),
        "perplexity": {"base": base_ppl, "tuned": tuned_ppl},
        "improvement": None
        if not base_ppl or not tuned_ppl
        else round((base_ppl - tuned_ppl) / base_ppl * 100, 1),
        "comparisons": comparisons,
        "duration_s": round(time.time() - started, 1),
    }
    log({"type": "done", "report": report})
    return report
