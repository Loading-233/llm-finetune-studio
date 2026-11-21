"""把消息样本编码成训练张量（仅对 assistant 回复计损失）。"""

from __future__ import annotations

from typing import Any, Sequence

IGNORE_INDEX = -100


def _extract_masks(enc: dict) -> list[int] | None:
    for key in ("assistant_masks", "assistant_tokens_mask"):
        value = enc.get(key)
        if value:
            return list(value)
    return None


def _prompt_length(tokenizer, messages: Sequence[dict]) -> int:
    """chat template 不含 {% generation %} 时，用「只编码提问前缀」来定位回复起点。"""
    if not messages or messages[-1].get("role") != "assistant":
        return 0
    try:
        prefix = tokenizer.apply_chat_template(
            list(messages[:-1]), tokenize=True, add_generation_prompt=True
        )
    except Exception:
        return 0
    return len(prefix) if isinstance(prefix, list) else 0


def build_example(tokenizer, messages: Sequence[dict], max_seq_len: int = 1024) -> dict[str, list[int]]:
    """返回 input_ids / attention_mask / labels。assistant 之外的位置标签置为 -100。"""
    try:
        enc = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        input_ids = list(enc["input_ids"])
        attention_mask = list(enc.get("attention_mask") or [1] * len(input_ids))
        assistant_mask = _extract_masks(enc)
    except Exception:
        enc = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )
        input_ids = list(enc["input_ids"])
        attention_mask = list(enc.get("attention_mask") or [1] * len(input_ids))
        assistant_mask = None

    if assistant_mask and len(assistant_mask) == len(input_ids):
        labels = [tok if flag else IGNORE_INDEX for tok, flag in zip(input_ids, assistant_mask)]
    else:
        labels = list(input_ids)
        prompt_len = _prompt_length(tokenizer, messages)
        for index in range(min(prompt_len, len(labels))):
            labels[index] = IGNORE_INDEX

    if all(label == IGNORE_INDEX for label in labels):
        labels = list(input_ids)

    input_ids = input_ids[:max_seq_len]
    attention_mask = attention_mask[:max_seq_len]
    labels = labels[:max_seq_len]

    # 至少保留一个有效标签，否则该样本无梯度
    if all(label == IGNORE_INDEX for label in labels):
        labels[-1] = input_ids[-1]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate(batch: Sequence[dict[str, list[int]]], pad_token_id: int) -> dict[str, Any]:
    import torch

    width = max(len(item["input_ids"]) for item in batch)
    input_ids, attention_mask, labels = [], [], []
    for item in batch:
        pad = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad)
        attention_mask.append(item["attention_mask"] + [0] * pad)
        labels.append(item["labels"] + [IGNORE_INDEX] * pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
