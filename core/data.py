"""数据管道：解析上传文件 -> 统一为 chat 格式 -> 校验/去重/切分。"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
from pathlib import Path
from typing import Any, Iterable

MAX_CHARS = 4000
SUPPORTED_ROLES = {"system", "user", "assistant"}


class DataError(ValueError):
    """数据格式不合法。"""


# ---------------------------------------------------------------- 规范化

def _s(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _pair_to_messages(user: str, assistant: str, system: str = "") -> list[dict]:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})
    return messages


def record_to_messages(rec: dict) -> list[dict] | None:
    """把多种常见样本格式统一成 messages 列表。"""
    if not isinstance(rec, dict):
        return None

    # 1) OpenAI / chat 格式
    if isinstance(rec.get("messages"), list):
        messages = []
        for item in rec["messages"]:
            if not isinstance(item, dict):
                continue
            role = _s(item.get("role")).lower()
            content = _s(item.get("content"))
            if role in SUPPORTED_ROLES and content:
                messages.append({"role": role, "content": content})
        if not any(m["role"] == "assistant" for m in messages):
            return None
        if not any(m["role"] == "user" for m in messages):
            return None
        return messages

    # 2) 指令微调格式
    if _s(rec.get("instruction")) or _s(rec.get("question")):
        instruction = _s(rec.get("instruction")) or _s(rec.get("question"))
        extra = _s(rec.get("input")) or _s(rec.get("context"))
        output = _s(rec.get("output")) or _s(rec.get("answer")) or _s(rec.get("response"))
        system = _s(rec.get("system"))
        if not output:
            return None
        user = f"{instruction}\n\n{extra}" if extra else instruction
        return _pair_to_messages(user, output, system)

    # 3) prompt / response 格式
    prompt = _s(rec.get("prompt")) or _s(rec.get("query"))
    response = _s(rec.get("response")) or _s(rec.get("completion")) or _s(rec.get("chosen"))
    if prompt and response:
        return _pair_to_messages(prompt, response, _s(rec.get("system")))

    return None


def normalize(records: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """返回 (合法样本, 错误列表)。"""
    kept: list[dict] = []
    errors: list[dict] = []
    seen: set[str] = set()

    for idx, rec in enumerate(records):
        messages = record_to_messages(rec)
        if messages is None:
            errors.append({"row": idx, "reason": "无法识别格式（需含 user/assistant 内容）"})
            continue
        if any(len(m["content"]) > MAX_CHARS for m in messages):
            errors.append({"row": idx, "reason": f"单条消息超过 {MAX_CHARS} 字符"})
            continue
        if not _s(messages[-1]["content"]):
            errors.append({"row": idx, "reason": "最后一条 assistant 内容为空"})
            continue

        digest = hashlib.sha1(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        kept.append({"messages": messages})

    return kept, errors


# ---------------------------------------------------------------- 解析入口

def parse_bytes(filename: str, raw: bytes) -> tuple[list[dict], list[dict]]:
    suffix = Path(filename).suffix.lower()
    text = raw.decode("utf-8-sig", errors="replace")

    if suffix in (".jsonl", ".ndjson"):
        records = _parse_jsonl(text)
    elif suffix == ".json":
        records = _parse_json(text)
    elif suffix in (".csv", ".tsv"):
        records = _parse_csv(text, delimiter="," if suffix == ".csv" else "\t")
    elif suffix in (".txt", ".text"):
        records = _parse_txt(text)
    else:
        raise DataError(f"不支持的文件类型：{suffix or '未知'}（支持 jsonl/json/csv/tsv/txt）")

    return normalize(records)


def _parse_jsonl(text: str) -> list[dict]:
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _parse_json(text: str) -> list[dict]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DataError(f"JSON 解析失败：{exc}") from exc
    if isinstance(payload, dict):
        for key in ("data", "samples", "records", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    if isinstance(payload, list):
        return payload
    raise DataError("JSON 顶层需要是数组或对象")


def _parse_csv(text: str, delimiter: str = ",") -> list[dict]:
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    rows = []
    for row in reader:
        rows.append({(k or "").strip().lower(): v for k, v in row.items()})
    return rows


def _parse_txt(text: str) -> list[dict]:
    """每行 `问题<TAB>答案` 或 `问题|||答案`。"""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            q, a = line.split("\t", 1)
        elif "|||" in line:
            q, a = line.split("|||", 1)
        else:
            continue
        rows.append({"instruction": q, "output": a})
    return rows


# ---------------------------------------------------------------- 落盘 / 切分

def save_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def split(records: list[dict], eval_ratio: float = 0.1, seed: int = 42) -> tuple[list[dict], list[dict]]:
    if not records:
        return [], []
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 8 or eval_ratio <= 0:
        return shuffled, []
    n_eval = max(1, min(int(len(shuffled) * eval_ratio), len(shuffled) - 1))
    return shuffled[n_eval:], shuffled[:n_eval]
