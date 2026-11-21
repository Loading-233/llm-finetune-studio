"""把公开指令数据集（alpaca-gpt4-zh）与本地多领域数据合并成一个训练集。"""

from __future__ import annotations

import json
import random
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, db  # noqa: E402
from core.data import load_jsonl  # noqa: E402

PUBLIC_FILE = ROOT / "data" / "public" / "alpaca_gpt4_zh.json"
MAX_PUBLIC = 2200
MAX_LOCAL = 700
SEED = 42


def build_public(limit: int) -> list[dict]:
    payload = json.loads(PUBLIC_FILE.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("data") or []
    random.Random(SEED).shuffle(payload)

    records, seen = [], set()
    for item in payload:
        instruction = str(item.get("instruction") or "").strip()
        extra = str(item.get("input") or "").strip()
        output = str(item.get("output") or "").strip()
        if not (5 <= len(instruction) <= 200) or not (20 <= len(output) <= 1200):
            continue
        key = (instruction, extra)
        if key in seen:
            continue
        seen.add(key)
        user = f"{instruction}\n\n{extra}" if extra else instruction
        records.append(
            {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": output}]}
        )
        if len(records) >= limit:
            break
    return records


def build_local(limit: int) -> list[dict]:
    files = sorted(
        (config.DATA_DIR / "datasets").glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in files:
        records = load_jsonl(path)
        if records:
            random.Random(SEED).shuffle(records)
            return records[:limit]
    return []


def main() -> int:
    if not PUBLIC_FILE.exists():
        print("公开数据集不存在，先下载")
        return 1
    public = build_public(MAX_PUBLIC)
    local = build_local(MAX_LOCAL)
    merged = public + local
    random.Random(SEED).shuffle(merged)

    dataset_id = uuid.uuid4().hex[:12]
    path = config.DATASET_DIR / f"{dataset_id}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for record in merged:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    db.init()
    db.insert(
        "datasets",
        {
            "id": dataset_id,
            "name": f"公开指令集 alpaca-gpt4 + 多领域（{len(merged)} 条）",
            "path": str(path),
            "num_samples": len(merged),
            "stats_json": json.dumps(
                {"public": len(public), "local": len(local), "source": "shibing624/alpaca-zh"},
                ensure_ascii=False,
            ),
            "created_at": db.now(),
        },
    )
    print(f"已生成 {dataset_id}：公开 {len(public)} + 本地 {len(local)} = {len(merged)} 条")
    print(f"路径：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
