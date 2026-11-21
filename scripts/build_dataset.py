"""把多个知识问答文件合成一个更适合训练的数据集：去重 + 少量问法改写。"""

from __future__ import annotations

import json
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, db  # noqa: E402

TEMPLATES = [
    "请问{q}",
    "我想了解一下：{q}",
    "{q}能简要说明一下吗？",
    "麻烦告诉我，{q}",
    "{q}（一句话回答即可）",
]

DEMO_DIR = ROOT / "data" / "demo"


def load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        messages = [m for m in (payload.get("messages") or []) if m.get("role") in ("user", "assistant")]
        if len(messages) < 2:
            continue
        user = messages[-2]["content"].strip()
        assistant = messages[-1]["content"].strip()
        if user and assistant:
            records.append({"user": user, "assistant": assistant})
    return records


def first_sentence(text: str) -> str:
    for mark in ("。", "！", "？"):
        index = text.find(mark)
        if 0 < index < 70:
            return text[: index + 1]
    return text


def main() -> int:
    sources = [DEMO_DIR / "knowledge_demo.jsonl", DEMO_DIR / "knowledge_v2.jsonl"]
    sources += sorted((DEMO_DIR / "domains").glob("*.jsonl"))
    merged: dict[str, dict] = {}
    for path in sources:
        if not path.exists():
            continue
        for record in load_records(path):
            merged.setdefault(record["user"], record)

    base = list(merged.values())
    print(f"去重后基础问答：{len(base)} 条")

    rng = random.Random(42)
    expanded: list[dict] = list(base)
    for index, record in enumerate(base):
        question = record["user"]
        template = TEMPLATES[index % len(TEMPLATES)]
        variant = template.format(q=question)
        if variant != question:
            expanded.append({"user": variant, "assistant": record["assistant"]})

        # 20% 的条目补一条「简短回答」，训练模型学会控制长度
        if index % 5 == 0:
            short = first_sentence(record["assistant"])
            if short != record["assistant"]:
                expanded.append({"user": question, "assistant": short})

    rng.shuffle(expanded)
    records = [
        {"messages": [{"role": "user", "content": item["user"]}, {"role": "assistant", "content": item["assistant"]}]}
        for item in expanded
    ]

    db.init()
    dataset_id = uuid.uuid4().hex[:12]
    path = config.DATASET_DIR / f"{dataset_id}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    db.insert(
        "datasets",
        {
            "id": dataset_id,
            "name": f"多领域指令集（{len(records)} 条）",
            "path": str(path),
            "num_samples": len(records),
            "stats_json": json.dumps({"source": "demo+v2", "base_facts": len(base)}, ensure_ascii=False),
            "created_at": db.now(),
        },
    )
    print(f"已生成数据集 {dataset_id}：{len(records)} 条 → {path}")
    print(f"基础事实 {len(base)} 条，问法变体 {len(records) - len(base)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
