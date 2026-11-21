"""固定评测集跑分：关键词命中率 + 通过率，每次训练后跑一次即可横向对比。"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, db  # noqa: E402
from core.runtime import runtime  # noqa: E402
from forge import retrieval  # noqa: E402

BENCH = config.DATA_DIR / "eval" / "benchmark.jsonl"
NOTE = "参考资料（与问题相关时优先据此回答，不要编造数字）：\n{block}"


def load_bench(limit: int | None = None) -> list[dict]:
    items = []
    for line in BENCH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items[:limit] if limit else items


def context_for(question: str) -> str:
    hits = retrieval.search(question, top_k=2)
    hits = [hit for hit in hits if hit.get("score", 0) >= 0.55]
    if not hits:
        return ""
    block = "\n".join(f"{i}. 问：{h['question']} 答：{h['answer']}" for i, h in enumerate(hits, 1))
    return NOTE.format(block=block)


def ask(question: str, use_retrieval: bool) -> str:
    system = (config.PERSONA.get("system") or "").strip() or "你是星尘笔记的产品助手，回答简洁准确。"
    if use_retrieval:
        extra = context_for(question)
        if extra:
            system = system + "\n\n" + extra
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    parts = []
    for chunk in runtime.stream(messages, max_new_tokens=220, temperature=0.0, top_p=1.0, use_adapter=True):
        if chunk.startswith("\x00"):
            continue
        parts.append(chunk)
    return "".join(parts).strip()


def score(answer: str, checks: list[list[str]]) -> tuple[float, bool]:
    """每个 check 是一组同义词，命中任意一个即算通过该检查点。"""
    text = answer.lower()
    hits = [group for group in checks if any(word.lower() in text for word in group)]
    ratio = len(hits) / max(1, len(checks))
    return ratio, ratio >= 0.999


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    adapter = db.one("adapters", args.adapter) if args.adapter else None
    if args.adapter and not adapter:
        print("适配器不存在")
        return 1
    base_model = (adapter or {}).get("base_model") or config.BASE_MODELS.get("default")
    if adapter:
        runtime.activate_adapter(adapter["path"], base_model, config.MODEL_DIR, adapter_name=args.adapter)
    else:
        runtime.ensure(base_model, config.MODEL_DIR)

    items = load_bench(args.limit or None)
    rows, totals = [], 0.0
    for item in items:
        started = time.time()
        answer = ask(item["question"], not args.no_retrieval)
        ratio, passed = score(answer, item["checks"])
        totals += ratio
        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "answer": answer,
                "hit": round(ratio, 2),
                "pass": passed,
                "seconds": round(time.time() - started, 1),
            }
        )
        print(f"{item['id']} {item['category']} hit={ratio:.2f} {'PASS' if passed else 'FAIL'} {rows[-1]['seconds']}s | {answer[:40]}")

    passed = sum(1 for row in rows if row["pass"])
    summary = {
        "adapter": args.adapter or "base",
        "retrieval": not args.no_retrieval,
        "items": len(rows),
        "pass_rate": round(passed / max(1, len(rows)), 3),
        "avg_hit": round(totals / max(1, len(rows)), 3),
        "rows": rows,
    }
    out = Path(args.out) if args.out else config.EVAL_DIR / f"bench_{uuid.uuid4().hex[:8]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    db.init()
    db.insert(
        "evals",
        {
            "id": uuid.uuid4().hex[:12],
            "adapter_id": args.adapter or None,
            "dataset_id": None,
            "status": "success",
            "report_json": json.dumps(summary, ensure_ascii=False),
            "created_at": db.now(),
        },
    )
    print(f"\n通过率 {summary['pass_rate'] * 100:.0f}%  平均命中 {summary['avg_hit'] * 100:.0f}%  报告：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
