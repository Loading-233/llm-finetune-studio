"""线上口径评测：直接请求 /api/chat，走与网页完全相同的链路（人设、检索、思考、历史）。

与 eval_bench.py 的区别：eval_bench 直接调 runtime，缺思考段、参数也不同，分数不可比。
本脚本只走 HTTP，不加载任何模型，因此不会和服务进程抢内存。

用法（默认即为对齐口径：无思考、贪心、320 token、normal）：
    .\\.venv\\Scripts\\python.exe scripts\\eval_live.py --limit 3
    .\\.venv\\Scripts\\python.exe scripts\\eval_live.py --thought --verbosity detail --max-new-tokens 768 --temperature 0.7
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "eval" / "benchmark.jsonl"
REPORT_DIR = ROOT / "data" / "eval" / "live_reports"
DEFAULT_URL = "http://127.0.0.1:8765/api/chat"


def load_bench(limit: int | None = None, only: str = "", bench: str = "") -> list[dict]:
    path = Path(bench) if bench else BENCH
    if not path.is_absolute():
        path = ROOT / path
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if only:
        wanted = {part.strip() for part in only.split(",") if part.strip()}
        items = [item for item in items if item["id"] in wanted]
    return items[:limit] if limit else items


def kb_doc_count() -> int:
    """检索索引里的文档条数：与 forge.retrieval._load_docs 同口径（按答案去重）。用于报告标注知识库版本。"""
    demo = ROOT / "data" / "demo"
    if not demo.exists():
        return 0
    answers: set[str] = set()
    for path in demo.rglob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                messages = json.loads(line).get("messages") or []
            except json.JSONDecodeError:
                continue
            if len(messages) >= 2:
                answers.add(str(messages[-1].get("content", "")).strip())
    return len(answers)


def ask(url: str, question: str, args) -> tuple[str, dict]:
    """发一次对话请求，返回（最终回答, 事件统计）。"""
    payload = json.dumps(
        {
            "messages": [{"role": "user", "content": question}],
            "adapter_id": args.adapter or None,
            "use_adapter": bool(args.adapter),
            "use_retrieval": not args.no_retrieval,
            "show_thought": args.thought,
            "verbosity": args.verbosity,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": 0.9,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    answer: list[str] = []
    stats = {"thought_chars": 0, "hits": 0, "hit_questions": [], "elapsed": "", "thought_elapsed": 0.0, "total": 0.0, "error": ""}
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            if kind == "delta":
                answer.append(event.get("text", ""))
            elif kind == "thought":
                stats["thought_chars"] += len(event.get("text", ""))
            elif kind == "thinking" and event.get("hits"):
                stats["hits"] = len(event["hits"])
                stats["hit_questions"] = [str(hit.get("question", "")) for hit in event["hits"]]
            elif kind == "done":
                stats["elapsed"] = str(event.get("elapsed", ""))
                stats["thought_elapsed"] = float(event.get("thought_elapsed") or 0.0)
                stats["total"] = float(event.get("total") or 0.0)
            elif kind == "error":
                stats["error"] = str(event.get("message") or event.get("text") or "未知错误")
    return "".join(answer).strip(), stats


def score(answer: str, checks: list[list[str]]) -> tuple[float, bool]:
    text = answer.lower()
    hits = [group for group in checks if any(word.lower() in text for word in group)]
    ratio = len(hits) / max(1, len(checks))
    return ratio, ratio >= 0.999


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", default="", help="只跑指定题号，逗号分隔，如 d04,t01")
    parser.add_argument("--bench", default="", help="题库路径，默认 data/eval/benchmark.jsonl")
    parser.add_argument("--verbosity", default="normal", choices=["short", "normal", "detail"])
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument("--thought", action="store_true", help="开启 4 段式思考（只用于测延迟）")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    items = load_bench(args.limit or None, args.only, args.bench)
    rows, total_hit = [], 0.0
    print(f"口径：retrieval={'off' if args.no_retrieval else 'on'} thought={'on' if args.thought else 'off'} "
          f"verbosity={args.verbosity} max_new_tokens={args.max_new_tokens} temperature={args.temperature} "
          f"kb_docs={kb_doc_count()}")

    for item in items:
        started = time.time()
        try:
            answer, stats = ask(args.url, item["question"], args)
            error = stats["error"]
        except urllib.error.HTTPError as exc:
            answer, stats, error = "", {"thought_chars": 0, "hits": 0, "elapsed": ""}, f"HTTP {exc.code}"
        except Exception as exc:  # noqa: BLE001
            answer, stats, error = "", {"thought_chars": 0, "hits": 0, "elapsed": ""}, str(exc)

        ratio, passed = (0.0, False) if error else score(answer, item["checks"])
        total_hit += ratio
        wall = round(time.time() - started, 1)
        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "subcategory": item.get("subcategory", ""),
                "question": item["question"],
                "answer": answer,
                "hit": round(ratio, 2),
                "pass": passed,
                "seconds": wall,
                "answer_s": float(stats.get("elapsed") or 0.0),
                "thought_s": stats.get("thought_elapsed", 0.0),
                "server_total_s": stats.get("total", 0.0),
                "chars": len(answer),
                "thought_chars": stats.get("thought_chars", 0),
                "retrieval_hits": stats.get("hits", 0),
                "hit_questions": stats.get("hit_questions", []),
                "error": error,
            }
        )
        flag = "ERR" if error else ("PASS" if passed else "FAIL")
        print(f"{item['id']} {item['category']} hit={ratio:.2f} {flag} {wall}s "
              f"thought={rows[-1]['thought_chars']}字 | {error or answer[:40]}")

    passed_count = sum(1 for row in rows if row["pass"])
    summary = {
        "url": args.url,
        "bench": args.bench or str(BENCH.relative_to(ROOT)).replace("\\", "/"),
        "kb_docs": kb_doc_count(),
        "adapter": args.adapter or "merged-gguf",
        "retrieval": not args.no_retrieval,
        "thought": args.thought,
        "verbosity": args.verbosity,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "items": len(rows),
        "pass_rate": round(passed_count / max(1, len(rows)), 3),
        "avg_hit": round(total_hit / max(1, len(rows)), 3),
        "avg_seconds": round(sum(row["seconds"] for row in rows) / max(1, len(rows)), 1),
        "avg_answer_s": round(sum(row["answer_s"] for row in rows) / max(1, len(rows)), 1),
        "avg_thought_s": round(sum(row["thought_s"] for row in rows) / max(1, len(rows)), 1),
        "avg_server_total_s": round(sum(row["server_total_s"] for row in rows) / max(1, len(rows)), 1),
        "avg_thought_chars": round(sum(row["thought_chars"] for row in rows) / max(1, len(rows)), 1),
        "errors": sum(1 for row in rows if row["error"]),
        "rows": rows,
    }

    out = Path(args.out) if args.out else REPORT_DIR / f"live_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n通过率 {summary['pass_rate'] * 100:.0f}%  平均命中 {summary['avg_hit'] * 100:.0f}%  "
          f"平均 {summary['avg_seconds']}s/题  思考均 {summary['avg_thought_chars']}字  "
          f"错误 {summary['errors']}  kb_docs={summary['kb_docs']}")
    print(f"报告：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
