"""重复评测：同一题跑 N 次，看通过率分布与耗时中位数，用来判断结果是稳定还是抽样。

只走 HTTP，口径与 eval_live.py 完全一致（复用它的 ask / score / load_bench）。

用法：
    .\\.venv\\Scripts\\python.exe scripts\\eval_repeat.py --only p02 --thought --verbosity detail --max-new-tokens 768 --repeat 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
from pathlib import Path

import eval_live
from eval_live import ask, kb_doc_count, load_bench, score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=eval_live.DEFAULT_URL)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--only", default="", help="题号，逗号分隔；留空表示全部")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bench", default="", help="题库路径，默认 data/eval/benchmark.jsonl")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--verbosity", default="normal", choices=["short", "normal", "detail"])
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument("--thought", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    items = load_bench(args.limit or None, args.only, args.bench)
    if not items:
        print("没有匹配的题目")
        return 1

    print(
        f"重复 {args.repeat} 次 | retrieval={'off' if args.no_retrieval else 'on'} "
        f"thought={'on' if args.thought else 'off'} verbosity={args.verbosity} "
        f"max_new_tokens={args.max_new_tokens} temperature={args.temperature} kb_docs={kb_doc_count()}"
    )

    results = []
    for item in items:
        runs = []
        for index in range(args.repeat):
            started = time.time()
            try:
                answer, stats = ask(args.url, item["question"], args)
                error = stats["error"]
            except urllib.error.HTTPError as exc:
                answer, stats, error = "", {}, f"HTTP {exc.code}"
            except Exception as exc:  # noqa: BLE001
                answer, stats, error = "", {}, str(exc)

            ratio, passed = (0.0, False) if error else score(answer, item["checks"])
            runs.append(
                {
                    "run": index + 1,
                    "answer": answer,
                    "hit": round(ratio, 2),
                    "pass": passed,
                    "seconds": round(time.time() - started, 1),
                    "answer_s": float(stats.get("elapsed") or 0.0),
                    "thought_s": stats.get("thought_elapsed", 0.0),
                    "thought_chars": stats.get("thought_chars", 0),
                    "retrieval_hits": stats.get("hits", 0),
                    "error": error,
                }
            )
            flag = "ERR" if error else ("PASS" if passed else "FAIL")
            print(f"  {item['id']} #{index + 1} {flag} hit={ratio:.2f} {runs[-1]['seconds']}s "
                  f"thought={runs[-1]['thought_chars']}字 | {error or answer[:36]}")

        passes = sum(1 for run in runs if run["pass"])
        seconds = [run["seconds"] for run in runs]
        summary = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "repeat": args.repeat,
            "pass_count": passes,
            "pass_rate": round(passes / max(1, len(runs)), 3),
            "hit_values": [run["hit"] for run in runs],
            "seconds_median": round(statistics.median(seconds), 1),
            "seconds_min": min(seconds),
            "seconds_max": max(seconds),
            "thought_chars": [run["thought_chars"] for run in runs],
            "retrieval_hits": sorted({run["retrieval_hits"] for run in runs}),
            "errors": sum(1 for run in runs if run["error"]),
            "runs": runs,
        }
        results.append(summary)
        print(f"  → {item['id']} {passes}/{len(runs)} 通过，耗时中位数 {summary['seconds_median']}s，"
              f"hit 分布 {summary['hit_values']}\n")

    payload = {
        "url": args.url,
        "kb_docs": kb_doc_count(),
        "adapter": args.adapter or "merged-gguf",
        "retrieval": not args.no_retrieval,
        "thought": args.thought,
        "verbosity": args.verbosity,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "repeat": args.repeat,
        "items": results,
    }
    out = Path(args.out) if args.out else eval_live.REPORT_DIR / f"repeat_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("汇总：")
    for row in results:
        print(f"  {row['id']}: {row['pass_count']}/{row['repeat']} 通过，"
              f"中位 {row['seconds_median']}s（{row['seconds_min']}–{row['seconds_max']}s）")
    print(f"报告：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
