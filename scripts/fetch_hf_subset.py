"""从 HF 镜像流式拉取公开指令数据集，抽样成项目训练格式。

用法：python scripts/fetch_hf_subset.py <repo> <file> <out_name> <count>
示例：python scripts/fetch_hf_subset.py m-a-p/COIG-CQIA COIG-CQIA-full.jsonl coig_600 600
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "eval"
MIRROR = "https://hf-mirror.com/datasets/{repo}/resolve/main/{file}"
SEED = 42


def pick(item: dict) -> dict | None:
    instruction = str(item.get("instruction") or "").strip()
    extra = str(item.get("input") or "").strip()
    output = str(item.get("output") or item.get("response") or item.get("answer") or "").strip()
    if not (8 <= len(instruction) <= 200):
        return None
    if not (40 <= len(output) <= 1500):
        return None
    user = f"{instruction}\n\n{extra}" if extra else instruction
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": output}]}


def main() -> int:
    repo, filename, out_name, count = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    url = MIRROR.format(repo=repo, file=filename)
    rng = random.Random(SEED)
    seen: set[str] = set()
    pool: list[dict] = []
    total = 0
    kept = 0

    with requests.get(url, stream=True, timeout=(15, 120)) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=False):
            if not raw:
                continue
            total += 1
            try:
                row = json.loads(raw.decode("utf-8", "ignore"))
            except json.JSONDecodeError:
                continue
            rec = pick(row)
            if not rec:
                continue
            key = rec["messages"][0]["content"]
            if key in seen:
                continue
            seen.add(key)
            kept += 1
            if len(pool) < count:
                pool.append(rec)
            else:
                j = rng.randrange(kept)
                if j < count:
                    pool[j] = rec

    rng.shuffle(pool)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{out_name}.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for rec in pool:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    avg = sum(len(r["messages"][1]["content"]) for r in pool) / max(1, len(pool))
    print(f"read={total} valid={kept} sampled={len(pool)} avg_answer={avg:.0f}chars")
    print(f"out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
