"""把数据库里记录的绝对路径改到当前项目目录，方便整个项目拷到别的机器/U 盘后直接使用。"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import db  # noqa: E402

PATTERN = re.compile(r"[A-Za-z]:[\\/].*?[\\/]llm-finetune-studio", re.I)
TARGETS = (("datasets", "path"), ("jobs", "output_dir"), ("adapters", "path"))


def main() -> int:
    db.init()
    changed = 0
    for table, column in TARGETS:
        for row in db.query(f"SELECT id, {column} AS value FROM {table}"):
            old = row["value"] or ""
            new = PATTERN.sub(lambda _match: str(ROOT), old)
            if new != old:
                db.execute(f"UPDATE {table} SET {column} = ? WHERE id = ?", (new, row["id"]))
                changed += 1
    print(f"已修正 {changed} 条路径 -> {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
