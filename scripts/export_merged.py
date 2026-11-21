"""把 1.5B 基座与 LoRA 适配器合并，导出成 fp16 HF 模型，供转 GGUF 使用。"""

from __future__ import annotations

import gc
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_DIR = ROOT / "models" / "Qwen__Qwen2.5-1.5B-Instruct"
OUT_DIR = ROOT / "models" / "gguf" / "merged-1.5b"
DB = ROOT / "data" / "studio.db"
ADAPTER_ID = sys.argv[1] if len(sys.argv) > 1 else "e58236ad9711"


def adapter_path() -> str:
    conn = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    row = conn.execute("select path from adapters where id = ?", (ADAPTER_ID,)).fetchone()
    conn.close()
    if not row:
        raise SystemExit(f"找不到适配器 {ADAPTER_ID}")
    return row[0]


def main() -> None:
    torch.set_num_threads(8)
    print("adapter:", adapter_path(), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=torch.float32, low_cpu_mem_usage=True
    )
    merged = PeftModel.from_pretrained(model, adapter_path())
    merged = merged.merge_and_unload()
    merged = merged.to(torch.float16)
    merged.eval()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(OUT_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(OUT_DIR))
    del merged, model
    gc.collect()
    total = sum(f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file())
    print(f"done {OUT_DIR} {round(total / 1024**3, 2)} GB", flush=True)


if __name__ == "__main__":
    main()
