"""应用配置与路径解析。"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}

# 让 huggingface_hub 走镜像（必须在导入 hub 之前生效）
_hf_endpoint = (CONFIG.get("hf") or {}).get("endpoint")
if _hf_endpoint:
    os.environ.setdefault("HF_ENDPOINT", _hf_endpoint)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_p = CONFIG.get("paths") or {}
DATA_DIR = ROOT / _p.get("data_dir", "data")
OUTPUT_DIR = ROOT / _p.get("output_dir", "outputs")
MODEL_DIR = ROOT / _p.get("model_dir", "models")
DB_PATH = ROOT / _p.get("db_path", "data/studio.db")
WEB_DIR = ROOT / "web"

DATASET_DIR = DATA_DIR / "datasets"
EVAL_DIR = OUTPUT_DIR / "evals"
JOB_DIR = OUTPUT_DIR / "jobs"

SERVER = CONFIG.get("server") or {}
BASE_MODELS = CONFIG.get("base_models") or {}
TRAIN_DEFAULTS = CONFIG.get("training") or {}
SANDBOX = CONFIG.get("sandbox") or {}
PERSONA = CONFIG.get("persona") or {}
INFERENCE = CONFIG.get("inference") or {}

for _dir in (DATA_DIR, DATASET_DIR, OUTPUT_DIR, MODEL_DIR, EVAL_DIR, JOB_DIR, DB_PATH.parent):
    _dir.mkdir(parents=True, exist_ok=True)


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__").replace("\\", "__")


def local_model_path(model_id: str) -> Path:
    return MODEL_DIR / model_slug(model_id)


def model_status(model_id: str) -> dict:
    path = local_model_path(model_id)
    ready = (path / "config.json").exists()
    size = 0
    if path.exists():
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return {"id": model_id, "downloaded": ready, "path": str(path), "bytes": size}
