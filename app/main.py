"""FastAPI 服务：数据集管理、训练任务、评测、对话。"""

from __future__ import annotations

import sys
from pathlib import Path

# 直接运行本文件时（如 PyCharm 里点运行），把项目根目录放到 sys.path 最前面，
# 避免被 site-packages 里同名的第三方包 app 遮蔽
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import re
import threading
import time
import traceback
import uuid
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import chat_actions, config, db
from core import llamacpp
from core.data import DataError, load_jsonl, parse_bytes, save_jsonl
from core.evaluate import evaluate_adapter
from core.finetune import TrainConfig, run_finetune
from core.runtime import device_report, load_system_messages, runtime
from forge import pipeline as forge_pipeline
from forge import retrieval as forge_retrieval
from sandbox import files as sandbox_io
from sandbox import guard as sandbox_guard

app = FastAPI(title="铸语 Cast", version="1.0.0")
db.init()

# CPU 推理优化：线程数，参数来自 config.yaml 的 inference 段
runtime.configure(threads=int(config.INFERENCE.get("threads") or 0))

TASK_LOCK = threading.Lock()
TASK_STATE: dict[str, Any] = {"busy": False, "kind": None, "id": None}
DOWNLOAD_STATE: dict[str, Any] = {"status": "idle", "model": None, "message": "", "bytes": 0, "total": 0}


# ---------------------------------------------------------------- 工具

class TaskLog:
    """把任务事件同时写进日志文件与数据库。"""

    def __init__(self, path: Path, job_id: str | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.job_id = job_id
        self._lock = threading.Lock()
        self.count = 0

    def emit(self, event: dict) -> None:
        line = {"ts": time.strftime("%H:%M:%S"), **event}
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            self.count += 1
        if self.job_id:
            if event.get("type") == "step" and event.get("max_steps"):
                progress = min(99.0, round(event["step"] / event["max_steps"] * 100, 1))
                db.update(
                    "jobs",
                    self.job_id,
                    {
                        "progress": progress,
                        "message": f"step {event['step']}/{event['max_steps']}",
                        "updated_at": db.now(),
                    },
                )
            elif event.get("type") == "stage":
                db.update(
                    "jobs", self.job_id, {"message": event.get("text", ""), "updated_at": db.now()}
                )


def _start_task(kind: str, task_id: str) -> bool:
    global TASK_STATE
    if not TASK_LOCK.acquire(blocking=False):
        return False
    TASK_STATE = {"busy": True, "kind": kind, "id": task_id}
    return True


def _finish_task() -> None:
    global TASK_STATE
    TASK_STATE = {"busy": False, "kind": None, "id": None}
    try:
        TASK_LOCK.release()
    except RuntimeError:
        pass


def _require_model(model_id: str) -> None:
    status = config.model_status(model_id)
    if not status["downloaded"]:
        raise HTTPException(
            status_code=409,
            detail=f"基座模型 {model_id} 尚未下载，请先在右上角点击“下载模型”",
        )


def _dataset_row(dataset_id: str) -> dict:
    row = db.one("datasets", dataset_id)
    if not row:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return row


# ---------------------------------------------------------------- 基础接口

@app.get("/")
def index():
    return FileResponse(config.WEB_DIR / "index.html")


@app.get("/api/system")
def system_info():
    return {
        "device": device_report(),
        "runtime": runtime.status(),
        "busy": TASK_STATE,
        "download": DOWNLOAD_STATE,
        "models": [config.model_status(item) for item in config.BASE_MODELS.get("options", [])],
        "default_model": config.BASE_MODELS.get("default"),
        "train_defaults": config.TRAIN_DEFAULTS,
    }


# ---------------------------------------------------------------- 数据集

@app.get("/api/datasets")
def datasets():
    return [{**row, "stats": json.loads(row.get("stats_json") or "{}")} for row in db.list_datasets()]


@app.get("/api/datasets/{dataset_id}/preview")
def preview(dataset_id: str, limit: int = 30):
    row = _dataset_row(dataset_id)
    records = load_jsonl(Path(row["path"]))
    return {"total": len(records), "samples": records[:limit]}


@app.post("/api/datasets/import")
async def import_dataset(file: UploadFile = File(...), name: str = ""):
    raw = await file.read()
    try:
        records, errors = parse_bytes(file.filename or "upload.jsonl", raw)
    except DataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not records:
        raise HTTPException(status_code=400, detail="没有解析到有效样本")

    dataset_id = uuid.uuid4().hex[:12]
    path = config.DATASET_DIR / f"{dataset_id}.jsonl"
    save_jsonl(path, records)
    dataset_name = name or Path(file.filename or "数据集").stem
    db.insert(
        "datasets",
        {
            "id": dataset_id,
            "name": dataset_name,
            "path": str(path),
            "num_samples": len(records),
            "stats_json": json.dumps(
                {"skipped": len(errors), "errors": errors[:5], "source": file.filename},
                ensure_ascii=False,
            ),
            "created_at": db.now(),
        },
    )
    return {"id": dataset_id, "num_samples": len(records), "skipped": len(errors)}


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    row = _dataset_row(dataset_id)
    Path(row["path"]).unlink(missing_ok=True)
    db.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
    return {"ok": True}


class SamplePayload(BaseModel):
    instruction: str = Field(min_length=1)
    output: str = Field(min_length=1)
    system: str = ""


@app.post("/api/datasets/{dataset_id}/samples")
def add_sample(dataset_id: str, payload: SamplePayload):
    row = _dataset_row(dataset_id)
    records = load_jsonl(Path(row["path"]))
    messages = []
    if payload.system.strip():
        messages.append({"role": "system", "content": payload.system.strip()})
    messages.append({"role": "user", "content": payload.instruction.strip()})
    messages.append({"role": "assistant", "content": payload.output.strip()})
    records.append({"messages": messages})
    save_jsonl(Path(row["path"]), records)
    db.update("datasets", dataset_id, {"num_samples": len(records)})
    return {"ok": True, "num_samples": len(records)}


@app.post("/api/datasets/demo")
def import_demo():
    demo_path = config.DATA_DIR / "demo" / "knowledge_demo.jsonl"
    if not demo_path.exists():
        raise HTTPException(status_code=404, detail="示例数据集缺失")
    records = load_jsonl(demo_path)
    dataset_id = uuid.uuid4().hex[:12]
    path = config.DATASET_DIR / f"{dataset_id}.jsonl"
    save_jsonl(path, records)
    db.insert(
        "datasets",
        {
            "id": dataset_id,
            "name": "示例：星尘笔记产品知识",
            "path": str(path),
            "num_samples": len(records),
            "stats_json": json.dumps({"source": "demo"}, ensure_ascii=False),
            "created_at": db.now(),
        },
    )
    return {"id": dataset_id, "num_samples": len(records)}


# ---------------------------------------------------------------- 训练

class TrainPayload(BaseModel):
    dataset_id: str
    base_model: str | None = None
    epochs: float | None = None
    max_steps: int | None = None
    batch_size: int | None = None
    grad_accum: int | None = None
    lr: float | None = None
    lora_r: int | None = None
    lora_alpha: int | None = None
    max_seq_len: int | None = None


@app.post("/api/train")
def start_train(payload: TrainPayload):
    dataset = _dataset_row(payload.dataset_id)
    defaults = config.TRAIN_DEFAULTS
    base_model = payload.base_model or config.BASE_MODELS.get("default")
    _require_model(base_model)

    job_id = uuid.uuid4().hex[:12]
    output_dir = config.JOB_DIR / job_id
    train_cfg = TrainConfig(
        base_model=base_model,
        dataset_path=Path(dataset["path"]),
        output_dir=output_dir,
        model_dir=config.MODEL_DIR,
        max_seq_len=payload.max_seq_len or defaults.get("max_seq_len", 1024),
        epochs=payload.epochs if payload.epochs is not None else defaults.get("epochs", 3),
        max_steps=payload.max_steps if payload.max_steps is not None else defaults.get("max_steps", -1),
        batch_size=payload.batch_size or defaults.get("batch_size", 1),
        grad_accum=payload.grad_accum or defaults.get("grad_accum", 8),
        lr=payload.lr or defaults.get("lr", 2e-4),
        lora_r=payload.lora_r or defaults.get("lora_r", 8),
        lora_alpha=payload.lora_alpha or defaults.get("lora_alpha", 16),
        lora_dropout=defaults.get("lora_dropout", 0.05),
        eval_ratio=defaults.get("eval_ratio", 0.1),
        logging_steps=defaults.get("logging_steps", 1),
    )

    if not _start_task("train", job_id):
        raise HTTPException(status_code=409, detail="已有任务在运行，请等待完成")

    log = TaskLog(output_dir / "train.log", job_id)
    db.insert(
        "jobs",
        {
            "id": job_id,
            "dataset_id": dataset["id"],
            "base_model": base_model,
            "status": "running",
            "progress": 0,
            "message": "排队中",
            "output_dir": str(output_dir),
            "config_json": json.dumps(train_cfg.to_dict(), ensure_ascii=False),
            "metrics_json": "{}",
            "created_at": db.now(),
            "updated_at": db.now(),
        },
    )

    def worker() -> None:
        try:
            log.emit({"type": "stage", "text": "释放推理模型，准备训练..."})
            runtime.unload()
            meta = run_finetune(train_cfg, log.emit)
            adapter_id = uuid.uuid4().hex[:12]
            adapter_name = f"{dataset['name']}-{time.strftime('%m%d-%H%M')}"
            db.insert(
                "adapters",
                {
                    "id": adapter_id,
                    "name": adapter_name,
                    "path": str(output_dir / "adapter"),
                    "base_model": base_model,
                    "dataset_id": dataset["id"],
                    "job_id": job_id,
                    "meta_json": json.dumps(meta, ensure_ascii=False),
                    "created_at": db.now(),
                },
            )
            db.update(
                "jobs",
                job_id,
                {
                    "status": "success",
                    "progress": 100,
                    "message": "训练完成",
                    "adapter_id": adapter_id,
                    "metrics_json": json.dumps(meta, ensure_ascii=False),
                    "updated_at": db.now(),
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.emit({"type": "error", "text": str(exc), "traceback": traceback.format_exc()})
            db.update(
                "jobs",
                job_id,
                {"status": "failed", "message": str(exc)[:300], "updated_at": db.now()},
            )
        finally:
            _finish_task()

    threading.Thread(target=worker, name=f"train-{job_id}", daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs")
def jobs():
    rows = db.list_jobs()
    for row in rows:
        row["config"] = json.loads(row.get("config_json") or "{}")
        row["metrics"] = json.loads(row.get("metrics_json") or "{}")
    return rows


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str):
    row = db.one("jobs", job_id)
    if not row:
        raise HTTPException(status_code=404, detail="任务不存在")
    row["config"] = json.loads(row.get("config_json") or "{}")
    row["metrics"] = json.loads(row.get("metrics_json") or "{}")
    return row


@app.get("/api/jobs/{job_id}/log")
def job_log(job_id: str, after: int = 0):
    row = db.one("jobs", job_id)
    if not row:
        raise HTTPException(status_code=404, detail="任务不存在")
    path = Path(row["output_dir"]) / "train.log"
    lines: list[dict] = []
    if path.exists():
        raw = path.read_text(encoding="utf-8").splitlines()
        for line in raw[after:]:
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        next_offset = len(raw)
    else:
        next_offset = after
    return {"lines": lines, "next": next_offset, "status": row["status"]}


# ---------------------------------------------------------------- 适配器

@app.get("/api/adapters")
def adapters():
    rows = db.list_adapters()
    for row in rows:
        row["meta"] = json.loads(row.get("meta_json") or "{}")
    return rows


@app.delete("/api/adapters/{adapter_id}")
def delete_adapter(adapter_id: str):
    row = db.one("adapters", adapter_id)
    if not row:
        raise HTTPException(status_code=404, detail="适配器不存在")
    path = Path(row["path"])
    if path.exists():
        for child in path.rglob("*"):
            if child.is_file():
                child.unlink(missing_ok=True)
        path.rmdir()
    db.execute("DELETE FROM adapters WHERE id = ?", (adapter_id,))
    return {"ok": True}


# ---------------------------------------------------------------- 评测

class EvalPayload(BaseModel):
    adapter_id: str


@app.post("/api/eval")
def start_eval(payload: EvalPayload):
    adapter = db.one("adapters", payload.adapter_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="适配器不存在")
    eval_id = uuid.uuid4().hex[:12]
    if not _start_task("eval", eval_id):
        raise HTTPException(status_code=409, detail="已有任务在运行，请等待完成")

    db.insert(
        "evals",
        {
            "id": eval_id,
            "adapter_id": adapter["id"],
            "dataset_id": adapter["dataset_id"],
            "status": "running",
            "report_json": "{}",
            "created_at": db.now(),
        },
    )

    dataset = db.one("datasets", adapter["dataset_id"])
    log = TaskLog(config.EVAL_DIR / f"{eval_id}.log")

    def worker() -> None:
        try:
            report = evaluate_adapter(
                adapter_dir=adapter["path"],
                base_model=adapter["base_model"],
                dataset_path=dataset["path"],
                model_dir=config.MODEL_DIR,
                emit=log.emit,
            )
            db.update(
                "evals",
                eval_id,
                {"status": "success", "report_json": json.dumps(report, ensure_ascii=False)},
            )
        except Exception as exc:  # noqa: BLE001
            log.emit({"type": "error", "text": str(exc), "traceback": traceback.format_exc()})
            db.update(
                "evals",
                eval_id,
                {"status": "failed", "report_json": json.dumps({"error": str(exc)}, ensure_ascii=False)},
            )
        finally:
            _finish_task()

    threading.Thread(target=worker, name=f"eval-{eval_id}", daemon=True).start()
    return {"eval_id": eval_id}


@app.get("/api/evals/{eval_id}")
def eval_detail(eval_id: str):
    row = db.one("evals", eval_id)
    if not row:
        raise HTTPException(status_code=404, detail="评测不存在")
    row["report"] = json.loads(row.get("report_json") or "{}")
    return row


# ---------------------------------------------------------------- 模型下载

class DownloadPayload(BaseModel):
    model_id: str


@app.post("/api/models/download")
def download_model(payload: DownloadPayload):
    global DOWNLOAD_STATE
    if DOWNLOAD_STATE["status"] == "downloading":
        raise HTTPException(status_code=409, detail="已有模型正在下载")
    model_id = payload.model_id
    if model_id not in (config.BASE_MODELS.get("options") or []):
        raise HTTPException(status_code=400, detail="未知模型")
    if config.model_status(model_id)["downloaded"]:
        return {"status": "ready"}

    DOWNLOAD_STATE = {"status": "downloading", "model": model_id, "message": "连接仓库...", "bytes": 0, "total": 0}
    target = config.local_model_path(model_id)

    def worker() -> None:
        global DOWNLOAD_STATE
        from huggingface_hub import HfApi, snapshot_download

        error = None
        api = HfApi()
        total = 0
        try:
            info = api.model_info(model_id, files_metadata=True)
            allowed = {"json", "safetensors", "txt", "model", "jinja"}
            total = sum(
                (sibling.size or 0)
                for sibling in (info.siblings or [])
                if sibling.rfilename.split(".")[-1] in allowed and not sibling.rfilename.startswith("original/")
            )
            DOWNLOAD_STATE["total"] = total
            done = threading.Event()
            box: dict[str, Any] = {}

            def download() -> None:
                try:
                    snapshot_download(
                        repo_id=model_id,
                        local_dir=str(target),
                        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
                        ignore_patterns=["*.bin", "*.pth", "original/*"],
                        max_workers=4,
                    )
                except Exception as exc:  # noqa: BLE001
                    box["error"] = exc
                finally:
                    done.set()

            threading.Thread(target=download, daemon=True).start()
            while not done.wait(1.0):
                size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) if target.exists() else 0
                DOWNLOAD_STATE["bytes"] = size
                DOWNLOAD_STATE["message"] = f"{size / 1024**3:.2f} / {total / 1024**3:.2f} GB"
            error = box.get("error")
        except Exception as exc:  # noqa: BLE001
            error = exc

        if error or not (target / "config.json").exists():
            DOWNLOAD_STATE = {
                "status": "error",
                "model": model_id,
                "message": str(error)[:300] if error else "下载不完整",
                "bytes": DOWNLOAD_STATE.get("bytes", 0),
                "total": total,
            }
        else:
            DOWNLOAD_STATE = {
                "status": "ready",
                "model": model_id,
                "message": "下载完成",
                "bytes": DOWNLOAD_STATE.get("bytes", 0),
                "total": total,
            }

    threading.Thread(target=worker, name="model-download", daemon=True).start()
    return {"status": "downloading"}


# ---------------------------------------------------------------- 对话

class ChatPayload(BaseModel):
    messages: list[dict]
    adapter_id: str | None = None
    use_adapter: bool = True
    use_retrieval: bool = True
    show_thought: bool = True
    verbosity: str = "normal"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    auto_approve: bool = False


VERBOSITY_HINTS = {
    "short": "本次回答请尽量简短，2 句以内说清即可。",
    "normal": "",
    "detail": "本次回答请详细展开：先给结论，再分点说明理由、步骤或例子，150-350 字。",
}

# 实时类问题：模型无法获取数据，必须明确拒答，避免编造气温/股价等
REALTIME_HINTS = re.compile(
    r"(天气|气温|温度|空气质量|新闻|股价|汇率|油价|金价|航班|比分|实时|最新消息|现在几点|今天是几号|明天.*(天气|下雨|温度))"
)
REALTIME_NOTE = (
    "\n\n【重要】你无法访问实时信息。涉及天气、新闻、股价、汇率、航班、比分等问题时，"
    "必须直接说明“无法获取实时信息”，不得编造具体数值、温度或日期。"
)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


RETRIEVAL_MIN_COVERAGE = 0.45


def _relevant_context(query: str, top_k: int = 3) -> list[dict]:
    """检索知识库：语义检索已做相关度过滤，这里只取前两条最相关的。"""
    if not query.strip():
        return []
    return forge_retrieval.search(query, top_k=top_k)[:2]


@app.post("/api/chat")
def chat(payload: ChatPayload):
    if TASK_STATE["busy"]:
        raise HTTPException(status_code=409, detail="训练或评测任务进行中，暂时无法对话")

    base_model = config.BASE_MODELS.get("default")
    adapter_path = None
    if payload.adapter_id:
        adapter = db.one("adapters", payload.adapter_id)
        if not adapter:
            raise HTTPException(status_code=404, detail="适配器不存在")
        base_model = adapter["base_model"]
        adapter_path = adapter["path"]
    _require_model(base_model)

    raw_messages = [
        {"role": m.get("role", "user"), "content": m.get("content", "")} for m in payload.messages
    ]
    max_history = int(config.PERSONA.get("max_history", 12))
    if len(raw_messages) > max_history:
        # 保留开场问题 + 最近若干轮，避免上下文无限膨胀
        raw_messages = raw_messages[:1] + raw_messages[-(max_history - 1):]

    messages = load_system_messages(raw_messages)
    query = next((str(m["content"]) for m in reversed(payload.messages) if m.get("role") == "user"), "")

    # 文件操作意图（创建/读取/列出）直接执行，不再绕一圈让模型“指导用户自己做”
    action = chat_actions.handle(query, auto_approve=payload.auto_approve)
    if action:
        def act_stream():
            started = time.time()
            yield _sse({"type": "thinking", "text": action["thinking"]})
            yield _sse({"type": "status", "text": "执行文件操作..."})
            yield _sse({"type": "delta", "text": action["reply"]})
            yield _sse(
                {
                    "type": "done",
                    "elapsed": "0.0",
                    "thought_elapsed": 0.0,
                    "total": round(time.time() - started, 1),
                }
            )

        return StreamingResponse(act_stream(), media_type="text/event-stream")

    persona = (config.PERSONA.get("system") or "").strip()
    if REALTIME_HINTS.search(query):
        persona = f"{persona}{REALTIME_NOTE}".strip()
    hint = VERBOSITY_HINTS.get(payload.verbosity, "")
    if hint:
        persona = f"{persona}\n{hint}".strip()
    if persona and messages and messages[0].get("role") == "system":
        messages[0]["content"] = persona

    retrieved = []
    note = ""
    instruction_task = False
    if payload.use_retrieval:
        instruction_task = forge_retrieval.looks_like_instruction(query)
        if not instruction_task:
            retrieved = _relevant_context(query)
            if retrieved:
                # 参考资料要放在 system 末尾（思考要点之后），否则会被分析文本稀释
                note = forge_retrieval.format_note(retrieved)

    def generate():
        started_at = time.time()
        thought_seconds = 0.0
        try:
            if instruction_task:
                yield _sse({"type": "thinking", "text": "识别为指令型任务，跳过知识检索，直接执行"})
            elif retrieved:
                yield _sse(
                    {
                        "type": "thinking",
                        "text": f"知识库命中 {len(retrieved)} 条资料",
                        "hits": [
                            {"question": hit["question"], "score": hit.get("score")}
                            for hit in retrieved
                        ],
                    }
                )
            else:
                yield _sse({"type": "thinking", "text": "知识库未命中，按模型自身知识作答"})
            if payload.adapter_id and llamacpp.enabled():
                yield _sse(
                    {
                        "type": "thinking",
                        "text": "GGUF 模式已内置合并后的微调权重，本次忽略所选适配器",
                    }
                )
            if persona:
                yield _sse({"type": "thinking", "text": "已加载人设与详细度要求"})
            yield _sse({"type": "status", "text": "加载模型..."})
            if adapter_path:
                runtime.activate_adapter(
                    adapter_path, base_model, config.MODEL_DIR, adapter_name=payload.adapter_id
                )
            else:
                runtime.ensure(base_model, config.MODEL_DIR)
            yield _sse({"type": "ready", "model": base_model})

            if payload.show_thought:
                question = next(
                    (str(m["content"]) for m in reversed(payload.messages) if m.get("role") == "user"), ""
                )
                yield _sse({"type": "thinking", "text": "开始分析问题（先思考，再作答）"})
                plan_messages = [
                    {"role": "system", "content": "你是严谨的中文分析助手，全程只使用中文，可以使用 Markdown 组织思路。"},
                    {
                        "role": "user",
                        "content": (
                            f"问题：{question}\n\n"
                            "先在内部理清思路，但不要直接作答。要求：\n"
                            "- 不要套固定模板，不要出现「问题拆解 / 关键信息 / 回答结构 / 风险点」这类小标题；\n"
                            "- 允许使用 Markdown：步骤用有序/无序列表，方案对比用表格，涉及代码或命令时用带语言标识的代码块；\n"
                            "- 寒暄或一句话能答的，简短说明思路即可；问题复杂时可以展开，写清楚判断依据、边界条件和取舍；\n"
                            "- 全程中文，篇幅控制在 800 字以内，不要给出最终答案。"
                        ),
                    },
                ]
                plan = ""
                for chunk in runtime.stream(
                    plan_messages,
                    max_new_tokens=1200,
                    temperature=0.0,
                    top_p=0.9,
                    use_adapter=payload.use_adapter and bool(adapter_path),
                ):
                    if chunk.startswith("\x00"):
                        thought_seconds = round(float(chunk[1:] or 0), 1)
                        continue
                    plan += chunk
                    yield _sse({"type": "thought", "text": chunk})
                if plan.strip() and messages and messages[0].get("role") == "system":
                    messages[0]["content"] = (
                        messages[0]["content"] + f"\n\n你刚才的分析要点（据此作答）：\n{plan.strip()}"
                    )
                yield _sse({"type": "thinking", "text": "分析完成，开始组织最终回答"})

            if note:
                if messages and messages[0].get("role") == "system":
                    messages[0]["content"] = messages[0]["content"] + "\n\n" + note
                else:
                    messages.insert(0, {"role": "system", "content": note})

            yield _sse({"type": "thinking", "text": "生成回答中"})
            for chunk in runtime.stream(
                messages,
                max_new_tokens=payload.max_new_tokens,
                temperature=payload.temperature,
                top_p=payload.top_p,
                use_adapter=payload.use_adapter and bool(adapter_path),
            ):
                if chunk.startswith("\x00"):
                    yield _sse(
                        {
                            "type": "done",
                            "elapsed": chunk[1:],
                            "thought_elapsed": thought_seconds,
                            "total": round(time.time() - started_at, 1),
                        }
                    )
                else:
                    yield _sse({"type": "delta", "text": chunk})
        except Exception as exc:  # noqa: BLE001
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/runtime/unload")
def unload_runtime():
    if TASK_STATE["busy"]:
        raise HTTPException(status_code=409, detail="任务运行中，无法释放模型")
    runtime.unload()
    return {"ok": True}


# ---------------------------------------------------------------- 文本工坊

class ForgePayload(BaseModel):
    task_id: str
    input: str
    variables: dict[str, str] = {}
    adapter_id: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


@app.get("/api/forge/tasks")
def forge_tasks():
    return [task.public() for task in forge_pipeline.reload_tasks()]


@app.post("/api/forge/run")
def forge_run(payload: ForgePayload):
    task = forge_pipeline.get_task(payload.task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务类型不存在")
    if TASK_STATE["busy"]:
        raise HTTPException(status_code=409, detail="训练或评测任务进行中，暂时无法生成")

    def generate():
        try:
            for event in forge_pipeline.run_forge(
                task,
                payload.input,
                payload.variables,
                payload.adapter_id,
                payload.temperature,
                payload.max_tokens,
            ):
                yield _sse(event)
        except Exception as exc:  # noqa: BLE001
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream")


class FeedbackPayload(BaseModel):
    question: str
    answer: str
    dataset_id: str | None = None


@app.post("/api/feedback")
def add_feedback(payload: FeedbackPayload):
    """数据飞轮：把对话里修正过的回答存成训练样本，并追加进最新数据集。"""
    question = payload.question.strip()
    answer = payload.answer.strip()
    if not question or not answer:
        raise HTTPException(status_code=400, detail="问题和回答都不能为空")

    record = {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }

    # 1) 写进知识库，RAG 立即生效
    demo_path = config.DATA_DIR / "demo" / "feedback.jsonl"
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    with demo_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 2) 追加到最新数据集，下次训练直接用到
    datasets = db.list_datasets()
    target = payload.dataset_id or (datasets[0]["id"] if datasets else None)
    total = 0
    if target:
        row = db.one("datasets", target)
        if row:
            records = load_jsonl(Path(row["path"]))
            records.append(record)
            save_jsonl(Path(row["path"]), records)
            db.update("datasets", target, {"num_samples": len(records)})
            total = len(records)

    return {"ok": True, "dataset_id": target, "num_samples": total}


# ---------------------------------------------------------------- 沙盒

class ExecPayload(BaseModel):
    command: str
    cwd: str | None = None
    timeout: int | None = None


class SandboxFilePayload(BaseModel):
    path: str
    content: str


@app.get("/api/sandbox/info")
def sandbox_info():
    return sandbox_guard.info()


@app.get("/api/sandbox/files")
def sandbox_files(path: str = ""):
    try:
        return {"workspace": str(sandbox_guard.WORKSPACE), "items": sandbox_io.listing(path)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sandbox/file")
def sandbox_read(path: str):
    try:
        return sandbox_io.read_file(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sandbox/file")
def sandbox_write(payload: SandboxFilePayload):
    try:
        return sandbox_io.write_file(payload.path, payload.content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/sandbox/file")
def sandbox_delete(path: str):
    try:
        return sandbox_io.delete_path(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sandbox/search")
def sandbox_search(q: str):
    return sandbox_io.search(q)


@app.post("/api/sandbox/exec")
def sandbox_exec(payload: ExecPayload):
    return sandbox_guard.execute(payload.command, payload.cwd, payload.timeout)


@app.get("/api/sandbox/pending")
def sandbox_pending():
    return sandbox_guard.pending_list()


@app.post("/api/sandbox/approve/{request_id}")
def sandbox_approve(request_id: str):
    return sandbox_guard.approve(request_id)


@app.post("/api/sandbox/deny/{request_id}")
def sandbox_deny(request_id: str):
    return sandbox_guard.deny(request_id)


app.mount("/static", StaticFiles(directory=config.WEB_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.SERVER.get("host", "127.0.0.1"),
        port=int(config.SERVER.get("port", 8765)),
    )
