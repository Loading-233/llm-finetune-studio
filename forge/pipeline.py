"""文本工坊流水线：渲染提示词 → 生成 → 后处理（结构化解析 / 代码块提取 / 重试）。"""

from __future__ import annotations

import json
import re
import time
from typing import Iterator

from core.runtime import runtime
from forge.tasks import ForgeTask, load_tasks, render

TASKS = load_tasks()


def get_task(task_id: str) -> ForgeTask | None:
    return TASKS.get(task_id)


def reload_tasks() -> list[ForgeTask]:
    """重新读取 tasks.yaml，支持不重启服务新增输出类型。"""
    global TASKS
    TASKS = load_tasks()
    return list(TASKS.values())


def _extract_json(text: str):
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", cleaned, re.S)
    if fence:
        cleaned = fence.group(1).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def _extract_code(text: str) -> str:
    fence = re.search(r"```[a-zA-Z0-9_+-]*\s*\n(.+?)```", text, re.S)
    return fence.group(1).strip() if fence else text.strip()


def postprocess(fmt: str, text: str) -> dict:
    raw = text.strip()
    if fmt == "json":
        value = _extract_json(raw)
        return {"text": json.dumps(value, ensure_ascii=False, indent=2) if value is not None else raw,
                "parsed": value, "valid": value is not None}
    if fmt == "code":
        return {"text": _extract_code(raw), "parsed": None, "valid": True}
    if fmt == "markdown":
        # 统一列表符号与空行
        normalised = re.sub(r"[ \t]+\n", "\n", raw)
        normalised = re.sub(r"\n{3,}", "\n\n", normalised)
        return {"text": normalised, "parsed": None, "valid": True}
    cleaned = re.sub(r"\n{3,}", "\n\n", raw)
    return {"text": cleaned, "parsed": None, "valid": True}


def _resolve(adapter_id: str | None) -> tuple[str, str | None]:
    from app import config, db

    base_model = config.BASE_MODELS.get("default")
    if not adapter_id:
        return base_model, None
    row = db.one("adapters", adapter_id)
    if not row:
        return base_model, None
    return row["base_model"], row["path"]


def _generate(messages: list[dict], max_new_tokens: int, temperature: float, use_adapter: bool,
              sink: dict) -> Iterator[str]:
    for chunk in runtime.stream(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        use_adapter=use_adapter,
    ):
        if chunk.startswith("\x00"):
            sink["elapsed"] = chunk[1:]
        else:
            sink["text"] += chunk
            yield chunk


def run_forge(
    task: ForgeTask,
    text: str,
    variables: dict[str, str] | None = None,
    adapter_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Iterator[dict]:
    from app import config

    if not text.strip():
        yield {"type": "error", "message": "输入内容为空"}
        return

    base_model, adapter_path = _resolve(adapter_id)
    yield {"type": "status", "text": f"准备模型：{base_model.split('/')[-1]}"}
    if adapter_path:
        runtime.activate_adapter(adapter_path, base_model, config.MODEL_DIR, adapter_name=adapter_id)
    else:
        runtime.ensure(base_model, config.MODEL_DIR)

    context = ""
    if task.retrieval:
        from forge.retrieval import search

        hits = search(text, top_k=3)
        if hits:
            context = "\n".join(
                f"{index}. 问：{hit['question']} 答：{hit['answer']}"
                for index, hit in enumerate(hits, start=1)
            )
            yield {"type": "status", "text": f"检索到 {len(hits)} 条参考资料"}

    prompt = render(task, text, variables, context=context)
    messages = [
        {"role": "system", "content": task.system},
        {"role": "user", "content": prompt},
    ]
    use_adapter = bool(adapter_path)
    kwargs = {
        "max_new_tokens": max_tokens or task.max_tokens,
        "temperature": task.temperature if temperature is None else temperature,
        "use_adapter": use_adapter,
    }

    yield {"type": "status", "text": f"生成中（{task.name} · {task.output_format}）"}
    sink = {"text": "", "elapsed": "0"}
    for chunk in _generate(messages, sink=sink, **kwargs):
        yield {"type": "delta", "text": chunk}

    result = postprocess(task.output_format, sink["text"])

    # 结构化输出不合格时自动重试一次，并把修复要求追加到对话里
    if task.output_format == "json" and not result["valid"]:
        yield {"type": "status", "text": "JSON 格式不合规，正在重试一次"}
        yield {"type": "reset"}
        retry_messages = messages + [
            {"role": "assistant", "content": sink["text"][:400]},
            {"role": "user", "content": "上面的输出不是合法 JSON。请只输出一个合法 JSON 对象，不要任何解释或代码围栏。"},
        ]
        sink["text"] = ""
        for chunk in _generate(retry_messages, sink=sink, **kwargs):
            yield {"type": "delta", "text": chunk}
        result = postprocess(task.output_format, sink["text"])

    yield {
        "type": "final",
        "task": task.id,
        "format": task.output_format,
        "text": result["text"],
        "valid": result["valid"],
        "elapsed": sink.get("elapsed", "0"),
        "time": time.strftime("%H:%M:%S"),
    }
