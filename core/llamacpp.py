"""GGUF 推理后端：用 llama.cpp 在 CPU 上跑量化模型，速度约为 PyTorch fp32 的 2-3 倍。"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Iterator, Sequence

from app import config

_LOCK = threading.Lock()
# llama.cpp 的 context 不是线程安全的；并发调用会在 ggml 里触发访问冲突直接崩溃
_GEN_LOCK = threading.Lock()
_LLM = None
_LOADED: Path | None = None


def _settings() -> dict:
    return config.INFERENCE


def gguf_path() -> Path | None:
    raw = _settings().get("gguf")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = config.ROOT / path
    return path if path.exists() else None


def enabled() -> bool:
    backend = str(_settings().get("backend") or "torch").lower()
    return backend in ("llamacpp", "llama.cpp", "gguf") and gguf_path() is not None


def status() -> dict:
    path = gguf_path()
    return {
        "loaded": _LLM is not None,
        "base_model": path.name if path else None,
        "adapter": "已合并进 GGUF",
        "device": "cpu",
        "backend": "llamacpp",
    }


def _get():
    """懒加载 GGUF 模型，整个进程只保留一份。"""
    global _LLM, _LOADED
    path = gguf_path()
    if path is None:
        raise RuntimeError("找不到 GGUF 模型文件")
    with _LOCK:
        if _LLM is not None and _LOADED == path:
            return _LLM
        from llama_cpp import Llama

        settings = _settings()
        _LLM = Llama(
            model_path=str(path),
            n_ctx=int(settings.get("n_ctx", 4096)),
            n_threads=int(settings.get("threads", 8)),
            verbose=False,
        )
        _LOADED = path
    return _LLM


def stream(
    messages: Sequence[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_adapter: bool = True,
) -> Iterator[str]:
    started = time.time()
    llm = _get()
    payload = [{"role": str(m.get("role", "user")), "content": str(m.get("content", ""))} for m in messages]
    options = {"max_tokens": max_new_tokens, "top_p": top_p}
    options["temperature"] = max(0.0, float(temperature))
    # 一次只让一个请求进入模型，避免并发导致 ggml 崩溃
    with _GEN_LOCK:
        for part in llm.create_chat_completion(messages=payload, stream=True, **options):
            choice = (part.get("choices") or [{}])[0]
            text = (choice.get("delta") or {}).get("content") or ""
            if text:
                yield text
    yield f"\x00{time.time() - started:.1f}"


def complete(
    messages: Sequence[dict],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.9,
    use_adapter: bool = True,
) -> str:
    parts = [
        chunk
        for chunk in stream(messages, max_new_tokens, temperature, top_p, use_adapter)
        if not chunk.startswith("\x00")
    ]
    return "".join(parts).strip()
