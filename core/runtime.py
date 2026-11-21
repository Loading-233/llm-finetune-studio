"""模型运行时：统一管理基座模型与 LoRA 适配器的加载、切换、生成。"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from core.tokens import build_example

DEFAULT_SYSTEM_PROMPT = "你是一个乐于助人的中文助手，回答准确、简洁。"


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__").replace("\\", "__")


def resolve_model_source(model_id: str, model_dir: Path | str) -> str:
    """优先使用本地已下载目录，否则回退到 HuggingFace 仓库名。"""
    local = Path(model_dir) / model_slug(model_id)
    if (local / "config.json").exists():
        return str(local)
    return model_id


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype_kwarg(dtype) -> dict:
    """transformers 4.56 起 torch_dtype 更名为 dtype。"""
    from transformers import __version__ as tf_version

    try:
        major, minor = (int(part) for part in tf_version.split(".")[:2])
    except ValueError:
        major, minor = (4, 44)
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def device_report() -> dict:
    device = pick_device()
    report = {
        "device": device,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "threads": torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        report["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    return report


class ModelRuntime:
    """单例：同一时刻只驻留一份模型，训练与推理互斥。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model = None
        self._tokenizer = None
        self._base_id: str | None = None
        self._adapter_name: str | None = None
        self._is_peft = False

    def configure(self, threads: int = 0) -> None:
        """启动时调用：设定 CPU 推理线程数。"""
        workers = threads or min(8, os.cpu_count() or 1)
        torch.set_num_threads(max(1, workers))

    # ------------------------------------------------------------ 状态

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def _gguf(self):
        """启用 GGUF（llama.cpp）后端时返回该模块，否则返回 None。"""
        from core import llamacpp

        return llamacpp if llamacpp.enabled() else None

    def status(self) -> dict:
        gguf = self._gguf()
        if gguf is not None:
            return gguf.status()
        with self._lock:
            return {
                "loaded": self._model is not None,
                "base_model": self._base_id,
                "adapter": self._adapter_name,
                "device": pick_device(),
            }

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._base_id = None
            self._adapter_name = None
            self._is_peft = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------ 加载

    def _dtype(self):
        device = pick_device()
        if device == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if device == "mps":
            return torch.float16
        return torch.float32

    def _load_base(self, model_id: str, model_dir: Path) -> None:
        source = resolve_model_source(model_id, model_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            source,
            **dtype_kwarg(self._dtype()),
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self._model.to(pick_device())
        self._model.eval()
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._base_id = model_id
        self._adapter_name = None
        self._is_peft = False

    def ensure(self, model_id: str, model_dir: Path) -> None:
        if self._gguf() is not None:
            return
        with self._lock:
            if self._model is not None and self._base_id == model_id:
                return
            self.unload()
            self._load_base(model_id, model_dir)

    def activate_adapter(
        self,
        adapter_path: str | Path,
        model_id: str,
        model_dir: Path,
        adapter_name: str | None = None,
    ) -> None:
        """加载并启用 LoRA 适配器。"""
        from peft import PeftModel

        if self._gguf() is not None:
            return
        with self._lock:
            # 所有训练产物的目录名都是 adapter，用任务目录名或显式名称区分
            name = adapter_name or Path(adapter_path).parent.name
            if (
                self._model is not None
                and self._base_id == model_id
                and self._is_peft
                and self._adapter_name == name
            ):
                return
            self.ensure(model_id, model_dir)

            if not self._is_peft:
                self._model = PeftModel.from_pretrained(self._model, str(adapter_path), adapter_name=name)
                self._is_peft = True
            else:
                loaded = set(getattr(self._model, "peft_config", {}) or {})
                if name not in loaded:
                    self._model.load_adapter(str(adapter_path), adapter_name=name)
            self._model.set_adapter(name)
            self._model.eval()
            self._adapter_name = name

    # ------------------------------------------------------------ 生成

    def _encode(self, messages: Sequence[dict]):
        with self._lock:
            tokenizer = self._tokenizer
        return tokenizer.apply_chat_template(
            list(messages),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    def _generate_kwargs(self, max_new_tokens: int, temperature: float, top_p: float) -> dict:
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.05,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
        }
        if temperature and temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            kwargs.update(do_sample=False)
        return kwargs

    def _adapter_scope(self, use_adapter: bool):
        """不启用适配器时临时关闭 LoRA（基座对照）。"""
        if self._is_peft and self._adapter_name and not use_adapter:
            return self._model.disable_adapter()
        return nullcontext()

    def stream(
        self,
        messages: Sequence[dict],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_adapter: bool = True,
    ) -> Iterator[str]:
        gguf = self._gguf()
        if gguf is not None:
            yield from gguf.stream(messages, max_new_tokens, temperature, top_p)
            return
        inputs = self._encode(messages)
        inputs = {k: v.to(pick_device()) for k, v in inputs.items()}
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        kwargs = dict(**inputs, streamer=streamer, **self._generate_kwargs(max_new_tokens, temperature, top_p))
        started = time.time()
        with self._adapter_scope(use_adapter):
            thread = threading.Thread(target=self._model.generate, kwargs=kwargs, daemon=True)
            thread.start()
            for chunk in streamer:
                if chunk:
                    yield chunk
            thread.join(timeout=5)
        yield f"\x00{time.time() - started:.1f}"

    def complete(
        self,
        messages: Sequence[dict],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
        use_adapter: bool = True,
    ) -> str:
        gguf = self._gguf()
        if gguf is not None:
            return gguf.complete(messages, max_new_tokens, temperature, top_p)
        inputs = self._encode(messages)
        inputs = {k: v.to(pick_device()) for k, v in inputs.items()}
        with self._adapter_scope(use_adapter), torch.inference_mode():
            output = self._model.generate(**inputs, **self._generate_kwargs(max_new_tokens, temperature, top_p))
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ------------------------------------------------------------ 评测指标

    def perplexity(self, records: Sequence[dict], max_seq_len: int = 1024, use_adapter: bool = True) -> float:
        if not records:
            return float("nan")
        if self._gguf() is not None:
            # GGUF 后端不支持困惑度计算，评测里会显示为「—」
            return float("nan")
        total_loss, total_tokens = 0.0, 0
        device = pick_device()
        with self._adapter_scope(use_adapter), torch.inference_mode():
            for rec in records:
                example = build_example(self._tokenizer, rec["messages"], max_seq_len)
                input_ids = torch.tensor([example["input_ids"]], device=device)
                attention_mask = torch.tensor([example["attention_mask"]], device=device)
                labels = torch.tensor([example["labels"]], device=device)
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                valid = int((labels != -100).sum())
                if valid:
                    total_loss += float(outputs.loss) * valid
                    total_tokens += valid
        if not total_tokens:
            return float("nan")
        return math.exp(total_loss / total_tokens)


runtime = ModelRuntime()


def load_system_messages(messages: Sequence[dict]) -> list[dict]:
    """没有 system 时补一条默认 system。"""
    if messages and messages[0].get("role") == "system":
        return list(messages)
    return [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, *messages]


def read_meta(path: str | Path) -> dict:
    meta_path = Path(path) / "training_meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) in ("1", "true", "True")

