"""中文语义检索：bge-small-zh 向量化 + 余弦相似度，比 BM25 更耐受同义改写。"""

from __future__ import annotations

import numpy as np

from app import config

MODEL_DIR = config.MODEL_DIR / "BAAI__bge-small-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
MIN_SCORE = 0.55

_STATE: dict[str, object] = {}


def available() -> bool:
    return (MODEL_DIR / "config.json").exists()


class SemanticIndex:
    def __init__(self, docs: list[dict]) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        self.model = AutoModel.from_pretrained(str(MODEL_DIR)).eval()
        self.docs = docs
        self.matrix = self._encode([f"{d['question']} {d['answer']}" for d in docs])

    def _encode(self, texts: list[str]) -> np.ndarray:
        import torch

        vectors = []
        with torch.no_grad():
            for start in range(0, len(texts), 16):
                batch = self.tokenizer(
                    texts[start:start + 16],
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
                vectors.append(torch.nn.functional.normalize(pooled, dim=-1).numpy())
        return np.vstack(vectors)

    def search(self, query: str, top_k: int = 3, min_score: float = MIN_SCORE) -> list[dict]:
        if not self.docs:
            return []
        query_vector = self._encode([QUERY_PREFIX + query])[0]
        similarity = self.matrix @ query_vector
        order = np.argsort(-similarity)[:top_k]
        return [
            {**self.docs[index], "score": round(float(similarity[index]), 3)}
            for index in order
            if similarity[index] >= min_score
        ]


def build(docs: list[dict], key: str) -> SemanticIndex | None:
    """按数据版本缓存索引，避免每次请求重新向量化。"""
    if not available():
        return None
    if _STATE.get("key") != key:
        _STATE["key"] = key
        try:
            _STATE["index"] = SemanticIndex(docs)
        except Exception:  # noqa: BLE001
            _STATE["key"] = None
            _STATE["index"] = None
            return None
    return _STATE["index"]  # type: ignore[return-value]
