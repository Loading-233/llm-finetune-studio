"""轻量检索：对知识数据集做 BM25（中文按字符二元组切分，无需额外依赖）。"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from app import config
from forge import semantic as semantic_module

_CACHE: dict[str, object] = {}

# 语义相似度低于这条线的一律不采纳：bge 对无关句子的分数普遍在 0.5~0.58
MIN_SEMANTIC_SCORE = 0.60
# 关键词兜底要求更高的覆盖率，否则「用/我/怎么」这类字也会凑出 0.6
KEYWORD_COVERAGE = 0.80

PUNCT = re.compile(r"[\s，。！？、；：\"'（）()\[\]{}<>《》,.!?;:~`|/\\\-—_=+*&^%$#@]+")

# 提问里的口水词，参与匹配只会制造噪声
FILLER = re.compile(
    r"(什么|怎么样|怎么|怎样|如何|为什么|为什么|一下|请问|请|解释|介绍|说明|告诉|讲讲|一句话|简单|具体|详细|"
    r"区别|是什么|是|的|了|吗|呢|呀|啊|我|你|能|可以|用|和|与|及|有|没有|多少|几个|怎样)"
)


def _tokens(text: str) -> list[str]:
    """中文按单字切分：不做分词也能容忍「费曼学习法 / 费曼技巧」这类近义改写。"""
    cleaned = PUNCT.sub("", text.lower())
    cleaned = FILLER.sub("", cleaned)
    return list(cleaned)


def _latest_dataset() -> Path | None:
    files = sorted(
        (config.DATA_DIR / "datasets").glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


class BM25:
    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1, self.b = k1, b
        self.doc_tokens = [_tokens(f"{d['question']} {d['answer']}") for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / max(1, len(self.doc_len))
        self.df: Counter = Counter()
        for tokens in self.doc_tokens:
            self.df.update(set(tokens))
        self.n = len(docs)

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 3, min_coverage: float = 0.35) -> list[dict]:
        terms = _tokens(query)
        term_set = set(terms)
        if not terms or not self.docs or not term_set:
            return []
        total_weight = sum(self.idf(term) for term in term_set) or 1.0
        scored = []
        for index, tokens in enumerate(self.doc_tokens):
            token_set = set(tokens)
            matched = term_set & token_set
            if not matched:
                continue
            coverage = sum(self.idf(term) for term in matched) / total_weight
            if coverage < min_coverage:
                continue
            tf = Counter(tokens)
            score = 0.0
            for term in matched:
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[index] / self.avg_len)
                score += self.idf(term) * freq * (self.k1 + 1) / denom
            scored.append((coverage, score, index))
        scored.sort(reverse=True)
        return [
            {**self.docs[index], "score": round(score, 3), "coverage": round(coverage, 2)}
            for coverage, score, index in scored[:top_k]
        ]


def _doc_paths() -> list[Path]:
    demo_dir = config.DATA_DIR / "demo"
    files = sorted(demo_dir.rglob("*.jsonl")) if demo_dir.exists() else []
    if files:
        return files
    latest = _latest_dataset()
    return [latest] if latest else []


def _load_docs() -> list[dict]:
    docs: dict[str, dict] = {}
    for path in _doc_paths():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            messages = payload.get("messages") or []
            if len(messages) < 2:
                continue
            question = messages[-2]["content"].strip()
            answer = messages[-1]["content"].strip()
            docs.setdefault(answer, {"question": question, "answer": answer})
    return list(docs.values())


def get_index() -> BM25 | None:
    paths = _doc_paths()
    if not paths:
        return None
    key = "|".join(f"{p}:{p.stat().st_mtime}" for p in paths)
    if _CACHE.get("key") != key:
        _CACHE["key"] = key
        _CACHE["index"] = BM25(_load_docs())
    return _CACHE["index"]  # type: ignore[return-value]


def semantic_index(docs: list[dict], key: str):
    return semantic_module.build(docs, key)


def search(query: str, top_k: int = 3) -> list[dict]:
    """混合检索：以语义结果为主，BM25 只做同文档内的加权；语义全空时才退回关键词。"""
    paths = _doc_paths()
    if not paths:
        return []
    key = "|".join(f"{path}:{path.stat().st_mtime}" for path in paths)
    docs = _load_docs()

    bm25 = get_index()
    semantic = semantic_index(docs, key)

    if semantic is not None:
        semantic_hits = [
            hit for hit in semantic.search(query, top_k * 2) if hit.get("score", 0) >= MIN_SEMANTIC_SCORE
        ]
        if semantic_hits:
            keyword_hits = bm25.search(query, top_k * 2) if bm25 is not None else []
            boost = {
                hit["answer"]: 1.0 / (60 + rank)
                for rank, hit in enumerate(keyword_hits)
                if hit.get("coverage", 0) >= KEYWORD_COVERAGE
            }
            for rank, hit in enumerate(semantic_hits):
                hit["rrf"] = round(1.0 / (60 + rank) + 0.4 * boost.get(hit["answer"], 0.0), 5)
            semantic_hits.sort(key=lambda hit: hit["rrf"], reverse=True)
            return semantic_hits[:top_k]

    if bm25 is None:
        return []
    # 太短的问题（如「你好」）只剩一两个字，关键词覆盖率会虚高成 100%，直接不检索
    if len(_tokens(query)) < 2:
        return []
    keyword_only = [
        hit for hit in bm25.search(query, top_k * 2) if hit.get("coverage", 0) >= KEYWORD_COVERAGE
    ]
    return keyword_only[:top_k]


def context_block(query: str, top_k: int = 3) -> str:
    hits = search(query, top_k)
    if not hits:
        return "（无）"
    lines = []
    for index, hit in enumerate(hits, start=1):
        lines.append(f"{index}. 问：{hit['question']} 答：{hit['answer']}")
    return "\n".join(lines)


NOTE_TEMPLATE = (
    "参考资料（事实以资料为准；分析只用于组织语言，资料没提到的功能不要臆测，不要编造数字）：\n{block}"
)

# 指令型任务：要的是执行，检索到的同类例句只会干扰，直接跳过检索
INSTRUCTION_HINTS = re.compile(
    r"(翻译|译成|译为|translate|润色|改写|改写成|改成英文|改成中文|压缩到|扩写到|转成|格式化为|纠错|语法错误)"
)
# 「xx 怎么翻译 / 英文怎么说」是在问术语译法，属于知识问题，不能跳过检索
TERM_QUESTION = re.compile(r"(怎么翻译|如何翻译|怎么译|译法|英文怎么说|用英文怎么说|英文是什么)")


def looks_like_instruction(query: str) -> bool:
    text = query.lower()
    if TERM_QUESTION.search(text):
        return False
    return bool(INSTRUCTION_HINTS.search(text))


def format_note(hits: list[dict]) -> str:
    """把检索结果拼成注入 system 的参考资料块，线上对话与评测共用同一份格式。"""
    block = "\n".join(f"{index}. 问：{hit['question']} 答：{hit['answer']}" for index, hit in enumerate(hits, 1))
    return NOTE_TEMPLATE.format(block=block)

