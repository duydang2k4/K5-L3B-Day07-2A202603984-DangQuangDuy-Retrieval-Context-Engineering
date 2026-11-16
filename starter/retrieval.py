"""Retrieval implementations used by the Day 7 submission.

The module is deliberately dependency-free so the benchmark can be reproduced
offline.  ``HashDenseIndex`` is a deterministic local dense-vector adapter;
when a Day 6 embedding service is available, ``EmbeddingRetriever`` can be
used with the same ``search`` contract without changing the agent.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from typing import Any, Iterable


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def _metadata_value(chunk: dict[str, Any], key: str) -> Any:
    if key in chunk:
        return chunk[key]
    return chunk.get("metadata", {}).get(key)


def matches_filters(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Apply only user-supplied exact metadata filters; never use benchmark gold."""
    if not filters:
        return True
    return all(_metadata_value(chunk, key) == value for key, value in filters.items())


def _result(chunk: dict[str, Any], score: float, strategy: str, rank: int) -> dict[str, Any]:
    return {**chunk, "score": float(score), "rank": rank, "strategy": strategy}


class BM25Index:
    """Small BM25 implementation suitable for the provided corpus."""

    def __init__(self, chunks: Iterable[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.tokens = [tokenize(c["text"]) for c in self.chunks]
        self.lengths = [len(row) for row in self.tokens]
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        df = Counter(term for row in self.tokens for term in set(row))
        n = len(self.chunks)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def _score(self, query_terms: list[str], index: int) -> float:
        counts = Counter(self.tokens[index])
        length = self.lengths[index]
        score = 0.0
        for term, qf in Counter(query_terms).items():
            if term not in counts:
                continue
            tf = counts[term]
            norm = tf + self.k1 * (1 - self.b + self.b * length / max(1, self.avgdl))
            score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / norm * min(qf, 2)
        return score

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        rows = [(self._score(tokenize(query), i), c) for i, c in enumerate(self.chunks)
                if matches_filters(c, filters)]
        rows = [(score, c) for score, c in rows if score > 0]
        rows.sort(key=lambda item: (-item[0], item[1]["id"]))
        return [_result(c, score, "bm25", rank) for rank, (score, c) in enumerate(rows[:top_k], 1)]


class HashDenseIndex:
    """Deterministic dense vectors from token and character n-gram features.

    This is a reproducible local adapter, not a claim that hashed features are
    equivalent to a hosted embedding model.  The index records its dimension
    and hash version so it cannot be silently mixed with another index.
    """

    def __init__(self, chunks: Iterable[dict[str, Any]], dimensions: int = 384,
                 hash_version: str = "sha256-sign-v1"):
        if dimensions < 32:
            raise ValueError("dimensions must be >= 32")
        self.chunks = list(chunks)
        self.dimensions = dimensions
        self.hash_version = hash_version
        self.vectors = [self._embed(c["text"]) for c in self.chunks]

    def _embed(self, text: str) -> list[float]:
        terms = tokenize(text)
        features = Counter(terms)
        for token in terms:
            padded = f"^{token}$"
            features.update(padded[i:i + 3] for i in range(max(0, len(padded) - 2)))
        vector = [0.0] * self.dimensions
        for feature, value in features.items():
            digest = hashlib.sha256(f"{self.hash_version}:{feature}".encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign * (1.0 + math.log1p(value))
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        qv = self._embed(query)
        rows = []
        for vector, chunk in zip(self.vectors, self.chunks):
            if matches_filters(chunk, filters):
                score = sum(a * b for a, b in zip(qv, vector))
                if score > 0:
                    rows.append((score, chunk))
        rows.sort(key=lambda item: (-item[0], item[1]["id"]))
        return [_result(c, score, "dense_hash", rank) for rank, (score, c) in enumerate(rows[:top_k], 1)]


class HybridIndex:
    """Reciprocal-rank fusion over BM25 and dense retrieval."""

    def __init__(self, chunks: Iterable[dict[str, Any]], candidate_k: int = 10, rrf_k: int = 60):
        self.chunks = list(chunks)
        self.candidate_k, self.rrf_k = candidate_k, rrf_k
        self.bm25 = BM25Index(self.chunks)
        self.dense = HashDenseIndex(self.chunks)

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        bm = self.bm25.search(query, filters=filters, top_k=max(top_k, self.candidate_k))
        de = self.dense.search(query, filters=filters, top_k=max(top_k, self.candidate_k))
        merged: dict[str, dict[str, Any]] = {}
        for rows in (bm, de):
            for item in rows:
                entry = merged.setdefault(item["id"], {**item, "score": 0.0})
                entry["score"] += 1.0 / (self.rrf_k + item["rank"])
                entry["retrieval_scores"] = {
                    **entry.get("retrieval_scores", {}), item["strategy"]: item["score"]
                }
        rows = sorted(merged.values(), key=lambda item: (-item["score"], item["id"]))[:top_k]
        return [{**item, "strategy": "hybrid_rrf", "rank": rank} for rank, item in enumerate(rows, 1)]


class RetrievalRouter:
    """Expose the common agent retriever contract for all local strategies."""

    def __init__(self, chunks: Iterable[dict[str, Any]]):
        self.indexes = {
            "dense": HashDenseIndex(chunks),
            "bm25": BM25Index(chunks),
            "hybrid": HybridIndex(chunks),
        }

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3, strategy: str = "hybrid") -> list[dict[str, Any]]:
        if strategy not in self.indexes:
            raise ValueError(f"unknown retrieval strategy: {strategy}")
        return self.indexes[strategy].search(query, filters=filters, top_k=top_k)
