"""Semantic ranking of agents with an embedding model (via OpenRouter's /embeddings endpoint).

Each agent is turned into one short text (name, description, people, topics), embedded once and cached on
disk by content hash; each request embeds only the message. Agents are ranked by cosine similarity.

If the embedding service is unavailable or slow the ranker returns nothing and the word ranker carries on,
so a failure here can never block a turn.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import httpx

from ...logging_config import logger
from ...openrouter_client.usage import record_call
from .roster import AgentRecord

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TOP_N = 20  # the semantic list is cut here so its long tail adds no noise to the fusion


def agent_text(record: AgentRecord) -> str:
    """What gets embedded for one agent."""
    parts = [record.name]
    if record.description:
        parts.append(record.description)
    if record.people:
        parts.append("People: " + ", ".join(record.people))
    if record.topics:
        parts.append("Topics: " + ", ".join(record.topics))
    return ". ".join(parts)


def _normalise(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


class OpenRouterEmbedder:
    def __init__(self, model: str, api_key: str, *, timeout: float = 5.0, base_url: str = OPENROUTER_BASE_URL):
        self.model, self.api_key, self.timeout, self.base_url = model, api_key, timeout, base_url.rstrip("/")

    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        started = time.monotonic()
        from ...openrouter_client.client import shared_client

        client = shared_client()  # the same connection pool as the chat calls
        try:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "input": list(texts)},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except Exception as exc:
            record_call(model=self.model, latency_seconds=time.monotonic() - started, response=None,
                        tools_offered=0, error=f"embeddings {type(exc).__name__}")
            raise
        data = response.json()
        record_call(model=self.model, latency_seconds=time.monotonic() - started, response=data, tools_offered=0)
        ordered = sorted(data["data"], key=lambda item: item.get("index", 0))
        return [_normalise(item["embedding"]) for item in ordered]


class VectorCache:
    """text -> unit vector, kept in memory and (optionally) in one JSON file per embedding model."""

    def __init__(self, path: Optional[Path] = None):
        self._path = path
        self._vectors: Dict[str, List[float]] = {}
        if path and path.exists():
            try:
                self._vectors = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("embedding cache unreadable, starting empty: %s", exc)

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        return self._vectors.get(self.key(text))

    def put_many(self, texts: Sequence[str], vectors: Sequence[List[float]]) -> None:
        for text, vector in zip(texts, vectors):
            self._vectors[self.key(text)] = [round(x, 5) for x in vector]
        if self._path:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self._vectors), encoding="utf-8")
                tmp.replace(self._path)
            except Exception as exc:  # the cache is an optimisation, never a requirement
                logger.warning("could not save embedding cache: %s", exc)


class EmbeddingRanker:
    name = "semantic"

    def __init__(self, embedder, cache: Optional[VectorCache] = None, top_n: int = DEFAULT_TOP_N, weight: float = 1.0):
        self.embedder, self.cache, self.top_n, self.weight = embedder, cache or VectorCache(), top_n, weight

    async def vectors(self, texts: Sequence[str]) -> List[List[float]]:
        """Unit vectors for these texts (cached ones are reused). Raises if the service is unavailable."""
        missing = [t for t in dict.fromkeys(texts) if self.cache.get(t) is None]
        if missing:
            self.cache.put_many(missing, await self.embedder.embed(missing))
        return [self.cache.get(t) for t in texts]

    async def rank(self, query: str, records: Sequence[AgentRecord]) -> List[AgentRecord]:
        if not query.strip() or not records:
            return []
        try:
            texts = [agent_text(r) for r in records]
            missing = [t for t in dict.fromkeys(texts) if self.cache.get(t) is None]
            if missing:  # first sight of an agent (or a changed description): one batched call
                self.cache.put_many(missing, await self.embedder.embed(missing))
            query_vector = (await self.vectors([query]))[0]  # cached: an agent's own text is often the query
        except Exception as exc:
            logger.warning("semantic ranking unavailable, using word matching only: %s", exc)
            return []
        scored = []
        for record, text in zip(records, texts):
            vector = self.cache.get(text)
            if vector:
                scored.append((sum(a * b for a, b in zip(query_vector, vector)), record))
        scored.sort(key=lambda pair: pair[1].last_active_at, reverse=True)
        scored.sort(key=lambda pair: -pair[0])
        return [record for _, record in scored[: self.top_n]]
