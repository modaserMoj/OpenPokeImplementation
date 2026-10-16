"""Choose which agents the main agent should see for a request.

The main agent used to receive EVERY agent name on EVERY turn. Now each ranker scores all agents against the
message, the rankings are merged with reciprocal rank fusion, and only a short list (plus the most recently
used agents) goes into the prompt. With a small roster nothing is hidden.

Rankers share one tiny interface (`rank(query, records) -> ordered records`), so a semantic (embedding)
ranker can be added next to the word ranker and fused with it.
"""

from __future__ import annotations

import inspect
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Protocol, Sequence

from .roster import ACTIVE, ARCHIVED, AgentRecord

DEFAULT_SHORTLIST = 10
DEFAULT_RECENT = 3
RRF_K = 60

# Words that carry no signal for picking an agent ("email" is in most agent names, so it cannot discriminate).
_STOP = frozenset(
    "a an and are as at be but by can could do does for from get give how i if in into is it just know let make me "
    "my need new of on or our please so that the their them then there this to up us want was we what when where "
    "which who will with would you your tell ask send email write about also any should com org net check follow update remind reminder status plan book look reply message follow-up".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")
# Field weights: a name or person match is worth more than a word in the description.
_FIELD_WEIGHTS = (("name", 3), ("people", 3), ("topics", 2), ("description", 1))


def tokenize(text: str) -> List[str]:
    """Lowercase words, minus stop words, with a light plural strip ("invoices" -> "invoice")."""
    tokens: List[str] = []
    for token in _TOKEN.findall((text or "").lower()):
        if token in _STOP or (len(token) < 2 and not token.isdigit()):
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


class Ranker(Protocol):
    def rank(self, query: str, records: Sequence[AgentRecord]) -> List[AgentRecord]:
        """Return the records that match, best first. Records with no signal are left out."""


def _weighted_terms(record: AgentRecord) -> Counter:
    fields = {
        "name": record.name,
        "people": " ".join(record.people),
        "topics": " ".join(record.topics),
        "description": record.description,
    }
    terms: Counter = Counter()
    for field, weight in _FIELD_WEIGHTS:
        for token in tokenize(fields[field]):
            terms[token] += weight
    return terms


class BM25Ranker:
    """Classic BM25 over each agent's name, people, topics and description."""

    name = "words"

    def __init__(self, k1: float = 1.5, b: float = 0.75, weight: float = 1.0):
        self.k1, self.b, self.weight = k1, b, weight

    def rank(self, query: str, records: Sequence[AgentRecord]) -> List[AgentRecord]:
        query_terms = list(dict.fromkeys(tokenize(query)))
        if not query_terms or not records:
            return []
        docs = [_weighted_terms(r) for r in records]
        lengths = [sum(d.values()) for d in docs]
        average = (sum(lengths) / len(lengths)) or 1.0
        n = len(records)
        document_frequency = Counter(term for d in docs for term in d)

        scored = []
        for record, doc, length in zip(records, docs, lengths):
            score = 0.0
            for term in query_terms:
                frequency = doc.get(term, 0)
                if not frequency:
                    continue
                idf = math.log(1 + (n - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
                score += idf * frequency * (self.k1 + 1) / (frequency + self.k1 * (1 - self.b + self.b * length / average))
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda pair: pair[1].last_active_at, reverse=True)  # ties: most recently used first
        scored.sort(key=lambda pair: -pair[0])  # sort is stable, so the tie order above is kept
        return [record for _, record in scored]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[AgentRecord]],
    *,
    k: int = RRF_K,
    weights: Optional[Sequence[float]] = None,
) -> List[AgentRecord]:
    """Merge several rankings: an agent's score is the sum of weight / (k + rank) over the lists it appears in."""
    scores: Dict[str, float] = {}
    by_id: Dict[str, AgentRecord] = {}
    first_seen: Dict[str, int] = {}
    for index, ranking in enumerate(rankings):
        weight = weights[index] if weights else 1.0
        for rank, record in enumerate(ranking, start=1):
            scores[record.id] = scores.get(record.id, 0.0) + weight / (k + rank)
            by_id[record.id] = record
            first_seen.setdefault(record.id, len(first_seen))
    order = sorted(scores, key=lambda rid: (-scores[rid], first_seen[rid]))
    return [by_id[rid] for rid in order]


def _assemble(live: Sequence[AgentRecord], fused: List[AgentRecord], limit: int, recent: int) -> List[AgentRecord]:
    chosen = fused[:limit]
    chosen_ids = {r.id for r in chosen}
    hot = sorted((r for r in live if r.status == ACTIVE), key=lambda r: r.last_active_at, reverse=True)
    for index, record in enumerate(hot):
        # Always keep the most recently used agents; then pad up to `limit` so a message the rankers cannot
        # match still sees a useful list instead of three agents.
        if record.id not in chosen_ids and (index < recent or len(chosen) < limit):
            chosen.append(record)
            chosen_ids.add(record.id)
    return chosen


def select_agents(
    records: Sequence[AgentRecord],
    query: str,
    *,
    limit: int = DEFAULT_SHORTLIST,
    recent: int = DEFAULT_RECENT,
    rankers: Optional[Sequence[Ranker]] = None,
) -> List[AgentRecord]:
    """The agents to show for this message: the best matches plus the most recently used ones.

    Synchronous: only for rankers that do not wait on anything (the word ranker). The running app uses
    `aselect_agents`, which can also await the embedding ranker."""
    live = [r for r in records if r.status != ARCHIVED]
    if len(live) <= limit + recent:
        return list(live)  # small roster: nothing is hidden, so retrieval cannot miss
    rankers = rankers or [BM25Ranker()]
    fused = reciprocal_rank_fusion([ranker.rank(query, live) for ranker in rankers],
                                   weights=[getattr(r, "weight", 1.0) for r in rankers])
    return _assemble(live, fused, limit, recent)


async def aselect_agents(
    records: Sequence[AgentRecord],
    query: str,
    *,
    limit: int = DEFAULT_SHORTLIST,
    recent: int = DEFAULT_RECENT,
    rankers: Optional[Sequence[Ranker]] = None,
) -> List[AgentRecord]:
    """Same as `select_agents`, but rankers may be async (e.g. calling an embedding service)."""
    live = [r for r in records if r.status != ARCHIVED]
    if len(live) <= limit + recent:
        return list(live)
    rankers = list(rankers or default_rankers())
    rankings = []
    for ranker in rankers:
        ranking = ranker.rank(query, live)
        rankings.append(await ranking if inspect.isawaitable(ranking) else ranking)
    fused = reciprocal_rank_fusion(rankings, weights=[getattr(r, "weight", 1.0) for r in rankers])
    return _assemble(live, fused, limit, recent)


_default_rankers: Optional[List[Ranker]] = None


def default_rankers() -> List[Ranker]:
    """Word ranker always; plus the semantic ranker when an embedding model is configured."""
    global _default_rankers
    if _default_rankers is None:
        from ...config import get_settings
        from .embeddings import EmbeddingRanker, OpenRouterEmbedder, VectorCache
        from .roster import _DATA_DIR

        settings = get_settings()
        rankers: List[Ranker] = [BM25Ranker()]
        if settings.embedding_model and settings.openrouter_api_key:
            rankers[0].weight = 0.5  # words help with exact names and IDs; meaning leads
            cache_file = _DATA_DIR / "execution_agents" / ("embeddings_" + re.sub(r"[^a-z0-9]+", "_", settings.embedding_model.lower()) + ".json")
            rankers.append(EmbeddingRanker(OpenRouterEmbedder(settings.embedding_model, settings.openrouter_api_key),
                                           VectorCache(cache_file)))
        _default_rankers = rankers
    return _default_rankers


# ---------------------------------------------------------------------- describing a new agent from its first request
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")  # no trailing "." from the end of a sentence
# A name only counts when something says it is a person: "email Alice", "with Pia Cole", "to Dana", "Alice's".
# (Matching any capitalised words turned topics like "Podcast Guest" and words like "Follow" into "people".)
_CUE = r"(?i:to|with|from|email|e-mail|send|write|contact|ping|tell|ask|message|text|call|cc|bcc|invite|meet|thank|remind|reply to|follow up with|for)"
_NAME_AFTER_CUE = re.compile(rf"\b{_CUE}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b")
_POSSESSIVE = re.compile(r"\b([A-Z][a-z]+)'s\b")
_NOT_NAMES = frozenset(
    "I Please Email Send Tell Ask Write Draft Reply Forward Check Find Search Set Remind Make Create Update "
    "Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July "
    "August September October November December Gmail Poke The This That These Those Can Could Would Will "
    "Today Tomorrow Tonight Me You Him Her Them Everyone".split()
)


def extract_people(text: str) -> List[str]:
    """Email addresses, and names that a cue marks as a person (rule based, no model call)."""
    people: List[str] = list(dict.fromkeys(_EMAIL.findall(text or "")))
    for match in [*_NAME_AFTER_CUE.findall(text or ""), *_POSSESSIVE.findall(text or "")]:
        name = " ".join(w for w in match.split() if w not in _NOT_NAMES)
        if name and name not in people:
            people.append(name)
    return people[:6]


def describe_request(text: str, limit: int = 160) -> str:
    """A one-line description of what an agent is for, taken from its first request."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"
