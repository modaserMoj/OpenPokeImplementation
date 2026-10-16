"""Semantic ranking and its fusion with the word ranker (issue 1). A fake embedder keeps this free and exact."""

import re

import pytest
from helpers import run

from server.services.execution import AgentRoster
from server.services.execution import retrieval
from server.services.execution.embeddings import EmbeddingRanker, VectorCache, agent_text
from server.services.execution.retrieval import BM25Ranker, aselect_agents

# Words that mean the same thing share a concept, the way an embedding model places them close together.
CONCEPTS = [
    {"pet", "dog", "cat", "vaccination", "shots", "vet"},
    {"passport", "travel", "document", "renewal", "renewed", "visa"},
    {"mortgage", "loan", "refinance", "restructuring", "home"},
    {"tax", "revenue", "receipts", "filing"},
]


class FakeEmbedder:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    async def embed(self, texts):
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("embedding service down")
        vectors = []
        for text in texts:
            words = set(re.findall(r"[a-z]+", text.lower()))
            vector = [float(len(words & concept)) for concept in CONCEPTS] + [0.01]
            norm = sum(x * x for x in vector) ** 0.5
            vectors.append([x / norm for x in vector])
        return vectors


def _roster(tmp_path, extra=60):
    roster = AgentRoster(tmp_path / "r.json")
    with roster.batch():
        roster.add_agent("Pet vaccination", description="Ongoing task: Pet vaccination.")
        roster.add_agent("Passport renewal", description="Ongoing task: Passport renewal.")
        roster.add_agent("Email to Tess", description="Emails with Tess Gibson about school pickup", people=["Tess Gibson"])
        for i in range(extra):
            roster.add_agent(f"Filler {i}", description=f"unrelated thing number {i}")
    return roster


def test_the_semantic_ranker_finds_a_paraphrase_the_word_ranker_cannot(tmp_path):
    records = _roster(tmp_path).list_records()
    query = "book the shots for my dog"
    assert BM25Ranker().rank(query, records) == []  # no shared word
    ranked = run(EmbeddingRanker(FakeEmbedder()).rank(query, records))
    assert ranked[0].name == "Pet vaccination"


def test_hybrid_keeps_exact_name_matches_and_paraphrase_matches(tmp_path):
    records = _roster(tmp_path).list_records()
    rankers = [BM25Ranker(), EmbeddingRanker(FakeEmbedder())]
    paraphrase = run(aselect_agents(records, "renew my travel document", limit=5, recent=0, rankers=rankers))
    exact = run(aselect_agents(records, "any word from Gibson about school pickup", limit=5, recent=0, rankers=rankers))
    assert "Passport renewal" in [r.name for r in paraphrase]
    assert exact[0].name == "Email to Tess"


def test_if_the_embedding_service_fails_the_word_ranker_still_answers(tmp_path):
    records = _roster(tmp_path).list_records()
    rankers = [BM25Ranker(), EmbeddingRanker(FakeEmbedder(fail=True))]
    shown = run(aselect_agents(records, "any word from Gibson about school pickup", limit=5, recent=0, rankers=rankers))
    assert shown[0].name == "Email to Tess"  # no exception, no lost turn


def test_agents_are_embedded_once_and_reused_from_the_cache(tmp_path):
    records = _roster(tmp_path).list_records()
    embedder = FakeEmbedder()
    ranker = EmbeddingRanker(embedder, VectorCache(tmp_path / "vectors.json"))
    run(ranker.rank("passport", records))
    run(ranker.rank("mortgage", records))
    agent_batches = [c for c in embedder.calls if len(c) > 1]
    query_calls = [c for c in embedder.calls if len(c) == 1]
    assert len(agent_batches) == 1 and len(query_calls) == 2  # one batch for every agent, then only the queries
    # and the cache survives a restart
    again = FakeEmbedder()
    run(EmbeddingRanker(again, VectorCache(tmp_path / "vectors.json")).rank("passport", records))
    assert [c for c in again.calls if len(c) > 1] == []


def test_a_changed_description_is_embedded_again(tmp_path):
    roster = _roster(tmp_path, extra=0)
    embedder = FakeEmbedder()
    ranker = EmbeddingRanker(embedder)
    run(ranker.rank("passport", roster.list_records()))
    record = roster.resolve("Passport renewal")
    record.description = "Now also about the visa"
    run(ranker.rank("passport", roster.list_records()))
    assert any(len(c) == 1 and "visa" in c[0] for c in embedder.calls[2:])  # the new text, not the unchanged ones


def test_agent_text_contains_what_identifies_the_agent(tmp_path):
    record = _roster(tmp_path, extra=0).resolve("Email to Tess")
    text = agent_text(record)
    assert "Email to Tess" in text and "school pickup" in text and "Tess Gibson" in text


def test_semantic_ranking_is_off_unless_an_embedding_model_is_configured(monkeypatch, tmp_path):
    from server import config
    from server.services.execution import roster as roster_module

    monkeypatch.setattr(roster_module, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(retrieval, "_default_rankers", None)
    assert [type(r).__name__ for r in retrieval.default_rankers()] == ["BM25Ranker"]  # tests run with it disabled

    configured = config.get_settings().model_copy(update={"embedding_model": "some/model"})
    monkeypatch.setattr(config, "get_settings", lambda: configured)
    monkeypatch.setattr(retrieval, "_default_rankers", None)
    assert [type(r).__name__ for r in retrieval.default_rankers()] == ["BM25Ranker", "EmbeddingRanker"]
