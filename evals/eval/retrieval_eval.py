"""Retrieval eval: is the right agent on the list the main agent sees?  (cheap: embeddings only, no chat model)

For every labeled request, each configuration (word ranker, semantic ranker, hybrid) ranks the 277 simulated
agents. We check whether an agent of the expected thread is in the top k, and on the list the main agent would
actually receive (top 10 + the 3 most recent). The 14 paraphrase requests share no words with their agent, so
they separate word matching from semantic matching.

    python evals/eval/retrieval_eval.py --label compare \
        --models perplexity/pplx-embed-v1-0.6b,voyageai/voyage-4-lite,perplexity/pplx-embed-v1-4b
    python evals/eval/retrieval_eval.py --label lexical            # words only, fully offline

Runs on a sandbox copy of server/ (live data untouched). Output: evals/results/retrieval_<label>/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
KS = (3, 5, 10)


def _api_key() -> str:
    if os.getenv("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    return ""


def outer(args) -> None:
    results = ROOT / "evals" / "results" / f"retrieval_{args.label}"
    shutil.rmtree(results, ignore_errors=True)
    results.mkdir(parents=True)
    sandbox = Path(tempfile.mkdtemp(prefix="openpoke-retrieval-"))
    shutil.copytree(ROOT / "server", sandbox / "server", ignore=shutil.ignore_patterns("data", "__pycache__", "*.pyc", "*.log"))
    for f in ("retrieval_eval.py", "dataset.py"):
        shutil.copy(HERE / f, sandbox / f)
    env = dict(os.environ, OPENROUTER_API_KEY=_api_key() or "unused", PYTHONPATH=str(sandbox),
               OPENPOKE_USAGE_LOG=str(results / "llm_usage.jsonl"), OPENPOKE_EMBEDDING_MODEL="")
    cmd = [sys.executable, str(sandbox / "retrieval_eval.py"), "--inner", "--label", args.label,
           "--results", str(results), "--models", args.models,
           "--lexical-weights", args.lexical_weights]
    rc = subprocess.run(cmd, cwd=sandbox, env=env).returncode
    shutil.rmtree(sandbox, ignore_errors=True)
    sys.exit(rc)


async def inner(args) -> None:
    sys.path.insert(0, os.getcwd())
    import dataset
    from server.agents.interaction_agent.agent import _format_agents
    from server.openrouter_client.usage import set_tag
    from server.services.execution import get_agent_roster
    from server.services.execution.embeddings import EmbeddingRanker, OpenRouterEmbedder, VectorCache
    from server.services.execution.retrieval import BM25Ranker, aselect_agents, reciprocal_rank_fusion

    truth = dataset.ground_truth()
    group_of = {a["name"]: a["group"] for a in truth["agents"]}
    primary_of = {}
    for a in truth["agents"]:
        primary_of.setdefault(a["group"], a["name"])
    roster = get_agent_roster()
    roster.clear()
    from server.services.execution.retrieval import extract_people

    with roster.batch():
        for agent in dataset.build_agents():  # descriptions from realistic first requests, as in production
            request = dataset.first_request(agent)
            roster.add_agent(agent.name, description=request, people=extract_people(request))
    records = roster.list_records()
    paraphrases = [asdict(r) for r in dataset.build_paraphrase_requests(dataset.build_agents())]
    wanted = [r for r in truth["requests"] + paraphrases if r["expected_group"]]
    cats = sorted({r["category"] for r in wanted})
    key = os.environ["OPENROUTER_API_KEY"]

    configs = [("words only", [BM25Ranker()])]
    for model in [m for m in args.models.split(",") if m]:
        embedder = OpenRouterEmbedder(model, key, timeout=20.0)
        semantic = EmbeddingRanker(embedder, VectorCache())
        configs.append((f"semantic {model.split('/')[-1]}", [semantic]))
        for weight in [float(w) for w in args.lexical_weights.split(",") if w]:
            configs.append((f"hybrid w={weight:<4} {model.split('/')[-1]}", [BM25Ranker(weight=weight), semantic]))

    def hit(names, group):
        return any(group_of[n] == group for n in names)

    out = {"label": args.label, "agents": len(records), "requests": len(wanted), "configs": {}}
    for name, rankers in configs:
        set_tag(f"retrieval:{name}")
        recall = {k: {c: [] for c in cats} for k in KS}
        shipped = {c: [] for c in cats}
        primary, missed = [], []
        for r in wanted:
            rankings = [await x.rank(r["text"], records) if asyncio.iscoroutinefunction(x.rank) else x.rank(r["text"], records)
                        for x in rankers]
            fused = [x.name for x in reciprocal_rank_fusion(rankings)]
            for k in KS:
                recall[k][r["category"]].append(hit(fused[:k], r["expected_group"]))
            shown = [x.name for x in await aselect_agents(records, r["text"], rankers=rankers)]
            ok = hit(shown, r["expected_group"])
            shipped[r["category"]].append(ok)
            primary.append(primary_of[r["expected_group"]] in shown)
            if not ok:
                missed.append(r["id"])
        flat = lambda d: [v for vs in d.values() for v in vs]  # noqa: E731
        out["configs"][name] = {
            "recall_at": {k: {"all": sum(flat(recall[k])) / len(flat(recall[k])),
                              **{c: sum(v) / len(v) for c, v in recall[k].items()}} for k in KS},
            "shipped": {"all": sum(flat(shipped)) / len(flat(shipped)), **{c: sum(v) / len(v) for c, v in shipped.items()}},
            "primary_agent_on_list": sum(primary) / len(primary),
            "missed": missed,
        }
    sample = _format_agents(records, await aselect_agents(records, wanted[0]["text"], rankers=configs[0][1]))
    full = "\n".join(f'<agent name="{a["name"]}" />' for a in truth["agents"])
    out["agent_list_tokens_est"] = {"every_name": round(len(full) / 3.5), "shipped": round(len(sample) / 3.5)}
    (Path(args.results) / "summary.json").write_text(json.dumps(out, indent=2))

    counts = ", ".join(c + " " + str(sum(1 for r in wanted if r["category"] == c)) for c in cats)
    lines = [f"RETRIEVAL EVAL  label={args.label}  agents={out['agents']}  requests={out['requests']} ({counts})", "=" * 100,
             f"Right thread on the list the main agent sees (top 10 + 3 most recent):",
             f"   {'configuration':<38}{'all':>7}" + "".join(f"{c:>11}" for c in cats) + f"{'main agent':>13}"]
    for name, c in out["configs"].items():
        lines.append(f"   {name:<38}{100*c['shipped']['all']:6.1f}%" + "".join(f"{100*c['shipped'][x]:10.0f}%" for x in cats)
                     + f"{100*c['primary_agent_on_list']:12.0f}%")
    lines += ["", "Ranker quality alone, right thread within the top k (all requests):",
              f"   {'configuration':<38}" + "".join(f"{'top '+str(k):>9}" for k in KS) + f"{'paraphrase top 5':>20}"]
    for name, c in out["configs"].items():
        lines.append(f"   {name:<38}" + "".join(f"{100*c['recall_at'][k]['all']:8.1f}%" for k in KS)
                     + f"{100*c['recall_at'][5].get('paraphrase', 0):19.0f}%")
    t = out["agent_list_tokens_est"]
    lines += ["", f"Agent list size (estimate): {t['shipped']} tokens shown vs {t['every_name']} tokens for every name (names only)."]
    (Path(args.results) / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="lexical")
    ap.add_argument("--models", default="", help="comma separated OpenRouter embedding models")
    ap.add_argument("--lexical-weights", default="1.0,0.5,0.25", help="weights of the word ranker in the hybrid")
    ap.add_argument("--inner", action="store_true")
    ap.add_argument("--results")
    args = ap.parse_args()
    asyncio.run(inner(args)) if args.inner else outer(args)


if __name__ == "__main__":
    main()
