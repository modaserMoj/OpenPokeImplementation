"""Lifecycle eval: does duplicate detection merge the right agents and NEVER the wrong ones?

Registry: the 277 simulated agents, but each with a description made from a realistic FIRST REQUEST for its own
name ("reply to Alice's last message", "follow up with Alice Chen on the lease renewal"...) and people extracted
from it the way production does - not the dataset's group-level descriptions, which are identical within a
group and would make merging trivially easy.

  A. consolidation: group existing near-duplicates; compare with the true thread groups.
     false merges (agents from DIFFERENT threads put together) must be 0.
  B. creation check: a new agent name + first request either reuses the right thread or creates a new agent.
     positives: new names for existing threads (incl. paraphrases with no shared words)
     negatives: brand-new people/topics, the same topic with another person, the same first name with another
                last name.

    python evals/eval/lifecycle_eval.py --label words                    # free: words and people only
    python evals/eval/lifecycle_eval.py --label voyage --model voyageai/voyage-4-lite --sweep

--sweep tries several similarity thresholds (embeddings are cached, so it costs nothing extra).
Runs on a sandbox copy of server/. Output: evals/results/lifecycle_<label>/
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
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


def _api_key() -> str:
    if os.getenv("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    return ""


def outer(args) -> None:
    results = ROOT / "evals" / "results" / f"lifecycle_{args.label}"
    shutil.rmtree(results, ignore_errors=True)
    results.mkdir(parents=True)
    sandbox = Path(tempfile.mkdtemp(prefix="openpoke-lifecycle-"))
    shutil.copytree(ROOT / "server", sandbox / "server", ignore=shutil.ignore_patterns("data", "__pycache__", "*.pyc", "*.log"))
    for f in ("lifecycle_eval.py", "dataset.py"):
        shutil.copy(HERE / f, sandbox / f)
    env = dict(os.environ, OPENROUTER_API_KEY=_api_key() if args.model else "unused", PYTHONPATH=str(sandbox),
               OPENPOKE_USAGE_LOG=str(results / "llm_usage.jsonl"), OPENPOKE_EMBEDDING_MODEL="")
    cmd = [sys.executable, str(sandbox / "lifecycle_eval.py"), "--inner", "--label", args.label,
           "--results", str(results), "--model", args.model] + (["--sweep"] if args.sweep else [])
    rc = subprocess.run(cmd, cwd=sandbox, env=env).returncode
    shutil.rmtree(sandbox, ignore_errors=True)
    sys.exit(rc)


def creation_cases(dataset, agents):
    by_group = {}
    for a in agents:
        by_group.setdefault(a.group, []).append(a)
    person_groups = [g for g, v in by_group.items() if v[0].kind == "person"]
    cases = []
    for g in person_groups[:15]:
        a = by_group[g][0]
        first = a.people[0].split()[0]
        cases.append(("new name, same thread", f"{first} - {a.topics[0]} update",
                      f"tell {first} the {a.topics[0]} is moving forward", g))
    for topic, text in list(dataset.PARAPHRASES.items())[:10]:
        group = next(x.group for x in agents if x.kind == "topic" and x.topics[0] == topic)
        cases.append(("paraphrase, same thread", text.split(" the ")[-1].title()[:40], text, group))
    for name, text in [("Email to Marguerite Okonkwo", "email Marguerite Okonkwo about the kayak rental"),
                       ("Sublet with Tobias", "ask Tobias Lindqvist about the sublet agreement"),
                       ("Violin lessons", "find violin lessons for the kids"),
                       ("Warehouse booking", "book the warehouse for March")]:
        cases.append(("brand new", name, text, None))
    used_topics = set()
    for g in person_groups[15:]:
        a = by_group[g][0]
        if a.topics[0] in used_topics or len(cases) >= 45:
            continue
        used_topics.add(a.topics[0])
        cases.append(("same topic, another person", f"Email to Zoltan Varga", f"email Zoltan Varga about the {a.topics[0]}", None))
    for g in person_groups[:6]:
        a = by_group[g][0]
        first = a.people[0].split()[0]
        cases.append(("same first name, another person", f"Email to {first} Okafor",
                      f"email {first} Okafor about the {a.topics[0]}", None))
    return cases


async def inner(args) -> None:
    sys.path.insert(0, os.getcwd())
    import dataset
    from server.services.execution import AgentRoster, lifecycle
    from server.services.execution.embeddings import EmbeddingRanker, OpenRouterEmbedder, VectorCache
    from server.services.execution.retrieval import BM25Ranker, extract_people

    agents = dataset.build_agents()
    group_of = {a.name: a.group for a in agents}
    roster = AgentRoster(Path(tempfile.mkdtemp()) / "roster.json")
    with roster.batch():
        for a in agents:
            request = dataset.first_request(a)
            roster.add_agent(a.name, description=request, people=extract_people(request))

    rankers = [BM25Ranker()]
    if args.model:
        semantic = EmbeddingRanker(OpenRouterEmbedder(args.model, os.environ["OPENROUTER_API_KEY"], timeout=30), VectorCache())
        rankers = [BM25Ranker(weight=0.5), semantic]

    true_pairs = {frozenset(p) for g in {a.group for a in agents}
                  for p in itertools.combinations([a.name for a in agents if a.group == g], 2)}
    cases = creation_cases(dataset, agents)

    async def evaluate():
        groups = await lifecycle.find_duplicate_groups(roster, rankers)
        predicted_pairs = {frozenset(p) for g in groups for p in itertools.combinations([r.name for r in g], 2)}
        wrong_groups = [[r.name for r in g] for g in groups if len({group_of[r.name] for r in g}) > 1]
        correct = predicted_pairs & true_pairs
        creation = []
        for kind, name, text, expected in cases:
            found = await lifecycle.find_duplicate(name, text, roster=roster, rankers=rankers)
            got = group_of.get(found.name) if found else None
            creation.append({"kind": kind, "name": name, "expected": expected, "got": got, "ok": got == expected})
        return {
            "consolidation": {
                "groups_formed": len(groups),
                "false_merge_groups": len(wrong_groups),
                "pair_precision": len(correct) / len(predicted_pairs) if predicted_pairs else 1.0,
                "pair_recall": len(correct) / len(true_pairs),
                "agents_after": len(agents) - sum(len(g) - 1 for g in groups),
                "wrong_groups": wrong_groups[:10],
            },
            "creation": {
                "reuse_when_should": sum(c["ok"] for c in creation if c["expected"]) / max(1, sum(1 for c in creation if c["expected"])),
                "wrong_reuse": sum(1 for c in creation if c["got"] and c["got"] != c["expected"]),
                "by_kind": {k: f"{sum(c['ok'] for c in creation if c['kind'] == k)}/{sum(1 for c in creation if c['kind'] == k)}"
                            for k in dict.fromkeys(c["kind"] for c in creation)},
                "mistakes": [c for c in creation if not c["ok"]],
            },
        }

    runs = {}
    settings_now = {k: getattr(lifecycle, k) for k in ("SAME_TOPIC_SIMILARITY", "DIFFERENT_TOPIC_SIMILARITY", "TOPIC_ONLY_SIMILARITY")}
    runs["defaults " + json.dumps(settings_now)] = await evaluate()
    if args.sweep and args.model:
        for same, topic_only in itertools.product((0.45, 0.55, 0.65), (0.70, 0.75, 0.80, 0.85)):
            lifecycle.SAME_TOPIC_SIMILARITY, lifecycle.TOPIC_ONLY_SIMILARITY = same, topic_only
            runs[f"same_topic={same} topic_only={topic_only}"] = await evaluate()
        for k, v in settings_now.items():
            setattr(lifecycle, k, v)

    usage = Path(os.environ.get("OPENPOKE_USAGE_LOG", ""))
    cost = sum((json.loads(l).get("cost_usd") or 0) for l in usage.read_text().splitlines()) if usage.exists() else 0.0
    out = {"label": args.label, "model": args.model or "(words and people only)", "agents": len(agents),
           "true_duplicate_pairs": len(true_pairs), "creation_cases": len(cases), "cost_usd": cost, "runs": runs}
    (Path(args.results) / "summary.json").write_text(json.dumps(out, indent=2))
    lines = [f"LIFECYCLE EVAL  label={args.label}  model={out['model']}  agents={len(agents)}  "
             f"true duplicate pairs={len(true_pairs)}  creation cases={len(cases)}  cost=${cost:.4f}", "=" * 118,
             f"   {'settings':<52}{'false merges':>13}{'precision':>10}{'recall':>8}{'agents left':>12}"
             f"{'creation: reuse ok':>20}{'wrong reuse':>12}"]
    for name, r in runs.items():
        c, k = r["consolidation"], r["creation"]
        lines.append(f"   {name[:52]:<52}{c['false_merge_groups']:>13}{100*c['pair_precision']:>9.0f}%{100*c['pair_recall']:>7.0f}%"
                     f"{c['agents_after']:>12}{100*k['reuse_when_should']:>19.0f}%{k['wrong_reuse']:>12}")
    first = next(iter(runs.values()))
    lines += ["", "Creation check by case (defaults): " + ", ".join(f"{k} {v}" for k, v in first["creation"]["by_kind"].items())]
    if first["consolidation"]["wrong_groups"]:
        lines += ["", "WRONG merges (defaults):", *[f"   {g}" for g in first["consolidation"]["wrong_groups"]]]
    (Path(args.results) / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="words")
    ap.add_argument("--model", default="", help="embedding model; empty = words and people only (free)")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--inner", action="store_true")
    ap.add_argument("--results")
    args = ap.parse_args()
    asyncio.run(inner(args)) if args.inner else outer(args)


if __name__ == "__main__":
    main()
