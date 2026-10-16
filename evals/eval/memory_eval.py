"""Memory eval: after compaction, does the agent still know what matters, and how much smaller is its prompt?

Builds one long-lived agent ("Email to Dana", 40 runs of realistic activity with long email-search results) and
plants facts in its older runs: identifiers (draft / thread / reminder ids, an email address) and facts that only
a summary can carry (a preference, an amount, a deadline, a rejection, a standing instruction). Then compacts
the memory and checks which facts are still in what the agent would see.

    python evals/eval/memory_eval.py --label offline --fake   # free: a summariser that keeps NOTHING
    python evals/eval/memory_eval.py --label haiku            # the configured summariser model (cents)
    python evals/eval/memory_eval.py --label sonnet --model anthropic/claude-sonnet-5.5

--fake gives the floor: what the code alone preserves (identifiers), with every summary-only fact lost.
Runs on a sandbox copy of server/. Output: evals/results/memory_<label>/
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
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TOKENS_PER_CHAR = 0.283  # measured with the real model on this codebase's prompts (cost_scaling_baseline)

# (run index, text placed in that run's final answer, what must survive, kind)
PLANTED = [
    (2, "Draft saved for the lease renewal, draft_id=r8553019871675242050.", "r8553019871675242050", "identifier"),
    (4, "Found the lease thread, thread_id=19a2f0c3d4e5b6a7.", "19a2f0c3d4e5b6a7", "identifier"),
    (6, "Reminder created to chase the deposit, trigger_id=7.", "trigger_id=7", "identifier"),
    (8, "Dana's new address is dana.cruz@newco.test (old one bounces).", "dana.cruz@newco.test", "identifier"),
    (10, "Dana said she prefers meetings on Tuesday afternoons.", "tuesday", "summary-only fact"),
    (12, "The invoice total Dana confirmed is $4,180.", "4,180", "summary-only fact"),
    (14, "The user rejected the first draft because it was too formal.", "formal", "summary-only fact"),
    (16, "The lease renewal deadline is November 30.", "november 30", "summary-only fact"),
    (18, "The user asked to always cc Priya on lease emails.", "priya", "summary-only fact"),
]
RUNS = 40


def _api_key() -> str:
    if os.getenv("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    return ""


def outer(args) -> None:
    results = ROOT / "evals" / "results" / f"memory_{args.label}"
    shutil.rmtree(results, ignore_errors=True)
    results.mkdir(parents=True)
    sandbox = Path(tempfile.mkdtemp(prefix="openpoke-memory-"))
    shutil.copytree(ROOT / "server", sandbox / "server", ignore=shutil.ignore_patterns("data", "__pycache__", "*.pyc", "*.log"))
    shutil.copy(HERE / "memory_eval.py", sandbox / "memory_eval.py")
    env = dict(os.environ, OPENROUTER_API_KEY=(_api_key() if not args.fake else "") or "unused",
               PYTHONPATH=str(sandbox), OPENPOKE_USAGE_LOG=str(results / "llm_usage.jsonl"), OPENPOKE_EMBEDDING_MODEL="")
    cmd = [sys.executable, str(sandbox / "memory_eval.py"), "--inner", "--label", args.label, "--results", str(results)]
    if args.fake:
        cmd.append("--fake")
    if args.model:
        env["OPENPOKE_MODEL_SUMMARIZER"] = args.model  # e.g. compare haiku and sonnet summaries
    rc = subprocess.run(cmd, cwd=sandbox, env=env).returncode
    shutil.rmtree(sandbox, ignore_errors=True)
    sys.exit(rc)


def _email_search_result(i: int) -> str:
    return json.dumps({"data": {"messages": [
        {"messageId": f"18f{i:02d}{j:04d}", "threadId": f"t{i:02d}{j:02d}", "from": f"person{j}@example.test",
         "subject": f"Re: lease item {i}-{j}", "date": "2026-10-01", "snippet": "lorem ipsum " * 15}
        for j in range(8)]}})


async def inner(args) -> None:
    sys.path.insert(0, os.getcwd())
    from server.services.execution import get_execution_agent_logs, memory
    from server.services.execution.memory import render_history

    name = "Email to Dana"
    logs = get_execution_agent_logs()
    planted = {run: text for run, text, _, _ in PLANTED}
    for i in range(RUNS):
        logs.record_request(name, f"Run {i}: check the lease thread with Dana and update the user.")
        memory.record_tool_call(name, "task_email_search", json.dumps({"search_query": "from:dana lease"}),
                                _email_search_result(i))
        logs.record_agent_response(name, f"Checked {8} new lease emails. {planted.get(i, 'Nothing new to report.')}")

    before_chars = len(logs.load_transcript(name))  # what the OLD code pasted into every prompt

    async def keep_nothing(agent_name, previous, runs_text):
        return "Purpose: lease emails with Dana."

    summariser = keep_nothing if args.fake else None
    rounds = 0
    while await memory.compact(name, summariser):
        rounds += 1
    view = render_history(name)
    lower = view.lower()

    rows = [(kind, text, must, must.lower() in lower) for _, text, must, kind in PLANTED]
    by_kind = {}
    for kind, _, _, ok in rows:
        by_kind.setdefault(kind, []).append(ok)
    usage_path = Path(os.environ.get("OPENPOKE_USAGE_LOG", ""))
    calls = [json.loads(l) for l in usage_path.read_text().splitlines()] if usage_path.exists() else []
    cost = sum(c.get("cost_usd") or 0 for c in calls)

    out = {
        "label": args.label,
        "summariser": "keeps nothing (floor)" if args.fake else __import__("server.config", fromlist=["x"]).get_settings().summarizer_model,
        "runs": RUNS, "compaction_rounds": rounds,
        "history_chars_before": before_chars, "history_chars_after": len(view),
        "history_tokens_est_before": round(before_chars * TOKENS_PER_CHAR),
        "history_tokens_est_after": round(len(view) * TOKENS_PER_CHAR),
        "retention": {k: sum(v) / len(v) for k, v in by_kind.items()},
        "facts": [{"kind": k, "fact": t, "kept": ok} for k, t, _, ok in rows],
        "summariser_calls": len(calls), "summariser_cost_usd": cost,
    }
    (Path(args.results) / "summary.json").write_text(json.dumps(out, indent=2))
    (Path(args.results) / "history_view.txt").write_text(view)
    lines = [
        f"MEMORY EVAL  label={args.label}  summariser: {out['summariser']}",
        "=" * 78,
        f"agent history after {RUNS} runs: {before_chars:,} chars (~{out['history_tokens_est_before']:,} tokens) in the old prompt",
        f"                       now:      {len(view):,} chars (~{out['history_tokens_est_after']:,} tokens)  "
        f"[{rounds} compaction rounds, ${cost:.4f}]",
        "",
        "Facts planted in older runs, still visible to the agent:",
        *[f"   {'KEPT ' if ok else 'LOST '} [{k}] {t}" for k, t, _, ok in rows],
        "",
        *[f"   {k}: {100 * v:.0f}% kept" for k, v in out["retention"].items()],
    ]
    (Path(args.results) / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="offline")
    ap.add_argument("--fake", action="store_true", help="summariser that keeps nothing: free, shows the floor")
    ap.add_argument("--model", default="", help="summariser model to use (default: the configured one)")
    ap.add_argument("--inner", action="store_true")
    ap.add_argument("--results")
    args = ap.parse_args()
    asyncio.run(inner(args)) if args.inner else outer(args)


if __name__ == "__main__":
    main()
