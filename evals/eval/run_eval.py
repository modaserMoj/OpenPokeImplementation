"""Routing eval for the interaction agent: does it reuse the RIGHT existing agent?

    python evals/eval/run_eval.py --label baseline --max-usd 1.6     # real model
    python evals/eval/run_eval.py --label smoke --fake               # free, no LLM
    python evals/eval/run_eval.py --label baseline --resume          # skip finished ids

What is real: the interaction agent runtime, its prompt, the roster rendering, the model.
What is stubbed: execution agents / Gmail (send_message_to_agent only records the choice),
and the conversation is reset before every request so requests are independent.
Only the interaction agent's routing calls cost money. The run stops as soon as it has
seen a send_message_to_agent call (or after MAX_CALLS LLM calls).

The server code is COPIED into a temp sandbox first, so the live app's data is never touched.
Results: evals/results/<label>/{results.jsonl,summary.txt,summary.json,llm_usage.jsonl}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MAX_CALLS = 4


# --------------------------------------------------------------------------- #
# outer: build sandbox, launch inner, summarise
# --------------------------------------------------------------------------- #
def _api_key() -> str:
    if os.getenv("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return ""


def outer(args) -> None:
    results = ROOT / "evals" / "results" / args.label
    results.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        for f in ("results.jsonl", "llm_usage.jsonl"):
            (results / f).unlink(missing_ok=True)

    sandbox = Path(tempfile.mkdtemp(prefix="openpoke-eval-"))
    shutil.copytree(
        ROOT / "server", sandbox / "server",
        ignore=shutil.ignore_patterns("data", "__pycache__", "*.pyc", "*.log"),
    )
    for f in ("run_eval.py", "dataset.py"):
        shutil.copy(HERE / f, sandbox / f)

    env = dict(os.environ)
    env.update(
        OPENROUTER_API_KEY=_api_key() or "fake",
        OPENPOKE_USAGE_LOG=str(results / "llm_usage.jsonl"),
        PYTHONPATH=str(sandbox),
    )
    for k in ("COMPOSIO_API_KEY", "COMPOSIO_GMAIL_AUTH_CONFIG_ID"):
        env.pop(k, None)
    cmd = [sys.executable, str(sandbox / "run_eval.py"), "--inner", "--label", args.label,
           "--results", str(results), "--max-usd", str(args.max_usd)]
    if args.fake:
        cmd.append("--fake")
        env["OPENPOKE_EMBEDDING_MODEL"] = ""  # the free mode must not call any service
    cmd += ["--set", args.set]
    cmd += ["--pace", str(args.pace)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    print(f"sandbox: {sandbox}")
    rc = subprocess.run(cmd, cwd=sandbox, env=env).returncode
    shutil.rmtree(sandbox, ignore_errors=True)
    summarise(results, args.label)
    sys.exit(rc)


def _pct(num: int, den: int) -> str:
    return f"{100 * num / den:5.1f}%" if den else "  n/a"


def _quantile(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def summarise(results: Path, label: str) -> dict:
    all_rows = [json.loads(l) for l in (results / "results.jsonl").read_text().splitlines() if l.strip()]
    api_errors = sum(r["outcome"] == "api_error" for r in all_rows)
    retries = sum(r.get("retries_429", 0) for r in all_rows)
    rows = [r for r in all_rows if r["outcome"] != "api_error"]
    n = len(rows)
    sys.path.insert(0, str(HERE))
    import dataset  # ground truth: which agent is the "primary" (main history) of each group

    agents = dataset.build_agents()
    primary = {}
    size = {}
    for a in agents:
        primary.setdefault(a.group, a.name)
        size[a.group] = size.get(a.group, 0) + 1
    frag = [r for r in rows if r["expected_group"] and size.get(r["expected_group"], 0) > 1 and r["routed"]]
    frag_hit = sum(r["routed"][0] == primary[r["expected_group"]] for r in frag)
    outcome = lambda r: r["outcome"]  # noqa: E731
    existing = [r for r in rows if r["expected_group"] is not None]
    cats = sorted({r["category"] for r in rows})
    correct = [r for r in rows if outcome(r) in ("correct_reuse", "correct_new")]
    rescued = [r for r in rows if r.get("safety_net") == "rescued"]
    import hashlib
    request_set = "paraphrase" if rows and rows[0]["id"].startswith("para-") else "main"
    if request_set == "main":
        fingerprint = hashlib.sha256(json.dumps(dataset.ground_truth(), sort_keys=True).encode()).hexdigest()[:12]
    else:
        from dataclasses import asdict
        fingerprint = "para-" + hashlib.sha256(json.dumps(
            [asdict(r) for r in dataset.build_paraphrase_requests(dataset.build_agents())], sort_keys=True).encode()).hexdigest()[:8]
    s = {
        "label": label,
        "dataset_sha": fingerprint,
        "requests": n,
        "api_errors_excluded": api_errors,
        "rate_limit_retries": retries,
        "roster_size": rows[0]["roster_size"] if rows else 0,
        "accuracy": len(correct) / n if n else 0,
        "accuracy_with_duplicate_check": (len(correct) + len(rescued)) / n if n else 0,
        "reuse_recall": (sum(outcome(r) == "correct_reuse" for r in existing) / len(existing)) if existing else 0,
        "duplicate_agent_rate": (sum(outcome(r) == "missed_reuse" for r in existing) / len(existing)) if existing else 0,
        "fragmented_requests": len(frag),
        "primary_hit_rate": (frag_hit / len(frag)) if frag else 0,
        "wrong_reuse_rate": sum(outcome(r) == "wrong_reuse" for r in rows) / n if n else 0,
        "no_route_rate": sum(outcome(r) == "no_route" for r in rows) / n if n else 0,
        "by_category": {
            c: sum(outcome(r) in ("correct_reuse", "correct_new") for r in rows if r["category"] == c)
               / max(1, sum(1 for r in rows if r["category"] == c))
            for c in cats
        },
        "outcomes": {o: sum(outcome(r) == o for r in rows) for o in
                     ("correct_reuse", "correct_new", "missed_reuse", "wrong_reuse", "no_route")},
        "prompt_tokens_first_call_avg": statistics.mean(r["prompt_tokens_first"] for r in rows) if rows else 0,
        "prompt_tokens_per_request_avg": statistics.mean(r["prompt_tokens_total"] for r in rows) if rows else 0,
        "llm_calls_per_request_avg": statistics.mean(r["llm_calls"] for r in rows) if rows else 0,
        "cost_usd_per_request_avg": statistics.mean(r["cost_usd"] for r in rows) if rows else 0,
        "cost_usd_total": sum(r["cost_usd"] for r in rows),
        "latency_s_per_request_p50": _quantile([r["latency_s"] for r in rows], 0.5),
        "latency_s_per_request_p95": _quantile([r["latency_s"] for r in rows], 0.95),
    }
    (results / "summary.json").write_text(json.dumps(s, indent=2))
    o = s["outcomes"]
    lines = [
        f"ROUTING EVAL  label={label}  dataset={fingerprint}  requests={n}  roster={s['roster_size']} agents   (api errors excluded: {api_errors}, 429 retries: {retries})",
        "=" * 70,
        f"accuracy                 {_pct(len(correct), n)}   (right agent reused, or new agent when none fits)",
        f"  + duplicate check      {_pct(len(correct) + len(rescued), n)}   (new names caught by the check behind the model: {len(rescued)})",
        f"reuse recall             {_pct(o['correct_reuse'], len(existing))}   (existing thread -> right agent)",
        f"duplicate-agent rate     {_pct(o['missed_reuse'], len(existing))}   (existing thread -> NEW agent created)",
        f"primary-agent hit rate  {_pct(frag_hit, len(frag))}   (thread split over 2-4 agents -> went to the one holding the main history; {len(frag)} requests)",
        f"wrong-reuse rate         {_pct(o['wrong_reuse'], n)}   (sent to an unrelated agent)",
        f"no-route rate            {_pct(o['no_route'], n)}   (never called send_message_to_agent)",
        "",
        "accuracy by category:",
        *[f"  {c:<10} {100*v:5.1f}%" for c, v in s["by_category"].items()],
        "",
        f"prompt tokens, 1st call   avg {s['prompt_tokens_first_call_avg']:.0f}",
        f"prompt tokens / request   avg {s['prompt_tokens_per_request_avg']:.0f}   ({s['llm_calls_per_request_avg']:.1f} LLM calls)",
        f"cost / request            ${s['cost_usd_per_request_avg']:.4f}   total ${s['cost_usd_total']:.3f}",
        f"latency / request         p50 {s['latency_s_per_request_p50']:.1f}s   p95 {s['latency_s_per_request_p95']:.1f}s",
        "",
        "per-request outcomes (id, outcome, routed-to):",
        *[f"  {r['id']:<12} {r['outcome']:<14} {', '.join(r['routed']) or '-'}" for r in rows],
    ]
    (results / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
    return s


# --------------------------------------------------------------------------- #
# inner: runs inside the sandbox
# --------------------------------------------------------------------------- #
class _Stop(Exception):
    pass


def _fake_completion(request_text: str, roster: list[str]):
    """Free rule-based router used by --fake to test the pipeline (NOT a model)."""
    first = re.findall(r"\b(?:email|tell|write|message|with|send)\s+([A-Z][a-z]+)", request_text)
    name = next((a for f in first for a in roster if f in a), None) or "Email to Someone New"
    call = {"id": "t1", "type": "function",
            "function": {"name": "send_message_to_agent",
                         "arguments": json.dumps({"agent_name": name, "instructions": request_text})}}
    return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [call]}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 20, "cost": 0.0}}


async def inner(args) -> None:
    sys.path.insert(0, os.getcwd())
    import dataset
    from server.agents.interaction_agent import runtime as rt
    from server.agents.interaction_agent import tools as itools
    try:
        from server.openrouter_client.usage import set_tag
    except ImportError:  # older code without the usage recorder
        def set_tag(_tag):
            return None
    from server.services.conversation import get_conversation_log
    from server.services.execution import get_agent_roster

    truth = dataset.ground_truth()
    if args.set == "paraphrase":  # requests that share no words with their target agent
        from dataclasses import asdict
        truth["requests"] = [asdict(r) for r in dataset.build_paraphrase_requests(dataset.build_agents())]
    name_to_group = {a["name"]: a["group"] for a in truth["agents"]}
    roster = get_agent_roster()
    # --- seed adapter: same ground-truth agents for every version of the code ---
    if hasattr(roster, "batch"):  # registry: names plus what production learns from each agent's first request
        from server.services.execution.retrieval import extract_people

        with roster.batch():
            for agent in dataset.build_agents():
                request = dataset.first_request(agent)
                roster.add_agent(agent.name, description=request, people=extract_people(request))
        try:  # the background maintenance job (merge duplicates) would have run on a real install
            from server.services.execution import lifecycle

            report = await lifecycle.consolidate()
            print(f"maintenance before the eval: merged {report.merged} duplicate agents into {len(report.groups)} threads")
        except ImportError:
            pass
    else:  # original roster: a plain list of names
        roster._agents = [a["name"] for a in truth["agents"]]
        roster.save()

    results_path = Path(args.results) / "results.jsonl"
    done = set()
    if results_path.exists():
        done = {json.loads(l)["id"] for l in results_path.read_text().splitlines() if l.strip()}
    requests = [r for r in truth["requests"] if r["id"] not in done]
    if args.limit:
        requests = requests[: args.limit]
    spent = sum(json.loads(l)["cost_usd"] for l in results_path.read_text().splitlines() if l.strip()) if done else 0.0

    state: dict = {}
    real_llm = rt.request_chat_completion
    roster_names = [a["name"] for a in truth["agents"]]

    async def wrapped(**kw):
        if state["routes"] or state["calls"] >= MAX_CALLS:
            raise _Stop()
        state["calls"] += 1
        started = time.monotonic()
        if args.fake:
            resp = _fake_completion(state["text"], roster_names)
        else:
            resp = None
            for attempt in range(7):
                try:
                    resp = await real_llm(**kw)
                    break
                except Exception as exc:  # noqa: BLE001
                    if "429" in str(exc) and attempt < 6:
                        state["retries"] += 1
                        await asyncio.sleep(min(60, 4 * 2 ** attempt))
                        continue
                    state["error"] = str(exc)[:200]
                    raise
        state["latency"] += time.monotonic() - started
        u = resp.get("usage") or {}
        state["prompt_tokens"].append(u.get("prompt_tokens") or 0)
        state["completion_tokens"] += u.get("completion_tokens") or 0
        state["cost"] += u.get("cost") or 0.0
        return resp

    def stub_send_message_to_agent(agent_name: str, instructions: str):
        state["routes"].append(agent_name)
        state["instructions"].append(instructions)
        return itools.ToolResult(success=True, payload={
            "status": "submitted", "agent_name": agent_name,
            "new_agent_created": agent_name not in name_to_group})

    rt.request_chat_completion = wrapped
    itools.send_message_to_agent = stub_send_message_to_agent

    with results_path.open("a", encoding="utf-8") as out:
        for req in requests:
            if spent >= args.max_usd:
                print(f"cost cap ${args.max_usd} reached (spent ${spent:.3f}); stopping. Use --resume to continue.")
                break
            state.update(routes=[], calls=0, latency=0.0, prompt_tokens=[], completion_tokens=0, cost=0.0,
                         text=req["text"], retries=0, error=None, instructions=[])
            get_conversation_log().clear()
            set_tag(f"{args.label}:{req['id']}")
            try:
                await rt.InteractionAgentRuntime().execute(req["text"])
            except _Stop:
                pass
            routed = state["routes"]
            first = routed[0] if routed else None
            exp = req["expected_group"]
            if first is None and state["error"]:
                outcome = "api_error"
            elif first is None:
                outcome = "no_route"
            elif exp is None:
                outcome = "wrong_reuse" if first in name_to_group else "correct_new"
            elif first not in name_to_group:
                outcome = "missed_reuse"
            else:
                outcome = "correct_reuse" if name_to_group[first] == exp else "wrong_reuse"
            safety_net = None
            if outcome == "missed_reuse":  # the model named a new agent: what would the duplicate check do?
                try:
                    from server.services.execution import lifecycle

                    found = await lifecycle.find_duplicate(first, state["instructions"][0] if state["instructions"] else "")
                    safety_net = "rescued" if found and name_to_group.get(found.name) == exp else \
                        ("wrong" if found else "not_caught")
                except ImportError:
                    pass
            row = {
                "id": req["id"], "category": req["category"], "text": req["text"], "safety_net": safety_net,
                "expected_group": exp, "routed": routed, "outcome": outcome,
                "roster_size": len(roster_names), "llm_calls": state["calls"],
                "prompt_tokens_first": state["prompt_tokens"][0] if state["prompt_tokens"] else 0,
                "prompt_tokens_total": sum(state["prompt_tokens"]),
                "completion_tokens": state["completion_tokens"],
                "cost_usd": state["cost"], "latency_s": round(state["latency"], 3),
                "retries_429": state["retries"], "error": state["error"],
            }
            spent += state["cost"]
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"{req['id']:<12} {outcome:<14} -> {first}   (${spent:.3f} spent, {state['retries']} retries)")
            if not args.fake:
                await asyncio.sleep(args.pace)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--max-usd", type=float, default=1.6, help="hard cap on LLM spend for this run")
    ap.add_argument("--fake", action="store_true", help="use a local rule-based router (no LLM, free)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", choices=("main", "paraphrase"), default="main",
                    help="main: the 40 baseline requests; paraphrase: 14 requests with no word in common with their agent")
    ap.add_argument("--pace", type=float, default=3.0, help="seconds to sleep between requests (rate limits)")
    ap.add_argument("--limit", type=int, default=0, help="only run the first N requests")
    ap.add_argument("--summarize-only", action="store_true", help="rebuild summary from results.jsonl")
    ap.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--results", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.summarize_only:
        summarise(ROOT / "evals" / "results" / args.label, args.label)
        return
    if args.inner:
        asyncio.run(inner(args))
    else:
        outer(args)


if __name__ == "__main__":
    main()
