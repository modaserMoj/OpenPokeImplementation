"""Cost scaling probes with the real model (small, capped spend).

  A. MAIN AGENT: how big does the interaction prompt get as conversation history grows (0..100 messages,
     where summarisation kicks in)?
  B. WORKER AGENT: how big does an execution agent's prompt get as its own log grows (0..50 prior runs)?

Texts are sampled from the real live session in evals/runtime/events.log, so lengths and token
density are realistic. Runs in a sandbox copy of server/ (live data untouched).

    python evals/eval/cost_scaling.py --label baseline --max-usd 0.5

Output: evals/results/cost_scaling_<label>/{results.json,summary.txt,llm_usage.jsonl}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
HISTORY_SIZES = [0, 10, 25, 50, 75, 100, 150, 200]
HISTORY_REAL_CALLS = {0, 50, 100}
RUN_COUNTS = [0, 5, 20, 50]


def _api_key() -> str:
    if os.getenv("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    return ""


def outer(args) -> None:
    results = ROOT / "evals" / "results" / f"cost_scaling_{args.label}"
    shutil.rmtree(results, ignore_errors=True)
    results.mkdir(parents=True)
    sandbox = Path(tempfile.mkdtemp(prefix="openpoke-cost-"))
    shutil.copytree(ROOT / "server", sandbox / "server",
                    ignore=shutil.ignore_patterns("data", "__pycache__", "*.pyc", "*.log"))
    shutil.copy(HERE / "cost_scaling.py", sandbox / "cost_scaling.py")
    env = dict(os.environ, OPENROUTER_API_KEY=_api_key() if not args.offline else "unused", PYTHONPATH=str(sandbox),
               OPENPOKE_EMBEDDING_MODEL="",
               OPENPOKE_USAGE_LOG=str(results / "llm_usage.jsonl"))
    for k in ("COMPOSIO_API_KEY", "COMPOSIO_GMAIL_AUTH_CONFIG_ID"):
        env.pop(k, None)
    cmd = [sys.executable, str(sandbox / "cost_scaling.py"), "--inner", "--label", args.label,
           "--results", str(results), "--max-usd", str(args.max_usd),
           "--samples", str(ROOT / "evals" / "runtime" / "events.log")] + (["--offline"] if args.offline else [])
    rc = subprocess.run(cmd, cwd=sandbox, env=env).returncode
    shutil.rmtree(sandbox, ignore_errors=True)
    sys.exit(rc)


def load_samples(events_log: str) -> dict:
    """Real payload texts per tag from the live session (conversation + non-email execution logs)."""
    out: dict = {}
    pat = re.compile(r'\S+ \[(conversation/poke_conversation\.log|execution_agents/[^\]]+)\] <(\w+) timestamp="[^"]*">(.*)</\2>\s*$')
    for line in open(events_log, errors="replace"):
        m = pat.match(line)
        if not m:
            continue
        src, tag, payload = m.groups()
        if src.startswith("execution_agents/") and "email-checker" in src:
            continue  # inbox search results contain real email content
        text = payload.replace("\\n", "\n")
        out.setdefault((src.split("/")[0], tag), []).append(text)
    return out


def cycle(items, n):
    return [items[i % len(items)] for i in range(n)]


async def inner(args) -> None:
    sys.path.insert(0, os.getcwd())
    from server.agents.execution_agent.agent import ExecutionAgent
    from server.agents.execution_agent.tools import get_tool_schemas as exec_tools
    from server.agents.interaction_agent.agent import build_system_prompt, prepare_message_with_history
    from server.agents.interaction_agent.tools import TOOL_SCHEMAS
    from server.config import get_settings
    from server.openrouter_client import request_chat_completion
    from server.openrouter_client.usage import set_tag
    from server.services.execution import get_execution_agent_logs

    s = get_settings()
    samples = load_samples(args.samples)
    need = [("conversation", "user_message"), ("conversation", "poke_reply"), ("conversation", "agent_message"),
            ("execution_agents", "agent_request"), ("execution_agents", "agent_action"),
            ("execution_agents", "tool_response"), ("execution_agents", "agent_response")]
    missing = [k for k in need if not samples.get(k)]
    if missing:
        raise SystemExit(f"not enough real samples in events.log for {missing}")

    spent = 0.0
    out = {"label": args.label, "model": s.interaction_agent_model, "history": [], "worker": []}

    async def real_call(model, system, messages, tools, tag):
        nonlocal spent
        if spent >= args.max_usd:
            return None
        set_tag(tag)
        r = await request_chat_completion(model=model, messages=messages, system=system,
                                          api_key=s.openrouter_api_key, tools=tools, max_tokens=16)
        u = r.get("usage") or {}
        spent += u.get("cost") or 0
        return u

    # ---------------- A. main agent: conversation history growth ----------------
    # The history is written into the conversation log and read back through the main agent's own loading path,
    # so whatever limit the code under test applies (none before the fix, a budget after) is what gets measured.
    # Background summarisation is switched off: this measures the largest prompt before any summary exists.
    from server.agents.interaction_agent.runtime import InteractionAgentRuntime
    from server.services.conversation import get_conversation_log
    from server.services.conversation.log import ConversationLog

    ConversationLog._notify_summarization = lambda self: None
    mix = ["user_message", "poke_reply", "poke_reply", "agent_message"]  # ~ real ratio 12:24:11
    sys_prompt = build_system_prompt()
    for n in HISTORY_SIZES:
        log = get_conversation_log()
        log.clear()
        pools = {t: cycle(samples[("conversation", t)], 400) for t in set(mix)}
        idx = {t: 0 for t in pools}
        writers = {"user_message": log.record_user_message, "poke_reply": log.record_reply,
                   "agent_message": log.record_agent_message}
        for i in range(n):
            tag = mix[i % len(mix)]
            writers[tag](pools[tag][idx[tag]])
            idx[tag] += 1
        transcript = InteractionAgentRuntime()._load_conversation_transcript()
        messages = prepare_message_with_history("remind me to call mom tomorrow", transcript, "user")
        chars = len(sys_prompt) + len(messages[0]["content"]) + len(json.dumps(TOOL_SCHEMAS))
        row = {"messages": n, "history_chars": len(transcript), "prompt_chars": chars}
        if n in HISTORY_REAL_CALLS and not args.offline:
            u = await real_call(s.interaction_agent_model, sys_prompt, messages, TOOL_SCHEMAS, f"history-{n}")
            if u:
                row.update(prompt_tokens=u.get("prompt_tokens"), cost_usd=u.get("cost"))
        out["history"].append(row)
        print("history", row, flush=True)

    # ---------------- B. worker agent: its own log growth ----------------
    store = get_execution_agent_logs()
    name = "Scaling Probe"
    pools = {t: cycle(samples[("execution_agents", t)], 200)
             for t in ("agent_request", "agent_action", "tool_response", "agent_response")}
    done = 0
    for k in RUN_COUNTS:
        while done < k:
            i = done
            store.record_request(name, pools["agent_request"][i])
            store.record_action(name, pools["agent_action"][i])
            store.record_tool_response(name, "tool", pools["tool_response"][i][:500])  # real cap
            store.record_agent_response(name, pools["agent_response"][i])
            done += 1
        system = ExecutionAgent(name).build_system_prompt_with_history()
        msgs = [{"role": "user", "content": "Remind the user to stretch in 5 minutes."}]
        tools = exec_tools()
        row = {"prior_runs": k, "system_chars": len(system)}
        u = None if args.offline else await real_call(s.execution_agent_model, system, msgs, tools, f"worker-{k}")
        if u:
            row.update(prompt_tokens=u.get("prompt_tokens"), cost_usd=u.get("cost"))
        out["worker"].append(row)
        print("worker", row, flush=True)

    out["spent_usd"] = spent
    (Path(args.results) / "results.json").write_text(json.dumps(out, indent=2))
    summarise(out, Path(args.results))


def summarise(out: dict, results: Path) -> None:
    real = [r for r in out["history"] if r.get("prompt_tokens")]
    ratio = (sum(r["prompt_tokens"] for r in real) / sum(r["prompt_chars"] for r in real)) if real else 0.283
    lines = [
        f"COST SCALING  label={out['label']}  model={out['model']}  spent=${out['spent_usd']:.3f}",
        "=" * 74,
        "A. MAIN AGENT prompt vs conversation history (roster empty; system prompt + tools + history)",
        f"   {'messages':>8} {'prompt chars':>13} {'tokens':>8}   {'cost/turn @ $3/M in':>20}",
    ]
    for r in out["history"]:
        tok = r.get("prompt_tokens")
        est = "" if tok else " (est)"
        tok = tok or round(r["prompt_chars"] * ratio)
        lines.append(f"   {r['messages']:>8} {r['prompt_chars']:>13,} {tok:>8,}{est:<6} ${tok*3/1e6:>8.4f}")
    lines += ["", f"   tokens/char from real calls: {ratio:.3f}  (est rows use this ratio)",
              "   (Background summarisation is off here: these are the largest prompts before any summary exists.)", "",
              "B. WORKER AGENT prompt vs number of prior runs in its own log",
              f"   {'prior runs':>10} {'system chars':>13} {'tokens':>8}   {'cost/call @ $3/M in':>20}"]
    for r in out["worker"]:
        tok = r.get("prompt_tokens")
        est = "" if tok else " (est)"
        tok = tok or round(r["system_chars"] * ratio)
        lines.append(f"   {r['prior_runs']:>10} {r['system_chars']:>13,} {tok:>8,}{est:<6} ${tok*3/1e6:>8.4f}")
    base = out["worker"][0].get("prompt_tokens") or round(out["worker"][0]["system_chars"] * ratio)
    last = out["worker"][-1].get("prompt_tokens") or round(out["worker"][-1]["system_chars"] * ratio)
    if base and last:
        lines.append(f"\n   worker prompt grew {last/base:.1f}x ({base:,} -> {last:,} tokens) over {out['worker'][-1]['prior_runs']} runs")
    (results / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--max-usd", type=float, default=0.5)
    ap.add_argument("--offline", action="store_true", help="no model calls: sizes only, tokens estimated")
    ap.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--results", help=argparse.SUPPRESS)
    ap.add_argument("--samples", help=argparse.SUPPRESS)
    args = ap.parse_args()
    asyncio.run(inner(args)) if args.inner else outer(args)


if __name__ == "__main__":
    main()
