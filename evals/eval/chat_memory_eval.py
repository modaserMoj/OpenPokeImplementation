"""Chat memory eval: after the main agent's chat is summarised (again and again), does it still know what matters?

memory_eval.py covers the WORKERS' memory. This covers the MAIN agent's chat history (issue 9): we now summarise
once the history passes ~4,000 tokens instead of after 110 messages, so the summary is rewritten more often and has
more chances to drop something.

Plays a long realistic chat (~150 messages) into the real conversation log, running the real summariser after
every message exactly as the app's background scheduler does (it only acts when the history is over budget). Facts
are planted early so they must survive several rewrites. Then checks what the main agent would actually see
(`render_transcript` with the prompt budget: summary + recent messages).

Also measured:
  * the summary length (the prompt asks for under ~500 words; code caps it at 5,000 characters)
  * a draft the user CANCELLED must not stay under "Pending" (the known open item: stale drafts)
  * an UPDATED fact: the new value must be there
  * a RELATIVE date ("tomorrow"): the summariser is told to turn it into a real date (the chat runs on today's date)

    python evals/eval/chat_memory_eval.py --label haiku              # configured summariser (Haiku 5.5), cents
    python evals/eval/chat_memory_eval.py --label haiku --repeats 3  # repeat to see run-to-run variation

Runs on a sandbox copy of server/. Output: evals/results/chat_memory_<label>/
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
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TOKENS_PER_CHAR = 0.283  # measured with the real model on this codebase's prompts (cost_scaling_baseline)
MAX_SPEND_USD = 0.50

# (turn, user message, assistant reply, label, kind, patterns: any one must appear in what the agent sees)
PLANTED = [
    (1, "Heads up: I'm vegetarian, so keep that in mind for any restaurant bookings.",
     "Noted - I'll only suggest vegetarian-friendly places.", "user is vegetarian", "preference", [r"vegetarian"]),
    (3, "My flight to Lisbon is booked, confirmation code QX7P2M, leaving 2026-11-12 at 07:40.",
     "Got it: flight QX7P2M on 2026-11-12 at 07:40.", "flight confirmation code QX7P2M", "identifier", [r"qx7p2m"]),
    (5, "The budget for the team offsite is $12,500, don't go over it.",
     "Understood, offsite budget capped at $12,500.", "offsite budget $12,500", "amount", [r"12,?500"]),
    (7, "Never schedule anything for me before 10am.",
     "Will do - nothing before 10:00.", "nothing before 10am", "standing rule", [r"10\s?(:00\s?)?am", r"10:00", r"before 10"]),
    (9, "Always cc Priya on anything about the apartment lease.",
     "Okay, Priya will be cc'd on all lease emails.", "always cc Priya on lease emails", "standing rule", [r"priya"]),
    (11, "Dentist appointment is on 2026-11-04 at 15:00 with Dr. Okafor.",
     "Saved: dentist with Dr. Okafor on 2026-11-04 at 15:00.", "dentist 2026-11-04 15:00", "dated commitment",
     [r"okafor", r"dentist"]),
    (13, "My sister Leila's birthday is 2026-12-03, I need to get her a gift.",
     "Noted Leila's birthday on 2026-12-03; I can help with gift ideas.", "Leila's birthday 2026-12-03", "person detail",
     [r"leila"]),
    (15, "My passport expires 2026-12-15, I need to renew it before the trip in January.",
     "Passport renewal is on your list, deadline before 2026-12-15.", "renew passport (expires 2026-12-15)", "open task",
     [r"passport"]),
    (16, "Call the landlord tomorrow at 14:00 about the broken heater.",
     "I'll remind you to call the landlord tomorrow at 14:00.", "call landlord tomorrow 14:00",
     "relative date", ["{tomorrow}"]),
    (17, "The offsite location is Lisbon.",
     "Offsite location: Lisbon.", "(setup for the update check)", "setup", []),
    (40, "Change of plan: the offsite moves from Lisbon to Porto.",
     "Updated: the offsite is now in Porto.", "offsite moved to Porto", "updated fact", [r"porto"]),
]
# A draft the user cancels: it must not survive as a pending item.
CANCELLED = [
    (20, "Draft an email to Marco asking to cancel my gym membership.",
     "Here's the draft to marco@gymflex.test: 'Hi Marco, please cancel my membership effective this month. Thanks!' "
     "Want me to send it?"),
    (22, "Actually never mind, don't send the Marco email - I'll keep the membership.",
     "Okay, I won't send it. The Marco draft is dropped."),
]

FILLER = [
    ("What's a quick dinner idea tonight?", "How about a chickpea and spinach curry? About 25 minutes with canned "
     "chickpeas, coconut milk, spinach and a spoon of curry paste; serve with rice or naan."),
    ("Can you remind me how to convert 180C to Fahrenheit?", "180°C is 356°F: multiply by 9/5 and add 32."),
    ("Any podcast recommendations for a long drive?", "Try a narrative history show, a long-form interview podcast, "
     "or an audio drama; pick something with episodes over an hour so you aren't switching constantly."),
    ("What's the difference between a Roth and a traditional IRA, briefly?", "Roth: you pay tax now, withdrawals in "
     "retirement are tax-free. Traditional: contributions may be deductible now, withdrawals are taxed later."),
    ("Thanks, that helps.", "Anytime!"),
    ("Give me three stretches for lower back pain.", "Child's pose, knee-to-chest, and cat-cow. Hold each for about "
     "30 seconds, breathe slowly, and stop if anything hurts sharply."),
    ("How long should I steep green tea?", "Two to three minutes at around 80°C; longer or hotter makes it bitter."),
    ("Summarise the plot of Hamlet in two lines.", "Prince Hamlet seeks revenge on his uncle Claudius, who murdered "
     "Hamlet's father and married his mother. His hesitation leads to a chain of deaths, including his own."),
    ("What's a good way to keep basil fresh?", "Trim the stems and keep it in a glass of water on the counter, loosely "
     "covered with a bag - not in the fridge, which blackens the leaves."),
    ("ok cool", "👍"),
    ("Explain what a VPN does in simple terms.", "It creates an encrypted tunnel between your device and a server, so "
     "your network provider sees only that you're connected to the VPN, not which sites you visit."),
    ("Suggest a name for a grey cat.", "Smokey, Ash, Pebble, Earl Grey, or Mochi."),
]
TURNS = 260  # user + assistant per turn: 520 messages, enough for several summary rewrites


def _api_key() -> str:
    if os.getenv("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    return ""


def outer(args) -> None:
    results = ROOT / "evals" / "results" / f"chat_memory_{args.label}"
    shutil.rmtree(results, ignore_errors=True)
    results.mkdir(parents=True)
    runs = []
    for repeat in range(1, args.repeats + 1):
        sandbox = Path(tempfile.mkdtemp(prefix="openpoke-chatmem-"))
        shutil.copytree(ROOT / "server", sandbox / "server",
                        ignore=shutil.ignore_patterns("data", "__pycache__", "*.pyc", "*.log"))
        shutil.copy(HERE / "chat_memory_eval.py", sandbox / "chat_memory_eval.py")
        out = results / f"run{repeat}"
        out.mkdir()
        env = dict(os.environ, OPENROUTER_API_KEY=_api_key() or "unused", PYTHONPATH=str(sandbox),
                   OPENPOKE_USAGE_LOG=str(out / "llm_usage.jsonl"), OPENPOKE_EMBEDDING_MODEL="")
        if args.model:
            env["OPENPOKE_MODEL_SUMMARIZER"] = args.model
        rc = subprocess.run([sys.executable, str(sandbox / "chat_memory_eval.py"), "--inner", "--results", str(out)],
                            cwd=sandbox, env=env).returncode
        shutil.rmtree(sandbox, ignore_errors=True)
        if rc != 0:
            sys.exit(rc)
        runs.append(json.loads((out / "summary.json").read_text()))
    _report(results, args.label, runs)


def _section(summary: str, header: str) -> str:
    """The text of one briefing section ("Pending & Follow-ups:" up to the next section header)."""
    match = re.search(re.escape(header) + r"(.*?)(?:\n\s*[A-Z][A-Za-z &]+:\s*\n|\Z)", summary, re.S)
    return match.group(1) if match else ""


async def inner(args) -> None:
    sys.path.insert(0, os.getcwd())
    from server.config import get_settings
    from server.services.conversation import log as log_module
    from server.services.conversation.summarization import summarizer
    from server.services.conversation.summarization.working_memory_log import get_working_memory_log

    # The app summarises in a background task after each message; here we run the same pass inline, in order.
    log_module.ConversationLog._notify_summarization = lambda self: None
    conversation = log_module.get_conversation_log()
    working_memory = get_working_memory_log()
    settings = get_settings()
    usage_path = Path(os.environ["OPENPOKE_USAGE_LOG"])

    def spent() -> float:
        if not usage_path.exists():
            return 0.0
        return sum(json.loads(l).get("cost_usd") or 0 for l in usage_path.read_text().splitlines())

    scripted = {turn: (u, a) for turn, u, a, *_ in PLANTED}
    scripted.update({turn: (u, a) for turn, u, a in CANCELLED})
    rounds, filler_index, total_chars = 0, 0, 0
    for turn in range(TURNS):
        if turn in scripted:
            user, reply = scripted[turn]
        else:
            user, reply = FILLER[filler_index % len(FILLER)]
            filler_index += 1
        for record, text in ((conversation.record_user_message, user), (conversation.record_reply, reply)):
            record(text)
            total_chars += len(text) + 50
            if await summarizer.summarize_conversation():
                rounds += 1
        if spent() > MAX_SPEND_USD:
            raise SystemExit(f"spending cap ${MAX_SPEND_USD} reached")

    state = working_memory.load_summary_state()
    summary = state.summary_text or ""
    view = working_memory.render_transcript(budget_chars=settings.conversation_history_budget_chars)
    lower = view.lower()

    from datetime import datetime, timedelta
    from server.utils.timezones import now_in_user_timezone

    tomorrow = (datetime.strptime(now_in_user_timezone("%Y-%m-%d"), "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    facts = []
    for turn, _, _, label, kind, patterns in PLANTED:
        patterns = [p.replace("{tomorrow}", tomorrow) for p in patterns]
        if not patterns:
            continue
        facts.append({"turn": turn, "fact": label, "kind": kind,
                      "kept": any(re.search(p, lower) for p in patterns),
                      "in_summary": any(re.search(p, summary.lower()) for p in patterns)})
    pending = _section(summary, "Pending & Follow-ups:").lower()
    out = {
        "summariser": settings.summarizer_model,
        "messages": TURNS * 2, "summary_rounds": rounds,
        "history_chars_without_summary": total_chars, "agent_view_chars": len(view),
        "summary_words": len(summary.split()), "summary_chars": len(summary),
        "facts": facts,
        "cancelled_draft_still_pending": "marco" in pending,
        "cost_usd": spent(),
    }
    results = Path(args.results)
    (results / "summary.json").write_text(json.dumps(out, indent=2))
    (results / "final_summary.txt").write_text(summary)
    (results / "agent_view.txt").write_text(view)


def _report(results: Path, label: str, runs) -> None:
    n = len(runs)
    labels = [f["fact"] for f in runs[0]["facts"]]
    kinds = {f["fact"]: f["kind"] for f in runs[0]["facts"]}
    kept = {lab: sum(1 for r in runs for f in r["facts"] if f["fact"] == lab and f["kept"]) for lab in labels}
    total_kept = sum(kept.values())
    cost = sum(r["cost_usd"] for r in runs)
    first = runs[0]
    view_sizes = ", ".join(f"{r['agent_view_chars']:,}" for r in runs)
    lines = [
        f"CHAT MEMORY EVAL (main agent)  label={label}  summariser: {first['summariser']}  runs: {n}",
        "=" * 78,
        f"chat: {first['messages']} messages, facts planted in the first {max(t for t, *_ in PLANTED)} turns; "
        f"summariser run after every message (as the app does)",
        f"summary rewrites per run: {', '.join(str(r['summary_rounds']) for r in runs)}",
        f"history if never summarised: {first['history_chars_without_summary']:,} chars "
        f"(~{round(first['history_chars_without_summary'] * TOKENS_PER_CHAR):,} tokens)",
        f"what the agent sees at the end: {view_sizes} chars",
        "",
        "Facts still visible to the main agent (kept in N of the runs):",
        *[f"   {kept[lab]}/{n}  [{kinds[lab]}] {lab}" for lab in labels],
        f"   total: {total_kept}/{len(labels) * n} ({100 * total_kept / (len(labels) * n):.0f}%)",
        "",
        "Summary length (prompt asks for under ~500 words; code caps at 5,000 chars):",
        f"   words: {', '.join(str(r['summary_words']) for r in runs)}   chars: "
        f"{', '.join(str(r['summary_chars']) for r in runs)}",
        "",
        "Cancelled draft (to Marco) still listed under Pending: "
        f"{sum(r['cancelled_draft_still_pending'] for r in runs)}/{n} runs",
        "",
        f"cost: ${cost:.4f} total, ${cost / n:.4f} per run",
        "Per-run files: run<N>/final_summary.txt, agent_view.txt, summary.json, llm_usage.jsonl",
    ]
    (results / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="haiku")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--model", default="", help="summariser model to use (default: the configured one)")
    ap.add_argument("--inner", action="store_true")
    ap.add_argument("--results")
    args = ap.parse_args()
    asyncio.run(inner(args)) if args.inner else outer(args)


if __name__ == "__main__":
    main()
