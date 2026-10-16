"""Print a before/after table from two eval summaries.

    python evals/eval/compare.py baseline after
"""

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
# (key, label, higher_is_better, formatter)
ROWS = [
    ("accuracy", "routing accuracy", True, lambda v: f"{100*v:.1f}%"),
    ("reuse_recall", "reuse recall", True, lambda v: f"{100*v:.1f}%"),
    ("primary_hit_rate", "primary-agent hit rate (context not split)", True, lambda v: f"{100*v:.1f}%"),
    ("duplicate_agent_rate", "duplicate-agent rate", False, lambda v: f"{100*v:.1f}%"),
    ("wrong_reuse_rate", "wrong-reuse rate", False, lambda v: f"{100*v:.1f}%"),
    ("no_route_rate", "no-route rate", False, lambda v: f"{100*v:.1f}%"),
    ("prompt_tokens_first_call_avg", "prompt tokens / routing call", False, lambda v: f"{v:,.0f}"),
    ("cost_usd_per_request_avg", "cost / request", False, lambda v: f"${v:.4f}"),
    ("latency_s_per_request_p50", "latency p50 (s)", False, lambda v: f"{v:.2f}"),
    ("latency_s_per_request_p95", "latency p95 (s)", False, lambda v: f"{v:.2f}"),
]


def main() -> None:
    a_label, b_label = sys.argv[1], sys.argv[2]
    a = json.loads((RESULTS / a_label / "summary.json").read_text())
    b = json.loads((RESULTS / b_label / "summary.json").read_text())
    if a.get("dataset_sha") != b.get("dataset_sha"):
        print(f"WARNING: datasets differ ({a.get('dataset_sha')} vs {b.get('dataset_sha')}); not comparable\n")
    print(f"{'metric':<46}{a_label:>12}{b_label:>12}   change")
    print("-" * 82)
    for key, label, higher_good, fmt in ROWS:
        va, vb = a.get(key, 0), b.get(key, 0)
        better = (vb > va) if higher_good else (vb < va)
        mark = "better" if better and va != vb else ("worse" if va != vb else "same")
        print(f"{label:<46}{fmt(va):>12}{fmt(vb):>12}   {mark}")
    print(f"\nrequests: {a['requests']} vs {b['requests']}   roster: {a['roster_size']} vs {b['roster_size']} agents")


if __name__ == "__main__":
    main()
