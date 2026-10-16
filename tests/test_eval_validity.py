"""The evaluator itself must be trustworthy: a fixed dataset with unambiguous labels, and a harness that
runs end to end. (The real-model run is evals/eval/run_eval.py; --fake exercises the same pipeline free.)"""

import collections
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals" / "eval"))
import dataset  # noqa: E402


def test_dataset_is_deterministic():
    assert json.dumps(dataset.ground_truth(), sort_keys=True) == json.dumps(dataset.ground_truth(), sort_keys=True)


def test_dataset_shape():
    truth = dataset.ground_truth()
    categories = collections.Counter(r["category"] for r in truth["requests"])
    assert categories == {"exact": 10, "indirect": 10, "dup_trap": 10, "new": 10}
    assert 250 <= len(truth["agents"]) <= 320
    names = [a["name"] for a in truth["agents"]]
    assert len(names) == len(set(names)), "duplicate agent names make routing labels ambiguous"


def test_every_label_points_at_a_real_group_and_new_requests_match_none():
    truth = dataset.ground_truth()
    groups = {a["group"] for a in truth["agents"]}
    for request in truth["requests"]:
        if request["category"] == "new":
            assert request["expected_group"] is None
        else:
            assert request["expected_group"] in groups, request["id"]


def test_indirect_person_requests_name_a_last_name_held_by_exactly_one_thread():
    truth = dataset.ground_truth()
    people = {}
    for agent in truth["agents"]:
        if agent["people"]:
            people[agent["group"]] = agent["people"][0]
    last_names = collections.Counter(p.split()[-1] for p in people.values())
    for request in truth["requests"]:
        group = request["expected_group"]
        if request["category"] == "indirect" and group in people:
            assert last_names[people[group].split()[-1]] == 1, request["id"]


def test_the_harness_runs_end_to_end_with_the_free_fake_router(tmp_path):
    label = "pytest_smoke"
    out = ROOT / "evals" / "results" / label
    try:
        proc = subprocess.run([sys.executable, str(ROOT / "evals" / "eval" / "run_eval.py"),
                               "--label", label, "--fake", "--limit", "12"],
                              capture_output=True, text=True, timeout=180)
        assert proc.returncode == 0, proc.stderr[-500:]
        summary = json.loads((out / "summary.json").read_text())
        assert summary["requests"] == 12 and 0 <= summary["accuracy"] <= 1
        assert summary["cost_usd_total"] == 0
    finally:
        import shutil
        shutil.rmtree(out, ignore_errors=True)


def test_paraphrase_requests_share_no_words_with_their_target_agents():
    """If a paraphrase request reuses a word from the agent, a word ranker gets it for free and the set proves nothing."""
    import re
    allowed = {"the", "a", "my", "of", "on", "to", "for", "with", "about", "me", "can", "you", "this", "up", "it", "is",
               "are", "did", "any", "get", "how", "what", "where", "do", "need", "i", "check", "follow"}
    agents = dataset.build_agents()
    by_group = collections.defaultdict(list)
    for a in agents:
        by_group[a.group].append(a)
    words = lambda text: set(re.findall(r"[a-z0-9]+", text.lower()))  # noqa: E731
    requests = dataset.build_paraphrase_requests(agents)
    assert len(requests) >= 10
    for request in requests:
        agent_words = set().union(*(words(a.name) | words(a.description) for a in by_group[request.expected_group]))
        assert not (words(request.text) & agent_words) - allowed, request.id
