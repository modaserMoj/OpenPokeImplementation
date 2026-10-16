# Tests and evaluators

Run everything that needs no keys and costs nothing:

    pip install -r tests/requirements.txt
    python -m pytest                 # 242 tests, ~25 seconds, no network, never touches server/data

## What the suite does

The model and Gmail are replaced by fakes (`conftest.py`): a scripted `FakeLLM` and a `FakeGmail` that
records every call. Every file-backed singleton (agent list, agent logs, conversation log) is pointed at a
temp folder, and any real network call fails the test.

How the suite was built:

* **Regression tests** protected behaviour that already worked (reminder timing, Gmail connect, usage
  logging, the evaluator's own dataset) so the fixes could not break it.
* **Known-bug tests** were written BEFORE any fix. Each stated the CORRECT behaviour for a problem in
  `evals/SUMMARY.txt` and was marked `xfail(strict=True)`, checked (`--runxfail`) to fail for the intended
  reason. Strict mode fails the run as soon as a fix makes one pass, so each marker was removed with its fix.
  Before: 23 passed, 20 expected failures. After: all pass. The `known_bug` helper is kept in `helpers.py`.

| File | Covers (SUMMARY.txt) |
|---|---|
| `test_send_guard.py` | S1 a hijacked worker cannot send |
| `test_approvals.py` | S1 the approval flow: shown, user-confirmed, single use, expiry, reply/forward |
| `test_roster.py`, `test_registry.py` | 2 identity and lifecycle, S3 safe save and `fcntl` |
| `test_retrieval.py`, `test_embeddings.py` | 1 the main agent sees a short, relevant agent list (word + semantic) |
| `test_execution_history.py`, `test_memory.py` | 3 bounded worker memory, compaction, full tool results kept |
| `test_conversation_history.py` | 9 the main agent's chat history stays under its budget |
| `test_runtime_robustness.py` | 7 bad tool JSON and phantom tool, 8 wasted call and repeated draft, silent failure |
| `test_batching.py`, `test_outbox.py` | 4 one turn for things that arrive together, Fibonacci email cooldown; 5 no waiting for the slowest agent, one agent never runs twice at once |
| `test_config.py` | 6 one model setting per role: Sonnet for the main agent and worker, Haiku for the rest |
| `test_latency.py` | no event-loop blocking on Gmail calls, one connection pool, prompt-caching markers |
| `test_triggers.py` | reminder scheduling and firing (regression) |
| `test_llm_client.py` | output cap and per-call usage log (regression) |
| `test_gmail_connect.py` | Composio `link()` flow (regression) |
| `test_eval_validity.py` | the evaluator's dataset and harness are sound |

## Evaluators (real model, capped spend; see `evals/README.txt`)

* `evals/eval/run_eval.py`: routing eval, 40 labeled requests on a 277-agent history. Measures accuracy,
  reuse of the right agent, duplicate agents, the share of requests reaching the agent with the main history,
  tokens, cost, latency. `--fake` runs the same pipeline free; `--max-usd` is a hard cap.
* `evals/eval/retrieval_eval.py`: is the right agent on the shortlist the main agent sees (embeddings only).
* `evals/eval/cost_scaling.py`: how the main agent's and a worker's prompts (and cost) grow with history.
* `evals/eval/lifecycle_eval.py`: duplicate merging (wrong merges, merges found) and the creation-time check.
* `evals/eval/memory_eval.py`: does a worker's memory keep IDs and facts after compaction.
* `evals/eval/chat_memory_eval.py`: does the main agent's chat summary keep facts over repeated rewrites.
* `evals/eval/classifier_eval.py`: email importance classifier (20 labeled emails incl. 2 attacks), per model.
* `evals/eval/compare.py baseline after`: before/after table (refuses mismatched datasets).

Results: `evals/results/tests_baseline/` is the suite before any fix (23 passed, 20 expected failures);
`evals/results/tests_after_all_fixes/` is after every fix (242 passed, 0 expected failures).

## Not covered

Reply quality (no judge eval yet). Behaviour that needs the real model and real Gmail (an injected email, the
email watcher, draft -> approve -> send, restart) is covered by the live tests in `evals/results/live_*`,
not by this free suite.
