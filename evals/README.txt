evals/ - the evidence: what was wrong with OpenPoke, what we changed, and how it was measured.

READ IN THIS ORDER
  1. SUMMARY.txt      the problems found in the original code (issues 1-9, S1-S3), with costs
  2. FIXES.txt        the fix for each problem, its measured impact, and why it was done that way
  3. RESULTS.txt      every before/after number and the file it comes from (the index for everything below)
  4. FINDINGS.txt     the full findings log, severities, bugs found while getting the app running

FOLDERS
  logs/               offline probes of the ORIGINAL code (real code, fake model); 00_SUMMARY.txt is the index
  eval/               the evaluators (see below)
  results/            every eval run, test-suite run and live test, one folder or file each

TOOLS
  run_baseline.py               10 offline probes of the original problems -> logs/. Free, ~10s.
  eval/run_eval.py              routing eval: 40 labeled requests on a 277-agent history. Real model, ~$0.25-0.9;
                                --fake is free; --max-usd is a hard cap; --resume continues a run.
  eval/compare.py               before/after table: python evals/eval/compare.py baseline after_s4
  eval/retrieval_eval.py        is the right agent on the shortlist (embeddings only, ~$0.0001)
  eval/cost_scaling.py          prompt size and cost as history grows
  eval/lifecycle_eval.py        duplicate merging and the creation-time duplicate check (~$0.0001)
  eval/memory_eval.py           worker memory: IDs and facts kept after compaction
  eval/chat_memory_eval.py      main agent chat summary: facts kept over repeated rewrites (~$0.006 per run)
  eval/classifier_eval.py       email importance classifier, 20 labeled emails incl. 2 attacks, per model
  eval/dataset.py               the deterministic eval history and labeled requests (fingerprinted)
  watch_events.py               during live testing, merges every app log into one timestamped file
                                (evals/runtime/, kept local: it holds real email contents)

Every eval runs on a sandbox copy of server/, never on your real data, and logs every model call
(tokens, cost, latency) next to its results.

METRICS (routing eval)
  accuracy              right agent reused, or a new agent when nothing fits
  primary-agent hit     thread split across several agents -> went to the one with the main history
  duplicate-agent rate  existing thread -> a NEW agent was created
  wrong-reuse rate      sent to an unrelated agent
  prompt tokens/call, cost/request, latency p50/p95

CAVEATS
  - Synthetic history and hand-labeled requests; n=40 for routing (one request = 2.5 points), 20 emails for
    the classifier. Good for direction and for before/after on the same data, not a production benchmark.
  - "before" and "after" routing runs use the same model (Sonnet 4) and the same requests, so only the code
    differs. Execution agents and Gmail are stubbed in the routing eval; only the routing decision is measured.
  - Live-test files had real email addresses replaced with placeholders (jane.doe@example.com etc.).
