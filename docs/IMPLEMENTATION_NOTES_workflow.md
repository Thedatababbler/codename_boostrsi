# Implementation notes: single-agent Boosted Workflow (branch `workflow-integration`)

Per §9.3, every place where the implementation departs from 《Boosted Workflow：单智能体版本实现文档》 is recorded
here. This file was `docs/IMPLEMENTATION_NOTES.md` on `main`. The multi-agent commit replaced that path with
boosttopo's notes, so it was restored here under a new name. The same applies to
`scripts/report.py` → `scripts/report_workflow.py`.

Labels: **[run]** = checked by running code here; **[inferred]** = reasoning only, not run.

## 0. Status of revision §12 and the acceptance checks

The revision decisions (start from y0, g = gap_closure, the run_subgraph loop semantics, tool boundary and diag
modes, reuse of boosttopo, seed schemes) are implemented. Section 12 of the design doc is **not** in this repo's
copy of the doc (`Boosted Workflow：单智能体版本实现文档.md` on `multiagent` ends at section 10), so the decisions were
taken from the request text.

**Acceptance (a)–(c) on the real data could not be run.** This machine has no `cache/`, no `runs/` and no
`third_party/SelfImprove-Bench` (all gitignored in the multi-agent repo; a search under the user's directories
found none). Every real-data check needs them. The two multi-agent tests that need SelfImprove-Bench
(`tests/test_bench.py`) error for the same reason, also before any change here. **[run]**

What was run instead, on boosttopo's own toy benchmark with its fake model (`tests/test_legacy_replay.py`, and
`scripts/replay_check.py` against a toy run made with `python -m boosttopo.run --baselines`):
- (b) the original code (boosttopo.infer) replays every stored row, both with the stored eval cache and from the
  LLM cache alone (empty eval cache), with 0 new model calls. **[run]**
- (a) writer once, best-of-n writer, the repair loop at a token budget that cuts loops short, the Stage A repair
  loop (≤ 5 repairs), and a GBDT ensemble (analyst + writer, writer×3 keep_best, repairer×2 until_legal, and a
  duplicate member with its occurrence shift) written as one workflow, replayed with seed_scheme legacy and
  diag_mode original. All match the original rows (score, g, tokens, validity) and the original code's final plan
  item by item, with 0 new model calls. **[run]**
- The test is not vacuous: breaking any of five rules makes it fail. The five are the loop's `>=` rule, the loop
  repeat-index stride, the fanout repeat index, the gate's strict-improvement rule, and gate-local notes. **[run]**
- (c) on the toy Farm adapter: simulate's profit equals the scorer's on 40 items, and no tool output contains the
  reference profit or the relative score. **[run]** On the real benchmark this is part of `replay_check.py`
  (not run).

Seed 0's accepted r4_d5 ensemble, according to the multi-agent notes (round 5), is analyst + writer, then
writer×8 keep_best, then writer×2 keep_best. Each of these is expressible (§R3). **[inferred]** The repeated
requests between the two loop stages, which the round-5 notes found, are reproduced by the legacy scheme by
construction, because both stages use the same repeat indices. **[inferred]**

## P. Plan of 2026-11-27: everything from scratch on this cluster (overrides §0 and parts of R1, R3)

The user's plan of 2026-11-27 replaces the hand-off to another server: no old cache or runs exist, so
- acceptance (a) and (b) on real data are dropped (nothing to replay); the toy replay tests stay;
- every sample is generated here, y0 included (R1's "never generate test y0" no longer applies; `require_cached_y0`
  is empty); every method uses seed_scheme `path` (legacy stays only for the tests);
- items: Farm d5 with the multi-agent generator seeds, train 1000-1011, heldout 2000-2029, test 6000-6039. The
  final-test seeds 9000-9009 are never generated (`final_tests: {}`, make_splits(final_test=False)). **[run]**

### P1. Setup (step 1)
- SelfImprove-Bench cloned into `third_party/SelfImprove-Bench` (github theseus-labs-rsi/SelfImprove-Bench, HEAD
  5694387, 2026-11-05; gitignored). `tests/test_bench.py` (the 2 former errors) passes. **[run]**
- **crop_stats fixed to the real catalog fields.** A real `crop_catalog` entry is {id, name, seed_price,
  maturity_days, yield_per_harvest, fruit_price}. The earlier guessed yield keys missed `yield_per_harvest`, so on
  real data crop_stats would have printed only grow days and seed prices. It now prints per crop (best per-day
  margin first) grow days, profit per plot per cycle = yield_per_harvest * fruit_price - seed_price, per day, and
  the last planting day that still matures. The toy adapter uses the same field names now. **[run]**
  Note: the per-day margin equals the task's `reference_daily_profits` (an input of the reference greedy), which
  the prompt omits. It is computed from the catalog the model sees, and it is neither the reference profit nor the
  relative score, so it is within the tool boundary as stated; flagged in case that boundary should be stricter.
  **[run]** (values compared on farm-d5-s1000)
- `tests/test_tools_real.py` (real benchmark, train seeds 1000/1003/1007/1011): on 24 hand-written legal plans
  (one crop, repeated plant-harvest-sell cycles, full and half fill, the three cheapest crops), simulate's profit =
  the scorer's profit = simulate_plan's profit; validate (clean) and simulate contain neither "reference", nor the
  reference profit, nor the relative score at 3+ decimals. At 2 decimals an idle rate (0.57) coincided with the
  relative score on s1007, which is not a leak, so the check uses 3+ decimals. An illegal plan's diagnosis carries
  the failing-day state and no reference profit, so clean = original there. **[run]**
- **Cross-process cache safety.** `CachedLLM._write_first` stores under a POSIX record lock (fcntl.lockf) on the
  shard's `.lock` file and makes the record appear through an atomic link() of a complete temporary file; the first
  write wins, keys are unchanged. `tests/test_llm_race.py::test_processes_sharing_one_cache_first_write_wins`: 6
  processes miss the same 6 keys at once, on local disk and on the NFS repo filesystem, with and without hard
  links; every process gets the same record. Removing the lock makes the no-link variant fail. **[run]**
- **Split by item.** One Slurm job holds all servers of an experiment and the driver
  (`slurm/serve_and_run.sbatch`). `workflow.llm.RoutedLLM` assigns items to servers round-robin over the sorted ids,
  so all requests of an item go to one server and no request is generated twice, even across servers.
  `tests/test_routing.py` (two stub HTTP servers). **[run]**
- **Why one job.** The account already had 5 jobs running or pending (other work), so under the 6-job limit only
  one more job fits at a time. The 6-GPU limit is met inside that one job. **[run]** (squeue, 2026-11-27)
- **Compute.** GPU type for every model server: H200 (partition aiscii). Weights: the local HF cache copy of
  Qwen/Qwen3-14B (snapshot 40c0698, offline; nothing downloaded). Server: vLLM 0.19.0 + torch cu128 in a new uv env
  (`envs/boostrsi-vllm-0.19.0`; the earlier scratch envs had been emptied). Identical settings on every server:
  bf16, --max-model-len 40960, --kv-cache-dtype fp8, --enable-prefix-caching, --gpu-memory-utilization 0.90,
  --seed 0; per request temperature 0.7, max_tokens 30000, thinking on, seed = the request's repeat index.
  llm.cache_tag = kvfp8-h200. If the servers do not start without --enforce-eager, all of them restart with it;
  the choice is in `runs/slurm/job-<id>/server_config.txt`.
- `models/env.sh` is not sourced (it may hold credentials); HF_HUB_CACHE is set directly.

### P2. Pre-check changes
- **B4, the fixed improve baseline:** `seq[draft, loop(k=c-1, stop=best)[improve]]` at each F's worst-case calls c.
- Paired comparisons report the mean difference, the two-sided 90% percentile-bootstrap interval, the one-sided
  10th percentile (boosttopo bootstrap_lower) and win/tie/loss (tie = equal g). The gate uses the one-sided 10th
  percentile (revision §12) against the stronger of B1 / B2; informational only, no search is started.
- Execution: y0 for every item, then one pool over all (method, item) pairs, a checkpoint line per finished pair
  (`checkpoint.jsonl`); resume skips finished pairs and re-reads finished calls from the cache. On the toy run a
  restart after 300 of 640 pairs made 0 new calls and gave identical paired results. **[run]**
- Common random numbers: path seeds hash (run seed, item, path inside the workflow, repeat, step) and not the
  method name, so identical requests are shared across methods: single draft's draft is also the first draft of
  F1, B3 and B4; best-of-2's samples are best-of-4's first two; F3's three drafts are B1's first three.
  **[inferred]** from the seed function; the pairing makes the comparisons tighter, and it means those methods are
  not independent samples.
- `scripts/smoke.py`: step 2 of the plan.

### P3. Runs of 2026-11-27 (all numbers **[run]** unless marked)
- Slurm bug fixed: aiscii nodes do not hide unallocated GPUs, so counting GPUs with nvidia-smi started a second server
  on another job's GPU (job 4320351, cancelled after 7 min). The job now uses only CUDA_VISIBLE_DEVICES.
- Servers started without --enforce-eager (CUDA graphs on), on H200 (aisciit01 for the smoke, 6 GPUs on one node for
  the pre-check).
- Smoke (job 4320374, 1 H200, 14.9 min): y0 legal 10/12, mean score 0.442; writer legal 9/12, mean score 0.446;
  0/29 truncated; 16.3k-18.3k tokens per call; 133-156 calls/h/GPU at 12 concurrent requests. improve from a legal
  plan: 1/5 legal (s1002: 0.94 -> 1.18); the 4 illegal ones break funds or maturity. No stop rule fired.
  File: runs/smoke_d5/smoke.json, runs/smoke_d5/improve_io.json.
- Pre-check (job 4320382, 6 H200, 75.8 min): 783 new calls, 0 truncated, 1820 logical calls; 105 calls/GPU-hour over
  the whole job (tail included). Results: runs/precheck_d5/summary.json. No F passes the one-sided gate.
- GPU-hours: 0.12 (cancelled) + 0.25 (smoke) + 7.58 (pre-check) = 7.95.

## R. Revision §12 (overrides §3.3, §4.1, §4.2, §7.3 where they disagree)

### R1. Start from y0; g
- Every workflow starts with plan = best_plan = y0 (`CachedLLM.call_y0` on the benchmark prompt, repeat 0) and
  best_score = s0. y0's tokens are charged to every method (`Cost.y0_tokens`, included in `Cost.tokens`), as in
  boosttopo.infer.
- `draft` is the writer role on the current plan. `start: empty` (workflow.start) is implemented but not run. It
  starts from the empty plan and charges no y0 tokens.
- A missing test-split y0 aborts (`Env.make_start(require_cached=True)`, `workflow.require_cached_y0:
  [test_d5]`). The check covers the y0 tags (`llm.y0_cache_tags: [null]`) and the client's own name.
- g = `boosttopo.evalx.gap_closure`, unchanged. best_plan is replaced only by a strictly higher score, so the
  output is the better of y0 and what the workflow produced, and g ≥ 0.
- seq stops early once g(best) = 1, which is the GBDT stage skip. Before the revision it stopped at score ≥ 1.

### R2. Loop and gate semantics (= graph.run_subgraph)
- loop: the lineage's working plan is replaced when the new plan's score is ≥ the current one. The output is the
  best plan of the lineage, which starts with the loop's input. The stop rule runs before each repetition:
  - g(best) = 1;
  - stop=legal (= until_legal): the lineage best is legal;
  - optional `budget`: the item's tokens so far, y0 included, reach it.

  `reset_to_best` is removed.
- The `budget` loop parameter exists for `infer.repair_loop` (token budget). A float budget B is stored as
  ceil(B), which gives the same decisions for integer token counts. **Deviation:** a new loop parameter. It is not
  in the edit set.
- **New block `gate`** (one multi-agent stage). The body's final plan replaces the input only if it scores
  strictly higher. Inside a gate, best_plan is not updated and seqs do not stop early, because a stage runs all
  its nodes. Notes and diag written inside a gate are local to it. It is needed to write GBDT ensembles: a stage
  whose output is worse must not become the next stage's input. **Deviation:** not in §2; not in the edit set.
- **Step.reads fix (legality check, §2.4).** Since the stop rule runs before the first repetition, a loop can run
  zero times, and a gate discards its diag and notes. The reads-before-write analysis therefore no longer counts
  fields written inside a loop or a gate as defined after it (a fanout body still counts, n ≥ 1). Example now
  rejected: `seq[loop(k=2, stop=legal)[validate, repair], repair]` (the last repair may read a diag that was never
  written). Test: tests/test_model_exec.py::test_constraints. **[run]**
- fanout branches get their own cost counters, so budget stops inside a branch see only the fork point plus
  their own tokens. Branches are merged in branch order.

### R3. Seeds
- `seed_scheme: path` (all new structures). Request.rep = hash(run seed, item id, structural path with
  loop / fanout indices, repetition, step name), as before.
- `seed_scheme: legacy` (baselines that must hit the old cache). base = rep + run_seed·1e6. Then:
  - fanout branch j → base + j;
  - loop repetition j (1..k) → base + j·7919;
  - gate(occ=o) → base + o·20011;
  - all steps of a stage share the stage's index.
- Request.rep is both the server seed and part of boosttopo's cache key, which is unchanged.
- Pre-check: single_draft, B1 and B3 are legacy-seeded. B3 keeps diag_mode clean, so its repairs of *legal*
  plans are new requests, because the original verdict on a legal plan contains the reference profit; repairs of
  illegal plans are served from the cache. F1–F3 and B2 are path-seeded.

### R4. What a workflow can express of a multi-agent member (`workflow/legacy.py`)
The node-to-step mapping is:
- writer → draft;
- repairer → validate, repair;
- analyst → analyze;
- critic → critique (a new skill = the critic role; it exists only to express ensembles and is not in the edit
  set);
- a loop node → loop(k, legal | best)[...].

The state carries one plan, so a member is expressible only if:
1. the writer-type nodes form one chain from IN to the output;
2. each reader feeds exactly one writer, and a critic reads that writer's plan;
3. a reader's notes reach no later writer of the same member, because notes persist until the end of the gate.

Anything else raises `NotExpressible` with the reason. Measured on the toy run: the hand topology (analyst →
writer → repairer) fails rule 3, and `[IN->repairer, IN->writer, writer->repairer]` fails rule 1 (the repairer
reads two plans). **[run]**

`best_of_n_matched` resamples the benchmark prompt (the y0 request), which is not a workflow skill, so it is not
written as a workflow.

### R5. Tools and checks
- **validate.** `diag_mode: clean` (default) returns FarmAdapter.score(...).diag with
  " (reference greedy profit=...)" removed, and keeps the failing-day state. `diag_mode: original` returns it
  byte-identical, and is used only by the replay checks.
- **The §3.3 import test is replaced** by output tests: no tool output contains "reference", the reference
  profit (when it differs from the plan's profit), or the relative score (`tests/test_tools.py`,
  `replay_check.py` (c)).
- **simulate** uses the benchmark's simulator (`FarmAdapter.simulate` = farm_planner.simulate_plan) and reports
  only these keys, which FarmAdapter itself reads: profit, final cash, idle plot-days, days with free plots,
  unsold fruit, leftover seeds, unharvested plants, plantings. Plot occupancy runs to maturity, as in FarmAdapter's
  idle rate. The output is ≤ 400 characters.
- **crop_stats** reads crop_catalog[c].maturity_days and seed_price. Profit per plot needs the catalog's fruit
  price and yield keys. Their names in the real catalog are **[inferred]** (tried: fruit_price / sell_price /
  price, and yield / fruit_yield / harvest_yield). Without them the line has only grow days, seed price and the
  last plant day. Check this on the real items before the pre-check.
- **status** reads ScoreResult.category of the current plan:
  - not valid → illegal;
  - valid_ok → legal_high;
  - valid_idle or valid_waste → legal_low.

  The 0.8 threshold and the "bound" mode are removed. The executor stores the category whenever a plan is
  written.
- **error_type** labels (reads the current plan's category):

  | FarmAdapter category | error_type |
  |---|---|
  | insufficient_funds | funds |
  | farmland_exceeded | farmland |
  | not_mature | not_mature |
  | missing_seeds, missing_fruit | inventory |
  | bad_day, unknown_crop, schema_error, parse_error, empty | other |
  | valid_* | none |

  The labels changed from (funds, not_mature, farmland, other) to add inventory and none.
- **Cost.** validate counts as one scorer call (a diagnosis), as does every plan score. Before the revision the
  diagnosis was not counted. The original's per-stage scorer-call totals are not compared.

### R6. Prompts
- draft / repair / analyze / critique are `boosttopo.roles.role_prompt`, unchanged:
  - the task text is the benchmark's build_prompt;
  - the ledger tail is included;
  - notes.analysis and notes.critique become the upstream ANALYST / CRITIC NOTES, in the order they were written.
- improve is new: the head is written here, in the role-prompt layout, and the writer's ledger tail is appended.
- **Deviation from §3.4 (fixed, possibly empty slots).** Tool outputs are added as sections only when the tool
  ran (before the ledger tail). An empty slot would change every original prompt and break cache replay. A
  with-tool vs without-tool comparison therefore differs by the section itself.
- Parsing: the plan is the model's answer after `strip_thinking`, where a truncated think block gives an empty
  answer. The benchmark parses it with the original `extract_json`.

### R7. Reuse of boosttopo
- `workflow/llm.py` builds boosttopo's CachedLLM / OpenAICompatLLM with api_key=None. No environment variable is
  read, unlike `build_llm`.
- Config is boosttopo's YAML + `--set` (`configs/workflow_d5.yaml`, `configs/workflow_toy.yaml`).
- Splits come from `boosttopo.run.make_splits` (never the final-test splits). The run directory uses `RunLog`.
  `precheck.py` and `run_boost.py` write `per_item.jsonl`.
- **Checkpoint / resume:** `fit_serial` and `fit_branch` write `checkpoint_<mode>.json` after every round. A rerun
  resumes after the last finished round, with the policy history, the proposer's tried set and lam restored. A
  resumed toy run equals a single run, with 0 new calls (`tests/test_boost.py`). **[run]**
- **Fresh samples after a rejection** (`boost.fresh_after_reject`, boosttopo's rep_base shift). After a rejected
  round, candidate stages run at repetition = number of rejections so far, and the frozen prefix stays at 0.
- **Defaults.** All of these are tested only on the fake model.
  - lam auto = 0.02·1000 / (mean tokens without y0 of `seq[analyze, draft, validate, repair]` on the first 8
    train items);
  - accept_margin 0.01;
  - min_gain 0.005;
  - max_calls 12;
  - concurrency 24.
- **Acceptance** is accepted iff the one-sided 90% lower bound (boosttopo's `bootstrap_lower`, 10th percentile,
  2000 resamples, seed 0) is > accept_margin. Reported intervals are two-sided 90%.

### R8. Cache race fix (boosttopo.llm.CachedLLM)
- Identical requests in flight at once are merged: the first one calls the model, the others wait and read its
  record.
- A record is never overwritten: it is stored by exclusive create (link), and if another process wrote first, the
  stored record is returned.
- An unreadable record is still replaced, as before.
- The cache key is unchanged (`tests/test_llm_race.py`). **[run]**
- The eval-stage cache (`evalx`) still overwrites. With the LLM fix its inputs no longer diverge, but it was not
  changed.

## 1. Representation (§2), unchanged parts
- The State has executor-maintained fields: plan_score, plan_legal, plan_cat (the current plan), and best_legal,
  best_cat.
- Block params are frozen (hashable). Branch cases are stored as children plus a labels param.
- Notes are addressed as `notes.<key>`; `notes` is always readable. **Revision:** plan and best_plan are readable
  from the start (y0).
- Depth counts blocks, with the root seq at depth 1. The limit is 3.
- The textual form (`parse` / `to_str`) gains `gate[...]`, `gate(occ=1)[...]` and
  `loop(k=.., stop=.., budget=..)[...]`.

## 2. Edits (§5), boosting (§6): unchanged parts
- `legal_edits` validates every candidate on the attached workflow. Structures are deduplicated across slots, and
  the proposer never re-proposes an evaluated (object, edit) pair. One slot per kind (fragment / control / other)
  is kept.
- The edit set's skills are draft, repair, improve and analyze. critique, gate and the loop budget are not in it.
- Serial attach splices fragments onto the root seq. Frozen-prefix evaluation is sample-identical to a full run
  and pays only for the new stage. **[run, toy]**
- Branch mode: one-block fragments become the case; heldout items with other labels contribute 0.
- Weights are 1/(ε + g) with ε = 0.1, capped at 5×mean and normalised. g ≥ 0 now, so the clip at 0 no longer
  matters.
- In branch mode the no-residual target draw is seeded per round, so it resumes; it no longer uses one stream.

## 3. Throughput assumptions **[inferred]**
- The design doc assumes about 64 calls per hour.
- The multi-agent notes measured about 200 calls per hour at d3 in thinking mode (about 13k tokens per call),
  before the fp8 KV restart.
- d5 calls are probably longer. Use the rate measured on the pre-check's first items.
