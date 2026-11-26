# Boosted Topology: implementation notes

Implementation of `Boosted Topology：实现文档.md`, milestones M0 to M3 (T1 to T8). M4 (playbook) is not implemented.

## Where each part of the design doc lives

| Doc section | Code |
|---|---|
| 3.3 Repository layout, T1 config / LLM / cache | `src/boosttopo/config.py`, `src/boosttopo/llm.py` (`OpenAICompatLLM`, `FakeLLM`, `CachedLLM`) |
| 4 SelfImprove-Bench adapter (T2) | `src/boosttopo/bench/farm.py` (the benchmark's own `farm_planner.generate_task`, `simulate_plan` and `game.build_prompt`), `bench/toy.py` (toy benchmark for tests) |
| 4.4 g | `evalx.gap_closure` |
| 5.1 Role library | `src/boosttopo/roles/__init__.py` |
| 5.2–5.3 Subgraph, edits, macros, gated executor (T3) | `src/boosttopo/graph/__init__.py` |
| 5.4 Evaluator + cache (T4) | `src/boosttopo/evalx/__init__.py` |
| 6.1 grow_subgraph, 9.3 Obs / f0, 9.4 EditOutcome (T5) | `search/grow.py`, `policy/__init__.py` |
| 6 GBDT (T6) | `boost/gbdt.py` |
| 7 AdaBoost, library, offline selection (T7) | `boost/adaboost.py`, `boost/library.py`, `boost/residual.py` |
| 8 Proposers, samplers, K branches, OOB selection (T8) | `search/proposers.py`, `boost/sampler.py`, `boost/common.pick_candidate` |
| 6.5 / 7.5 Inference, 11.1 baselines | `infer.py` |
| 10.3 Logs | `runs/<id>/{rounds,edit_outcomes,library,budget}.jsonl`, `checkpoint.json`, `manifest.json`, `config.json` |
| End-to-end driver | `python -m boosttopo.run --config configs/<preset>.yaml [--set a.b=v] [--baselines]` |
| Report | `python scripts/report.py <run ids>` writes `runs/REPORT.md` and `runs/heldout_g_vs_tokens.png` |

Tests (section 10.4, toy benchmark + FakeLLM, about 2 s): `.venv/bin/python -m pytest -q`.

## Choices made where the design doc left room, and deviations from it

- **Who wrote the code.** I (Claude Code) wrote it directly, not through mini-swe-agent task sheets (section 3). The acceptance tests of T1–T8 are in `tests/`.
- **Benchmark interface (open item 12.1-1).** Farm generation and scoring run without Docker: the adapter imports `benches/farm/farm_planner.py` from the cloned repo. The reference profit it recomputes matches the official case files (checked on `farm_d3_001.json`). Scores below 0 (legal plans that lose money) are floored at 0, which is the same as the empty plan.
- **Simulator verdict (diag).** When a plan is illegal, the verdict adds the simulator state at the start of the failing day (cash, free plots, inventories, growing crops), computed by replaying the plan's legal prefix through the official simulator. Without it, the repairer fixed the first error and then hit the next one, so every repair scored 0.
- **Which plan a node treats as its "current solution".** In the doc's pseudo-code every node edits `y_in`. Here a node edits the output of its latest writer-type upstream node (or `y_in` if it has none). Other writer upstreams are shown to it as alternative plans, and reader outputs as notes. The repairer's verdict is computed on that current plan.
- **Repeats.** Each (subgraph, item) is run `n_repeats` times. The score is the mean over repeats. In GBDT mode the plan carried to the next stage is repeat 0's gated output.
- **Random-subspace proposer.** For macros, `writer` always counts as an allowed role. Otherwise, with `role_frac = 0.5`, `critique_and_rewrite` and `repair_tail` would almost never be available.
- **K-way search branches** run one after another in the same process; the evaluation calls inside each branch run in parallel. There is no Slurm job array or `grow_worker`.
- **Combat (G3)** is not connected, because it is a multi-step interactive game (section 4.1 also defers it).
- **Experiment scale.** See `runs/REPORT.md` and the experiment section below for the sizes actually used and why.
- **Rejected rounds draw new samples.** In the default preset (`sampler=all`, `k=1`, deterministic f0) nothing is random, so a rejected round would re-run exactly the same cached evaluations, and "retry with a new random seed" (section 6.3) would do nothing. After each rejection the evaluator shifts the repeat index (`Evaluator.rep_base`), so the next round draws fresh LLM samples. Inference uses `rep_base = 0`.
- **Top-k candidates run concurrently** inside `grow_subgraph`, and are logged in rank order. The local GPU is limited by its slowest generations, so running candidates one after another roughly tripled wall time.
- **Cache keys.** LLM cache: (model and generation settings, prompt hash, temperature, repeat). Evaluation cache: (subgraph key, item, input hash, repeat, model, prompt-template hash).

## LLM and difficulty (open item 12.1-4: model choice)

- Model: Qwen3-14B served locally with vLLM 0.32 on the job's A6000 (`--max-model-len 16384`, prefix caching), non-thinking mode, temperature 0.7, roles capped at 4096 tokens. The endpoint, model, key variable and generation settings all come from the `llm:` block of the config, so switching to an API model needs no code change.
- Measurements that shaped the setup (8 to 12 training items, 2 repeats, gated on y0):

| setting | y0 (single writer) | writer | repairer | writer->repairer | analyst->writer | analyst->writer->repairer |
|---|---|---|---|---|---|---|
| d3, JSON-only roles | score 0.04, 5% legal (37/40 "insufficient funds") | g 0 | 0 | 0 | 0 | 0 |
| d3, ledger roles (+ failure state) | same | 0.066 | 0.052 | 0.066 | 0.000 | 0.057 |
| d3, thinking mode | score ~0.53 | 0.166, but ~13k tokens per call (~200 calls/h on this GPU) | - | - | - | - |
| **d2, ledger roles (used)** | score 0.13 | 0.072 | 0.143 | 0.179 | 0.075 | 0.155 |

- "Ledger roles": writer-type roles first write a brief day-by-day plain-text ledger, then give the plan in a final ```json block. The adapter parses the first ```json block in the reply (`bench.farm.extract_json`).
- Because of this, the experiments use **difficulty 2** for training, heldout and G1, with 24 / 20 / 20 items. G2 is d3 and d4 (10 items each). On d3, a first GBDT run with this model rejected both rounds (heldout gain lower bound 0.000) and stopped with an empty ensemble.

## Round 2: making M1–M3 trustworthy (no new features, no M4)

### Best-of-N = 0.000 on G1 (step 1): not a code bug
- Checked by re-scoring every cached best-of-16 sample on heldout and G1 (`best_of_n` asked for N samples,
  the budget was N x y0 tokens, all N requests reached the LLM cache with distinct repeat indices).
- Cause: with the benchmark's own prompt (the y0 / single-writer prompt, JSON only, no ledger), Qwen3-14B in
  non-thinking mode is close to deterministic. Many items return the same byte-identical plan for every
  seed (verified by calling vLLM directly with seeds 0–3 and with no seed: same 735-char plan). Across 16
  samples there were only 1–16 distinct plans per item (mean ≈ 6).
- By contrast the ledger role prompts sample diversely: writer and repairer each gave 4 distinct outputs in
  4 seeds on all 6 heldout items probed. So the best-of-N baseline (benchmark prompt) explores far less than
  the searched writers do; that is a property of the baseline as the doc defines it, not of the code.
- On heldout, 3 of 20 items had a legal sample other than y0, giving best-of-16 g = 0.108. On G1 the only item
  with any legal sample already had a legal y0 (no gain possible), so g = 0.000. 3/20 vs 0/19 eligible items is
  well within chance (Fisher p ≈ 0.23). No earlier number changes because of a bug here; the honest reading is
  that every heldout/G1 difference in the last report rests on a handful of items.
- Regression tests: `tests/test_round2.py::test_best_of_n_*` (N distinct samples drawn, tokens = N x y0, the
  best one picked, same behaviour from any split, seed shifts the extra samples).

### Prompts and diagnosis used by each method (step 1, checked by reading the code + tests)
- single writer = y0 and best-of-N use the benchmark prompt unchanged (`roles.y0_request`). They do **not** use
  the ledger-before-JSON role prompt and see no diagnosis. This is by design (y0 is the doc's start solution),
  but it means best-of-N samples a different, lower-diversity prompt than the searched writers.
- The hand topology and every searched subgraph go through the same `graph.run_subgraph`: identical role
  prompts (`roles.role_prompt`, ledger tail for writer/repairer) and the identical repairer diagnosis
  (`FarmAdapter.score(...).diag`, with the failing-day state).
- All methods use the true scorer at test time: best-of-N to pick a sample, the subgraphs to gate each stage's
  output and to produce repairer diagnoses. Scorer calls per item are now reported
  (`infer.stage_scorer_calls`: one per repairer node + one gate per stage, + 1 for y0; best-of-N: N).

### Other changes
- **Reporting** (`scripts/report.py`): per split and method, mean g, per-seed g, paired bootstrap 90% interval
  of Δg against the single writer and against best-of-N on the same items, legal rate, mean score among legal
  plans, tokens and scorer calls per item. Runs `<preset>_s<seed>` are pooled per preset. Every run now writes
  `runs/<id>/per_item.jsonl` (one line per split, method, item, repeat). Since g is measured against y0,
  the single writer has g = 0 by construction and "Δg vs single writer" is just g.
- **Splits**: `bench.tests` is renamed to `bench.dev_tests` (old G1 → `dev_d2`, G2_d3 → `dev_d3`,
  G2_d4 → `dev_d4`); using the old key is an error. New `bench.final_tests` with fresh, never-used seeds and
  the design doc's sizes/difficulties: `final_G1_d3` (d3, seeds 7000–7019), `final_G2_d4` (d4, 8000–8009),
  `final_G2_d5` (d5, 9000–9009). They are only generated into the run and evaluated with
  `python -m boosttopo.run ... --final-test` (which needs an already fitted ensemble, evaluates only the final
  splits and writes `final_final_test.json` / `per_item_final_test.jsonl`). Id overlap with every other
  split is checked either way. Test: `test_run_without_flag_never_touches_final_items`.
  **Note**: the design doc's final G1 is d3, while all current experiments train at d2, where this model is
  off the floor; I kept the doc's difficulties as asked. Decide before using it.
- **Config merge**: maps of named splits (`dev_tests`, `final_tests`) now replace the default instead of
  being merged into it, which is what added the unplanned d5 split (`config.REPLACE_KEYS`).
- **Sizes**: heldout d2 seeds 2000–2059 (60 items), `dev_d2` seeds 3000–3059 (60). Train stays at 24.
- **f0 rule 1** now promotes only edits that increase the number of repairer nodes (`policy._adds_repairer`).
  Before, `change_role` turning a repairer into another role also got the +100. Test:
  `test_f0_rule1_promotes_only_edits_that_add_a_repairer`. This changes the candidate order, so searches
  after this fix do not reproduce earlier runs' cached search paths.
- **Run seed**: `seed` now shifts the repeat index of every subgraph-stage LLM sample
  (`Evaluator.rep_seed = seed * 1e6`, fit and inference) and best-of-N's extra samples. Before, with the default
  presets (deterministic f0, sampler all, k = 1) the seed changed nothing. Seed 0 gives exactly the old cache
  keys. y0 is not shifted: all seeds start from the same y0.
- **Difficulty calibration**: `scripts/calibrate_difficulty.py --config configs/calibrate.yaml` (model and
  endpoint from the `llm:` block; seeds 6000–6019 per difficulty, not used by any other split). Output in
  `runs/calibration_<model>.{json,md}`.
- **Round-2 reruns**: `scripts/run_round2.sh` (GBDT and AdaBoost default presets, seeds 0–2, baselines),
  report in `runs/REPORT_round2.md`.
- `docs/interfaces.md` referenced in the round-2 request does not exist in this repository; no public
  signature of the modules was changed except: `infer.best_of_n(..., seed=0)` (new optional argument),
  `run.make_splits(cfg, bench, final_test=False)` (new optional argument), and the dicts returned by `infer` /
  `best_of_n` gained `scorer_calls_per_item`, `legal_mean_score` and `rows`.
- Cached results: the fixes do not change any LLM-cache or eval-cache key for seed 0; cached stage outputs
  stay valid. What changes is which candidates the f0 search evaluates (rule 1) and the larger heldout.

## Round 3

- **Writer-role baselines** (`infer.writer_role`, `best_of_n_subgraph`, `repair_loop`; in every `--baselines` run):
  - `writer_once`: `[IN->writer]` (ledger prompt) on y0, one sample per repeat (n_repeats repeats averaged,
    like `ours`), gated against y0 like every stage. It reads y0 because `IN` is y0 in this framework; a
    from-scratch writer (empty current plan) is what the calibration script's `--prompt writer` measures.
  - `best_of_n_writer`: the same subgraph sampled N times (repeat indices 0..N-1, so sample 0/1 are the
    `writer_once` samples); N = ceil((our tokens − y0 tokens) / writer-stage tokens) per split.
  - `repair_loop`: writer role on y0, then the repairer repeatedly repairs the writer's own lineage (the
    best-scoring plan of the lineage so far, newest on ties) with the scorer's diagnosis until the per-item
    token count reaches our method's mean tokens per item on that split (max 40 repairs; stops at g = 1).
    The answer is the best of y0 and the lineage. Scorer calls: y0 + writer output + 2 per repair
    (diagnosis + score), the same convention as `stage_scorer_calls`.
  - The report gives Δg vs y0, vs `writer_once` and vs best-of-N (y0 prompt) with paired 90% intervals.
- **Fresh heldout re-evaluation** (`fresh_heldout_eval`, default on; `Evaluator.run_fresh`). Deviation from the
  design doc. Candidate *selection* never uses heldout (search and OOB pick run on train only; checked by the
  leak test), so the accept decision is the only use of the heldout sample. With the switch on, the accept
  decision is still made on that sample, but once a round is accepted the accepted subgraph is re-run on
  heldout with unused repeat indices (`rep_base + 500000`). That fresh sample gives the reported heldout g and,
  in GBDT, the heldout solutions carried to the next stage. In AdaBoost the members' heldout g used for later
  rounds comes from the fresh sample. `rounds.jsonl` records both `ho_gain_mean` (decision sample) and
  `ho_gain_mean_fresh`. Offline selection still compares on the decision-style sample (unchanged).
- **Calibration script**: `--prompt benchmark|writer` (writer = writer role from an empty plan) and
  `configs/calibrate_thinking.yaml` (thinking on, `max_tokens` 13000, same temperature). Outputs are
  `runs/calibration_<model>_<prompt>_<think|nothink>.{json,md}`.
- **Sampling settings** (checked by capturing the request bodies the client sends, and the vLLM server log):
  every request (y0, best-of-N, every role) sends `temperature 0.7, max_tokens 4096, seed = repeat index,
  chat_template_kwargs.enable_thinking = false`. No top_p/top_k is sent, so vLLM applies the model's
  `generation_config.json` defaults `top_k 20, top_p 0.95` (server log: "Default vLLM sampling parameters have
  been overridden by the model's generation_config.json"). The settings are identical for all methods.
- `scripts/members.py`: accepted member/stage structures, and for AdaBoost the per-item "which member solves
  which item" tables on heldout and dev_d2.
- `docs/interfaces.md`: written from the code (round 3) as the reference for public signatures.

## Round 4

### Part 1 tooling
- `scripts/calibrate_v2.py`: per difficulty, y0 (benchmark prompt), the writer role from an empty plan, and the
  writer role once on y0, 1 sample each, seeds 6000+. Reports legal rate, truncated share (`finish_reason ==
  "length"`), unparseable share (parse / schema / empty), the unclipped simulator `relative_score` among legal plans,
  failure shares among illegal plans, tokens (prompt + completion, thinking included) and seconds per item.
  Criterion 2 ("legal plans leave room to improve") is operationalised as: at most half of the legal plans score
  >= 0.95 of the reference. This threshold is my choice.
- vLLM restarted with `--max-model-len 40960` (the model's limit); thinking configs use `max_tokens 30000`
  (`configs/calibrate_thinking_long.yaml`), leaving room for prompts up to ~10k tokens.
- `OpenAICompatLLM.call_meta` / `CachedLLM.call_meta` return `(text, tokens, meta)` with finish_reason, completion and
  prompt tokens and seconds; new cache records store `meta`. `call()` and cache keys are unchanged.
- Thinking outputs: the cache now returns the text after the `<think>...</think>` block (`llm.strip_thinking`); an
  unterminated block (truncated inside the reasoning) yields an empty answer. The raw output stays in the cache
  record. Without this, a downstream role would receive the whole reasoning of the plan it works on. All 33,246
  non-thinking cache records were checked: none contains a think tag, so no earlier result changes.

### Decision a: duplicate members draw fresh samples (`fresh_duplicate_stages`, default true)
- Implemented as the member's **occurrence index among earlier members with the same structure** (not its absolute
  stage position): `rep += occ * 20011` in `Evaluator._one`, so it enters the seed and the stage cache key. Every
  non-duplicate member keeps occurrence 0 and its earlier key; fitting uses the same index the member will have at
  inference (`occurrence(h, current ensemble)`), which keeps AdaBoost's offline selection consistent with inference.
  With absolute positions, a member chosen by offline selection would be re-sampled at inference with a different
  seed from the one it was fitted with.
- Runs before round 4 are reproduced with `fresh_duplicate_stages=false` (checked, cache only: GBDT seed 2 and
  AdaBoost seed 0 reproduce every stored row of ours and the hand topology on heldout and dev_d2 exactly).
  Scripts reading old runs (`attempts.py`, `dup_stages.py`) set the switch off when the stored config lacks it.
- Inference rows now carry `stage_runs` and `dup_stage_runs` (stage runs that exactly repeat an earlier one of
  the same item and repeat); the report shows their share (should be 0 with the switch on).
- Offsets cannot collide (test): r + 1000*rejections + 20011*occ + 7919*j + 500000*fresh < 1e6 for the ranges used.

### Decision b: loop attribute (`loop.enabled`, default false)
- `Node(nid, role, k=1, stop="")`; a writer-type node with k > 1 runs its role up to k times on the best plan of its
  own lineage (newest on ties; the lineage starts with the node's input). Stop rules: `until_legal` (stop once the
  lineage holds a legal plan) and `keep_best` (run all k, stop early only at g = 1). The check runs before every
  repetition, so an input that already qualifies costs nothing. Repetition j uses repeat index rep + 7919*j, the
  repair-loop baseline's schedule. Output = the best plan of the lineage; then the usual stage gate.
- Every repetition's tokens are added to the stage cost; scorer calls: 2 per repairer repetition (diagnosis + score
  of the new plan), 1 per writer repetition, plus the stage gate. `StageOutput.scorer_calls` now carries the count
  (stage results cached before this round have no field and use the static count, which is exact for loop-free
  stages).
- `[IN->writer, writer->repairer*k keep_best]` reproduces the repair-loop baseline with k repairs and no token cap
  exactly: same plans, scores, tokens and scorer calls per (item, repeat) (test, k = 2, 4, 8). The baseline itself
  still stops on the token budget, the node on k.
- Edit `set_loop(nid, k, stop)`, k from `loop.ks` = (2, 4, 8) (k = 1 clears). It is not in `EDIT_KINDS`, so the
  random-subspace proposer's sampling is unchanged when loops are off; the loop edit is proposed only when enabled.
- Loop-free subgraphs keep their keys, `to_dict` and `describe` (tested against keys recorded with the old code).
  Looped nodes are written `n1:repairer*4keep_best`.
- **Deviation:** with loops on, one evaluation slot per step is reserved for the best-ranked loop edit
  (`loop.reserve_slot`, default true). Reason: f0 rule 1 adds 100 to every edit that adds a repairer, which is the
  usual case with this model, so a loop edit would never reach the top-3 slots. The f0 rules are unchanged.

### Macro switch (`search.macro_slot`, default false)
- The atomic proposer also emits every legal expansion of the three macros, and one evaluation slot per step is
  reserved for the best-ranked macro (`search.grow.reserve_slots`; with both switches off it is exactly
  `ranked[:top_k]`).

### Baselines
- `baselines.bon_round`: "ceil" (rounds 2-3, default) or "floor" (round-4 presets) for best-of-N's sample count.

### Reduced Part 2 (Stage A / Stage B), as confirmed after Part 1
Setup: Qwen3-14B, thinking on, temperature 0.7, max_tokens 30000, main difficulty d5. Preset `configs/r4_d5.yaml`.
- **Server change (fp8 KV cache).** vLLM was restarted with `--kv-cache-dtype fp8`; the KV cache went from 93.6k to
  185.6k tokens, so up to 4.53 requests of 40,960 tokens fit at once, up from 2.29. Otherwise the server is unchanged.
  fp8 KV storage can change outputs slightly, so the cache-key model name now carries a server tag
  (`llm.cache_tag: kvfp8`, new; null = the earlier names). Every Stage A/B output is generated under the tag.
- **Fixed y0 inputs.** The test split is the 40 calibration items at d5 (seeds 6000–6039). Their y0 plans are the
  calibration's cached records, generated before the restart. They are reused read-only through
  `llm.y0_cache_tags: [null]` (`CachedLLM.call_y0`). The calibration's writer-on-y0 records are *not* reused, because
  those requests are keyed under the new tag. y0 of the new train/heldout items is generated under the tag.
  `stage_a.py` aborts if any test-split y0 would be generated.
- **Repeats.** There is 1 repeat per evaluation everywhere (fit, heldout and test), including the Stage A baselines,
  so every comparison is paired on (item, sample).
- **Stage A** (`scripts/stage_a.py`):
  - writer once;
  - best-of-5 of the writer role;
  - the repair loop with at most 5 repairs, keeping the best (no token cap; it stops early at g = 1);
  - the hand topology, at its actual tokens.
  - writer once, best-of-5 sample 0 and the repair loop's writer call are the same request, so they share one
    sample.
- **Stage B slots** (`search.slot_mode: one_per_kind`, new, default `top_k`). Each step evaluates exactly the
  best-ranked macro, the best-ranked atomic edit (not a loop edit) and one loop edit.
  - **Deviation:** f0 gives every loop edit the same score. Its order among them is only the string tie-break, so
    the loop slot would always try the same (node, k, stop rule). With `loop.slot_pick: random` (new, default
    `rank`), the loop edit is instead drawn uniformly from all legal loop edits, seeded by
    (search seed, step).
- **Stage B other settings:**
  - 12 train and 30 heldout items, new d5 items (seeds 1000–1011 and 2000–2029, never used at d5);
  - `max_steps 2`, `rounds 3`, `patience 1`;
  - `lam: auto` (the GBDT default), with bon_round floor;
  - the test split is evaluated only for `ours` and `repair_loop`; the repair loop runs at ours' actual tokens per
    item, as in run.py.
- **Report scores.** "Mean score among legal plans" in Stage A/B tables is the scorer's score, clipped at 0 below and
  not clipped above. It differs from the unclipped value only for legal plans with negative profit.

### Round 5 Part 1 findings (cache-only checks on Stage B seed 0, `scripts/r5_checks.py`)
These are observations only. No behaviour has been changed yet.

- **The macro slot duplicated the atomic slot in all 6 steps.** In every step the best-ranked macro and the
  best-ranked atomic edit produced the same structure:
  - analyze_first = add_reader(analyst);
  - repair_tail = add_node(repairer).

  Each step therefore evaluated only 2 distinct structures.
- **The two copies were sampled twice, under a race.** They were evaluated concurrently on the same stage and LLM
  cache keys. Both missed the cache and both generated, and vLLM gave different outputs for the same seed (batch
  composition). The cache kept the last write. Consequences:
  - in round 0, add_reader (fit 0.397) and analyze_first (0.342) evaluated the same structure;
  - the accepted stage is the add_reader one;
  - its stored bag fit came from the sample that was later overwritten: oob_fit 0.341 equals the macro's value.
  
  Heldout acceptance used separate samples, so the accept decision is not affected. Train-side selection is
  effectively a best of two samples.
- **Stages with different structures can repeat requests.** Two stages with different structures that receive the
  same input repeat each other's requests: writer*8keep_best followed by writer*2keep_best, with repetition j at
  rep + j·7919. Effects:
  - on the test split, ours made 12 such repeated requests out of 438, all in the second loop stage;
  - for 5 items, stage 1 left the plan unchanged, so stage 2's first repetitions were identical to stage 1's;
  - the tokens were still charged to ours.
  
  `fresh_duplicate_stages` only covers members with the same structure.

## Round 5 Part 2: step 8 ablations and step 9 items

- 8a (`scripts/r5_ablations.py --step 8a`): seed-0 ensemble with stage 0 replaced by a plain writer call
  `[IN->writer]`, stages 1-2 unchanged, run through `infer` (same gating / early stop). The plain writer call at
  repeat 0 is the same request as writer_once / best-of-N sample 0, so that sample is shared from the cache.
- 8b (`--step 8b`): stage 0's cached gated output, then independent plain-writer samples on it (repeat indices
  0, 1, ... of that stage; disjoint from ours' loop indices j*7919), keep best by score, items at g = 1 skipped.
  Drawn round by round; a new round only if the projected mean tokens/item (current + last round's mean cost x
  open share) stays within ours' 203.6k. Result: 10 rounds, mean 9.43 samples/item, 201.5k tokens/item.
- Step 9 replication items: difficulty 5, generator seeds 10000-10039 (split `rep_d5`, configs/r5_rep_d5.yaml).
  Never used before (used d5 seeds: 1000-1011, 2000-2029, 5000-5009, 6000-6039; final-test 9000-9009); checked
  40 distinct prompts with 0 overlap with any used or final-test item. Ensemble = seed-0 ensemble.json copied
  unchanged into runs/r5_rep_s0 (no refit). Methods: ours, best_of_n_writer (floor-matched); no repair loop.
