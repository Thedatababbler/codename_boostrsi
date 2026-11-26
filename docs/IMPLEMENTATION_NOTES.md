# Implementation notes — deviations from 《Boosted Workflow：单智能体版本实现文档》

Per §9.3, every place where the implementation departs from the design doc is recorded here.
Status labels: **[measured]** = checked by running code / tests here; **[inferred]** = reasoning only.

## 0. The original repo was not available (largest deviation)

The doc says to modify `~/projects/boostrsi_next`. That path does not exist on this machine, and
no copy was found under the user's project directories. So this is a **self-contained
implementation** in `boostrsi/` with thin adapters where the original code would plug in:

| Doc expects | What exists here | To integrate with the original repo |
|---|---|---|
| Farm benchmark, real rules, scorer, reference solution | `farm/` — a **toy** Farm game (`rules.py`), greedy reference (`reference.py`), scorer (`bench.py`) | Implement `Bench.score(item, plan) -> ScoreResult(score, raw, legal, profit, error, error_msg)`; point `workflow/tools.py` at the real rules module (never at the reference solution) |
| Original `writer` / `repairer` / `analyst` prompts, unchanged | Stand-in prompts in `workflow/skills.py` | Paste the originals verbatim into `TEMPLATES`, keeping the `### SKILL:` header line and the `{crops}/{sim}/{analysis}` slots |
| `evalx/` evaluator + cache | `workflow/evaluate.py`, `workflow/llm.py` (`CachedLLM`, JSONL) | Wrap the existing cache or reuse its key format; keys must include the structural step path (see §4) |
| `search/grow.py`, `search/proposers.py`, `policy/`, `boost/`, `baselines.py`, `scripts/report.py` | `workflow/grow.py`, `workflow/proposer.py`, `workflow/policy.py`, `boost/{serial,branch,global_search}.py`, `workflow/baselines.py`, `scripts/report.py` | Move as needed |
| Old role-subgraph code / old configs | Not present | — |

Consequences:
- **Not done: "old configs reproduce old results line for line" (§1.2, S0).** There is no old code.
- **Not done: "repair loop / best-of-n written as workflows replay from cache and match the stored
  baseline results per item" (W1, S0).** There are no stored results. As a substitute, the
  tests check that a cache replay is identical to the live run with zero new calls, and that
  best-of-2's samples are a prefix of best-of-4's (`tests/test_model_exec.py`). **[measured]**
- **Not done: "y0 from the cache".** `Item.y0_score` takes the cached value when present. Otherwise
  y0 is the score of one draft at a dedicated seed (`run_seed="y0"`), and its cost is never charged
  to any method. **[measured]**
- **Toy items.** `farm/bench.py:standard_splits` generates train 12 / heldout 30 / test 40 /
  final 40 for the split sizes in §8.4. The real 40 calibration items must replace `test`.

## 1. Representation (§2)

- **State has extra executor-maintained fields**: `plan_score`, `plan_legal` (score and legality of
  the *current* plan), `raw_best` (uncapped score of `best_plan`, for the §8.2 breakdown), and
  `best_legal`. Needed by `loop(stop=legal)`, the `status` check, and the report.
- **Block params** are stored frozen (`frozen_params`) so blocks are hashable. `.params` gives a
  dict view. Branch cases are stored as `children` plus a parallel `labels` param. `.cases` gives
  the dict.
- **Notes fields** are addressed as `notes.<key>`. `notes` itself is always readable, and missing
  keys render as empty slots (§3.4). `best_plan` counts as written whenever `plan` is.
- **Depth** counts blocks, with the root seq at depth 1. The §2.5 example has depth 3, which is
  the limit.
- **Default `max_calls = 8`.** The doc gives no number. Every precheck method and the doc example
  fit under it. Settable with `--max-calls`.
- A textual form (`parse` / `to_str`) is included. It is the form the doc uses, so every handwritten
  workflow is written exactly as in the doc.

## 2. Tools and checks (§3)

- **`status` vs §2.4.** The doc's default ("by the current score reaching the threshold, 0.8")
  needs the scorer, which depends on the reference solution. That is not inference-time
  information, so it conflicts with §2.4. Both are implemented: `--status-mode score` (doc default)
  and `--status-mode bound`, which uses only simulated profit divided by a naive upper bound from
  the problem. **Needs a decision.** **[inferred]**
- `best_plan` tracking uses the scorer, as the doc's executor sketch does. It is the gate equivalent.
- `reset_to_best` is a tool with signature `best_plan -> plan` (§4.1). It is a no-op if no best
  exists yet.
- `simulate` output (first version, §10.2-3): profit, idle plot-days, ranges of days with idle land,
  the days with the most idle cash plus land, and the first harvests. It is compressed to
  ≤ 400 characters, so it is not a full per-day table.
- Tool boundary (§3.3) is enforced by tests. `workflow/tools.py` imports only `farm.rules`, checked
  both statically with AST and at runtime. **[measured]**

## 3. Skills (§3.1)

- `improve` is newly written (`workflow/skills.py`). Per §10.2-2 it **must be read by a human on a
  few training items before the precheck**. Its output has not been checked with a real model.

## 4. Executor and seeds (§4)

- **Seed = hash(run seed, item id, structural path, rep, step name).** The doc lists
  (run seed, path, loop/fanout index, rep). The **item id was added** because, without it, all items
  share one random stream at a given step. That made the fake model's results strongly correlated
  across items. With a real model the prompt already differs per item, so this is harmless.
  **[measured on the fake model]**
- **Structural path.** Each element is a child index, a branch label, or `("it", i)` / `("fo", j)`
  for loop / fanout iterations. Labels are used for branch cases so a case keeps its seeds when
  other cases are added.
- **Consequence for the cache.** Identical prefixes share samples across workflows. For example,
  `seq[draft, ...]` reuses the single-draft sample, and best-of-n reuses best-of-m (m < n).
  Prepending a step such as `crop_stats` shifts paths, which gives fresh samples. The prompt
  changes in that case anyway.
- **Ties in fanout** go to the first branch. A legal plan with score 0 replaces an illegal one as
  `best_plan`.
- **Cache hits are charged their original usage**, so costs are identical to a live run.
  `new_llm_calls` counts real invocations.

## 5. Edits (§5)

- `legal_edits` generates candidates structurally, then **filters each one through the full
  validator**, applied to the *attached* workflow when a fragment is being grown. This guarantees
  that every edit maps a legal workflow to a legal one. Random-sequence tests cover it. **[measured]**
- `add_branch` children are drawn from a small fixed set: repair_loop, sim_improve,
  `seq[validate, repair]`, `seq[improve]`.
- `stats_first` and `multi_draft` apply only at an existing `draft` step. The serial tail grows from
  an empty seq, so there they appear only after a draft is inserted.
- The proposer never re-proposes an (object, edit) pair that was already evaluated.

## 6. Boosting (§6)

- **Serial attach splices** the fragment's children onto the root seq, so a stage does not add a
  nesting level. Frozen-prefix evaluation runs those children at the same structural paths as the
  full workflow. As a result, candidate evaluation is sample-identical to a full run and pays only
  for the new stage. **[measured]** (`tests/test_boost.py`)
- **Branch-mode cases.** A fragment with a single block becomes the case itself (`normalize_case`),
  so `branch{illegal: loop[...]}` stays within depth 3. A multi-node case is a seq, and a seq case
  cannot contain loops under depth 3. Raising `max_depth` to 4 for branch mode may be needed.
  **[inferred]**
- **Fitness** = (weighted) mean g gain minus `lam` × tokens per 1k. In branch mode this is the
  *delta* tokens relative to the current case. The first version's `lam` is unknown, so the
  default is `lam = 0.002`.
- **Heldout acceptance in branch mode.** Heldout items with other labels contribute a 0 difference,
  because they run an unchanged case.
- **Weights**: 1/(ε + max(g, 0)), with ε = 0.1, capped at 5 × mean, then normalised. ε is not
  given in the doc.
- **"No stages" ablation** (`boost/global_search.py`) uses the same proposer, policy and edit set
  over the whole workflow. Pass `--eval-budget` equal to the boosting run's `evals` to match budgets.
  Its default is rounds × max_steps × 3.

## 7. Scripts and gate (§7, §9.3)

- `scripts/run_boost.py` **refuses to run without a passing precheck summary**, which implements
  the S1 gate. `--skip-gate` exists only for fake-model dry runs.
- The final-test split is generated but never used by any script.
- The B3 baseline in the precheck uses k = calls − 1 (one draft plus k repairs), so its calls match.
- The §7.3 decision table is printed as *advisory* diagnostics, not as a decision.
- Concurrency: the `CachedLLM` semaphore (`--concurrency`) limits all model requests, including
  those from fanout branches and items running in parallel.
