# Interfaces (reference)

Written from the code as of round 3 (2026-11-19); round-4 additions marked (r4). These are the public signatures other modules, scripts and
tests rely on. Change one only on purpose, and update this file and `IMPLEMENTATION_NOTES.md` in the same change.
Everything lives under `src/boosttopo/`.

## Data types

```python
# llm.py
@dataclass
class Request:
    prompt: str; role: str = ""; item: Any = None; y_cur: str = ""; upstream: list = []
    diag: str | None = None; rep: int = 0; system: str | None = None
    # rep = sample index; it is part of the LLM cache key and is sent to the server as `seed`.

# bench/base.py
@dataclass(frozen=True)
class Item:        item_id: str; game: str; difficulty: int; seed: int; prompt: str; payload: dict
@dataclass
class ScoreResult: score: float; valid: bool; diag: str; category: str = ""
    # score: normalised profit (profit / reference profit, floored at 0); valid: legal plan;
    # diag: the simulator verdict shown to the repairer; category: invalid_* / valid_{idle,waste,ok}

# graph/__init__.py
IN = "IN"                                    # the input node (the stage's input solution)
@dataclass(frozen=True) class Node:     nid: str; role: str; k: int = 1; stop: str = ""
    # role in roles.ROLES; (r4) k > 1 = loop "repeat up to k times", stop in LOOP_STOPS = ("until_legal", "keep_best")
@dataclass(frozen=True) class Limits:   max_nodes: int = 3; max_indegree: int = 2; loop_ks: tuple = (); loop_stops: tuple = ()
@dataclass(frozen=True) class Subgraph: nodes: tuple[Node, ...]; edges: tuple[tuple[str, str], ...]; out: str
    .role(nid) -> str; .ids() -> list[str]; .preds(nid) / .succs(nid) -> list[str]
    .key() -> str                 # canonical, permutation-invariant hash
    .canonical() -> Subgraph; .to_dict() -> dict; Subgraph.from_dict(d); Subgraph.from_describe(s)
    .describe() -> str            # e.g. "[IN->n1:writer, n1:writer->n0:repairer] out=n0:repairer"
    .role_counts() -> dict[str, int]; .depth() -> int; (r4) .node(nid) -> Node; .loops() -> [(role, k, stop)]
@dataclass(frozen=True) class Edit:      kind: str; args: tuple      # Edit.make(kind, **args); .a -> dict
@dataclass(frozen=True) class MacroEdit: steps: tuple[Edit, ...]; name: str = ""
AnyEdit = Edit | MacroEdit
class IllegalEdit(Exception)
@dataclass
class StageOutput: solution: str; score: float; raw_score: float; cost: float; diag: str = ""; category: str = ""; trace: list = []
                   scorer_calls: int = -1   # (r4) verifier queries of the stage; -1 in records cached before r4
    # solution/score: gated (input kept unless the output is strictly better); raw_score: ungated output; cost: tokens

# evalx/__init__.py
@dataclass
class ItemResult: item_id; score; raw_score; g; cost; diag; category; solution; sol_score; sol_g; scores
    # means over repeats; solution / sol_* = repeat 0; scores = per-repeat gated scores

# boost/common.py
@dataclass class Ctx:       cfg; ev; proposer; policy; sampler; log; library: Library; stats: EditStats
@dataclass class FitResult: mode: str; ensemble: list[Subgraph]; history: list; extra: dict = {}

# policy/__init__.py
@dataclass class Obs: subgraph: dict; residual: dict; edit_stats: dict; round: int; step: int; budget_left: float = inf; leaf: str = ""
```

## Protocols (what a plug-in must implement)

```python
class LLM:           def call(self, req: Request) -> tuple[str, int]          # (text, total tokens)
class BenchAdapter:  def make_items(self, game: str, difficulty: int, seeds: list[int]) -> list[Item]
                     def score(self, item: Item, solution: str) -> ScoreResult
                     def features(self, item: Item) -> dict
class EditProposer:  def start_search(self, seed: int) -> None
                     def propose(self, h: Subgraph, ctx: GrowContext) -> list[AnyEdit]
                     # register with @register_proposer("name"); selected by cfg.search.proposer
class Policy:        def rank(self, h: Subgraph, cands: list[AnyEdit], obs: Obs) -> list[AnyEdit]
Sampler:             .split(items, weights=None, seed=0) -> (bag, oob)        # AllSampler, Subsample, WeightedBootstrap
```

## Functions by module

**config**: `load_config(path=None, overrides=None) -> Cfg` (YAML deep-merged over `DEFAULTS`; split maps in
`REPLACE_KEYS` = {`dev_tests`, `final_tests`} replace instead of merge; the old key `bench.tests` raises);
`deep_merge(base, over)`, `parse_override("a.b=v") -> dict`, `cfg_hash(dict) -> str`, `sub_seed(root, *parts) -> int`.
`Cfg` = dict with attribute access, `.to_dict()`.

**llm**: `OpenAICompatLLM(api_base, model, api_key=None, temperature=0.7, max_tokens=4096, timeout=600,
max_concurrency=32, extra_body=None, retries=6)`; `FakeLLM(rule, name="fake", temperature=0.0)`;
`CachedLLM(inner, cache_dir, budget=None)` with `.key(req)`, `.call(req)`, (r4) `.call_meta(req) -> (text, tokens,
meta)`, `.budget: Budget`; (r4) `OpenAICompatLLM.call_meta`, `strip_thinking(text) -> (answer, reasoning_chars)`;
`Budget(logical_calls, logical_tokens, new_calls, new_tokens, score_calls).snapshot()`;
`build_llm(cfg, cache_dir, fake_rule=None) -> CachedLLM`; `sha(s)`, `atomic_write_text(path, text)`,
`atomic_write_json(path, obj)`.

**bench**: `make_bench(cfg) -> BenchAdapter`; `farm.FarmAdapter(repo)` (+ `.simulate(item, solution) -> dict`),
`farm.extract_json(text)`, `farm.categorize_error(msg)`; `toy.ToyBench()`, `toy.toy_y0(item)`, `toy.toy_rule(req)`.

**roles**: `WRITERS = ("writer", "repairer")`, `READERS = ("critic", "analyst")`, `ROLES`;
`role_prompt(role, item, y_cur, upstream: list[(role, text)], diag, rep=0) -> Request`;
`y0_request(item, rep=0) -> Request` (the benchmark prompt unchanged); `prompt_version() -> str`; `is_writer(role)`.

**graph**: `single_writer()`, `topo_order(h)`, `is_valid(h, limits)`, `apply_edit(h, e, limits=None)`
(raises `IllegalEdit`), `try_apply(h, e, limits) -> Subgraph | None`, `legal_edits(h, limits)`,
`expand_macro(name, h, limits)`, `edit_roles(h, e)`, `edit_type(e)`, `edit_summary(h, e)`,
`run_subgraph(h, item, y_in, llm, bench, rep=0, s_in=None, g_of=None) -> StageOutput`;
(r4) `static_scorer_calls(h)`, `LOOP_EDIT = "set_loop"` (edit args nid, k, stop), `LOOP_STOPS`, `LOOP_REP_STRIDE = 7919`.

**evalx**: `gap_closure(s, s0) -> float`;
`Evaluator(bench, llm, cache_dir, n_repeats=2, lam=0.0, max_workers=32)` with
`.make_start(items) -> {id: y0}`, `.g(item_id, score)`, `.score_of(item, solution)`,
`.run(h, items, inputs, n_repeats=None, occ=0) -> {id: ItemResult}`, `.run_fresh(..., occ=0)` (same, on unused
repeat indices), `.mean_cost(res)`, `.cost_penalty(res)`, `._one(h, item, y_in, rep, occ=0) -> StageOutput` (cached
stage), (r4) `.occurrence(h, earlier_members) -> int` (0 unless `dedup_stages`), and the attributes `y0`, `s0`,
`rep_base`, `rep_seed`, `lam`, `pool`, `FRESH_OFFSET`, (r4) `dedup_stages`, `DUP_STRIDE = 20011`.
Effective repeat of a stage sample = `rep + rep_base + rep_seed + occ * DUP_STRIDE`.

**boost**: `fit_gbdt(train, heldout, ctx, resume=True) -> FitResult`; `fit_adaboost(train, heldout, ctx,
resume=True) -> FitResult`; `offline_selection(train, heldout, ensemble, ctx) -> dict`;
`fit_ada(res, items, g_ens, r)`, `ensemble_g(ev, ensemble, items, y0)`;
`residual.residual_weights(g_ens, cfg)`, `high_weight_ids(weights, mass=0.5)`, `residual_report(g, weights, cats)`,
`dominant_failure(report)`, `bootstrap_lower(values, level=0.9, n_boot=2000, seed=0)`, `paired(new, old)`, `corr(a, b)`;
`sampler.make_sampler(cfg)`; `library.Library(log=None)` (`.add(h, mode, stage=None, per_item=None, fit=None,
source="")`, `.load_jsonl(path, subgraph_of)`, `.entries(mode=None)`), `weighted_gain(e, g_ens, r)`,
`pick_seed(library, r, g_ens, cfg, exclude=())`, `select_ensemble(library, train_ids, cfg)`;
`common.lower_ci(values, cfg, seed=0)`, `pick_candidate(cands, ens, tol) -> (fit, Subgraph)`.

**search**: `grow_subgraph(seed, items, inputs, fitness, proposer, policy, cfg, *, limits, residual, stats, rnd,
search_seed, log=None, tag="", cost_of=None) -> GrowResult(h, fit, kept, n_evals)`, where
`fitness(h, items, inputs) -> (fit, cost_penalty)`; `make_obs(h, fit_h, cost_pen, residual, stats, rnd, step)`;
`reserve_slots(ranked, cfg)` (r4), `proposers.make_proposer(cfg)`, `AtomicProposer(macros=())`, `RandomSubspaceProposer(role_frac=0.5, edit_type_frac=0.67,
resample="per_search", macro_rate=0.5, macros=(...))`.

**policy**: `make_policy(cfg)`, `F0Policy(explore=0.02, repair_share=0.5)`, `RandomPolicy(root_seed=0)`,
`EditStats()` (`.add(etype, fail, gain)`, `.view()`).

**infer** (inference and baselines; each returns a summary dict with `n, mean_g, g_ci95, mean_score,
tokens_per_item, scorer_calls_per_item, valid_rate, legal_mean_score, rows` (+ method fields); `rows` = one dict
per (item, repeat) with `item_id, rep, g, score, tokens, scorer_calls, valid, category` (+ (r4) `stage_runs,
dup_stage_runs` for ensembles)):
`infer(mode, ensemble, items, ev, n_repeats)`; `best_of_n(items, ev, n, seed=0)` (benchmark prompt);
`best_of_n_subgraph(h, items, ev, n)`; `repair_loop(items, ev, budget, n_repeats, max_repairs=40)`;
`writer_role()`, `hand_topology()`, `stage_scorer_calls(h)`, `y0_tokens(ev, item)`; `SEED_STRIDE = 1_000_000`.

**runlog**: `RunLog(run_dir)` with `.append(name, rec)` (jsonl), `.write(name, obj)`, `.read(name)`.

**run** (driver): `python -m boosttopo.run --config C [--set k=v ...] [--no-test] [--refit] [--baselines] [--final-test]`;
`build(cfg, run_dir) -> (bench, llm, ev, log)`, `make_splits(cfg, bench, final_test=False) -> {name: [Item]}`,
`eval_splits(cfg, final_test) -> [names]`, `calibrate_lam(cfg, ev, train)`, `fit(cfg, splits, ev, log)`.

## Run directory files (`runs/<id>/`)

`config.json`, `manifest.json` (`manifest_final.json`), `checkpoint.json`, `ensemble.json`, `rounds.jsonl`,
`edit_outcomes.jsonl`, `library.jsonl`, `budget.jsonl`, `final.json` (`final_final_test.json`),
`per_item.jsonl` (`per_item_final_test.jsonl`): one line per split, method, item, repeat.

## Scripts

`scripts/report.py [--out F] RUN...`, `scripts/members.py [--runs-dir D] RUN...`,
`scripts/calibrate_difficulty.py [--config C] [--prompt benchmark|writer] [--n 20]`,
`scripts/run_round2.sh`, `scripts/probe_topologies.py`, `scripts/m0_check.py`; (r4) `scripts/calibrate_v2.py
--config C --difficulties 2,3,4,5 --n 40`, `scripts/attempts.py RUN...`, `scripts/dup_stages.py RUN...`.
