# Boosted Topology (multi-agent) and Boosted Workflow (single-agent)

Two implementations share this repo:

- `src/boosttopo/`: the multi-agent version (role subgraphs, GBDT / AdaBoost). Its notes are in
  `docs/IMPLEMENTATION_NOTES.md`, its interfaces in `docs/interfaces.md`, its presets in `configs/*.yaml`.
- `workflow/` + `boost/`: the single-agent version of 《Boosted Workflow：单智能体版本实现文档》, integrated with
  boosttopo (branch `workflow-integration`, revision §12). **Read `docs/IMPLEMENTATION_NOTES_workflow.md` first.**

The single-agent version reuses boosttopo's benchmark adapter (`FarmAdapter`), LLM client and cache
(`CachedLLM`, keys unchanged), config (YAML + `--set`), `RunLog` and `per_item.jsonl`. The role prompts are the
multi-agent ones, unchanged. `farm/` is a toy Farm game for tests and fake dry runs only.

## Layout (single-agent)

| Path | Doc § | What |
|---|---|---|
| `workflow/model.py` | §2 | `State`, `Step`, `Block` (seq / loop / branch / fanout / gate), legality checks, text form |
| `workflow/exec.py` | §4, §12 | interpreter: start from y0, loop / gate semantics, seed schemes `path` / `legacy` |
| `workflow/tools.py` | §3, §12 | `validate` (diag modes clean / original), `simulate`, `crop_stats`; checks `status`, `error_type` |
| `workflow/skills.py` | §3.1, §12 | requests: writer / repairer / analyst / critic role prompts unchanged; `improve` new |
| `workflow/llm.py` | §4.3 | builds boosttopo's `CachedLLM`; `CacheOnly` for no-model-call replay |
| `workflow/evaluate.py` | §4.2 | y0 via `call_y0` (missing test y0 aborts), g = `evalx.gap_closure`, frozen prefixes |
| `workflow/legacy.py` | §12 | multi-agent methods / ensembles written as workflows (for cache replay) |
| `workflow/edits.py`, `proposer.py`, `policy.py`, `grow.py` | §5, §6.1 | edit set, 3-slot proposer, f0 |
| `boost/serial.py`, `boost/branch.py`, `boost/global_search.py` | §6, §8.3 | boosting modes, checkpoint / resume |
| `workflow/baselines.py` | §7, §8.1 | F1–F3, B1–B3, token-aligned baselines |
| `scripts/precheck.py` | §7 | S1 pre-check and gate; `--estimate` = cache-only call estimate |
| `scripts/run_boost.py` | S2–S4 | search, then the test report against the cost-aligned baselines |
| `scripts/replay_check.py` | §12 | cache-only acceptance checks (a)–(c) against a stored multi-agent run |
| `scripts/report_workflow.py` | §8.2 | tables from a saved precheck.json / test.json |
| `farm/` | — | toy Farm game with FarmAdapter's interface, fake model (tests only) |

## Run

```bash
# one test runner for both packages (needs pyyaml, pytest)
python -m pytest -q

# acceptance before any model call (needs the stored cache/ and runs/, and third_party/SelfImprove-Bench)
python scripts/replay_check.py --run-dir runs/r4_d5 --split test_d5
python scripts/precheck.py --config configs/workflow_d5.yaml --estimate

# fake dry runs
python scripts/precheck.py --config configs/workflow_toy.yaml
python scripts/run_boost.py --config configs/workflow_toy.yaml --mode serial --skip-gate

# the real pre-check (one model job at a time), then S2-S4 (refuse to start unless it passed)
python scripts/precheck.py --config configs/workflow_d5.yaml
python scripts/run_boost.py --config configs/workflow_d5.yaml --mode branch --precheck runs/precheck_workflow_d5/summary.json
```

Workflows are written in the doc's notation, for example:

```
seq[crop_stats, draft, branch(status){illegal: loop(k=3, stop=legal)[validate, repair], legal_low: seq[simulate, improve]}]
```
