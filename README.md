# Boosted Workflow — single-agent version

This implements 《Boosted Workflow：单智能体版本实现文档》. The object being optimised is a workflow
of one agent calling skills, tools and checks over a shared state. The workflow is grown stage by
stage, with serial (GBDT-style) or branch (AdaBoost-style) boosting. Pure Python 3.11 standard
library; no dependencies.

**Read `docs/IMPLEMENTATION_NOTES.md` first.** The original repo (`~/projects/boostrsi_next`) was
not available, so the benchmark here is a *toy* Farm game and the old prompts are stand-ins.

## Layout

| Path | Doc § | What |
|---|---|---|
| `workflow/model.py` | §2 | `State`, `Step`, `Block`, legality checks, `worst_case_calls`, text form `parse`/`to_str` |
| `workflow/exec.py` | §4 | recursive interpreter, cost accounting, path-derived seeds |
| `workflow/tools.py` | §3.2 | `validate`, `simulate`, `crop_stats`, `reset_to_best`; checks `status`, `error_type` |
| `workflow/skills.py` | §3.1 | prompt templates `draft`, `repair`, `improve`, `analyze` with fixed tool slots |
| `workflow/llm.py` | §4.3 | JSONL cache + concurrency limit; OpenAI-compatible client (vLLM / Qwen3-14B) |
| `workflow/fake.py` | — | deterministic fake model for tests and dry runs |
| `workflow/evaluate.py` | §4.2 | evaluation entry, g, frozen-prefix reuse |
| `workflow/edits.py` | §5.1–5.2 | atomic edits, fragments, `legal_edits`, `apply_edit` |
| `workflow/proposer.py`, `policy.py` | §5.3 | 3-slot `WorkflowProposer`, residual-source policy f0 |
| `workflow/grow.py` | §6.1 | `grow_fragment` |
| `boost/serial.py`, `boost/branch.py` | §6.2–6.4 | the two boosting modes and the acceptance test |
| `boost/global_search.py` | §8.3 | "no stages" ablation |
| `workflow/baselines.py` | §7.1–7.2, §8.1 | F1–F3, B1–B3, token-aligned baselines (all are workflows) |
| `scripts/precheck.py` | §7 | S1 pre-check and gate |
| `scripts/run_boost.py` | S2–S4 | search, then a test report against the cost-aligned baselines |
| `scripts/report.py` | §8.2 | metrics, breakdown, structure and usage from a saved JSON |
| `farm/` | — | toy benchmark: rules, scorer, reference solver (tools may not import it) |

## Run

```bash
# tests (W1–W7 acceptance checks that can be run without the original repo)
PYTHONPATH=.:tests python3.11 -m unittest discover -s tests

# S1 pre-check: estimate first, then the fake dry run, then the real model
python3.11 scripts/precheck.py --estimate --calls-per-hour 64
python3.11 scripts/precheck.py --llm fake --out runs/precheck_fake
python3.11 scripts/precheck.py --llm openai --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen3-14B --concurrency 2 --cache runs/cache/qwen3.jsonl --out runs/precheck
python3.11 scripts/report.py runs/precheck/precheck.json

# S2 / S3 / S4 (refuse to start unless the pre-check passed)
python3.11 scripts/run_boost.py --mode serial --precheck runs/precheck/summary.json [model args]
python3.11 scripts/run_boost.py --mode branch --precheck runs/precheck/summary.json [model args]
python3.11 scripts/run_boost.py --mode branch --no-residual ...   # ablation: no residual
python3.11 scripts/run_boost.py --mode global --eval-budget N ... # ablation: no stages
python3.11 scripts/run_boost.py --mode serial --no-tools ...      # ablation: no tools
```

Workflows are written in the doc's notation, for example:

```
seq[crop_stats, draft, branch(status){illegal: loop(k=3, stop=legal)[validate, repair], legal_low: seq[simulate, improve]}]
```
