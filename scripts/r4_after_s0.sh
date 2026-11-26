#!/bin/sh
# Runs after Stage B seed 0: matched best-of-N writer (LLM, sequential), then the cache-only reports.
cd "$(dirname "$0")/.."
until grep -q "END exit" runs/r4_stageB_s0.log; do sleep 120; done
export PYTHONPATH=src
echo "[$(date)] best_of_n_writer start"
.venv/bin/python -u -m boosttopo.run --config configs/r4_d5.yaml --set run_id=r4_gbdt_lm_s0 --methods best_of_n_writer --splits test_d5; echo "bon exit=$?"
echo "[$(date)] stage A breakdown"
.venv/bin/python -u scripts/stage_a_breakdown.py --config configs/r4_d5.yaml; echo "breakdown exit=$?"
.venv/bin/python -u scripts/stage_b_report.py r4_gbdt_lm_s0; echo "report exit=$?"
echo "[$(date)] QUEUE END"
