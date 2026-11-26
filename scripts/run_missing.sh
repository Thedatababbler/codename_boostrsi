#!/bin/bash
# The two items missing from round 3, strictly sequential.
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
L=runs/run_missing.log
until curl -s -m 5 http://127.0.0.1:8401/v1/models | grep -q qwen3-14b; do sleep 15; done
echo "$(date +%T) server up" >> $L
step() { echo "$(date +%T) START $*" >> $L; "$@" >> runs/missing_steps.log 2>&1; echo "$(date +%T) END exit=$? $*" >> $L; }
R="--config configs/farm_adaboost.yaml --set seed=0 --set run_id=r2_farm_adaboost_s0"
step $PY -m boosttopo.run $R --methods best_of_n_writer,repair_loop --splits heldout,dev_d2
step $PY -m boosttopo.run $R --methods ours,single_writer,hand_topology,best_of_n_matched --splits heldout,dev_d2,dev_d3,dev_d4
$PY scripts/report.py --out REPORT_round3.md r2_farm_gbdt_s0 r2_farm_gbdt_s1 r2_farm_gbdt_s2 r2_farm_adaboost_s0 r2_farm_adaboost_s1 r2_farm_adaboost_s2 > /dev/null 2>> $L
echo "$(date +%T) ALL_DONE" >> $L
