#!/bin/bash
# Round-3 follow-ups, strictly sequential (one LLM job at a time), in the user's priority order.
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
L=runs/run_round3.log
G="r2_farm_gbdt_s0 r2_farm_gbdt_s1 r2_farm_gbdt_s2"
A="r2_farm_adaboost_s0 r2_farm_adaboost_s1 r2_farm_adaboost_s2"
step() { echo "$(date +%T) START $*" >> $L; "$@" >> runs/round3_steps.log 2>&1; echo "$(date +%T) END exit=$? $*" >> $L; }
run() {  # $1 run id, $2 preset, rest: args
  id=$1; preset=$2; shift 2; seed=${id##*_s}
  step $PY -m boosttopo.run --config configs/$preset.yaml --set seed=$seed --set run_id=$id "$@"
}
report() { $PY scripts/report.py --out REPORT_round3.md $G $A > /dev/null 2>> $L; }
# 1. writer role once, heldout + dev_d2, all seeds
for id in $G; do run $id farm_gbdt --methods writer_once --splits heldout,dev_d2; done
for id in $A; do run $id farm_adaboost --methods writer_once --splits heldout,dev_d2; done
report; echo "$(date +%T) DONE_1" >> $L
# 2. best-of-N writer + repair loop at the GBDT budget, all GBDT seeds
for id in $G; do run $id farm_gbdt --methods best_of_n_writer,repair_loop --splits heldout,dev_d2; done
report; echo "$(date +%T) DONE_2" >> $L
# 3. calibration (reduced)
step $PY scripts/calibrate_difficulty.py --prompt writer --n 20 --difficulties 1,2,3,4,5
step $PY scripts/calibrate_difficulty.py --config configs/calibrate_thinking.yaml --prompt benchmark --n 10 --difficulties 1,2,3
step $PY scripts/calibrate_difficulty.py --config configs/calibrate_thinking.yaml --prompt writer --n 10 --difficulties 1,2,3
echo "$(date +%T) DONE_3" >> $L
# 4. best-of-N writer + repair loop at the AdaBoost budget, seed 0 only
run r2_farm_adaboost_s0 farm_adaboost --methods best_of_n_writer,repair_loop --splits heldout,dev_d2
report; echo "$(date +%T) DONE_4" >> $L
# 5. E analysis (cache only)
$PY scripts/members.py r2_farm_gbdt_s0 r2_farm_gbdt_s1 r2_farm_gbdt_s2 $A > runs/MEMBERS_round3.md 2>> $L
echo "$(date +%T) DONE_5" >> $L
echo ALL_DONE >> $L
