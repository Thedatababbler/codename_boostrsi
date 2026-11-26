#!/bin/bash
# Round-2 reruns: GBDT and AdaBoost default presets at d2, seeds 0-2, all baselines; then the report.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONPATH=src
lane() {  # $1 = preset
  for s in 0 1 2; do
    id="r2_$1_s$s"
    $PY -m boosttopo.run --config configs/$1.yaml --set seed=$s --set run_id=$id --baselines > runs/$id.log 2>&1
    echo "$id exit=$?" >> runs/run_round2.log
  done
}
lane farm_gbdt & lane farm_adaboost & wait
$PY scripts/report.py --out REPORT_round2.md r2_farm_gbdt_s0 r2_farm_gbdt_s1 r2_farm_gbdt_s2 \
  r2_farm_adaboost_s0 r2_farm_adaboost_s1 r2_farm_adaboost_s2 > /dev/null 2>> runs/run_round2.log
echo ALL_DONE >> runs/run_round2.log
