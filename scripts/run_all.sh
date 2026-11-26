#!/bin/bash
# Runs the presets two at a time against the shared vLLM server and cache; logs in runs/<id>.log.
# Usage: scripts/run_all.sh [extra --set overrides...]
cd "$(dirname "$0")/.."
run() { .venv/bin/python -m boosttopo.run --config "configs/$1.yaml" --baselines "${@:2}" > "runs/$1.log" 2>&1; }
export PYTHONPATH=src
(run farm_gbdt "$@"; run farm_gbdt_forest "$@"; run farm_gbdt_randpolicy "$@") &
(run farm_adaboost "$@"; run farm_adaboost_forest "$@"; run farm_adaboost_uniform "$@") &
wait
.venv/bin/python scripts/report.py farm_gbdt farm_adaboost farm_gbdt_forest farm_adaboost_forest farm_gbdt_randpolicy farm_adaboost_uniform > /dev/null
echo ALL_DONE
