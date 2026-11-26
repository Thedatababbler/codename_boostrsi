#!/bin/sh
cd "$(dirname "$0")/.."
export PYTHONPATH=src
echo "[$(date)] 8a start"; .venv/bin/python -u scripts/r5_ablations.py --step 8a; echo "8a exit=$?"
echo "[$(date)] 8b start"; .venv/bin/python -u scripts/r5_ablations.py --step 8b; echo "8b exit=$?"
.venv/bin/python -u scripts/r5_ablations.py --step report; echo "report exit=$?"
echo "[$(date)] STEP8 END"
