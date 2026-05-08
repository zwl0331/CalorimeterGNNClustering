#!/bin/bash
# Build per-disk graphs from MixLow v2 ROOT files for all three splits.
# Mirrors scripts/build_all_graphs.sh but points at the MixLow config.
# Run on CPU node (or build node):
#     bash scripts/build_all_mixlow.sh

cd /exp/mu2e/app/users/wzhou2/projects/calorimeter/GNN
source setup_env.sh
set -e

CFG=configs/run1b_mixlow_default.yaml
N_EVENTS=500

echo "=== Clearing old MixLow processed data ==="
find data/processed_run1b_mixlow/ -name '*.pt' -delete 2>/dev/null || true
find data/processed_run1b_mixlow/ -name '*.csv' -delete 2>/dev/null || true

echo
echo "=== Train (134 files × $N_EVENTS events, computing norm stats) ==="
python3 scripts/build_graphs.py --split train --config "$CFG" --n-events "$N_EVENTS" --compute-norm

echo
echo "=== Val (29 files × $N_EVENTS events) ==="
python3 scripts/build_graphs.py --split val --config "$CFG" --n-events "$N_EVENTS"

echo
echo "=== Test (29 files × $N_EVENTS events) ==="
python3 scripts/build_graphs.py --split test --config "$CFG" --n-events "$N_EVENTS"

echo
echo "=== Pack into split bundles ==="
python3 scripts/pack_graphs.py --config "$CFG"

echo
echo "=== Done ==="
echo "Total .pt files:"
find data/processed_run1b_mixlow/ -name '*.pt' | wc -l
echo "Disk usage: $(du -sh data/processed_run1b_mixlow/ | cut -f1)"
echo "Bundles:"
ls -lh data/processed_run1b_mixlow/{train,val,test}.pt 2>/dev/null
