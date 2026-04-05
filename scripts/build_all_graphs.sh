#!/bin/bash
# Build graphs for all splits from local ROOT files.
# Run on CPU node:  bash scripts/build_all_graphs.sh
#
# Uses 500 events/file → ~50K graphs total, takes ~10 min.
# To use all events, remove the --n-events flag (will take ~4 hours).

cd /exp/mu2e/app/users/wzhou2/projects/calorimeter/GNN
source setup_env.sh
set -e

ROOT_DIR=/exp/mu2e/data/users/wzhou2/GNN/root_files_v2
N_EVENTS=500

echo "=== Clearing old processed data ==="
find data/processed/ -name '*.pt' -delete 2>/dev/null || true
find data/processed/ -name '*.csv' -delete 2>/dev/null || true

echo ""
echo "=== Building train graphs (35 files, $N_EVENTS events each) ==="
python3 scripts/build_graphs.py --split train --root-dir "$ROOT_DIR" --n-events "$N_EVENTS" --compute-norm

echo ""
echo "=== Building val graphs (7 files, $N_EVENTS events each) ==="
python3 scripts/build_graphs.py --split val --root-dir "$ROOT_DIR" --n-events "$N_EVENTS"

echo ""
echo "=== Building test graphs (8 files, $N_EVENTS events each) ==="
python3 scripts/build_graphs.py --split test --root-dir "$ROOT_DIR" --n-events "$N_EVENTS"

echo ""
echo "=== Done ==="
echo "Total .pt files: $(find data/processed/ -name '*.pt' | wc -l)"
echo "Disk usage: $(du -sh data/processed/ | cut -f1)"
