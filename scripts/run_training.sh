#!/bin/bash
# Train SimpleEdgeNet on GPU after graphs are built.
# Run with: nohup bash scripts/run_training.sh &> logs/train.log &
# Monitor:  tail -f logs/train.log

set -e
cd /exp/mu2e/app/users/wzhou2/projects/calorimeter/GNN
source setup_env.sh

# Verify graphs exist
N_TRAIN=$(ls data/processed/*_00000035_*.pt data/processed/*_00000065_*.pt 2>/dev/null | wc -l)
N_VAL=$(ls data/processed/*_00000059_*.pt data/processed/*_00000044_*.pt 2>/dev/null | wc -l)
echo "Train graphs (sample): $N_TRAIN"
echo "Val graphs (sample):   $N_VAL"

if [ "$N_VAL" -eq 0 ]; then
    echo "ERROR: No val graphs found. Wait for build_all_graphs.sh to finish."
    exit 1
fi

python3 scripts/train_gnn.py \
    --config configs/default.yaml \
    --device cuda \
    --epochs 100 \
    --batch-size 64 \
    --run-name simple_edge_net_v1
