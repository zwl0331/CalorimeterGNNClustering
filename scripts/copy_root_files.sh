#!/bin/bash
# Copy all MDC2025-002 ROOT files from tape to local data directory.
# Run on a CPU node with tape access:
#   bash /exp/mu2e/app/users/wzhou2/projects/calorimeter/GNN/scripts/copy_root_files.sh

DEST=/exp/mu2e/data/users/wzhou2/GNN/root_files
mkdir -p "$DEST"

# All 50 files: 35 train + 7 val + 8 test
FILES=(
  # Train (35)
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/18/72/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000035.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/1c/24/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000065.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/1d/bb/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000062.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/25/fc/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000165.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/2b/2f/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000020.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/2e/02/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000171.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/3d/89/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000022.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/3e/ae/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000060.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/44/6c/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000046.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/48/70/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000004.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/69/4e/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000029.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/69/b8/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000201.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/78/50/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000006.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/7a/c7/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000009.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/7a/df/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000013.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/84/bb/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000079.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/86/e2/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000042.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/88/f2/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000033.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/8e/3b/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000184.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/90/08/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000007.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/93/a5/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000000.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/99/5d/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000038.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/9f/08/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000111.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/ad/de/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000072.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/b4/d6/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000066.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/b6/a0/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000089.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/bb/08/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000002.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/ca/7a/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000181.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/d3/67/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000053.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/da/ef/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000014.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/db/31/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000028.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/dd/b7/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000024.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/e1/97/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000008.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/e3/62/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000015.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/f7/e7/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000130.root
  # Val (7)
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/04/13/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000059.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/20/aa/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000044.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/34/dd/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000040.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/36/72/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000056.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/58/14/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000138.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/73/8f/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000032.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/ab/80/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000051.root
  # Test (8)
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/04/2a/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000003.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/17/7a/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000001.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/29/7e/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000198.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/37/25/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000121.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/3c/20/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000012.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/9b/db/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000019.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/ab/a0/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000027.root
  /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/f1/fd/nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_00000037.root
)

TOTAL=${#FILES[@]}
COUNT=0
FAILED=0

for f in "${FILES[@]}"; do
  COUNT=$((COUNT + 1))
  NAME=$(basename "$f")
  if [ -f "$DEST/$NAME" ]; then
    echo "[$COUNT/$TOTAL] SKIP (exists) $NAME"
    continue
  fi
  echo "[$COUNT/$TOTAL] Copying $NAME..."
  if cp "$f" "$DEST/"; then
    echo "  OK"
  else
    echo "  FAILED"
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "Done. $((TOTAL - FAILED))/$TOTAL succeeded. Destination: $DEST"
ls -lh "$DEST/" | tail -5
echo "Total size: $(du -sh "$DEST" | cut -f1)"
