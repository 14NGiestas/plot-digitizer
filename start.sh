#!/usr/bin/env bash
set -euo pipefail
(
WORKERS=${WORKERS:-6}
SAMPLES=${SAMPLES:-500}
EPOCHS=${EPOCHS:-}
BATCH=${BATCH:-}
OUTPUT=${OUTPUT:-curriculum-run}

export OMP_NUM_THREADS=$WORKERS
export MKL_NUM_THREADS=$WORKERS

echo "==> Syncing progress from existing checkpoints…"
digitizer train --output-dir ${OUTPUT} --sync

echo ""
echo "==> Curriculum plan:"
digitizer train --output-dir ${OUTPUT} --chain-info --resume

echo ""
echo "==> Starting curriculum pipeline…"
echo "    output=$OUTPUT  samples=$SAMPLES  workers=$WORKERS"

CMD="digitizer train \
  --output-dir ${OUTPUT} \
  --samples-per-stage ${SAMPLES} \
  --workers ${WORKERS} \
  --resume"

if [ -n "$EPOCHS" ]; then
  CMD="$CMD --epochs $EPOCHS"
fi
if [ -n "$BATCH" ]; then
  CMD="$CMD --batch $BATCH"
fi

nix develop .#rocm -c sh -c "$CMD"

echo ""
echo "Training complete. Best model: ${OUTPUT}/stage4/train/seg*/weights/best.pt"
echo "MLflow UI: mlflow ui --backend-store-uri file:${OUTPUT}/mlruns"
) || echo "Press [Enter] to continue..."
read
