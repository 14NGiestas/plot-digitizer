#!/usr/bin/env bash
set -euo pipefail
(
WORKERS=${WORKERS:-6}
SAMPLES=${SAMPLES:-500}
EPOCHS=${EPOCHS:-}
BATCH=${BATCH:-}
OUTPUT=${OUTPUT:-runs}

export OMP_NUM_THREADS=$WORKERS
export MKL_NUM_THREADS=$WORKERS

echo "==> Syncing progress from existing checkpoints…"
nix develop .#rocm -c digitizer train --sync --output-dir "$OUTPUT"

echo ""
echo "==> Curriculum plan:"
nix develop .#rocm -c digitizer train --chain-info --resume --output-dir "$OUTPUT"

echo ""
echo "==> Starting curriculum pipeline…"
echo "    samples=$SAMPLES  workers=$WORKERS"

CMD=(
  digitizer train
  --output-dir "$OUTPUT"
  --samples-per-stage "$SAMPLES"
  --workers "$WORKERS"
  --resume
)

if [ -n "$EPOCHS" ]; then
  CMD+=(--epochs "$EPOCHS")
fi
if [ -n "$BATCH" ]; then
  CMD+=(--batch "$BATCH")
fi

nix develop .#rocm -c "${CMD[@]}"

echo ""
echo "Training complete. Best model: ${OUTPUT}/stage4/train/seg*/weights/best.pt"
echo "MLflow UI: mlflow ui --backend-store-uri file:${OUTPUT}/mlruns"
) || echo "Training failed."
echo "Press [Enter] to continue..."
read
