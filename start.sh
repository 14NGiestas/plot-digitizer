#!/usr/bin/env bash
set -euo pipefail
(
WORKERS=${WORKERS:-6}
SAMPLES=${SAMPLES:-500}
EPOCHS=${EPOCHS:-}
BATCH=${BATCH:-}
OUTPUT=${OUTPUT:-curriculum-run}

# ─ thread tuning for generation ─────────────────────────────────────────────
export OMP_NUM_THREADS=$WORKERS
export MKL_NUM_THREADS=$WORKERS

echo "==> Syncing progress from existing checkpoints…"
digitizer curriculum --output-dir ${OUTPUT} --sync

echo ""
echo "==> Curriculum plan:"
digitizer curriculum --output-dir ${OUTPUT} --chain-info --resume

echo ""
echo "==> Starting curriculum pipeline…"
echo "    output=$OUTPUT  samples=$SAMPLES  workers=$WORKERS"

CMD="digitizer curriculum \
  --output-dir ${OUTPUT} \
  --samples-per-stage ${SAMPLES} \
  --workers ${WORKERS} \
  --resume \
  --execute"

if [ -n "$EPOCHS" ]; then
  CMD="$CMD --epochs $EPOCHS"
fi
if [ -n "$BATCH" ]; then
  CMD="$CMD --batch $BATCH"
fi

nix develop .#rocm -c sh -c "$CMD"

echo ""
echo "Training complete. Best model: ${OUTPUT}/stage4/train/synthetic_plot_digitizer/weights/best.pt"
echo "Fine-tuned model: ${OUTPUT}/interpret-finetune/best.pt"
) || echo "Press [Enter] to continue..."
read
