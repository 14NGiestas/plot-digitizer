#!/usr/bin/env bash
set -euo pipefail

WORKERS=${WORKERS:-$(nproc)}
SAMPLES=${SAMPLES:-500}

# ── data generation ──────────────────────────────────────────────────────────
export OMP_NUM_THREADS=$WORKERS
export MKL_NUM_THREADS=$WORKERS

echo "==> Generating per-stage datasets ($SAMPLES samples each)…"
nix develop -c digitizer generate --output-dir synthetic-data/stage1 --count $SAMPLES --difficulty 1 --workers $WORKERS
nix develop -c digitizer generate --output-dir synthetic-data/stage2 --count $SAMPLES --difficulty 2 --workers $WORKERS
nix develop -c digitizer generate --output-dir synthetic-data/stage3 --count $SAMPLES --difficulty 3 --workers $WORKERS
nix develop -c digitizer generate --output-dir synthetic-data/stage4 --count $SAMPLES --difficulty 4 --workers $WORKERS

# ── training ─────────────────────────────────────────────────────────────────
# One DataLoader worker per GPU process avoids OMP thread contention.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "==> Stage 1 – easy curves…"
nix develop .#rocm -c digitizer train \
  --dataset-dir synthetic-data/stage1 \
  --output-dir   training-runs/stage1 \
  --hyp-yaml     runs/curriculum_stage1.yml \
  --workers      2 \
  --execute

echo "==> Stage 2 – basic bars / annotations…"
nix develop .#rocm -c digitizer train \
  --dataset-dir synthetic-data/stage2 \
  --output-dir   training-runs/stage2 \
  --weights      training-runs/stage1/synthetic_plot_digitizer/weights/last.pt \
  --hyp-yaml     runs/curriculum_stage2.yml \
  --workers      2 \
  --execute

echo "==> Stage 3 – full annotation mix…"
nix develop .#rocm -c digitizer train \
  --dataset-dir synthetic-data/stage3 \
  --output-dir   training-runs/stage3 \
  --weights      training-runs/stage2/synthetic_plot_digitizer/weights/last.pt \
  --hyp-yaml     runs/curriculum_stage3.yml \
  --workers      2 \
  --execute

echo "==> Stage 4 – dense / degraded…"
nix develop .#rocm -c digitizer train \
  --dataset-dir synthetic-data/stage4 \
  --output-dir   training-runs/stage4 \
  --weights      training-runs/stage3/synthetic_plot_digitizer/weights/last.pt \
  --hyp-yaml     runs/curriculum_stage4.yml \
  --workers      2 \
  --execute

echo ""
echo "Training complete. Best model: training-runs/stage4/synthetic_plot_digitizer/weights/best.pt"
