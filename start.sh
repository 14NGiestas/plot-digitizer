#!/usr/bin/env bash
set -euo pipefail
(
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

validate_stage() {
    local stage=$1
    local weights=$2
    local dataset_dir="synthetic-data/${stage}/images"
    echo "==> Validating Stage ${stage} (Digitizing 5 sample images with overlays)…"
    mkdir -p "validation-runs/${stage}"
    # Pick first 5 images for a quick sanity check visualization
    find "${dataset_dir}" -maxdepth 1 -name '*.png' | head -n 5 | xargs \
        nix develop .#rocm -c digitizer digitize \
        --output-dir "validation-runs/${stage}" \
        --weights "${weights}" \
        --overlay \
        --workers "${WORKERS}"
}

echo "==> Stage 1 – easy curves…"
nix develop .#rocm -c digitizer train \
  --dataset-dir synthetic-data/stage1 \
  --output-dir   training-runs/stage1 \
  --hyp-yaml     runs/curriculum_stage1.yml \
  --workers      2 \
  --execute
validate_stage "stage1" "training-runs/stage1/synthetic_plot_digitizer/weights/last.pt"

echo "==> Stage 2 – basic bars / annotations…"
nix develop .#rocm -c digitizer train \
  --dataset-dir synthetic-data/stage2 \
  --output-dir   training-runs/stage2 \
  --weights      training-runs/stage1/synthetic_plot_digitizer/weights/last.pt \
  --hyp-yaml     runs/curriculum_stage2.yml \
  --workers      2 \
  --execute
validate_stage "stage2" "training-runs/stage2/synthetic_plot_digitizer/weights/last.pt"

echo "==> Stage 3 – full annotation mix…"
nix develop .#rocm -c digitizer train \
  --dataset-dir synthetic-data/stage3 \
  --output-dir   training-runs/stage3 \
  --weights      training-runs/stage2/synthetic_plot_digitizer/weights/last.pt \
  --hyp-yaml     runs/curriculum_stage3.yml \
  --workers      2 \
  --execute
validate_stage "stage3" "training-runs/stage3/synthetic_plot_digitizer/weights/last.pt"

echo "==> Stage 4 – dense / degraded…"
nix develop .#rocm -c digitizer train \
  --dataset-dir synthetic-data/stage4 \
  --output-dir   training-runs/stage4 \
  --weights      training-runs/stage3/synthetic_plot_digitizer/weights/last.pt \
  --hyp-yaml     runs/curriculum_stage4.yml \
  --workers      2 \
  --execute
validate_stage "stage4" "training-runs/stage4/synthetic_plot_digitizer/weights/last.pt"

echo ""
echo "Training complete. Best model: training-runs/stage4/synthetic_plot_digitizer/weights/best.pt"
) && echo "Press [Enter] to continue..."
read
