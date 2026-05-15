# plot-digitizer

Automatic AI-assisted plot digitizer (synthetic data generation, training, digitization, validation).

## Quick start

### Nix (recommended)

```bash
# CPU
nix develop

# AMD ROCm
nix develop .#rocm

# NVIDIA CUDA
nix develop .#cuda
```

Run tests:

```bash
nix develop --command sh -c "PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v"
```

### Local install (uv)

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Core commands

```bash
# Generate synthetic data
digitizer generate --output-dir synthetic-data --count 200

# Train (plan only)
digitizer train --dataset-dir synthetic-data --output-dir training-runs --epochs 50

# Train (execute)
digitizer train --dataset-dir synthetic-data --output-dir training-runs --epochs 50 --execute

# Digitize images
digitizer digitize synthetic-data/images --output-dir digitized --overlay

# Validate against truth
digitizer validate \
  --prediction-csv digitized/plot_0000.digitized.csv \
  --truth-csv synthetic-data/csv/plot_0000.csv
```

## Curriculum dataset generation

`generate` now supports:

- `--difficulty {0,1,2,3,4}`
- `--curriculum` (round-robin stages 1→2→3→4)

Examples:

```bash
# Fixed stage
digitizer generate --output-dir synthetic-stage1 --count 200 --difficulty 1

# Balanced curriculum mix
digitizer generate --output-dir synthetic-curriculum --count 800 --curriculum
```

## Curriculum training presets (`runs/*.yml`)

Added per-stage training presets:

- `runs/curriculum_stage1.yml`
- `runs/curriculum_stage2.yml`
- `runs/curriculum_stage3.yml`
- `runs/curriculum_stage4.yml`

Use them with `--hyp-yaml`:

```bash
# Stage 1
digitizer train \
  --dataset-dir synthetic-stage1 \
  --output-dir training-runs-stage1 \
  --weights yolov8n-seg.pt \
  --hyp-yaml runs/curriculum_stage1.yml \
  --execute

# Stage 2 (continue)
digitizer train \
  --dataset-dir synthetic-stage2 \
  --output-dir training-runs-stage2 \
  --weights training-runs-stage1/synthetic_plot_digitizer/weights/last.pt \
  --hyp-yaml runs/curriculum_stage2.yml \
  --execute
```

Repeat similarly for stages 3 and 4 with their matching YAML files.

## Useful options

- `digitizer digitize --weights model.pt` supports `.pt` or `.onnx`.
- `digitizer digitize --x-reference "...,...\" --y-reference \"...,...\"` for known axis points.
- `digitizer digitize --interactive-axis-selection` for GUI point picking.
- `digitizer digitize` now requires axis bounds from references/sidecar/interactive selection (no implicit 0:1 default).
- `digitizer digitize` is AI-only for curve extraction and fails when no curve-class masks are detected.
- `digitizer train --workers N --execute` sets both dataloader workers and torch CPU thread pools.

## Commands summary

- `generate`: synthetic plots + labels + metadata + CSV
- `train`: plan or execute YOLO segmentation training
- `digitize`: segment and convert plots to numeric CSV
- `validate`: compare predicted CSV to ground truth
