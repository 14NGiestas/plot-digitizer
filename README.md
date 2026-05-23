# plot-digitizer

Automatic AI-assisted plot digitizer with synthetic-data generation, curriculum training, MLflow tracking, digitization, validation, and annotation tools.

## Recommended workflow

The current recommended training flow is the built-in curriculum pipeline driven by `digitizer train`. It:

- generates stage datasets automatically
- chains weights across stages 1 → 4
- uses the checked-in curriculum presets from `hyps/curriculum_stage*.yml`
- logs MLflow data when `mlflow` is installed in the active environment

The older single-stage `digitizer train --dataset-dir ... --execute` path still exists for compatibility, but it is deprecated.

## Quick start (Nix, recommended)

### 1. Enter a dev shell

```bash
# CPU
nix develop

# AMD ROCm
nix develop .#rocm

# NVIDIA CUDA
nix develop .#cuda
```

### 2. Install MLflow in the active environment

MLflow tracking is optional at runtime, but recommended for the curriculum workflow:

```bash
uv pip install mlflow
```

### 3. Run the test suite

```bash
nix develop --command sh -c "python -m unittest discover -s tests -p 'test_*.py' -v"
```

### 4. Inspect the curriculum plan

```bash
digitizer train --status --output-dir runs
digitizer train --chain-info --resume --output-dir runs
```

### 5. Run curriculum training

This is the main training entrypoint. It resumes from existing checkpoints when possible and writes MLflow data under `runs/mlruns` when MLflow is installed.

```bash
digitizer train \
  --output-dir runs \
  --samples-per-stage 500 \
  --workers 6 \
  --resume
```

Useful overrides:

```bash
digitizer train \
  --output-dir runs \
  --samples-per-stage 500 \
  --workers 6 \
  --epochs 50 \
  --batch 16 \
  --resume
```

### 6. Open MLflow

```bash
mlflow ui --backend-store-uri file:runs/mlruns
```

### 7. Digitize images with the latest trained weights

For a fresh curriculum run, stage 4 weights land under `runs/stage4/train/seg*/weights/best.pt`.

```bash
BEST_RUN="$(ls -dt runs/stage4/train/seg* | head -n1)"
digitizer digitize runs/stage4/data/images \
  --output-dir digitized \
  --weights "$BEST_RUN/weights/best.pt" \
  --overlay
```

When digitizing generated synthetic images, the metadata sidecars in `runs/stage4/data/images` provide axis ranges automatically.

### 8. Validate against ground truth

```bash
digitizer validate \
  --prediction-csv digitized/csv/plot_0000.csv \
  --truth-csv runs/stage4/data/csv/plot_0000.csv
```

`validate` exits with a non-zero status when the result does not pass the built-in threshold, so it can be used in scripts and CI.

## Local install with `uv`

If you are not using Nix, the simplest verified local setup is the CPU path:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,ai-cpu]"
uv pip install mlflow
```

`PyYAML` is included in the core dependency set, so YAML-backed training config loading works in the standard install.

## Current CLI commands

### Curriculum training (recommended)

```bash
# Show detected progress
digitizer train --status --output-dir runs

# Show which weights will be chained
digitizer train --chain-info --resume --output-dir runs

# Re-scan checkpoints and rebuild progress.json
digitizer train --sync --output-dir runs

# Run or resume the curriculum pipeline
digitizer train --output-dir runs --samples-per-stage 500 --workers 6 --resume
```

### Synthetic dataset generation

Use this when you want a standalone dataset without running the full curriculum trainer:

```bash
# Fixed difficulty
digitizer generate --output-dir synthetic-stage1 --count 200 --difficulty 1

# Balanced curriculum-style mix
digitizer generate --output-dir synthetic-curriculum --count 800 --curriculum
```

### Digitization

```bash
# Generated images: axis ranges come from metadata sidecars
digitizer digitize train-dataset/images --output-dir digitized --overlay

# External images: provide calibration explicitly
digitizer digitize my-plot.png \
  --output-dir digitized \
  --x-range "0,10" \
  --y-range "-2,2" \
  --weights model.pt \
  --overlay
```

Notes:

- `--weights` accepts `.pt` or `.onnx`
- without `--weights`, `digitizer digitize` falls back to the CV segmentation path
- external plots need axis calibration from `--x-range` / `--y-range`, `--x-reference` / `--y-reference`, or a metadata sidecar

### Annotation

```bash
digitizer annotate my-plot.png --output-dir train-dataset
```

This opens the interactive annotation workflow and writes image, labels, and metadata sidecars.
