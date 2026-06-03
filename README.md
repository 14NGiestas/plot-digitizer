# plot-digitizer

Automatic AI-assisted plot digitizer (synthetic data generation, curriculum training, AI inference, manual annotation).

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
python -m unittest discover -s tests -p 'test_*.py' -v
```

### Local install (uv)

```bash
uv venv
source .venv/bin/activate
uv pip install -e '.[dev,ai]'   # full (includes YOLO/torch)
uv pip install -e '.[dev]'      # CPU-only
```

## CLI commands

| Command | Purpose |
|---|---|
| `digitizer generate` | Generate synthetic plots + YOLO labels + CSV + metadata |
| `digitizer train` | Full 4-stage curriculum (generate data → train → chain weights → MLflow) |
| `digitize digitize <INPUTS...>` | AI segmentation → axis calibration → CSV export |
| `digitizer annotate <IMAGE>` | Interactive matplotlib GUI for manual YOLO annotation |

### Generate

```bash
# Basic
digitizer generate --output-dir synthetic-data --count 200

# Per-difficulty
digitizer generate --output-dir synthetic-stage1 --count 200 --difficulty 1

# Balanced curriculum mix (round-robin stages 1→2→3→4)
digitizer generate --output-dir synthetic-curriculum --count 800 --curriculum
```

### Train (full curriculum)

`digitizer train` runs all 4 stages automatically, chaining `best.pt` between stages:

```bash
# Full run
digitizer train --output-dir curriculum-run

# Resume from last completed stage
digitizer train --output-dir curriculum-run --resume

# Check progress / plan only
digitizer train --output-dir curriculum-run --status
digitizer train --output-dir curriculum-run --chain-info --resume

# Sync existing checkpoints into progress.json
digitizer train --output-dir curriculum-run --sync
```

Training presets live in `hyps/` (stage1–4). MLflow tracks locally at `file:<output-dir>/mlruns`.

### Digitize

```bash
# AI segmentation (with trained weights)
digitizer digitize bandstructure_target.png --output digitized_data.json \
  --weights curriculum-run/stage4/train/seg*/weights/best.pt

# OpenCV fallback (no weights)
digitizer digitize bandstructure_target.png --output digitized_data.json

# Known axis calibration
digitizer digitize plot.png --output-dir digitized \
  --x-reference "100:0,500:10" --y-reference "80:0,420:50"

# Batch directory
digitizer digitize plots/ --output-dir digitized --overlay
```

### Annotate

```bash
digitizer annotate my_plot.png --output-dir train-dataset
```

Opens an interactive matplotlib GUI to draw polygon annotations and save YOLO-format labels.

## Useful options

- `--weights` supports `.pt` or `.onnx`. Without weights, uses OpenCV fallback.
- `--x-reference` / `--y-reference`: `"px0:real0,px1:real1"` for known axis points.
- `--x-scale` / `--y-scale`: `linear` (default) or `log`.
- `--invert-y`: flip Y axis direction.
- `--overlay`: write segmentation overlay images.
- `--workers N`: parallel workers for generation/training (also sets `OMP_NUM_THREADS`).

## Convenience script

`start.sh` runs the full curriculum inside `nix develop .#rocm` with auto-resume:

```bash
./start.sh
```

After training, view MLflow UI:

```bash
mlflow ui --backend-store-uri file:curriculum-run/mlruns
```
