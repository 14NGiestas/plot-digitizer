# AGENTS.md — plot-digitizer

## Project
Python CLI (`digitizer`) for AI-assisted plot digitization. Synthetic data generation → curriculum training (YOLO segmentation) → inference. Python >=3.10,<3.13.

## Exactly 4 CLI commands

| Command | Purpose |
|---|---|
| `digitizer annotate <IMAGE>` | Interactive matplotlib GUI for manual YOLO annotation |
| `digitizer generate` | Synthetic plots + YOLO labels + CSV + metadata |
| `digitizer train` | Full 4-stage curriculum (generate data → train → chain weights → MLflow) |
| `digitizer digitize <INPUTS...>` | AI segmentation → axis calibration → CSV export |

**Removed:** `validate`, `curriculum`, `import-annotations` are no longer CLI commands. Their logic lives in internal modules only.

## Environment

### Nix (recommended)
```bash
nix develop                     # CPU
nix develop .#rocm              # AMD GPU (gfx1103 / Radeon 780M)
nix develop .#cuda              # NVIDIA GPU
```
After entering `.#rocm` or `.#cuda`, install PyTorch:
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2   # ROCm
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124     # CUDA
```

### uv (alternative)
```bash
uv pip install -e '.[dev,ai]'   # full
uv pip install -e '.[dev]'      # CPU-only
```

## Key commands

```bash
# Tests (unittest, NOT pytest)
python -m unittest discover -s tests -p 'test_*.py' -v

# Generate synthetic data
digitizer generate --output-dir synthetic-data --count 200 --difficulty 3

# Train (full curriculum, auto-resume)
digitizer train --output-dir curriculum-run --resume

# Train (status / sync / plan only)
digitizer train --output-dir curriculum-run --status
digitizer train --output-dir curriculum-run --sync
digitizer train --output-dir curriculum-run --chain-info --resume

# Digitize
digitizer digitize bandstructure_target.png --output-dir digitized --weights curriculum-run/stage4/train/seg*/weights/best.pt

# MLflow UI (local tracking)
mlflow ui --backend-store-uri file:curriculum-run/mlruns
```

## Training (`digitizer train`) — not a single-stage trainer

`train` runs the full 4-stage curriculum automatically:
1. Generates synthetic data per stage (difficulty 1→4)
2. Trains YOLO segmentation, chaining `best.pt` from each stage
3. Logs artifacts to MLflow and registers in model registry

**Flags:** `--resume` (continue from last checkpoint), `--sync` (scan existing checkpoints), `--status` (show progress), `--chain-info` (show weight chain plan), `--from-stage N` (skip to stage N).

`start.sh` is the convenience wrapper: runs `--sync`, `--chain-info`, then `--resume` inside `nix develop .#rocm`.

## Architecture

```
src/digitizer/
  __init__.py          — CLI entry point (main) + curriculum orchestration (432 lines)
  __main__.py          — Enables `python -m digitizer`
  parser.py            — argparse with 4 subparsers
  training.py          — YOLO training + MLflow artifact logging + _find_latest_run_dir
  digitize_workflow.py — Core pipeline: _run_segmentation() + digitize_image()
  ai_segmentation.py   — YOLO inference (run_ai_segmentation)
  cv_segmentation.py   — OpenCV fallback (no weights)
  models.py            — Data classes: PlotBox, AxisCalibration, SegmentationResult, DigitizeResult
  calibration.py       — Axis calibration (detect_axis_anchor_pixels, calibrate_axes)
  axis_parsing.py      — Parse --x-range/--y-range/--reference CLI values
  points.py            — extract_curve_points, convert_points (mask → dataframe)
  annotation_io.py     — YOLO label I/O, polygon geometry, CLASS_MAPPING dict
  constants.py         — Shared constants (LOGGER, thresholds, defaults)
  image_ops.py         — load_image, preprocess_image, resolve_plot_box, discover_images
  cli_support.py       — _set_matplotlib_backend, configure_logging
  plotting.py          — build_replot_frame, create_overlay, create_replot
  curve_utils.py       — Curve smoothing/resampling helpers
  math_utils.py        — _norm_to_scale and friends
  validation.py        — Internal validation (no CLI)
  interactive_annotator.py — Annotation GUI (digitizer annotate)
  interactive_axis.py      — Interactive axis reference selection
  synth/               — Synthetic data generation
    dataset.py         — generate_synthetic_dataset()
    example.py         — _write_synthetic_example()
    render.py          — mask rendering functions
    degrade.py         — _apply_degradation_filters()
hyps/                  — Training presets (curriculum_stage1–4.yml + 3 experimental .yml)
```

## Gotchas

- **Ultralytics auto-increments run dirs** (`seg`, `seg2`, `seg3`…). Always use `_find_latest_run_dir()` to locate `best.pt`.
- **Test patch paths for synth functions** go through `digitizer.synth.example.*`, not `digitizer.*`. The `__init__.py` re-exports them, but `example.py` imports them locally.
- **`train` uses `unittest`**, not pytest. `pytest` is in `dev` deps but the test suite uses `unittest.TestCase`.
- **Hyp configs** are in `hyps/`, not `runs/`. Architecture keys (`model`, `nc`, `names`, etc.) are stripped before passing to ultralytics.
- **MLflow tracking URI** is `file:<output-dir>/mlruns`. Model registry name: `YOLO_Plot_Digitizer_Curriculum`.
- **`digitize` without `--weights`** uses CV fallback (OpenCV). With weights, AI-only — fails if no curve-class masks detected.
- **`OMP_NUM_THREADS` / `MKL_NUM_THREADS`** should match `--workers` for training (set in `start.sh`).
- **`--output` is NOT a valid flag** — the parser uses `--output-dir` for both `digitize` and `generate` (README has `--output` in examples; this is a known doc error).
- **Curriculum stage completion** is determined by MLflow (tag `curriculum_stage`). The `--resume` flag queries `is_stage_complete()` against MLflow runs to find the next incomplete stage.
