# plot-digitizer

Automatic AI-assisted plot digitizer available as a Python package.

## Installation

### Install directly from Git with uv

```bash
# Install the base package (CV-based digitization)
uv add git+https://github.com/14NGiestas/plot-digitizer.git

# Or install with AI segmentation support (requires YOLO weights)
uv add "git+https://github.com/14NGiestas/plot-digitizer.git[ai]"

# Or install for development
uv add --dev "git+https://github.com/14NGiestas/plot-digitizer.git[dev]"
```

### Install from local source

```bash
# Create a virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Or install with optional dependencies
uv pip install -e ".[ai]"      # AI segmentation support
uv pip install -e ".[dev]"     # Development tools
```

## Usage

Once installed, use the `digitizer` command:

```bash
# Generate synthetic plots for training/testing
digitizer generate --output-dir synthetic-data --count 8

# Digitize plot images
digitizer digitize synthetic-data/images --output-dir digitized --overlay

# Validate digitized results against ground truth
digitizer validate \
  --prediction-csv digitized/plot_0000.digitized.csv \
  --truth-csv synthetic-data/ground_truth/plot_0000.csv
```

### Training workflow (new, continue, and fine-tune)

```bash
# 1) Generate a synthetic dataset for segmentation training
digitizer generate --output-dir synthetic-data --count 200

# 2) Print a training plan (no training yet)
digitizer train \
  --dataset-dir synthetic-data \
  --output-dir training-runs \
  --weights yolov8n-seg.pt \
  --epochs 50

# 3) Execute a new training run
digitizer train \
  --dataset-dir synthetic-data \
  --output-dir training-runs \
  --weights yolov8n-seg.pt \
  --epochs 50 \
  --execute

# 4) Continue training from a previous checkpoint (resume)
digitizer train \
  --dataset-dir synthetic-data \
  --output-dir training-runs \
  --weights training-runs/synthetic_plot_digitizer/weights/last.pt \
  --epochs 30 \
  --execute

# 5) Fine-tune from best checkpoint on updated data
digitizer train \
  --dataset-dir synthetic-data \
  --output-dir training-runs-finetune \
  --weights training-runs/synthetic_plot_digitizer/weights/best.pt \
  --epochs 20 \
  --execute
```

### Predict on a real plot image

```bash
# Use a trained model to predict (segment + digitize) one or more real plots
digitizer digitize real-plots \
  --output-dir predictions \
  --weights training-runs/synthetic_plot_digitizer/weights/best.pt \
  --overlay
```

### Axis calibration with known points (instead of extremities)

If axis limits are not at the visible extremities, pass two known points per axis:

```bash
# Format:
# --x-reference "x_pixel_1:x_real_1,x_pixel_2:x_real_2"
# --y-reference "y_pixel_1:y_real_1,y_pixel_2:y_real_2"
digitizer digitize real-plots/plot.png \
  --output-dir predictions \
  --weights training-runs/synthetic_plot_digitizer/weights/best.pt \
  --x-reference "120:0,880:10" \
  --y-reference "710:0,120:100"
```

You can also select those axis points interactively:

```bash
digitizer digitize real-plots/plot.png \
  --output-dir predictions \
  --interactive-axis-selection \
  --weights training-runs/synthetic_plot_digitizer/weights/best.pt
```

### Using without installation (uv run from Git)

You can also run commands directly from the Git repository without installing:

```bash
uv run --from git+https://github.com/14NGiestas/plot-digitizer.git digitizer generate --output-dir synthetic-data --count 8
```

## Commands

- `generate`: create synthetic plots, YOLO segmentation labels, sidecar metadata, and ground-truth CSV files
- `train`: print or execute an Ultralytics YOLOv8 segmentation training plan
- `digitize`: run AI segmentation when weights are provided and fall back to deterministic CV clustering otherwise; supports known-point axis calibration via `--x-reference/--y-reference` or interactive selection
- `validate`: compare a digitized CSV with ground truth and report error metrics

## Notes

- Axis ranges are resolved from CLI hints first, then synthetic sidecar metadata, then safe `0:1` defaults.
- Pass `--x-range min:max` and `--y-range min:max` when auto-detection is unavailable.
- `digitize --weights model.pt` supports `.pt` or `.onnx` Ultralytics-compatible weights.
- The `ai` optional dependency adds `ultralytics` for YOLO-based segmentation.
