# plot-digitizer

Automatic AI-assisted plot digitizer runnable with `uv`.

## Quick start

```bash
uv run ./digitizer.py generate --output-dir synthetic-data --count 8
uv run ./digitizer.py digitize synthetic-data/images --output-dir digitized --overlay
uv run ./digitizer.py validate \
  --prediction-csv digitized/plot_0000.digitized.csv \
  --truth-csv synthetic-data/ground_truth/plot_0000.csv
```

## Commands

- `generate`: create synthetic plots, YOLO segmentation labels, sidecar metadata, and ground-truth CSV files
- `train`: print or execute an Ultralytics YOLOv8 segmentation training plan
- `digitize`: run AI segmentation when weights are provided and fall back to deterministic CV clustering otherwise
- `validate`: compare a digitized CSV with ground truth and report error metrics

## Notes

- Axis ranges are resolved from CLI hints first, then synthetic sidecar metadata, then safe `0:1` defaults.
- Pass `--x-range min:max` and `--y-range min:max` when auto-detection is unavailable.
- `digitize --weights model.pt` supports `.pt` or `.onnx` Ultralytics-compatible weights.
