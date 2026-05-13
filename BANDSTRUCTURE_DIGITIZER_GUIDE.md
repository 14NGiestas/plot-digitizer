# Enhanced Plot Digitizer for Bandstructures and Scientific Graphs

This guide explains how to use the enhanced synthetic data generation and training pipeline for digitizing bandstructures and graphs from old scientific articles.

## New Features

### 1. Multi-Class Segmentation Support

The system now supports detecting multiple annotation types:
- **0: curve** - Data curves/lines (including bandstructure bands)
- **1: vbar** - Vertical bars (high-symmetry point markers in bandstructures)
- **2: hbar** - Horizontal bars (Fermi levels, threshold lines)
- **3: arrow** - Arrow annotations pointing to features
- **4: error_bar** - Error bar markers

### 2. Bandstructure-Specific Synthetic Data Generation

Generate realistic bandstructure plots with:
- Multiple energy bands (4-10 bands per plot)
- Avoided crossings (band gaps)
- High-symmetry point markers (vertical bars)
- Fermi level lines (horizontal bars)
- Parabolic band dispersion
- Umklapp-like oscillations

### 3. Manual Annotation Integration

Convert your manually collected annotations into YOLO format for training.

---

## Usage Guide

### Step 1: Generate Synthetic Training Data

#### For Bandstructures Only
```bash
python -m digitizer generate \
    --output-dir datasets/synthetic_bandstructure \
    --count 100 \
    --seed 42 \
    --plot-type bandstructure
```

#### For General Scientific Plots
```bash
python -m digitizer generate \
    --output-dir datasets/synthetic_general \
    --count 100 \
    --plot-type general
```

#### Mixed Dataset (Recommended)
```bash
python -m digitizer generate \
    --output-dir datasets/synthetic_mixed \
    --count 200 \
    --plot-type mixed
```

### Step 2: Convert Manual Annotations

If you have manually annotated points from real bandstructures/graphs:

#### CSV Format Example
Create a CSV file like `manual_annotations.csv`:
```csv
image_name,type,x,y_top,y_bottom,width,y,x_left,x_right,height,start_x,start_y,end_x,end_y,y_error
old_paper_fig1.png,vbar,150,50,700,3,,,,,,,,,
old_paper_fig1.png,hbar,,,,,400,100,1100,2,,,,,
old_paper_fig1.png,arrow,,,,5,,,,,200,300,400,500,
```

#### JSON Format Example
Create `annotations.json`:
```json
{
  "images": {
    "old_paper_fig1.png": {"width": 1200, "height": 800}
  },
  "annotations": [
    {"image": "old_paper_fig1.png", "type": "vbar", "x": 150, "y_top": 50, "y_bottom": 700, "width": 3},
    {"image": "old_paper_fig1.png", "type": "hbar", "y": 400, "x_left": 100, "x_right": 1100, "height": 2},
    {"image": "old_paper_fig1.png", "type": "arrow", "start_x": 200, "start_y": 300, "end_x": 400, "end_y": 500}
  ]
}
```

#### Convert and Merge with Synthetic Data
```bash
python scripts/convert_manual_annotations.py \
    --input manual_annotations.csv \
    --merge-with datasets/synthetic_mixed \
    --output-dir datasets/combined_dataset
```

### Step 3: Train the Model

```bash
python -m digitizer train \
    --dataset-dir datasets/combined_dataset \
    --output-dir training-runs \
    --weights yolov8n-seg.pt \
    --epochs 50 \
    --imgsz 640 \
    --batch 8 \
    --execute
```

### Step 4: Distribute the Trained Model

After training, your model weights are saved at:
```
training-runs/synthetic_plot_digitizer/weights/best.pt
```

#### Distribution Options:

1. **Direct File Sharing**
   ```bash
   # Share the .pt file directly
   scp training-runs/synthetic_plot_digitizer/weights/best.pt user@server:/models/
   ```

2. **Export to ONNX for Broader Compatibility**
   ```python
   from ultralytics import YOLO
   model = YOLO('training-runs/synthetic_plot_digitizer/weights/best.pt')
   model.export(format='onnx')
   ```

3. **Upload to Model Registry**
   - Hugging Face Hub
   - AWS S3 / Google Cloud Storage
   - Internal model registry

#### Usage by End Users:
```bash
digitize real-plots/ \
    --output-dir predictions \
    --weights path/to/best.pt \
    --overlay
```

---

## Dataset Structure

Generated datasets follow YOLO segmentation format:

```
datasets/synthetic_mixed/
├── dataset.yaml          # Dataset configuration
├── images/
│   ├── plot_0000.png
│   ├── plot_0000.metadata.json
│   └── ...
├── labels/
│   ├── plot_0000.txt     # YOLO format labels
│   └── ...
└── ground_truth/
    ├── plot_0000.csv     # Curve data in real coordinates
    └── ...
```

### Label File Format (YOLO Segmentation)
```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

Example with multiple classes:
```
0 0.486905 0.293367 0.480357 0.287415 ...   # curve
1 0.185714 0.670918 0.184524 0.670918 ...   # vbar
2 0.832143 0.633503 0.830952 0.633503 ...   # hbar
```

---

## Tips for Old Article Graphs

### Handling Low-Quality Scans
1. Generate more training data with noise:
   ```bash
   python -m digitizer generate --count 500 --plot-type mixed
   ```

2. Add your own annotated examples from similar old papers

### Dealing with Specific Styles
- **Bandstructures**: Use `--plot-type bandstructure` for physics-style plots
- **Log-scale axes**: The digitizer supports `--x-scale log --y-scale log`
- **Inverted Y-axis**: Use `--invert-y` flag during digitization

### Calibration for Real Coordinates
For accurate digitization, provide axis calibration:
```bash
digitize old_paper_figure.png \
    --x-reference "100:0,900:10" \
    --y-reference "700:-5,100:5" \
    --weights path/to/best.pt
```

Or use interactive mode:
```bash
digitize old_paper_figure.png \
    --interactive-axis-selection \
    --weights path/to/best.pt
```

---

## Complete Workflow Example

```bash
# 1. Generate synthetic bandstructure dataset
python -m digitizer generate \
    --output-dir datasets/bandstructure_synthetic \
    --count 150 \
    --plot-type bandstructure

# 2. Convert your manual annotations from old papers
python scripts/convert_manual_annotations.py \
    --input my_manual_annotations.json \
    --merge-with datasets/bandstructure_synthetic \
    --output-dir datasets/bandstructure_combined

# 3. Train the model
python -m digitizer train \
    --dataset-dir datasets/bandstructure_combined \
    --output-dir training-runs \
    --epochs 75 \
    --execute

# 4. Test on real old article scans
digitize scanned_papers/ \
    --output-dir digitized_results \
    --weights training-runs/synthetic_plot_digitizer/weights/best.pt \
    --overlay

# 5. Share the model
cp training-runs/synthetic_plot_digitizer/weights/best.pt \
   /shared/models/bandstructure_digitizer_v1.pt
```

---

## Troubleshooting

### Model Not Detecting Vertical Bars
- Ensure your training data has enough vbar examples
- Increase the proportion of bandstructure plots in training
- Try training for more epochs

### Poor Performance on Old Scans
- Add more diverse training data (different resolutions, noise levels)
- Include actual annotated examples from old papers
- Consider fine-tuning with lower learning rate

### Memory Issues During Training
- Reduce batch size: `--batch 4`
- Use smaller model: `--weights yolov8n-seg.pt` (nano) instead of larger variants
