import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from typing import Any

from .ai_segmentation import run_ai_segmentation, _select_digitization_segmentations
from .models import PlotBox, SegmentationResult, AxisCalibration as Calibration
from .points import convert_points, extract_curve_points
from .constants import LOGGER

class DigitizerModel:
    """Wrapper to encapsulate YOLO inference and point conversion logic."""
    
    def __init__(self, weights_path: str | Path, conf_threshold: float = 0.25):
        self.weights_path = str(weights_path)
        self.conf_threshold = conf_threshold
        # Initialize YOLO. Note: this might load weights into memory twice
        # if the same weights are used by other parts, so keep an eye on VRAM.
        self.model = YOLO(self.weights_path)

    def digitize(
        self, 
        image: np.ndarray, 
        plot_box: PlotBox, 
        calibration: Calibration, 
        image_path: Path,
        workers: int = 1,
        imgsz: int | None = None
    ) -> tuple[pd.DataFrame, list[SegmentationResult]]:
        """
        Runs inference and returns a tuple of (digitized points DataFrame, raw segmentations).
        Raises RuntimeError if no curves are isolated.
        """
        # 1. Inference using the AI segmentation helper
        segmentations = run_ai_segmentation(
            image, 
            plot_box, 
            self.weights_path, 
            self.conf_threshold, 
            workers=workers,
            imgsz=imgsz
        )
        
        if segmentations:
            segmentations = _select_digitization_segmentations(segmentations)
        
        if not segmentations:
            raise RuntimeError(
                f"Unable to isolate curves in {image_path}. "
                "AI segmentation returned no curve-class masks."
            )

        # 2. Extraction and Point Conversion
        point_frames = []
        for seg in segmentations:
            frame = extract_curve_points(seg, plot_box)
            if not frame.empty:
                point_frames.append(frame)
        
        if not point_frames:
            raise RuntimeError(f"No digitized points were extracted from {image_path}.")
        
        combined = pd.concat(point_frames, ignore_index=True)
        # Sort and clean
        combined = combined.dropna().sort_values(["dataset_id", "x_px"]).reset_index(drop=True)
        
        # 3. Convert to Real Units
        return convert_points(combined, calibration, plot_box), segmentations
