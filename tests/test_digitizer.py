"""Focused tests for the plot digitizer CLI."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import digitizer


class DigitizerWorkflowTests(unittest.TestCase):
    """Exercise the synthetic generation and CV digitization flow."""

    def test_generate_digitize_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "synthetic"
            output_dir = root / "digitized"

            digitizer.generate_synthetic_dataset(dataset_dir, count=1, seed=7, image_format="png")
            image_path = next((dataset_dir / "images").glob("*.png"))
            truth_csv = next((dataset_dir / "ground_truth").glob("*.csv"))

            result = digitizer.digitize_image(
                image_path=image_path,
                output_dir=output_dir,
                x_range=None,
                y_range=None,
                x_scale="linear",
                y_scale="linear",
                invert_y=False,
                weights=None,
                conf_threshold=0.25,
                create_overlay_image=True,
            )

            self.assertTrue(result.csv_path.exists())
            self.assertTrue(result.metadata_path.exists())
            self.assertTrue(result.overlay_path and result.overlay_path.exists())

            summary = digitizer.validate_digitization(result.csv_path, truth_csv)
            self.assertLess(summary["mean_absolute_percentage_error_proxy"], 30.0)

            frame = pd.read_csv(result.csv_path)
            self.assertIn("dataset_id", frame.columns)
            self.assertIn("confidence", frame.columns)
            self.assertGreater(len(frame), 50)

            metadata = json.loads(result.metadata_path.read_text())
            self.assertIn("segmentation", metadata)
            self.assertTrue(all(isinstance(value, int) for value in metadata["segmentation"]["method_counts"].values()))

    def test_render_curve_mask_marks_curve_not_background(self) -> None:
        x_values = np.linspace(0.0, 10.0, 200)
        y_values = np.sin(x_values)
        mask = digitizer._render_curve_mask(
            fig_size=(4.0, 3.0),
            dpi=100,
            x_values=x_values,
            y_values=y_values,
            x_range=(0.0, 10.0),
            y_range=(-1.5, 1.5),
            style={"linewidth": 2.0, "linestyle": "-"},
        )
        self.assertGreater(mask.mean(), 0.001)
        self.assertLess(mask.mean(), 0.2)

    def test_validate_digitization_enforces_unique_curve_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth_csv = root / "truth.csv"
            prediction_csv = root / "prediction.csv"

            x_values = np.linspace(0.0, 1.0, 20)
            truth = pd.DataFrame(
                {
                    "dataset_id": ["truth_a"] * len(x_values) + ["truth_b"] * len(x_values),
                    "x_real": np.concatenate([x_values, x_values]),
                    "y_real": np.concatenate([x_values, 1.0 - x_values]),
                }
            )
            prediction = pd.DataFrame(
                {
                    "dataset_id": ["pred_shared"] * len(x_values) + ["pred_bad"] * len(x_values),
                    "x_real": np.concatenate([x_values, x_values]),
                    "y_real": np.concatenate([x_values, np.full_like(x_values, 5.0)]),
                    "confidence": 0.9,
                }
            )

            truth.to_csv(truth_csv, index=False)
            prediction.to_csv(prediction_csv, index=False)

            summary = digitizer.validate_digitization(prediction_csv, truth_csv)
            assigned_predictions = [row["predicted_dataset_id"] for row in summary["per_curve"]]
            self.assertEqual(len(set(assigned_predictions)), len(assigned_predictions))
            self.assertFalse(summary["passed_under_5_percent"])


if __name__ == "__main__":
    unittest.main()
