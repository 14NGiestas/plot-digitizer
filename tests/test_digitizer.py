"""Focused tests for the plot digitizer CLI."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
