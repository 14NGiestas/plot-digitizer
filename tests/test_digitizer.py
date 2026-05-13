"""Focused tests for the plot digitizer CLI."""

from __future__ import annotations

import json
import builtins
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

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

            digitizer.generate_synthetic_dataset(
                dataset_dir,
                count=1,
                seed=1,
                image_format="png",
                plot_type="general",
            )
            image_path = next((dataset_dir / "images").glob("*.png"))
            truth_csv = next((dataset_dir / "ground_truth").glob("*.csv"))

            result = digitizer.digitize_image(
                image_path=image_path,
                output_dir=output_dir,
                x_range=None,
                y_range=None,
                x_reference=None,
                y_reference=None,
                x_scale="linear",
                y_scale="linear",
                invert_y=False,
                weights=None,
                conf_threshold=0.25,
                create_overlay_image=True,
            )

            self.assertTrue(result.csv_path.exists())
            self.assertTrue(result.replot_csv_path.exists())
            self.assertTrue(result.metadata_path.exists())
            self.assertTrue(result.replot_path.exists())
            self.assertTrue(result.overlay_path and result.overlay_path.exists())

            summary = digitizer.validate_digitization(result.csv_path, truth_csv)
            self.assertLess(summary["mean_absolute_percentage_error_proxy"], 30.0)

            frame = pd.read_csv(result.csv_path)
            self.assertIn("dataset_id", frame.columns)
            self.assertIn("confidence", frame.columns)
            self.assertGreater(len(frame), 50)

            replot_frame = pd.read_csv(result.replot_csv_path)
            self.assertIn("x_real", replot_frame.columns)
            self.assertGreaterEqual(len(replot_frame.columns), 2)
            self.assertGreater(len(replot_frame), 10)

            metadata = json.loads(result.metadata_path.read_text())
            self.assertIn("segmentation", metadata)
            self.assertIn("exports", metadata)
            self.assertEqual(metadata["exports"]["replot_csv"], str(result.replot_csv_path))
            self.assertEqual(metadata["exports"]["replot_image"], str(result.replot_path))
            method_counts_are_ints = all(isinstance(value, int) for value in metadata["segmentation"]["method_counts"].values())
            self.assertTrue(method_counts_are_ints)

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

    def test_build_replot_frame_uses_log_grid_for_log_x_scale(self) -> None:
        points = pd.DataFrame(
            {
                "dataset_id": ["dataset_0"] * 5,
                "x_real": np.geomspace(1.0, 100.0, 5),
                "y_real": np.linspace(0.0, 1.0, 5),
            }
        )

        replot_frame = digitizer.build_replot_frame(points, x_scale="log", max_points=5)

        np.testing.assert_allclose(replot_frame["x_real"].to_numpy(), np.geomspace(1.0, 100.0, 5))

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

    def test_parse_reference_pair_parses_expected_format(self) -> None:
        parsed = digitizer.parse_reference_pair("20:0,120:10", "x")
        self.assertEqual(parsed, ((20.0, 0.0), (120.0, 10.0)))

    def test_generate_writes_consistent_multiclass_dataset_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "synthetic"
            digitizer.generate_synthetic_dataset(
                dataset_dir,
                count=8,
                seed=7,
                image_format="png",
                plot_type="mixed",
            )

            yaml_text = (dataset_dir / "dataset.yaml").read_text().splitlines()
            path_value = next((line.split(":", 1)[1].strip() for line in yaml_text if line.startswith("path:")), None)
            nc_value = int(next(line.split(":", 1)[1].strip() for line in yaml_text if line.startswith("nc:")))
            name_lines = [line for line in yaml_text if line.startswith("  ")]
            names = {int(line.split(":", 1)[0].strip()): line.split(":", 1)[1].strip() for line in name_lines}

            self.assertIsNotNone(path_value)
            self.assertEqual(Path(path_value), dataset_dir.resolve())
            self.assertEqual(nc_value, len(names))
            self.assertEqual(sorted(names.keys()), list(range(nc_value)))

            max_class_id = -1
            for label_file in (dataset_dir / "labels").glob("*.txt"):
                for raw_line in label_file.read_text().splitlines():
                    if not raw_line.strip():
                        continue
                    max_class_id = max(max_class_id, int(raw_line.split()[0]))
            self.assertGreaterEqual(max_class_id, 0)
            self.assertLess(max_class_id, nc_value)

    def test_run_training_raises_import_error_for_missing_torch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "synthetic"
            output_dir = Path(tmp) / "runs"
            digitizer.generate_synthetic_dataset(
                dataset_dir,
                count=1,
                seed=3,
                image_format="png",
                plot_type="general",
            )

            real_import = builtins.__import__

            def import_without_torch(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
                if name == "torch":
                    raise ImportError("No module named 'torch'")
                return real_import(name, globals, locals, fromlist, level)

            with patch("builtins.__import__", side_effect=import_without_torch):
                with self.assertRaisesRegex(ImportError, "torch and torchvision"):
                    digitizer.run_training(
                        dataset_dir=dataset_dir,
                        output_dir=output_dir,
                        epochs=1,
                        imgsz=640,
                        weights="yolov8n-seg.pt",
                        batch=1,
                        execute=True,
                    )

    def test_run_training_raises_import_error_for_missing_ai_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "synthetic"
            output_dir = Path(tmp) / "runs"
            digitizer.generate_synthetic_dataset(
                dataset_dir,
                count=1,
                seed=3,
                image_format="png",
                plot_type="general",
            )

            real_import = builtins.__import__
            # Inject a dummy torch module so the preflight torch check passes, then
            # block only ultralytics to exercise the ultralytics-missing error path.
            dummy_torch = types.ModuleType("torch")

            def import_without_ultralytics(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
                # Only block ultralytics so the preflight torch check passes and we exercise
                # the ultralytics-missing error path.
                if name == "ultralytics":
                    raise ImportError("No module named 'ultralytics'")
                if name == "torch":
                    return dummy_torch
                return real_import(name, globals, locals, fromlist, level)

            with patch("builtins.__import__", side_effect=import_without_ultralytics):
                expected_message = (
                    "Training requires ultralytics. Install digitizer with the 'ai' extra: "
                    "`uv pip install -e \".[ai]\"`"
                )
                with self.assertRaisesRegex(ImportError, re.escape(expected_message)):
                    digitizer.run_training(
                        dataset_dir=dataset_dir,
                        output_dir=output_dir,
                        epochs=1,
                        imgsz=640,
                        weights="yolov8n-seg.pt",
                        batch=1,
                        execute=True,
                    )

    def test_run_training_includes_optional_hyp_yaml_in_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "synthetic"
            output_dir = root / "runs"
            hyp_yaml = root / "hyp.yaml"
            hyp_yaml.write_text("mosaic: 0.0\nclose_mosaic: 0\n")
            digitizer.generate_synthetic_dataset(
                dataset_dir,
                count=1,
                seed=3,
                image_format="png",
                plot_type="general",
            )

            plan = digitizer.run_training(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                epochs=5,
                imgsz=640,
                weights="yolov8n-seg.pt",
                batch=2,
                execute=False,
                hyp_yaml=hyp_yaml,
            )
            self.assertEqual(plan["cfg"], str(hyp_yaml.resolve()))

    def test_calibrate_axes_uses_reference_points_for_non_extreme_axis_points(self) -> None:
        image_path = Path("nonexistent.png")
        plot_box = digitizer.PlotBox(left=10, top=10, right=110, bottom=210)
        calibration, metadata = digitizer.calibrate_axes(
            image_path=image_path,
            plot_box=plot_box,
            x_range=None,
            y_range=None,
            x_reference=((30.0, 2.0), (80.0, 7.0)),
            y_reference=((170.0, 20.0), (70.0, 70.0)),
            x_scale="linear",
            y_scale="linear",
            invert_y=False,
        )
        self.assertAlmostEqual(calibration.x_min, 0.0, places=6)
        self.assertAlmostEqual(calibration.x_max, 10.0, places=6)
        self.assertAlmostEqual(calibration.y_min, 0.0, places=6)
        self.assertAlmostEqual(calibration.y_max, 100.0, places=6)
        self.assertEqual(metadata["axis_detection"]["x_range_source"], "reference")
        self.assertEqual(metadata["axis_detection"]["y_range_source"], "reference")

    def test_gpu_shells_auto_install_torch_via_venv(self) -> None:
        flake_text = (Path(__file__).resolve().parents[1] / "flake.nix").read_text()
        self.assertIn(
            "rocmPkgs = if pkgs ? pkgsRocm then pkgs.pkgsRocm else pkgs;",
            flake_text,
        )
        self.assertIn("cudaPkgs = if pkgs ? pkgsCuda", flake_text)
        self.assertIn("then pkgs.pkgsCuda else pkgs;", flake_text)
        self.assertIn(
            "aiPythonPkgs = defaultPs: ps:",
            flake_text,
        )
        # All three GPU shells must use the AI Python packages.
        self.assertEqual(
            flake_text.count("extraPythonPkgs = aiPythonPkgs python.pkgs;"),
            3,
        )
        # mkAiVenvHook helper must be present and wired up for each accelerator.
        self.assertIn("mkAiVenvHook", flake_text)
        self.assertIn("venv-ai-rocm", flake_text)
        self.assertIn("venv-ai-cuda", flake_text)
        self.assertIn("venv-ai-cuda-legacy", flake_text)
        self.assertIn('-c "import torch; import torchvision; import numpy"', flake_text)
        self.assertIn("rocm6.2", flake_text)
        self.assertIn("cu124", flake_text)
        self.assertIn("cu118", flake_text)
        # Old manual-install hint echoes must be gone.
        self.assertNotIn(
            'echo "Install torch/torchvision for your accelerator before training (see README)."',
            flake_text,
        )

    def test_dev_shell_exposes_digitizer_command_wrapper(self) -> None:
        flake_text = (Path(__file__).resolve().parents[1] / "flake.nix").read_text()
        self.assertIn('writeShellScriptBin "digitizer"', flake_text)
        self.assertIn('exec python -m digitizer "$@"', flake_text)


if __name__ == "__main__":
    unittest.main()
