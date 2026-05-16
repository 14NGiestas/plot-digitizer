"""Focused tests for the plot digitizer CLI."""

from __future__ import annotations

import json
import builtins
import io
import re
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

import digitizer
from digitizer.annotation_io import (
    annotation_to_yolo_line,
    load_training_sample_annotations,
    polygon_from_arrow,
    polygon_from_curve,
    polygon_from_error_bar,
    polygon_from_hbar,
    polygon_from_line,
    polygon_from_point,
    polygon_from_rectangle,
    polygon_from_vbar,
    save_training_sample,
    scale_annotation_points,
)
from digitizer.parser import build_parser


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
            truth_csv = next((dataset_dir / "csv").glob("*.csv"))

            with patch("digitizer.digitize_workflow.run_ai_segmentation", side_effect=lambda img, box, w, c, workers=None, imgsz=None: digitizer.cv_segmentation.run_cv_segmentation(img, box)):
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
                    weights="dummy.pt",
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
            self.assertEqual(metadata["exports"]["csv"], str(result.csv_path))
            self.assertIsNotNone(result.label_path)
            self.assertTrue(result.label_path.exists())
            method_counts_are_ints = all(isinstance(value, int) for value in metadata["segmentation"]["method_counts"].values())
            self.assertTrue(method_counts_are_ints)

    def test_digitize_image_ignores_non_curve_ai_detections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "plot.png"
            output_dir = root / "digitized"
            image = np.zeros((10, 10, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), image)
            processed_gray = np.zeros((10, 10), dtype=np.uint8)
            plot_box = digitizer.PlotBox(left=0, top=0, right=10, bottom=10)
            calibration = digitizer.AxisCalibration(
                x_min=1.0,
                x_max=10.0,
                y_min=0.0,
                y_max=1.0,
                x_scale="log",
            )
            non_curve = digitizer.SegmentationResult(
                dataset_id="plot_area",
                mask=np.ones((10, 10), dtype=bool),
                confidence=0.99,
                method="ai",
                class_id=5,
            )
            curve = digitizer.SegmentationResult(
                dataset_id="curve",
                mask=np.ones((10, 10), dtype=bool),
                confidence=0.88,
                method="ai",
                class_id=0,
            )
            curve_points = pd.DataFrame(
                {
                    "dataset_id": ["curve_a", "curve_a"],
                    "x_px": [0.0, 9.0],
                    "y_px": [9.0, 0.0],
                    "confidence": [0.88, 0.88],
                    "segmentation_method": ["ai", "ai"],
                }
            )
            replot_frame = pd.DataFrame({"x_real": [1.0, 10.0], "curve_a": [0.0, 1.0]})

            with (
                patch("digitizer.digitize_workflow.load_image", return_value=image),
                patch("digitizer.digitize_workflow.preprocess_image", return_value=(processed_gray, {})),
                patch("digitizer.digitize_workflow.resolve_plot_box", return_value=plot_box),
                patch("digitizer.digitize_workflow.calibrate_axes", return_value=(calibration, {})),
                patch("digitizer.digitize_workflow.run_ai_segmentation", return_value=[non_curve, curve]),
                patch("digitizer.digitize_workflow.extract_curve_points", return_value=curve_points) as mock_extract,
                patch("digitizer.digitize_workflow.build_replot_frame", return_value=replot_frame),
                patch("digitizer.digitize_workflow.create_replot"),
                patch("digitizer.digitize_workflow.create_overlay") as mock_overlay,
                patch("digitizer.digitize_workflow._segmentations_to_yolo_label", return_value="0 0.1 0.1 0.9 0.9"),
            ):
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
                    weights="best.pt",
                    conf_threshold=0.25,
                    create_overlay_image=True,
                )

            mock_extract.assert_called_once()
            self.assertEqual(mock_extract.call_args.args[0].class_id, 0)
            overlay_segmentations = mock_overlay.call_args.args[2]
            self.assertEqual(len(overlay_segmentations), 1)
            self.assertEqual(overlay_segmentations[0].class_id, 0)
            exported = pd.read_csv(result.csv_path)
            self.assertEqual(list(exported["dataset_id"].unique()), ["curve_a"])
            self.assertTrue(np.issubdtype(exported["x_real"].dtype, np.floating))

    def test_digitize_image_raises_when_ai_returns_no_curve_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "plot.png"
            output_dir = root / "digitized"
            image = np.zeros((10, 10, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), image)
            processed_gray = np.zeros((10, 10), dtype=np.uint8)
            plot_box = digitizer.PlotBox(left=0, top=0, right=10, bottom=10)
            calibration = digitizer.AxisCalibration(
                x_min=0.0,
                x_max=1.0,
                y_min=0.0,
                y_max=1.0,
            )
            non_curve = digitizer.SegmentationResult(
                dataset_id="plot_area",
                mask=np.ones((10, 10), dtype=bool),
                confidence=0.95,
                method="ai",
                class_id=5,
            )

            with (
                patch("digitizer.digitize_workflow.load_image", return_value=image),
                patch("digitizer.digitize_workflow.preprocess_image", return_value=(processed_gray, {})),
                patch("digitizer.digitize_workflow.resolve_plot_box", return_value=plot_box),
                patch("digitizer.digitize_workflow.calibrate_axes", return_value=(calibration, {})),
                patch("digitizer.digitize_workflow.run_ai_segmentation", return_value=[non_curve]),
            ):
                with self.assertRaises(RuntimeError) as exc:
                    digitizer.digitize_image(
                        image_path=image_path,
                        output_dir=output_dir,
                        x_range=None,
                        y_range=None,
                        x_reference=None,
                        y_reference=None,
                        x_scale="linear",
                        y_scale="linear",
                        invert_y=False,
                        weights="best.pt",
                        conf_threshold=0.25,
                        create_overlay_image=False,
                    )

            self.assertIn("no curve-class masks", str(exc.exception).lower())

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

    def test_render_curve_mask_padding_expands_training_region(self) -> None:
        x_values = np.linspace(0.0, 10.0, 200)
        y_values = np.sin(x_values)
        base_mask = digitizer._render_curve_mask(
            fig_size=(4.0, 3.0),
            dpi=100,
            x_values=x_values,
            y_values=y_values,
            x_range=(0.0, 10.0),
            y_range=(-1.5, 1.5),
            style={"linewidth": 1.2, "linestyle": "-"},
            curve_mask_padding_pixels=0,
        )
        padded_mask = digitizer._render_curve_mask(
            fig_size=(4.0, 3.0),
            dpi=100,
            x_values=x_values,
            y_values=y_values,
            x_range=(0.0, 10.0),
            y_range=(-1.5, 1.5),
            style={"linewidth": 1.2, "linestyle": "-"},
            curve_mask_padding_pixels=2,
        )
        self.assertGreater(padded_mask.sum(), base_mask.sum())
        self.assertTrue(np.all(~base_mask | padded_mask))

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

    def test_format_reference_pair_cli_value_is_reusable_by_parser(self) -> None:
        original = ((20.25, -1.5), (120.75, 10.125))
        formatted = digitizer._format_reference_pair_cli_value(original)
        parsed = digitizer.parse_reference_pair(formatted, "x")
        for parsed_point, original_point in zip(parsed, original, strict=True):
            # Formatter uses .6f for pixel values and .15g for real values.
            self.assertAlmostEqual(parsed_point[0], original_point[0], places=6)
            self.assertAlmostEqual(parsed_point[1], original_point[1], places=14)

    def test_main_logs_reproducible_args_after_interactive_axis_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "plot.png"
            image_path.write_bytes(b"stub")
            output_dir = root / "digitized"
            fake_result = digitizer.DigitizeResult(
                csv_path=output_dir / "plot.digitized.csv",
                replot_csv_path=output_dir / "plot.replot.csv",
                metadata_path=output_dir / "plot.metadata.json",
                replot_path=output_dir / "plot.replot.png",
                overlay_path=None,
                point_count=1,
                dataset_count=1,
            )
            x_reference = ((10.0, 0.0), (90.0, 10.0))
            y_reference = ((180.0, 0.0), (20.0, 100.0))

            with (
                patch("digitizer._set_matplotlib_backend", return_value=None),
                patch("digitizer.interactive_reference_selection", return_value=(x_reference, y_reference)),
                patch("digitizer.digitize_image", return_value=fake_result),
                patch.object(digitizer.LOGGER, "info") as mock_info,
            ):
                exit_code = digitizer.main(
                    [
                        "digitize",
                        str(image_path),
                        "--output-dir",
                        str(output_dir),
                        "--interactive-axis-selection",
                    ]
                )

            self.assertEqual(exit_code, 0)
            interactive_log_calls = [
                call
                for call in mock_info.call_args_list
                if "Interactive axis selection complete. Reuse with:" in call.args[0]
            ]
            self.assertEqual(len(interactive_log_calls), 1)
            self.assertEqual(
                interactive_log_calls[0].args[1],
                digitizer._format_reference_pair_cli_value(x_reference),
            )
            self.assertEqual(
                interactive_log_calls[0].args[2],
                digitizer._format_reference_pair_cli_value(y_reference),
            )

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

    def test_parallel_generation_matches_sequential(self) -> None:
        """Parallel generation (workers=2) must produce identical files to sequential (workers=1)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq_dir = root / "sequential"
            par_dir = root / "parallel"

            digitizer.generate_synthetic_dataset(
                seq_dir, count=4, seed=99, image_format="png", plot_type="mixed", workers=1
            )
            digitizer.generate_synthetic_dataset(
                par_dir, count=4, seed=99, image_format="png", plot_type="mixed", workers=2
            )

            for subdir in ("images", "labels", "csv"):
                seq_files = sorted(f.name for f in (seq_dir / subdir).iterdir())
                par_files = sorted(f.name for f in (par_dir / subdir).iterdir())
                self.assertEqual(seq_files, par_files, f"File list mismatch in {subdir}/")

            seq_relative_paths = sorted(path.relative_to(seq_dir) for path in seq_dir.rglob("*") if path.is_file())
            par_relative_paths = sorted(path.relative_to(par_dir) for path in par_dir.rglob("*") if path.is_file())
            self.assertEqual(seq_relative_paths, par_relative_paths)

            for rel_path in seq_relative_paths:
                seq_file = seq_dir / rel_path
                par_file = par_dir / rel_path
                if rel_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    self.assertEqual(
                        seq_file.read_bytes(),
                        par_file.read_bytes(),
                        f"File content differs for {rel_path}",
                    )
                else:
                    # Several generated text files embed absolute output paths; normalize root prefixes
                    # so content comparisons focus on deterministic data rather than directory names.
                    seq_text = seq_file.read_text().replace(str(seq_dir), "__DATASET_ROOT__")
                    par_text = par_file.read_text().replace(str(par_dir), "__DATASET_ROOT__")
                    self.assertEqual(seq_text, par_text, f"File content differs for {rel_path}")

    def test_generate_rejects_non_positive_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "synthetic"

            with self.assertRaisesRegex(ValueError, "workers must be >= 1"):
                digitizer.generate_synthetic_dataset(
                    dataset_dir, count=1, seed=7, image_format="png", plot_type="mixed", workers=0
                )

            with self.assertRaisesRegex(ValueError, "workers must be >= 1"):
                digitizer.generate_synthetic_dataset(
                    dataset_dir, count=1, seed=7, image_format="png", plot_type="mixed", workers=-2
                )

    def test_generate_parser_rejects_non_positive_workers(self) -> None:
        parser = digitizer.build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            for invalid_workers in ("0", "-2"):
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    with self.assertRaises(SystemExit):
                        parser.parse_args(["generate", "--output-dir", tmp, "--workers", invalid_workers])
                self.assertIn("must be >= 1", stderr.getvalue())

    def test_digitize_parser_rejects_non_positive_workers(self) -> None:
        parser = digitizer.build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "input.png"
            image_path.write_bytes(b"stub")
            for invalid_workers in ("0", "-2"):
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    with self.assertRaises(SystemExit):
                        parser.parse_args(["digitize", str(image_path), "--workers", invalid_workers])
                self.assertIn("must be >= 1", stderr.getvalue())

    def test_run_ai_segmentation_resizes_mask_to_image_shape(self) -> None:
        """Masks at model resolution must be resized to original image shape."""
        image_h, image_w = 400, 600
        mask_h, mask_w = 160, 160  # model resolution smaller than image

        class FakeTensor:
            def __init__(self, arr: np.ndarray) -> None:
                self._arr = arr

            def cpu(self) -> "FakeTensor":
                return self

            def numpy(self) -> np.ndarray:
                return self._arr

            def item(self) -> float:
                return float(self._arr.flat[0])

        class FakeMasks:
            def __init__(self) -> None:
                self.data = [FakeTensor(np.ones((mask_h, mask_w), dtype=np.float32))]

        class FakeBoxes:
            def __init__(self) -> None:
                self.conf = [FakeTensor(np.array(0.9))]
                self.cls = [FakeTensor(np.array(0.0))]

        class FakeResult:
            def __init__(self) -> None:
                self.masks = FakeMasks()
                self.boxes = FakeBoxes()

        class FakeYOLO:
            def __init__(self, weights: str) -> None:
                pass

            def predict(self, _image: np.ndarray, **kwargs: object) -> list[object]:
                return [FakeResult()]

        fake_ultralytics = types.SimpleNamespace(YOLO=FakeYOLO)
        image = np.ones((image_h, image_w, 3), dtype=np.uint8) * 200
        plot_box = digitizer.PlotBox(left=10, top=10, right=image_w - 10, bottom=image_h - 10)

        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            segs = digitizer.run_ai_segmentation(
                image=image,
                plot_box=plot_box,
                weights="fake.pt",
                conf_threshold=0.25,
            )

        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].mask.shape, (image_h, image_w),
                         "Mask must be resized to original image shape, not model resolution")

    def test_run_ai_segmentation_forwards_workers_to_predict(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeYOLO:
            def __init__(self, _weights: str):
                pass

            def predict(self, _image: np.ndarray, **kwargs: object) -> list[object]:
                calls.append(kwargs)
                return []

        fake_ultralytics = types.SimpleNamespace(YOLO=FakeYOLO)
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        plot_box = digitizer.PlotBox(left=0, top=0, right=16, bottom=16)

        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            result = digitizer.run_ai_segmentation(
                image=image,
                plot_box=plot_box,
                weights="fake.pt",
                conf_threshold=0.25,
                workers=16,
            )

        self.assertEqual(result, [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["workers"], 16)

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
                        weights="yolo11s-seg.pt",
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
                        weights="yolo11s-seg.pt",
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
                weights="yolo11s-seg.pt",
                batch=2,
                execute=False,
                hyp_yaml=hyp_yaml,
            )
            self.assertEqual(plan["cfg"], str(hyp_yaml.resolve()))

    def test_run_training_execute_applies_workers_to_torch_threads_and_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "synthetic"
            output_dir = root / "runs"
            digitizer.generate_synthetic_dataset(
                dataset_dir,
                count=1,
                seed=3,
                image_format="png",
                plot_type="general",
            )

            calls: dict[str, object] = {}

            class FakeTorch:
                @staticmethod
                def set_num_threads(value: int) -> None:
                    calls["num_threads"] = value

                @staticmethod
                def set_num_interop_threads(value: int) -> None:
                    calls["num_interop_threads"] = value

            class FakeTrainResult:
                save_dir = Path(tmp) / "yolo-run"

            class FakeYOLO:
                def __init__(self, _weights: str):
                    pass

                def train(self, **kwargs: object) -> FakeTrainResult:
                    calls["train_kwargs"] = kwargs
                    return FakeTrainResult()

            fake_ultralytics = types.SimpleNamespace(YOLO=FakeYOLO)
            with patch.dict(sys.modules, {"torch": FakeTorch, "ultralytics": fake_ultralytics}):
                plan = digitizer.run_training(
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                    epochs=1,
                    imgsz=640,
                    weights="yolo11s-seg.pt",
                    batch=1,
                    execute=True,
                    workers=16,
                )

            self.assertEqual(calls["num_threads"], 16)
            self.assertEqual(calls["num_interop_threads"], 16)
            self.assertEqual(calls["train_kwargs"]["workers"], 16)
            self.assertEqual(plan["workers"], 16)
            # amp defaults to False
            self.assertFalse(calls["train_kwargs"]["amp"])
            self.assertFalse(plan["amp"])

    def test_calibrate_axes_uses_reference_points_for_non_extreme_axis_points(self) -> None:
        image_path = Path("nonexistent.png")
        plot_box = digitizer.PlotBox(left=10, top=10, right=110, bottom=210)
        calibration, metadata = digitizer.calibrate_axes(
            image_path=image_path,
            plot_box=plot_box,
            processed_gray=None,
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

    def test_calibrate_axes_can_auto_detect_axis_anchor_pixels(self) -> None:
        image_path = Path("nonexistent.png")
        plot_box = digitizer.PlotBox(left=10, top=10, right=110, bottom=210)
        processed_gray = np.full((220, 120), 255, dtype=np.uint8)
        # Simulate dark y-axis and x-axis lines inside the detected plot box.
        processed_gray[25:195, 24] = 0
        processed_gray[186, 24:101] = 0

        calibration, metadata = digitizer.calibrate_axes(
            image_path=image_path,
            plot_box=plot_box,
            processed_gray=processed_gray,
            x_range=(0.0, 10.0),
            y_range=(0.0, 100.0),
            x_reference=None,
            y_reference=None,
            x_scale="linear",
            y_scale="linear",
            invert_y=False,
            auto_axis_anchors=True,
        )
        self.assertAlmostEqual(calibration.x_min, 0.0, places=6)
        self.assertAlmostEqual(calibration.x_max, 10.0, places=6)
        self.assertAlmostEqual(calibration.y_min, 0.0, places=6)
        self.assertAlmostEqual(calibration.y_max, 100.0, places=6)
        self.assertEqual(metadata["axis_detection"]["x_range_source"], "auto-anchor")
        self.assertEqual(metadata["axis_detection"]["y_range_source"], "auto-anchor")
        self.assertIsNotNone(metadata["axis_anchor_pixels"])
        x_anchor_left, x_anchor_right = metadata["axis_anchor_pixels"]["x"]
        y_anchor_bottom, y_anchor_top = metadata["axis_anchor_pixels"]["y"]
        x_anchor_real, y_coords_at_x_anchor = calibration.pixel_to_real(
            np.array([x_anchor_left, x_anchor_right]),
            np.array([y_anchor_bottom, y_anchor_bottom]),
            plot_box,
        )
        x_coords_at_y_anchor, y_anchor_real = calibration.pixel_to_real(
            np.array([x_anchor_left, x_anchor_left]),
            np.array([y_anchor_bottom, y_anchor_top]),
            plot_box,
        )
        self.assertAlmostEqual(float(x_anchor_real[0]), 0.0, places=4)
        self.assertAlmostEqual(float(x_anchor_real[1]), 10.0, places=4)
        self.assertAlmostEqual(float(y_anchor_real[0]), 0.0, places=4)
        self.assertAlmostEqual(float(y_anchor_real[1]), 100.0, places=4)
        self.assertTrue(np.all(np.isfinite(y_coords_at_x_anchor)))
        self.assertTrue(np.all(np.isfinite(x_coords_at_y_anchor)))
        self.assertEqual(calibration.x_pixel_min, x_anchor_left)
        self.assertEqual(calibration.x_pixel_max, x_anchor_right)
        self.assertEqual(calibration.y_pixel_bottom, y_anchor_bottom)
        self.assertEqual(calibration.y_pixel_top, y_anchor_top)
        self.assertNotEqual(calibration.x_pixel_min, plot_box.left)

    def test_calibrate_axes_requires_axis_bounds_when_no_inputs_available(self) -> None:
        image_path = Path("nonexistent.png")
        plot_box = digitizer.PlotBox(left=10, top=10, right=110, bottom=210)
        with self.assertRaises(RuntimeError) as exc:
            digitizer.calibrate_axes(
                image_path=image_path,
                plot_box=plot_box,
                processed_gray=None,
                x_range=None,
                y_range=None,
                x_reference=None,
                y_reference=None,
                x_scale="linear",
                y_scale="linear",
                invert_y=False,
            )
        self.assertIn("x-axis bounds", str(exc.exception).lower())

    def test_write_synthetic_example_draws_vbar_hbar_and_error_bar_on_main_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for subdir in ("images", "labels", "csv", "annotations"):
                (root / subdir).mkdir()
            rng = np.random.default_rng(7)
            solid_mask = np.zeros((120, 120), dtype=bool)
            solid_mask[20:100, 30:90] = True
            with (
                patch("digitizer._apply_degradation_filters", return_value=None),
                patch("digitizer._render_curve_mask", return_value=solid_mask),
                patch("digitizer._render_vbar_mask", return_value=solid_mask),
                patch("digitizer._render_hbar_mask", return_value=solid_mask),
                patch("digitizer._render_arrow_mask", return_value=solid_mask),
                patch("digitizer._render_error_bar_mask", return_value=solid_mask),
                patch("matplotlib.axes.Axes.axvline") as mock_axvline,
                patch("matplotlib.axes.Axes.axhline") as mock_axhline,
                patch("matplotlib.axes.Axes.errorbar") as mock_errorbar,
            ):
                digitizer._write_synthetic_example(0, root, rng, image_format="png", plot_type="general")

            metadata = json.loads((root / "images" / "plot_0000.metadata.json").read_text())
            annotation_counts: dict[str, int] = {}
            for annotation in metadata["annotations"]:
                annotation_counts[annotation["type"]] = annotation_counts.get(annotation["type"], 0) + 1

            self.assertEqual(mock_axvline.call_count, annotation_counts.get("vbar", 0))
            self.assertEqual(mock_axhline.call_count, annotation_counts.get("hbar", 0))
            self.assertEqual(mock_errorbar.call_count, annotation_counts.get("error_bar", 0))

    def test_ai_shells_auto_install_torch_via_venv(self) -> None:
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
        # CPU + GPU shells must use the AI Python packages.
        self.assertEqual(
            flake_text.count("extraPythonPkgs = aiPythonPkgs python.pkgs;"),
            4,
        )
        # mkAiVenvHook helper must be present and wired up for each accelerator.
        self.assertIn("mkAiVenvHook", flake_text)
        self.assertIn("venv-ai-cpu", flake_text)
        self.assertIn("venv-ai-rocm", flake_text)
        self.assertIn("venv-ai-cuda", flake_text)
        self.assertIn("venv-ai-cuda-legacy", flake_text)
        self.assertIn('-c "import torch; import torchvision; import numpy"', flake_text)
        self.assertIn("https://download.pytorch.org/whl/cpu", flake_text)
        self.assertIn("rocm6.2", flake_text)
        self.assertIn("cu124", flake_text)
        self.assertIn("cu114", flake_text)
        # Old manual-install hint echoes must be gone.
        self.assertNotIn(
            'echo "Install torch/torchvision for your accelerator before training (see README)."',
            flake_text,
        )

    def test_dev_shell_exposes_digitizer_command_wrapper(self) -> None:
        flake_text = (Path(__file__).resolve().parents[1] / "flake.nix").read_text()
        self.assertIn('writeShellScriptBin "digitizer"', flake_text)
        self.assertIn('exec python -m digitizer "$@"', flake_text)


class AnnotationIOTests(unittest.TestCase):
    """Tests for annotation_io polygon geometry and training sample export."""

    def test_polygon_from_vbar_normalises_x_center(self) -> None:
        # vbar at x=100 in a 200×200 image, line_width=4 → half-width = 2px
        poly = polygon_from_vbar(100, 0, 200, 4, 200, 200)
        # poly = [x0, y0, x1, y1, x2, y2, x3, y3] (4 corners)
        self.assertEqual(len(poly), 8)
        xs = poly[0::2]
        self.assertAlmostEqual(min(xs), (100 - 2) / 200, places=5)
        self.assertAlmostEqual(max(xs), (100 + 2) / 200, places=5)

    def test_polygon_from_hbar_normalises_y_center(self) -> None:
        poly = polygon_from_hbar(60, 0, 120, 6, 120, 120)
        ys = poly[1::2]
        self.assertAlmostEqual(min(ys), (60 - 3) / 120, places=5)
        self.assertAlmostEqual(max(ys), (60 + 3) / 120, places=5)

    def test_polygon_from_arrow_returns_four_corners(self) -> None:
        poly = polygon_from_arrow((0, 0), (100, 0), 4, 200, 200)
        self.assertEqual(len(poly), 8)

    def test_polygon_from_arrow_degenerate_returns_empty(self) -> None:
        poly = polygon_from_arrow((50, 50), (50, 50), 4, 200, 200)
        self.assertEqual(poly, [])

    def test_polygon_from_curve_two_points(self) -> None:
        poly = polygon_from_curve([(10, 10), (90, 10)], 4, 100, 100)
        # upper + lower sides → 4 corners = 8 values
        self.assertGreaterEqual(len(poly), 8)

    def test_polygon_from_curve_single_point_empty(self) -> None:
        self.assertEqual(polygon_from_curve([(50, 50)], 3, 100, 100), [])

    def test_polygon_from_error_bar_shape(self) -> None:
        poly = polygon_from_error_bar(50, 10, 90, 20, 4, 100, 100)
        # 8 vertices × 2 coords = 16 values
        self.assertEqual(len(poly), 16)

    def test_polygon_from_rectangle(self) -> None:
        poly = polygon_from_rectangle((10, 20), (90, 80), 100, 100)
        self.assertEqual(len(poly), 8)

    def test_polygon_from_line(self) -> None:
        poly = polygon_from_line((10, 10), (90, 10), 3, 100, 100)
        self.assertEqual(len(poly), 8)

    def test_polygon_from_point(self) -> None:
        poly = polygon_from_point((50, 50), 6, 100, 100)
        self.assertEqual(len(poly), 8)

    def test_annotation_to_yolo_line_vbar_format(self) -> None:
        ann = {"type": "vbar", "points": [(100, 50)]}
        line = annotation_to_yolo_line(ann, 200, 200)
        self.assertIsNotNone(line)
        assert line is not None
        self.assertTrue(line.startswith("1 "))
        parts = line.split()
        self.assertEqual(parts[0], "1")
        # Remaining values should all parse as floats in [0, 1]
        for v in parts[1:]:
            self.assertGreaterEqual(float(v), 0.0)
            self.assertLessEqual(float(v), 1.0)

    def test_annotation_to_yolo_line_arrow_format(self) -> None:
        ann = {"type": "arrow", "points": [(10, 10), (90, 90)]}
        line = annotation_to_yolo_line(ann, 100, 100)
        self.assertIsNotNone(line)
        assert line is not None
        self.assertTrue(line.startswith("3 "))

    def test_annotation_to_yolo_line_frame_classes(self) -> None:
        cases = [
            ({"type": "plot_area", "points": [(5, 5), (95, 95)]}, "5 "),
            ({"type": "x_axis", "points": [(5, 95), (95, 95)]}, "6 "),
            ({"type": "y_axis", "points": [(5, 5), (5, 95)]}, "7 "),
            ({"type": "x_anchor", "points": [(50, 95)]}, "8 "),
            ({"type": "y_anchor", "points": [(5, 50)]}, "9 "),
        ]
        for ann, prefix in cases:
            with self.subTest(ann_type=ann["type"]):
                line = annotation_to_yolo_line(ann, 100, 100)
                self.assertIsNotNone(line)
                assert line is not None
                self.assertTrue(line.startswith(prefix))

    def test_annotation_to_yolo_line_unknown_type_returns_none(self) -> None:
        self.assertIsNone(annotation_to_yolo_line({"type": "unknown", "points": [(1, 1)]}, 100, 100))

    def test_annotation_to_yolo_line_insufficient_points_returns_none(self) -> None:
        # arrow needs 2 points
        self.assertIsNone(annotation_to_yolo_line({"type": "arrow", "points": [(1, 1)]}, 100, 100))

    def test_scale_annotation_points_scales_proportionally(self) -> None:
        ann = {"type": "vbar", "points": [(100, 200)], "line_width": 3.0}
        scaled = scale_annotation_points(ann, 0.5, 0.5)
        self.assertAlmostEqual(scaled["points"][0][0], 50.0)
        self.assertAlmostEqual(scaled["points"][0][1], 100.0)
        # Original should be unmodified
        self.assertAlmostEqual(ann["points"][0][0], 100.0)

    def test_save_training_sample_creates_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Create a minimal synthetic image
            import cv2
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img_path = root / "test_image.png"
            cv2.imwrite(str(img_path), img)

            annotations = [
                {"type": "vbar", "points": [(50, 0)]},
                {"type": "hbar", "points": [(0, 50)]},
                {"type": "arrow", "points": [(10, 10), (80, 80)]},
            ]
            result = save_training_sample(img_path, annotations, root / "out")

            self.assertTrue(Path(result["image_path"]).exists())
            self.assertTrue(Path(result["label_path"]).exists())
            self.assertTrue(Path(result["metadata_path"]).exists())
            self.assertTrue(Path(result["annotations_path"]).exists())

            label_text = Path(result["label_path"]).read_text()
            lines = [l for l in label_text.splitlines() if l.strip()]
            self.assertEqual(len(lines), 3)
            # First line should be class 1 (vbar)
            self.assertTrue(lines[0].startswith("1 "))
            # Second line should be class 2 (hbar)
            self.assertTrue(lines[1].startswith("2 "))
            # Third line should be class 3 (arrow)
            self.assertTrue(lines[2].startswith("3 "))

            metadata = json.loads(Path(result["metadata_path"]).read_text())
            self.assertEqual(metadata["image_width"], 100)
            self.assertEqual(metadata["image_height"], 100)
            self.assertEqual(metadata["label_count"], 3)
            self.assertNotIn("annotations", metadata)
            self.assertIn("annotations_path", metadata)

            ann_data = json.loads(Path(result["annotations_path"]).read_text())
            self.assertEqual(len(ann_data["annotations"]), 3)
            self.assertEqual(ann_data["image_width"], 100)

    def test_save_training_sample_resize_scales_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            import cv2
            img = np.zeros((100, 200, 3), dtype=np.uint8)
            img_path = root / "test_image.png"
            cv2.imwrite(str(img_path), img)

            annotations = [{"type": "x_anchor", "points": [(100, 50)], "point_size": 6.0}]
            result = save_training_sample(img_path, annotations, root / "out", resize_to=(100, 50))
            metadata = json.loads(Path(result["metadata_path"]).read_text())
            self.assertEqual(metadata["source_image_width"], 200)
            self.assertEqual(metadata["source_image_height"], 100)
            self.assertEqual(metadata["image_width"], 100)
            self.assertEqual(metadata["image_height"], 50)

            ann_data = json.loads(Path(result["annotations_path"]).read_text())
            scaled_pt = ann_data["annotations"][0]["points"][0]
            self.assertAlmostEqual(float(scaled_pt[0]), 50.0, places=3)
            self.assertAlmostEqual(float(scaled_pt[1]), 25.0, places=3)

    def test_load_training_sample_annotations_rescales_to_target_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "out"
            ann_dir = output_dir / "annotations"
            ann_dir.mkdir(parents=True, exist_ok=True)
            ann_path = ann_dir / "plot.json"
            ann_path.write_text(
                json.dumps(
                    {
                        "image_width": 100,
                        "image_height": 50,
                        "annotations": [{"type": "x_anchor", "points": [(50, 25)], "point_size": 6.0}],
                    }
                )
            )
            loaded = load_training_sample_annotations(
                image_path=root / "plot.png",
                output_dir=output_dir,
                target_size=(200, 100),
            )
            self.assertEqual(len(loaded), 1)
            point = loaded[0]["points"][0]
            self.assertAlmostEqual(float(point[0]), 100.0, places=3)
            self.assertAlmostEqual(float(point[1]), 50.0, places=3)

    def test_load_training_sample_annotations_legacy_metadata_fallback(self) -> None:
        """Metadata with embedded annotations is still readable (backward compat)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "out"
            images_dir = output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = images_dir / "plot.metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "image_width": 100,
                        "image_height": 50,
                        "annotations": [{"type": "x_anchor", "points": [(50, 25)], "point_size": 6.0}],
                    }
                )
            )
            loaded = load_training_sample_annotations(
                image_path=root / "plot.png",
                output_dir=output_dir,
                target_size=(200, 100),
            )
            self.assertEqual(len(loaded), 1)
            point = loaded[0]["points"][0]
            self.assertAlmostEqual(float(point[0]), 100.0, places=3)
            self.assertAlmostEqual(float(point[1]), 50.0, places=3)

    def test_load_training_sample_annotations_missing_metadata_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = load_training_sample_annotations(root / "missing.png", root / "out", target_size=(100, 100))
            self.assertEqual(loaded, [])


class AnnotateParserTests(unittest.TestCase):
    """Tests for the `annotate` CLI sub-command parser."""

    def test_annotate_parser_accepts_image_input(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["annotate", "my_plot.png"])
        self.assertEqual(args.command, "annotate")
        self.assertEqual(str(args.input), "my_plot.png")
        self.assertAlmostEqual(args.line_width, 3.0)

    def test_annotate_parser_accepts_custom_output_dir(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["annotate", "img.png", "--output-dir", "/tmp/out", "--line-width", "5.0"])
        self.assertEqual(str(args.output_dir), "/tmp/out")
        self.assertAlmostEqual(args.line_width, 5.0)

    def test_annotate_parser_accepts_resize_dimensions(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["annotate", "img.png", "--resize-width", "640", "--resize-height", "480"])
        self.assertEqual(args.resize_width, 640)
        self.assertEqual(args.resize_height, 480)

    def test_annotate_parser_accepts_update_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["annotate", "img.png", "--update"])
        self.assertTrue(args.update)


class GenerateDegradationsTests(unittest.TestCase):
    """Tests for multi-degradation variant generation."""

    def test_multi_degradation_produces_N_images_per_base_plot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            solid_mask = np.zeros((120, 120), dtype=bool)
            solid_mask[20:100, 30:90] = True
            with (
                patch("digitizer._apply_degradation_filters", return_value=None),
                patch("digitizer._render_curve_mask", return_value=solid_mask),
                patch("digitizer._render_vbar_mask", return_value=solid_mask),
                patch("digitizer._render_hbar_mask", return_value=solid_mask),
                patch("digitizer._render_arrow_mask", return_value=solid_mask),
                patch("digitizer._render_error_bar_mask", return_value=solid_mask),
            ):
                digitizer.generate_synthetic_dataset(
                    root / "out", count=2, seed=7, image_format="png",
                    plot_type="general", workers=1, degradations=3,
                )

            images = sorted((root / "out" / "images").glob("*.png"))
            labels = sorted((root / "out" / "labels").glob("*.txt"))
            # 2 base plots × 3 degradations = 6 images, 6 label files
            self.assertEqual(len(images), 6)
            self.assertEqual(len(labels), 6)
            # Variant naming: plot_0000_deg00, plot_0000_deg01, plot_0000_deg02 ...
            stems = {p.stem for p in images}
            self.assertIn("plot_0000_deg00", stems)
            self.assertIn("plot_0000_deg02", stems)
            self.assertIn("plot_0001_deg00", stems)
            # Base clean image should NOT be present
            self.assertNotIn("plot_0000", stems)
            # Base label file should also be removed
            self.assertFalse((root / "out" / "labels" / "plot_0000.txt").exists())
            # Shared files: 2 annotations + 2 csv (one per base plot)
            ann_files = list((root / "out" / "annotations").glob("*.json"))
            csv_files = list((root / "out" / "csv").glob("*.csv"))
            self.assertEqual(len(ann_files), 2)
            self.assertEqual(len(csv_files), 2)

    def test_single_degradation_preserves_old_naming(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            solid_mask = np.zeros((120, 120), dtype=bool)
            solid_mask[20:100, 30:90] = True
            with (
                patch("digitizer._apply_degradation_filters", return_value=None),
                patch("digitizer._render_curve_mask", return_value=solid_mask),
                patch("digitizer._render_vbar_mask", return_value=solid_mask),
                patch("digitizer._render_hbar_mask", return_value=solid_mask),
                patch("digitizer._render_arrow_mask", return_value=solid_mask),
                patch("digitizer._render_error_bar_mask", return_value=solid_mask),
            ):
                digitizer.generate_synthetic_dataset(
                    root / "out", count=2, seed=7, image_format="png",
                    plot_type="general", workers=1, degradations=1,
                )
            images = sorted((root / "out" / "images").glob("*.png"))
            # degradations=1: 2 images, no _deg suffix
            stems = {p.stem for p in images}
            self.assertIn("plot_0000", stems)
            self.assertIn("plot_0001", stems)
            # No _deg suffix files
            self.assertFalse(any("_deg" in s for s in stems))

    def test_generate_parser_rejects_non_positive_degradations(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["generate", "--degradations", "0"])

    def test_generate_parser_accepts_degradations_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["generate", "--degradations", "5"])
        self.assertEqual(args.degradations, 5)


class TrainingAmpTests(unittest.TestCase):
    """Tests for amp=False default in training."""

    def test_train_parser_amp_defaults_to_false(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["train", "--dataset-dir", "/tmp/ds"])
        self.assertFalse(args.amp)

    def test_train_parser_accepts_amp_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["train", "--dataset-dir", "/tmp/ds", "--amp"])
        self.assertTrue(args.amp)

    def test_run_training_plan_includes_amp_false_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "ds"
            digitizer.generate_synthetic_dataset(
                dataset_dir, count=1, seed=1, image_format="png", plot_type="general",
            )
            plan = digitizer.run_training(
                dataset_dir=dataset_dir,
                output_dir=Path(tmp) / "runs",
                epochs=1, imgsz=320, weights="yolo11s-seg.pt", batch=1, execute=False,
            )
            self.assertFalse(plan["amp"])

    def test_run_training_plan_includes_amp_true_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "ds"
            digitizer.generate_synthetic_dataset(
                dataset_dir, count=1, seed=1, image_format="png", plot_type="general",
            )
            plan = digitizer.run_training(
                dataset_dir=dataset_dir,
                output_dir=Path(tmp) / "runs",
                epochs=1, imgsz=320, weights="yolo11s-seg.pt", batch=1, execute=False,
                amp=True,
            )
            self.assertTrue(plan["amp"])


class TickLabelAnnotationTests(unittest.TestCase):
    """Tests for tick-label annotation extraction during synthetic generation."""

    def test_generated_annotations_include_tick_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            solid_mask = np.zeros((120, 120), dtype=bool)
            solid_mask[20:100, 30:90] = True
            with (
                patch("digitizer._apply_degradation_filters", return_value=None),
                patch("digitizer._render_curve_mask", return_value=solid_mask),
                patch("digitizer._render_vbar_mask", return_value=solid_mask),
                patch("digitizer._render_hbar_mask", return_value=solid_mask),
                patch("digitizer._render_arrow_mask", return_value=solid_mask),
                patch("digitizer._render_error_bar_mask", return_value=solid_mask),
            ):
                digitizer.generate_synthetic_dataset(
                    root / "out", count=1, seed=42, image_format="png",
                    plot_type="general", workers=1,
                )
            ann_path = root / "out" / "annotations" / "plot_0000.json"
            self.assertTrue(ann_path.exists())
            ann_data = json.loads(ann_path.read_text())
            types_found = {a["type"] for a in ann_data["annotations"]}
            self.assertIn("x_tick_label", types_found)
            self.assertIn("y_tick_label", types_found)
            # Every tick label must have a non-empty text value
            for ann in ann_data["annotations"]:
                if ann["type"] in ("x_tick_label", "y_tick_label"):
                    self.assertIn("text", ann)
                    self.assertTrue(ann["text"].strip())

    def test_tick_labels_appear_in_yolo_label_file(self) -> None:
        from digitizer.annotation_io import CLASS_MAPPING
        x_tick_class = CLASS_MAPPING["x_tick_label"]
        y_tick_class = CLASS_MAPPING["y_tick_label"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            solid_mask = np.zeros((120, 120), dtype=bool)
            solid_mask[20:100, 30:90] = True
            with (
                patch("digitizer._apply_degradation_filters", return_value=None),
                patch("digitizer._render_curve_mask", return_value=solid_mask),
                patch("digitizer._render_vbar_mask", return_value=solid_mask),
                patch("digitizer._render_hbar_mask", return_value=solid_mask),
                patch("digitizer._render_arrow_mask", return_value=solid_mask),
                patch("digitizer._render_error_bar_mask", return_value=solid_mask),
            ):
                digitizer.generate_synthetic_dataset(
                    root / "out", count=1, seed=42, image_format="png",
                    plot_type="general", workers=1,
                )
            label_text = (root / "out" / "labels" / "plot_0000.txt").read_text()
            classes_in_labels = {int(line.split()[0]) for line in label_text.splitlines() if line.strip()}
            self.assertIn(x_tick_class, classes_in_labels)
            self.assertIn(y_tick_class, classes_in_labels)


class ImportAnnotationsTests(unittest.TestCase):
    """Tests for import-annotations subcommand."""

    def test_import_from_old_metadata_json(self) -> None:
        from digitizer.annotation_io import import_annotations_from_old_format
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_path = root / "plot_0000.metadata.json"
            metadata_path.write_text(json.dumps({
                "image": str(root / "plot_0000.png"),
                "image_width": 100,
                "image_height": 80,
                "annotations": [
                    {"type": "x_anchor", "points": [(20, 60)], "point_size": 6.0},
                    {"type": "vbar", "points": [(50, 0)]},
                ],
            }))
            out = import_annotations_from_old_format(metadata_path, root / "train-dataset")
            self.assertTrue(out.exists())
            data = json.loads(out.read_text())
            self.assertEqual(len(data["annotations"]), 2)
            self.assertEqual(data["image_width"], 100)
            self.assertEqual(data["image_height"], 80)
            self.assertIn("imported_from", data)

    def test_import_skips_annotations_without_points(self) -> None:
        from digitizer.annotation_io import import_annotations_from_old_format
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_path = root / "plot.metadata.json"
            metadata_path.write_text(json.dumps({
                "image_width": 100,
                "image_height": 80,
                "annotations": [
                    {"type": "vbar", "x_pos": 0.5},          # old descriptor, no points
                    {"type": "x_anchor", "points": [(50, 70)]},  # valid
                ],
            }))
            out = import_annotations_from_old_format(metadata_path, root / "out")
            data = json.loads(out.read_text())
            # Only the annotation with points should be imported
            self.assertEqual(len(data["annotations"]), 1)
            self.assertEqual(data["annotations"][0]["type"], "x_anchor")

    def test_import_raises_when_no_annotations_found(self) -> None:
        from digitizer.annotation_io import import_annotations_from_old_format
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_path = root / "empty.metadata.json"
            metadata_path.write_text(json.dumps({"image_width": 100, "image_height": 80}))
            with self.assertRaises(ValueError):
                import_annotations_from_old_format(metadata_path, root / "out")

    def test_import_annotations_cli_subcommand(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_path = root / "plot.metadata.json"
            metadata_path.write_text(json.dumps({
                "image_width": 50, "image_height": 40,
                "annotations": [{"type": "vbar", "points": [(25, 0)]}],
            }))
            parser = build_parser()
            args = parser.parse_args([
                "import-annotations", str(metadata_path),
                "--output-dir", str(root / "train-dataset"),
            ])
            self.assertEqual(args.command, "import-annotations")
            self.assertEqual(args.source, metadata_path)

    def test_import_annotations_main_missing_metadata_sidecar_reports_cli_error(self) -> None:
        missing_source = "nonexistent_input_image.png"
        err = io.StringIO()
        with redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                digitizer.main(["import-annotations", missing_source])
        self.assertEqual(ctx.exception.code, 2)
        stderr = err.getvalue()
        self.assertIn(f"No metadata sidecar found for {missing_source}", stderr)
        self.assertIn("Provide a .metadata.json path", stderr)

    def test_import_annotations_main_missing_metadata_json_reports_cli_error(self) -> None:
        missing_metadata = "nonexistent_plot.metadata.json"
        err = io.StringIO()
        with redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                digitizer.main(["import-annotations", missing_metadata])
        self.assertEqual(ctx.exception.code, 2)
        stderr = err.getvalue()
        self.assertIn(f"Could not parse metadata file {missing_metadata}", stderr)


class CurriculumAndNewFeaturesTests(unittest.TestCase):
    """Tests for legend/axis-label extraction, colour inversion, and curriculum difficulty."""

    def _run_generate(self, tmp: str, **kwargs: object) -> Path:
        root = Path(tmp)
        solid_mask = np.zeros((120, 120), dtype=bool)
        solid_mask[20:100, 30:90] = True
        with (
            patch("digitizer._apply_degradation_filters", return_value=None),
            patch("digitizer._render_curve_mask", return_value=solid_mask),
            patch("digitizer._render_vbar_mask", return_value=solid_mask),
            patch("digitizer._render_hbar_mask", return_value=solid_mask),
            patch("digitizer._render_arrow_mask", return_value=solid_mask),
            patch("digitizer._render_error_bar_mask", return_value=solid_mask),
        ):
            digitizer.generate_synthetic_dataset(
                root / "out", image_format="png", workers=1, **kwargs
            )
        return root / "out"

    # ------------------------------------------------------------------
    # Legend annotation
    # ------------------------------------------------------------------

    def test_legend_annotation_present_at_difficulty_4(self) -> None:
        """Difficulty 4 always shows a legend; it must appear in annotations."""
        with tempfile.TemporaryDirectory() as tmp:
            out = self._run_generate(tmp, count=2, seed=5, plot_type="general", difficulty=4)
            ann_files = list((out / "annotations").glob("*.json"))
            self.assertGreater(len(ann_files), 0)
            legend_found = any(
                a["type"] == "legend"
                for af in ann_files
                for a in json.loads(af.read_text())["annotations"]
            )
            self.assertTrue(legend_found, "Expected at least one 'legend' annotation at difficulty=4")

    def test_legend_class_id_in_class_mapping(self) -> None:
        from digitizer.annotation_io import CLASS_MAPPING
        self.assertIn("legend", CLASS_MAPPING)
        self.assertEqual(CLASS_MAPPING["legend"], 12)

    def test_legend_annotation_to_yolo_line_produces_rectangle(self) -> None:
        from digitizer.annotation_io import annotation_to_yolo_line
        ann = {"type": "legend", "points": [(10.0, 5.0), (60.0, 40.0)]}
        line = annotation_to_yolo_line(ann, 100, 100)
        self.assertIsNotNone(line)
        assert line is not None
        parts = line.split()
        self.assertEqual(int(parts[0]), 12)
        # 4 corner pairs → 8 coord values + 1 class id = 9 parts total
        self.assertEqual(len(parts), 9)

    # ------------------------------------------------------------------
    # Axis label annotations (x_axis_label / y_axis_label)
    # ------------------------------------------------------------------

    def test_axis_label_classes_in_class_mapping(self) -> None:
        from digitizer.annotation_io import CLASS_MAPPING
        self.assertIn("x_axis_label", CLASS_MAPPING)
        self.assertIn("y_axis_label", CLASS_MAPPING)
        self.assertEqual(CLASS_MAPPING["x_axis_label"], 13)
        self.assertEqual(CLASS_MAPPING["y_axis_label"], 14)

    def test_axis_labels_appear_in_generated_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = self._run_generate(tmp, count=2, seed=11, plot_type="general")
            ann_files = list((out / "annotations").glob("*.json"))
            all_types = {
                a["type"]
                for af in ann_files
                for a in json.loads(af.read_text())["annotations"]
            }
            self.assertIn("x_axis_label", all_types)
            self.assertIn("y_axis_label", all_types)

    def test_axis_labels_have_text_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = self._run_generate(tmp, count=2, seed=22, plot_type="general")
            for af in (out / "annotations").glob("*.json"):
                data = json.loads(af.read_text())
                for ann in data["annotations"]:
                    if ann["type"] in ("x_axis_label", "y_axis_label"):
                        self.assertIn("text", ann)
                        self.assertTrue(ann["text"].strip())

    def test_axis_labels_appear_in_yolo_label_file(self) -> None:
        from digitizer.annotation_io import CLASS_MAPPING
        x_class = CLASS_MAPPING["x_axis_label"]
        y_class = CLASS_MAPPING["y_axis_label"]
        with tempfile.TemporaryDirectory() as tmp:
            out = self._run_generate(tmp, count=2, seed=33, plot_type="general")
            all_classes = {
                int(line.split()[0])
                for lf in (out / "labels").glob("*.txt")
                for line in lf.read_text().splitlines()
                if line.strip()
            }
            self.assertIn(x_class, all_classes)
            self.assertIn(y_class, all_classes)

    # ------------------------------------------------------------------
    # Tick labels: 2 per axis with scale_type field
    # ------------------------------------------------------------------

    def test_tick_labels_at_most_two_per_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = self._run_generate(tmp, count=4, seed=42, plot_type="general")
            for af in (out / "annotations").glob("*.json"):
                data = json.loads(af.read_text())
                x_ticks = [a for a in data["annotations"] if a["type"] == "x_tick_label"]
                y_ticks = [a for a in data["annotations"] if a["type"] == "y_tick_label"]
                self.assertLessEqual(len(x_ticks), 2, f"Too many x_tick_labels in {af.name}")
                self.assertLessEqual(len(y_ticks), 2, f"Too many y_tick_labels in {af.name}")

    def test_tick_labels_carry_scale_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = self._run_generate(tmp, count=4, seed=55, plot_type="general")
            found = False
            for af in (out / "annotations").glob("*.json"):
                for ann in json.loads(af.read_text())["annotations"]:
                    if ann["type"] in ("x_tick_label", "y_tick_label"):
                        self.assertIn("scale_type", ann, f"Missing scale_type in {ann}")
                        self.assertIn(ann["scale_type"], ("linear", "log"))
                        found = True
            self.assertTrue(found, "No tick label annotations found to check scale_type")

    # ------------------------------------------------------------------
    # Colour inversion degradation
    # ------------------------------------------------------------------

    def test_color_inversion_produces_inverted_image(self) -> None:
        """_apply_degradation_filters with intensity='heavy' and rng state that
        triggers only inversion should produce a pixel-inverted image."""
        import cv2 as _cv2
        from digitizer.synth_degrade import _apply_degradation_filters
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "test.png"
            original = np.full((50, 50, 3), 200, dtype=np.uint8)
            _cv2.imwrite(str(img_path), original)

            # At intensity="heavy" the order of rng.random() calls is:
            #   1: apply_jpeg   > 0.1
            #   2: apply_noise  > 0.2
            #   3: apply_blur   > 0.3
            #   4: apply_contrast > 0.3
            #   5: apply_bw     > 0.6
            #   6: apply_salt_pepper > 0.4
            #   7: apply_invert > (1 - COLOR_INVERT_PROBABILITY*2) ≈ 0.56
            # Returning 0.0 for calls 1–6 skips them; returning 1.0 for call 7
            # ensures inversion fires.
            call_count = [0]

            class ControlledRng:
                def random(self_) -> float:
                    call_count[0] += 1
                    return 1.0 if call_count[0] == 7 else 0.0

                def normal(self_, *a: object, **kw: object) -> np.ndarray:
                    return np.zeros((50, 50, 3))

                def uniform(self_, low: float, high: float) -> float:
                    return high

                def integers(self_, *a: object, **kw: object) -> int:
                    return 0

                def choice(self_, arr: object) -> object:
                    return arr[0]  # type: ignore[index]

            rng = ControlledRng()  # type: ignore[assignment]
            _apply_degradation_filters(img_path, rng, intensity="heavy")  # type: ignore[arg-type]
            result = _cv2.imread(str(img_path))
            self.assertIsNotNone(result)
            # All channels were 200 → after bitwise_not they should be 255-200 = 55.
            self.assertTrue(
                np.all(result == 255 - 200),
                f"Expected inverted pixels (55) but got mean {result.mean():.1f}",
            )

    def test_apply_degradation_filters_intensity_none_leaves_image_unchanged(self) -> None:
        """intensity='none' must not modify the image at all."""
        import cv2 as _cv2
        from digitizer.synth_degrade import _apply_degradation_filters
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "test.png"
            original = np.full((40, 40, 3), 128, dtype=np.uint8)
            _cv2.imwrite(str(img_path), original)
            original_bytes = img_path.read_bytes()
            rng = np.random.default_rng(0)
            _apply_degradation_filters(img_path, rng, intensity="none")
            self.assertEqual(img_path.read_bytes(), original_bytes)

    # ------------------------------------------------------------------
    # Curriculum difficulty
    # ------------------------------------------------------------------

    def test_curriculum_generates_all_four_difficulty_levels(self) -> None:
        """With --curriculum and count=8, sample indices 0,4 are difficulty=1 (easy),
        while indices 3,7 are difficulty=4 (hard with legend)."""
        with tempfile.TemporaryDirectory() as tmp:
            out = self._run_generate(tmp, count=8, seed=77, plot_type="general", curriculum=True)
            ann_files = sorted((out / "annotations").glob("*.json"))
            self.assertEqual(len(ann_files), 8)
            # Round-robin: index→difficulty: 0→1, 1→2, 2→3, 3→4, 4→1, 5→2, 6→3, 7→4
            easy_files = [ann_files[0], ann_files[4]]   # difficulty=1: no annotation layers
            hard_files = [ann_files[3], ann_files[7]]   # difficulty=4: legend always present
            annotation_layer_types = {"vbar", "hbar", "arrow", "error_bar"}
            types_easy = {
                a["type"]
                for af in easy_files
                for a in json.loads(af.read_text())["annotations"]
            }
            types_hard = {
                a["type"]
                for af in hard_files
                for a in json.loads(af.read_text())["annotations"]
            }
            self.assertTrue(
                types_easy.isdisjoint(annotation_layer_types),
                f"Easy samples (difficulty=1) should have no annotation layers but found: "
                f"{types_easy & annotation_layer_types}",
            )
            self.assertIn("legend", types_hard, "difficulty=4 samples should include a legend annotation")

    def test_difficulty_parser_accepts_valid_values(self) -> None:
        parser = build_parser()
        for level in (0, 1, 2, 3, 4):
            args = parser.parse_args(["generate", "--difficulty", str(level)])
            self.assertEqual(args.difficulty, level)

    def test_curriculum_parser_flag(self) -> None:
        parser = build_parser()
        args_off = parser.parse_args(["generate"])
        self.assertFalse(args_off.curriculum)
        args_on = parser.parse_args(["generate", "--curriculum"])
        self.assertTrue(args_on.curriculum)

    def test_difficulty_4_annotations_richer_than_difficulty_1(self) -> None:
        """Difficulty 4 should produce annotations beyond just curves and tick labels."""
        with tempfile.TemporaryDirectory() as tmp:
            out1 = self._run_generate(tmp + "/d1", count=4, seed=88, plot_type="general", difficulty=1)
            out4 = self._run_generate(tmp + "/d4", count=4, seed=88, plot_type="general", difficulty=4)

            ann_types_d1 = {
                a["type"]
                for af in (out1 / "annotations").glob("*.json")
                for a in json.loads(af.read_text())["annotations"]
            }
            ann_types_d4 = {
                a["type"]
                for af in (out4 / "annotations").glob("*.json")
                for a in json.loads(af.read_text())["annotations"]
            }
            # Difficulty 4 must contain legend; difficulty 1 must not.
            self.assertIn("legend", ann_types_d4)
            self.assertNotIn("vbar", ann_types_d1)
            self.assertNotIn("hbar", ann_types_d1)
            self.assertNotIn("arrow", ann_types_d1)


if __name__ == "__main__":
    unittest.main()
