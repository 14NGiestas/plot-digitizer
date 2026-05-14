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

            for subdir in ("images", "labels", "ground_truth"):
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
                    weights="yolov8n-seg.pt",
                    batch=1,
                    execute=True,
                    workers=16,
                )

            self.assertEqual(calls["num_threads"], 16)
            self.assertEqual(calls["num_interop_threads"], 16)
            self.assertEqual(calls["train_kwargs"]["workers"], 16)
            self.assertEqual(plan["workers"], 16)

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

    def test_write_synthetic_example_draws_vbar_hbar_and_error_bar_on_main_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for subdir in ("images", "labels", "ground_truth"):
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
