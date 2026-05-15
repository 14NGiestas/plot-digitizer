"""Validation helpers for digitized output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .constants import VALIDATION_THRESHOLD
from .curve_utils import _interp_curve

def validate_digitization(prediction_csv: Path, truth_csv: Path, output_json: Path | None = None) -> dict[str, Any]:
    """Compare digitized results against ground truth curves."""
    predicted = pd.read_csv(prediction_csv)
    truth = pd.read_csv(truth_csv)
    predicted_groups = list(predicted.groupby("dataset_id"))
    truth_groups = list(truth.groupby("dataset_id"))

    if not predicted_groups or not truth_groups:
        raise ValueError("Validation requires at least one predicted and one truth dataset.")

    truth_ranges = {
        dataset_id: max(1e-6, float(np.ptp(group["y_real"])))
        for dataset_id, group in truth_groups
    }

    truth_ids = [dataset_id for dataset_id, _ in truth_groups]
    predicted_ids = [dataset_id for dataset_id, _ in predicted_groups]
    assignment_matrix_size = max(len(truth_groups), len(predicted_groups))
    cost_matrix = np.full((len(truth_groups), assignment_matrix_size), np.inf, dtype=float)

    for truth_index, (truth_id, truth_frame) in enumerate(truth_groups):
        reference_x = truth_frame["x_real"].to_numpy()
        truth_y = truth_frame["y_real"].to_numpy()
        for predicted_index, (_, predicted_frame) in enumerate(predicted_groups):
            aligned = _interp_curve(predicted_frame, reference_x)
            cost_matrix[truth_index, predicted_index] = float(np.mean(np.abs(aligned - truth_y)))
        # Only dummy prediction columns are needed: every truth curve must be assigned,
        # while extra predicted curves can remain unused in the rectangular cost matrix.
        for dummy_index in range(len(predicted_groups), assignment_matrix_size):
            cost_matrix[truth_index, dummy_index] = truth_ranges[truth_id]

    truth_assignment, predicted_assignment = linear_sum_assignment(cost_matrix)
    metrics: list[dict[str, Any]] = []
    total_error: list[float] = []
    for truth_index, assigned_index in zip(truth_assignment.tolist(), predicted_assignment.tolist(), strict=True):
        truth_id = truth_ids[truth_index]
        predicted_id = predicted_ids[assigned_index] if assigned_index < len(predicted_ids) else None
        mae = float(cost_matrix[truth_index, assigned_index])
        metrics.append(
            {
                "truth_dataset_id": truth_id,
                "predicted_dataset_id": predicted_id,
                "mae": mae,
            }
        )
        total_error.append(mae)

    summary = {
        "mean_absolute_error": float(np.mean(total_error)),
        "mean_absolute_percentage_error_proxy": float(
            np.mean([row["mae"] / truth_ranges[row["truth_dataset_id"]] for row in metrics])
            * 100.0
        ),
        "per_curve": metrics,
        "passed_under_5_percent": bool(
            np.mean([row["mae"] / truth_ranges[row["truth_dataset_id"]] for row in metrics])
            < VALIDATION_THRESHOLD
        ),
    }
    if output_json is not None:
        output_json.write_text(json.dumps(summary, indent=2))
    return summary

