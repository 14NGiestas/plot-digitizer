"""Synthetic dataset task orchestration helpers."""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..annotation_io import CLASS_MAPPING
from ..constants import DEFAULT_GENERATE_WORKERS_CAP
from .example import _write_synthetic_example

_NAMES_BY_ID = {class_id: name for name, class_id in CLASS_MAPPING.items()}
if sorted(_NAMES_BY_ID.keys()) != list(range(len(_NAMES_BY_ID))):
    raise ValueError("CLASS_MAPPING must be contiguous from 0..nc-1")


@dataclass(frozen=True, slots=True)
class SampleGenerationTask:
    """Container for one synthetic sample's parallel generation parameters."""

    index: int
    output_dir: Path
    child_seed: np.random.SeedSequence
    image_format: str
    plot_type: str
    degradations: int = 1
    difficulty: int = 0


def _generate_one_sample(args: SampleGenerationTask) -> None:
    """Worker function for parallel synthetic sample generation."""
    rng = np.random.default_rng(args.child_seed)
    _write_synthetic_example(args.index, args.output_dir, rng, args.image_format, args.plot_type, args.degradations, difficulty=args.difficulty)


def generate_synthetic_dataset(
    output_dir: Path,
    count: int,
    seed: int,
    image_format: str,
    plot_type: str = "mixed",
    workers: int | None = None,
    degradations: int = 1,
    difficulty: int = 0,
    curriculum: bool = False,
) -> None:
    """Generate a synthetic plot dataset with YOLO segmentation labels.

    Samples are generated in parallel using :class:`ProcessPoolExecutor` by
    default (one process per CPU core).  Each sample receives an independent
    :class:`numpy.random.SeedSequence` child so results are fully deterministic
    and identical regardless of the number of workers used.

    Args:
        output_dir: Output directory for the dataset.
        count: Number of **base** plots to generate.  When *degradations* > 1
            the total number of training images is ``count × degradations``.
        seed: Random seed for reproducibility.
        image_format: Image format (``"png"`` or ``"jpg"``).
        plot_type: Type of plots to generate – ``"general"``,
            ``"bandstructure"``, or ``"mixed"``.
        workers: Number of worker processes.  ``None`` (default) uses
            ``min(os.cpu_count(), count, 8)``. Pass ``1`` for strictly
            sequential execution (useful for debugging). The cap keeps
            process and memory overhead reasonable on high-core systems.
        degradations: Number of independently degraded image variants to
            produce per base plot.  All variants share the same YOLO labels,
            annotations, and ground-truth CSV.  Defaults to ``1``.
        difficulty: Curriculum difficulty level.  ``0`` (default) disables
            per-level restrictions (full-complexity plots, backward-compatible).
            Pass ``1``–``4`` to fix all samples at one difficulty level.
        curriculum: When ``True``, override *difficulty* and distribute samples
            evenly across levels 1–4 in round-robin order (1,2,3,4,1,2,3,4,…).
            Ignored when *difficulty* != 0.
    """
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if degradations < 1:
        raise ValueError(f"degradations must be >= 1, got {degradations}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("images", "labels", "csv", "annotations"):
        (output_dir / subdir).mkdir(exist_ok=True)

    ss = np.random.SeedSequence(seed)
    all_children = ss.spawn(count + 1)
    type_rng = np.random.default_rng(all_children[0])
    sample_seeds = all_children[1:]

    plot_types: list[str] = [
        type_rng.choice(["general", "bandstructure"]) if plot_type == "mixed" else plot_type
        for _ in range(count)
    ]

    # difficulty=0 → no restrictions (backward compat).
    # curriculum=True → override with round-robin 1→2→3→4 across samples.
    if curriculum:
        difficulties: list[int] = [((i % 4) + 1) for i in range(count)]
    else:
        difficulties = [difficulty] * count

    tasks: list[SampleGenerationTask] = [
        SampleGenerationTask(
            index=i,
            output_dir=output_dir,
            child_seed=sample_seeds[i],
            image_format=image_format,
            plot_type=plot_types[i],
            degradations=degradations,
            difficulty=difficulties[i],
        )
        for i in range(count)
    ]

    cpu_count = os.cpu_count() or 1
    n_workers = workers if workers is not None else min(cpu_count, count, DEFAULT_GENERATE_WORKERS_CAP)
    if n_workers <= 1:
        for task in tasks:
            _generate_one_sample(task)
    else:
        mp_ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as executor:
            # Consume the iterator to propagate any worker exceptions.
            for _ in executor.map(_generate_one_sample, tasks):
                pass

    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: images",
                "val: images",
                "test: images",
                f"nc: {len(_NAMES_BY_ID)}",
                "names:",
                *[f"  {index}: {_NAMES_BY_ID[index]}" for index in range(len(_NAMES_BY_ID))],
            ]
        )
    )
