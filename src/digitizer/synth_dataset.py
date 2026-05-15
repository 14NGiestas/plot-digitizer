"""Synthetic dataset task orchestration helpers."""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import DEFAULT_GENERATE_WORKERS_CAP
from .synth_example import _write_synthetic_example

@dataclass(frozen=True, slots=True)
class SampleGenerationTask:
    """Parameters for generating one synthetic sample."""

    index: int
    output_dir: Path
    child_seed: np.random.SeedSequence
    image_format: str
    plot_type: str


def _generate_one_sample(args: SampleGenerationTask) -> None:
    """Worker function for parallel synthetic sample generation.

    Accepts a :class:`SampleGenerationTask` dataclass instance so it can be
    passed through :func:`ProcessPoolExecutor.map` with explicit named fields.
    """
    rng = np.random.default_rng(args.child_seed)
    _write_synthetic_example(args.index, args.output_dir, rng, args.image_format, args.plot_type)


def generate_synthetic_dataset(
    output_dir: Path,
    count: int,
    seed: int,
    image_format: str,
    plot_type: str = "mixed",
    workers: int | None = None,
) -> None:
    """Generate a synthetic plot dataset with YOLO segmentation labels.

    Samples are generated in parallel using :class:`ProcessPoolExecutor` by
    default (one process per CPU core).  Each sample receives an independent
    :class:`numpy.random.SeedSequence` child so results are fully deterministic
    and identical regardless of the number of workers used.

    Args:
        output_dir: Output directory for the dataset.
        count: Number of images to generate.
        seed: Random seed for reproducibility.
        image_format: Image format (``"png"`` or ``"jpg"``).
        plot_type: Type of plots to generate – ``"general"``,
            ``"bandstructure"``, or ``"mixed"``.
        workers: Number of worker processes.  ``None`` (default) uses
            ``min(os.cpu_count(), count, 8)``. Pass ``1`` for strictly
            sequential execution (useful for debugging). The cap keeps
            process and memory overhead reasonable on high-core systems.
    """
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("images", "labels", "ground_truth"):
        (output_dir / subdir).mkdir(exist_ok=True)

    # Derive independent child seeds from the master SeedSequence.
    # children[0] drives plot-type assignment; children[1:] drive per-sample rngs.
    ss = np.random.SeedSequence(seed)
    all_children = ss.spawn(count + 1)
    type_rng = np.random.default_rng(all_children[0])
    sample_seeds = all_children[1:]

    plot_types: list[str] = [
        type_rng.choice(["general", "bandstructure"]) if plot_type == "mixed" else plot_type
        for _ in range(count)
    ]

    tasks: list[SampleGenerationTask] = [
        SampleGenerationTask(
            index=i,
            output_dir=output_dir,
            child_seed=sample_seeds[i],
            image_format=image_format,
            plot_type=plot_types[i],
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

    # Multi-class segmentation labels must stay contiguous from 0..nc-1
    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: images",
                "val: images",
                "test: images",
                "nc: 5",
                "names:",
                "  0: curve",
                "  1: vbar",
                "  2: hbar", 
                "  3: arrow",
                "  4: error_bar",
            ]
        )
    )

