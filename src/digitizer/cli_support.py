"""Logging and matplotlib backend helpers."""

from __future__ import annotations

import logging

import matplotlib

_MATPLOTLIB_BACKEND_SET = False


def _set_matplotlib_backend(backend: str) -> None:
    """Set matplotlib backend before importing pyplot."""
    global _MATPLOTLIB_BACKEND_SET
    if not _MATPLOTLIB_BACKEND_SET:
        matplotlib.use(backend)
        _MATPLOTLIB_BACKEND_SET = True


def configure_logging(verbose: bool) -> None:
    """Configure structured logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
