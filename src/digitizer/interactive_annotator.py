"""Interactive matplotlib annotation session.

Lets the user paint YOLO training annotations (curves, bars, arrows, frame and
axis elements) directly on top of a real plot image. Keyboard shortcuts:

    1 / v  →  vbar     (one click, auto-commits)
    2 / h  →  hbar     (one click, auto-commits)
    3 / a  →  arrow    (two clicks, auto-commits)
    4 / c  →  curve    (many clicks, press F to commit segment)
    5 / e  →  error_bar (two clicks: top-cap then bottom-cap, auto-commits)
    6 / p  →  plot_area (two clicks: top-left & bottom-right, auto-commits)
    7 / x  →  x_axis    (two clicks, auto-commits)
    8 / y  →  y_axis    (two clicks, auto-commits)
    9      →  x_anchor  (one click, auto-commits)
    0      →  y_anchor  (one click, auto-commits)
    z      →  undo last committed annotation
    Enter  →  save annotations and close
    Esc    →  close without saving
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from .annotation_io import load_training_sample_annotations, save_training_sample
from .constants import LOGGER
from .image_ops import load_image

_MODE_COLORS: dict[str, str] = {
    "vbar": "mediumpurple",
    "hbar": "darkorange",
    "arrow": "crimson",
    "curve": "royalblue",
    "error_bar": "forestgreen",
    "plot_area": "black",
    "x_axis": "saddlebrown",
    "y_axis": "teal",
    "x_anchor": "goldenrod",
    "y_anchor": "mediumseagreen",
}
_MIN_ZOOM_HALF_SIZE = 30.0
_ZOOM_HALF_SIZE_SCALE = 0.075
_KEY_TO_MODE: dict[str, str] = {
    "1": "vbar", "v": "vbar",
    "2": "hbar", "h": "hbar",
    "3": "arrow", "a": "arrow",
    "4": "curve", "c": "curve",
    "5": "error_bar", "e": "error_bar",
    "6": "plot_area", "p": "plot_area",
    "7": "x_axis", "x": "x_axis",
    "8": "y_axis", "y": "y_axis",
    "9": "x_anchor",
    "0": "y_anchor",
}
# None means variable-length (curve); commit with F key.
_POINTS_NEEDED: dict[str, int | None] = {
    "vbar": 1,
    "hbar": 1,
    "arrow": 2,
    "curve": None,
    "error_bar": 2,
    "plot_area": 2,
    "x_axis": 2,
    "y_axis": 2,
    "x_anchor": 1,
    "y_anchor": 1,
}
_HELP = (
    "1/v=vbar 2/h=hbar 3/a=arrow 4/c=curve 5/e=err_bar 6/p=plot_area 7/x=x_axis 8/y=y_axis 9=x_anchor 0=y_anchor | "
    "F=commit curve | Z=undo | Enter=save | Esc=cancel"
)


class _AnnotatorSession:
    """Stateful matplotlib annotation session."""

    def __init__(
        self,
        image: np.ndarray,
        image_width: int,
        image_height: int,
        line_width: float,
        initial_annotations: list[dict[str, Any]] | None = None,
    ) -> None:
        self._image = image
        self._w = image_width
        self._h = image_height
        self._line_width = line_width
        self._mode: str = "curve"
        self._current: list[tuple[float, float]] = []
        self._committed: list[dict[str, Any]] = list(initial_annotations or [])
        self._do_save = False
        self._active_point: tuple[float, float] | None = None
        self._zoom_half_size = max(_MIN_ZOOM_HALF_SIZE, max(self._w, self._h) * _ZOOM_HALF_SIZE_SCALE)
        self._fig, (self._ax, self._zoom_ax) = plt.subplots(
            ncols=2,
            figsize=(13, 7),
            gridspec_kw={"width_ratios": [3.2, 1.4]},
        )
        self._fig.subplots_adjust(bottom=0.06)
        self._fig.text(0.5, 0.01, _HELP, ha="center", fontsize=8, color="dimgray")

    # ------------------------------------------------------------------ helpers

    def _clamp(self, x: float, y: float) -> tuple[float, float]:
        return float(np.clip(x, 0, self._w - 1)), float(np.clip(y, 0, self._h - 1))

    def _commit_current(self) -> None:
        needed = _POINTS_NEEDED[self._mode]
        min_pts = 2 if needed is None else needed
        if len(self._current) >= min_pts:
            self._committed.append({
                "type": self._mode,
                "points": list(self._current),
                "line_width": self._line_width,
            })
            LOGGER.debug("Committed %s annotation (%d pts)", self._mode, len(self._current))
        self._current = []

    # ------------------------------------------------------------------ drawing

    def _draw_annotation(self, ann: dict[str, Any], alpha: float = 0.75) -> None:
        t = ann["type"]
        color = _MODE_COLORS[t]
        pts = ann["points"]
        lw = 2.0
        if t == "vbar" and pts:
            self._ax.axvline(pts[0][0], color=color, alpha=alpha, linewidth=lw)
        elif t == "hbar" and pts:
            self._ax.axhline(pts[0][1], color=color, alpha=alpha, linewidth=lw)
        elif t == "arrow" and len(pts) >= 2:
            self._ax.annotate(
                "", xy=pts[1], xytext=pts[0],
                arrowprops={"arrowstyle": "->", "color": color, "lw": lw},
            )
        elif t == "plot_area" and len(pts) >= 2:
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            self._ax.plot(
                [left, right, right, left, left],
                [top, top, bottom, bottom, top],
                color=color,
                alpha=alpha,
                linewidth=lw,
            )
        elif t in ("x_axis", "y_axis") and len(pts) >= 2:
            self._ax.plot(
                [pts[0][0], pts[1][0]],
                [pts[0][1], pts[1][1]],
                color=color,
                alpha=alpha,
                linewidth=lw,
            )
            self._ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], "o", color=color, alpha=alpha, markersize=5)
        elif t in ("x_anchor", "y_anchor") and pts:
            self._ax.plot(pts[0][0], pts[0][1], marker="o", color=color, alpha=alpha, markersize=8)
        elif t in ("curve", "error_bar") and len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            self._ax.plot(xs, ys, color=color, alpha=alpha, linewidth=lw)
            self._ax.plot(xs, ys, "o", color=color, alpha=alpha, markersize=5)

    def _draw_current(self) -> None:
        if not self._current:
            return
        color = _MODE_COLORS[self._mode]
        xs = [p[0] for p in self._current]
        ys = [p[1] for p in self._current]
        self._ax.plot(xs, ys, "o--", color=color, alpha=0.9, markersize=8, linewidth=1.5)

    def _refresh_zoom(self) -> None:
        self._zoom_ax.clear()
        self._zoom_ax.imshow(self._image)
        self._zoom_ax.axis("off")
        if self._active_point is None:
            self._zoom_ax.set_title("Zoom (hover/click on plot)", fontsize=10)
            return

        x_coord, y_coord = self._active_point
        x_min = max(0.0, x_coord - self._zoom_half_size)
        x_max = min(self._w, x_coord + self._zoom_half_size)
        y_min = max(0.0, y_coord - self._zoom_half_size)
        y_max = min(self._h, y_coord + self._zoom_half_size)
        self._zoom_ax.set_xlim(x_min, x_max)
        self._zoom_ax.set_ylim(y_max, y_min)
        self._zoom_ax.axvline(x_coord, color="yellow", linestyle="--", linewidth=1.0)
        self._zoom_ax.axhline(y_coord, color="yellow", linestyle="--", linewidth=1.0)
        self._zoom_ax.set_title(f"Zoom ({x_coord:.1f}, {y_coord:.1f})", fontsize=10)

    def _redraw(self) -> None:
        self._ax.clear()
        self._ax.imshow(self._image)
        self._ax.axis("off")
        for ann in self._committed:
            self._draw_annotation(ann)
        self._draw_current()
        needed = _POINTS_NEEDED[self._mode]
        if needed is None:
            pts_hint = f"({len(self._current)} pts; F=commit)"
        else:
            pts_hint = f"({len(self._current)}/{needed} pts)"
        self._ax.set_title(
            f"Mode: {self._mode.upper()} {pts_hint}  |  "
            f"{len(self._committed)} annotation(s) committed",
            fontsize=10,
        )
        self._refresh_zoom()
        self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------ events

    def _on_click(self, event: Any) -> None:
        if event.inaxes is not self._ax or event.xdata is None:
            return
        x, y = self._clamp(float(event.xdata), float(event.ydata))
        if event.button == 1:
            self._current.append((x, y))
            self._active_point = (x, y)
            needed = _POINTS_NEEDED[self._mode]
            if needed is not None and len(self._current) >= needed:
                self._commit_current()
            self._redraw()
        elif event.button == 3 and self._current:
            self._current.pop()
            self._active_point = (x, y)
            self._redraw()

    def _on_motion(self, event: Any) -> None:
        if event.inaxes is not self._ax or event.xdata is None or event.ydata is None:
            return
        x, y = self._clamp(float(event.xdata), float(event.ydata))
        self._active_point = (x, y)
        self._refresh_zoom()
        self._fig.canvas.draw_idle()

    def _on_key(self, event: Any) -> None:
        key = event.key or ""
        if key in _KEY_TO_MODE:
            self._commit_current()
            self._mode = _KEY_TO_MODE[key]
            self._redraw()
        elif key == "f":
            self._commit_current()
            self._redraw()
        elif key == "z":
            if self._committed:
                self._committed.pop()
            elif self._current:
                self._current.pop()
            self._redraw()
        elif key == "enter":
            self._commit_current()
            self._do_save = True
            plt.close(self._fig)
        elif key == "escape":
            plt.close(self._fig)

    # ------------------------------------------------------------------ public

    def run(self) -> list[dict[str, Any]]:
        """Show the annotation window; return committed annotations (empty on cancel)."""
        self._redraw()
        self._fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)
        # Disable default matplotlib navigation shortcuts so they do not overlap
        # with annotation keys such as c/p/x/y and number mappings.
        keymap_overrides = {
            key: [] for key in mpl.rcParams if key.startswith("keymap.")
        }
        with mpl.rc_context(rc=keymap_overrides):
            plt.show()
        return list(self._committed) if self._do_save else []


def interactive_annotation_session(
    image_path: Path,
    output_dir: Path,
    line_width: float = 3.0,
    resize_to: tuple[int, int] | None = None,
    update_existing: bool = False,
) -> dict[str, str]:
    """Annotate *image_path* interactively and save a training sample.

    Opens a matplotlib window for the user to draw vbars, hbars, arrows,
    curves, and error bars.  On save (Enter), writes the image copy, YOLO
    label file, and metadata sidecar to *output_dir*.

    Returns the paths dict from :func:`~digitizer.annotation_io.save_training_sample`,
    or an empty dict when the user cancels.
    """
    image_bgr = load_image(image_path)
    import cv2
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    initial_annotations: list[dict[str, Any]] = []
    if update_existing:
        initial_annotations = load_training_sample_annotations(
            image_path=image_path,
            output_dir=output_dir,
            target_size=(w, h),
        )
        LOGGER.info("Loaded %d existing annotation(s) for update mode.", len(initial_annotations))
    session = _AnnotatorSession(image_rgb, w, h, line_width, initial_annotations=initial_annotations)
    annotations = session.run()
    if not annotations:
        LOGGER.info("Annotation session cancelled — nothing saved.")
        return {}
    result = save_training_sample(
        image_path=image_path,
        annotations=annotations,
        output_dir=output_dir,
        default_line_width=line_width,
        resize_to=resize_to,
    )
    LOGGER.info(
        "Saved %d annotation(s) → %s", len(annotations), result["label_path"]
    )
    return result
