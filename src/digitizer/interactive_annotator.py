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

import json
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
    "paint_mask": "magenta",
}
_MIN_ZOOM_HALF_SIZE = 30.0
_ZOOM_HALF_SIZE_SCALE = 0.075
_ANNOTATOR_FIGSIZE = (13, 7)
_ANNOTATOR_WIDTH_RATIOS = [3.2, 1.4]
_KEYMAP_OVERRIDES = {key: [] for key in mpl.rcParams if key.startswith("keymap.")}
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
    "m": "paint_mask",
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
    "paint_mask": None,
}
_HELP = (
    "1/v=vbar 2/h=hbar 3/a=arrow 4/c=curve 5/e=err_bar 6/p=plot_area 7/x=x_axis 8/y=y_axis 9=x_anchor 0=y_anchor m=paint | "
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
            figsize=_ANNOTATOR_FIGSIZE,
            gridspec_kw={"width_ratios": _ANNOTATOR_WIDTH_RATIOS},
        )
        self._fig.subplots_adjust(bottom=0.06)
        self._fig.text(0.5, 0.01, _HELP, ha="center", fontsize=8, color="dimgray")

    # ------------------------------------------------------------------ helpers

    def _clamp(self, x: float, y: float) -> tuple[float, float]:
        return float(np.clip(x, 0, self._w - 1)), float(np.clip(y, 0, self._h - 1))

    def _extract_painted_curve(self, radius: float = 10.0) -> list[tuple[float, float]]:
        if len(self._current) < 2:
            return []
        import cv2
        from .cv_segmentation import _foreground_mask, _saturated_mask
        
        paint_mask = np.zeros((self._h, self._w), dtype=np.uint8)
        pts = np.array(self._current, dtype=np.int32)
        for i in range(len(pts) - 1):
            cv2.line(paint_mask, tuple(pts[i]), tuple(pts[i+1]), 255, int(radius * 2))
            
        ys, xs = np.nonzero(paint_mask)
        if len(xs) == 0: return []
        left, right = max(0, xs.min() - 5), min(self._w, xs.max() + 5)
        top, bottom = max(0, ys.min() - 5), min(self._h, ys.max() + 5)
        
        crop = self._image[top:bottom, left:right]
        fg = _saturated_mask(crop)
        if fg.sum() < 10:
            fg = _foreground_mask(crop)
            
        fg_global = np.zeros((self._h, self._w), dtype=bool)
        fg_global[top:bottom, left:right] = fg
        final_mask = fg_global & (paint_mask > 0)
        
        fys, fxs = np.nonzero(final_mask)
        if len(fxs) == 0: return []
        
        unique_x = np.unique(fxs)
        extracted = []
        for x in unique_x:
            y = fys[fxs == x].mean()
            extracted.append((float(x), float(y)))
            
        if len(extracted) > 150:
            step = max(1, len(extracted) // 150)
            extracted = extracted[::step]
            
        return extracted

    def _commit_current(self) -> None:
        if self._mode == "paint_mask":
            extracted = self._extract_painted_curve()
            if extracted:
                self._committed.append({
                    "type": "curve",
                    "points": extracted,
                    "line_width": self._line_width,
                })
                LOGGER.debug("Committed extracted curve (%d pts)", len(extracted))
            self._current = []
            return

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
        pts = ann.get("points", [])
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
        elif not pts:
            LOGGER.debug("Skipping annotation of type %r with no points in display", t)

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
        if event.inaxes is not self._ax or event.xdata is None or event.ydata is None:
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
        if self._mode == "paint_mask" and getattr(event, 'button', None) == 1:
            if not self._current or np.hypot(x - self._current[-1][0], y - self._current[-1][1]) > 5:
                self._current.append((x, y))
                self._redraw()
                return
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
        with mpl.rc_context(rc=_KEYMAP_OVERRIDES):
            plt.show()
        return list(self._committed) if self._do_save else []


def interactive_annotation_session(
    image_path: Path,
    output_dir: Path,
    line_width: float = 3.0,
    resize_to: tuple[int, int] | None = None,
    update_existing: bool = False,  # Deprecated; existing annotations are always loaded automatically.
) -> dict[str, str]:
    """Annotate *image_path* interactively and save a training sample.

    Opens a matplotlib window for the user to draw vbars, hbars, arrows,
    curves, and error bars.  Existing annotations for the image are loaded
    automatically — from ``output_dir/annotations/<stem>.json`` if present,
    otherwise from a sibling ``annotations/`` directory or an older metadata
    sidecar — so the session always resumes from the last saved state.

    On save (Enter), writes the image copy, YOLO label file, annotations file,
    and metadata sidecar to *output_dir*.

    Args:
        update_existing: **Deprecated.** Existing annotations are always
            loaded automatically; this argument has no effect and is retained
            only for backward compatibility with older call sites.

    Returns the paths dict from :func:`~digitizer.annotation_io.save_training_sample`,
    or an empty dict when the user cancels.
    """
    image_bgr = load_image(image_path)
    import cv2
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    # Always try to load existing annotations so the session resumes from the
    # last saved state regardless of whether --update was passed.
    initial_annotations = load_training_sample_annotations(
        image_path=image_path,
        output_dir=output_dir,
        target_size=(w, h),
    )
    if initial_annotations:
        LOGGER.info("Loaded %d existing annotation(s).", len(initial_annotations))

    # Auto-detect an associated CSV (ground truth or digitized data) to
    # preserve alongside the new annotations in output_dir/csv/.
    csv_src: Path | None = _find_associated_csv(image_path, output_dir)

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
        csv_path=csv_src,
    )
    LOGGER.info(
        "Saved %d annotation(s) → %s", len(annotations), result["label_path"]
    )
    return result


def _find_associated_csv(image_path: Path, output_dir: Path) -> Path | None:
    """Return an associated CSV file for *image_path*, or ``None`` if not found.

    Search order:
    1. ``csv_path`` / ``ground_truth_csv`` recorded in any nearby metadata sidecar.
    2. Conventional ``csv/`` sibling directory of *output_dir*.
    3. Conventional ``csv/`` sibling directory one level above *image_path*.
    """
    stem = image_path.stem

    # 1. Read metadata sidecars (output_dir primary, then image sibling).
    for meta_candidate in (
        output_dir / "images" / f"{stem}.metadata.json",
        image_path.parent / f"{stem}.metadata.json",
    ):
        if not meta_candidate.exists():
            continue
        try:
            meta = json.loads(meta_candidate.read_text())
        except Exception:
            continue
        for key in ("csv_path", "ground_truth_csv"):
            ref = meta.get(key)
            if ref:
                p = Path(ref)
                if p.exists():
                    return p
        break  # only inspect the first found sidecar

    # 2. output_dir/csv/<stem>.csv
    candidate = output_dir / "csv" / f"{stem}.csv"
    if candidate.exists():
        return candidate

    # 3. <image_parent>/../csv/<stem>.csv  (handles images inside images/ subdir)
    candidate = image_path.parent.parent / "csv" / f"{stem}.csv"
    if candidate.exists():
        return candidate

    return None
