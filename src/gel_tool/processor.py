"""
Gel image processing engine.

Faithfully replicates the pipeline from reference/gel_conversion.ipynb.
"""

from __future__ import annotations

import cv2
import numpy as np


class GelResult:
    """Container for all processed outputs of a single gel image."""

    def __init__(
        self,
        raw_bgr: np.ndarray,
        crop_bgr: np.ndarray,
        green: np.ndarray,
        boosted: np.ndarray,
        residual_u8: np.ndarray,
        gain: float,
    ):
        self.raw_bgr = raw_bgr
        self.crop_bgr = crop_bgr
        self.green = green
        self.boosted = boosted
        self.residual_u8 = residual_u8
        self.gain = gain


def _auto_crop_gel(raw_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect the orange gel tray and crop tightly around the gel region.

    Returns (crop_bgr, green_channel).
    """
    hsv = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV)
    orange_mask = cv2.inRange(hsv, (5, 30, 60), (35, 255, 255))
    k_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, k_morph)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, k_morph)

    contours, _ = cv2.findContours(
        orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return raw_bgr, raw_bgr[:, :, 1]

    bx, by, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))

    mx, my = int(bw * 0.04), int(bh * 0.04)
    cx0, cy0 = bx + mx, by + my
    cw0, ch0 = bw - 2 * mx, bh - 2 * my

    crop0_bgr = raw_bgr[cy0 : cy0 + ch0, cx0 : cx0 + cw0]

    g0 = cv2.cvtColor(crop0_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(g0, (0, 0), 1.2)
    sx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.abs(sx) + np.abs(sy)

    row_energy = edge.mean(axis=1)
    col_energy = edge.mean(axis=0)
    row_energy = cv2.GaussianBlur(row_energy[:, None], (1, 51), 0).ravel()
    col_energy = cv2.GaussianBlur(col_energy[None, :], (51, 1), 0).ravel()

    rt = np.percentile(row_energy, 80)
    ct = np.percentile(col_energy, 80)
    rows = np.where(row_energy >= rt)[0]
    cols = np.where(col_energy >= ct)[0]

    rows = rows[rows < int(ch0 * 0.60)]

    if len(rows) == 0 or len(cols) == 0:
        x1, y1, x2, y2 = 0, 0, cw0, ch0
    else:
        y1, y2 = int(rows.min()), int(rows.max() + 1)
        x1, x2 = int(cols.min()), int(cols.max() + 1)

    pad_x = int((x2 - x1) * 0.015)
    pad_y = int((y2 - y1) * 0.02)
    x1 = max(0, x1 - pad_x)
    x2 = min(cw0, x2 + pad_x)

    TOP_TRIM_FRAC = 0.09
    min_y1 = int(ch0 * TOP_TRIM_FRAC)
    y1 = max(min_y1, y1 - pad_y)
    y2 = min(ch0, y2 + pad_y)

    crop_bgr = crop0_bgr[y1:y2, x1:x2]
    green = crop_bgr[:, :, 1]

    return crop_bgr, green


def _band_boost(green: np.ndarray, gain: float = 4.0) -> np.ndarray:
    """Green channel with band-boost: smooth background + amplified positive residual."""
    img = green.astype(np.float32)
    baseline = cv2.GaussianBlur(img, (1, 301), 0)
    residual = img - baseline
    band = cv2.GaussianBlur(residual, (31, 9), 0)
    band_pos = np.clip(band, 0, None)
    boosted = np.clip(img + gain * band_pos, 0, 255).astype(np.uint8)
    return boosted


def _residual(green: np.ndarray) -> np.ndarray:
    """Column-detrended, H-smoothed residual converted to 8-bit."""
    g = green.astype(np.float32)
    baseline = cv2.GaussianBlur(g, (1, 301), 0)
    residual = g - baseline
    residual_smooth = cv2.GaussianBlur(residual, (31, 9), 0)

    vmin, vmax = np.percentile(residual_smooth, (1, 99))
    if vmax - vmin < 1e-6:
        return np.zeros_like(green, dtype=np.uint8)
    result = np.clip((residual_smooth - vmin) / (vmax - vmin) * 255, 0, 255)
    return result.astype(np.uint8)


def process_gel(path: str, gain: float = 4.0) -> GelResult:
    """Full pipeline: load image, crop gel, compute all three output modes."""
    raw_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if raw_bgr is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    crop_bgr, green = _auto_crop_gel(raw_bgr)
    boosted = _band_boost(green, gain=gain)
    residual_u8 = _residual(green)

    return GelResult(
        raw_bgr=raw_bgr,
        crop_bgr=crop_bgr,
        green=green,
        boosted=boosted,
        residual_u8=residual_u8,
        gain=gain,
    )
