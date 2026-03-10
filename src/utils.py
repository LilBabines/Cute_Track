from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6 import QtGui

import warnings


def ensure_bgr_u8(img: np.ndarray) -> np.ndarray:
    """Convert an image (8/16-bit, mono/RGBA) to BGR uint8 for display and processing.
       - 16-bit → scaled to 0..255 (min-max normalization)
       - 1 channel → BGR
       - 4 channels (BGRA) → BGR
    """
    if img is None:
        return img

    # 16-bit → 8-bit via min-max scaling
    if img.dtype == np.uint16:
        i_min, i_max = int(img.min()), int(img.max())
        if i_max > i_min:
            img8 = ((img - i_min) * 255.0 / (i_max - i_min)).astype(np.uint8)
        else:
            img8 = (img / 256).astype(np.uint8)
        img = img8
    elif img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)

    # Convert grayscale or BGRA to BGR
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def cvimg_to_qimage(img_bgr: np.ndarray) -> QtGui.QImage:
    """Convert a BGR numpy array to a QImage (RGB888 format)."""
    if img_bgr is None:
        return QtGui.QImage()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    return QtGui.QImage(img_rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)


# ---------------------------------------------------------------------------
# Annotation data classes
# ---------------------------------------------------------------------------

@dataclass
class PolyClass:
    """Annotation container for generic polygonal regions."""
    poly: np.ndarray           # shape (n, 2) float32, image coordinates
    cls_id: int
    conf: float
    verified: bool = False
    deleted: bool = False

    def to_json(self) -> dict:
        return {
            "poly": self.poly.tolist(),
            "cls_id": int(self.cls_id),
            "conf": float(self.conf),
            "verified": bool(self.verified),
            "deleted": bool(self.deleted),
        }


@dataclass
class OBBOX(PolyClass):
    """Annotation container for oriented bounding boxes (4-point polygons)."""
    poly: np.ndarray           # shape (4, 2) float32, image coordinates


# ---------------------------------------------------------------------------
# Mask I/O and conversion
# ---------------------------------------------------------------------------

def load_mask_png(path: str) -> Optional[np.ndarray]:
    """Load a mask PNG (RGBA or grayscale) and return a single-channel uint8 array.
       - RGBA/BGRA → uses the alpha channel
       - Grayscale → thresholds dark pixels (< 50) as foreground (255)
    """
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        warnings.warn(f"Could not read mask image at {path}.")
        return None

    if mask.ndim == 3 and mask.shape[2] == 4:
        # RGBA / BGRA → extract alpha channel
        m = mask[..., 3].astype(np.uint8)
    elif mask.ndim == 2:
        # Grayscale: dark pixels are foreground
        m = np.where(mask < 50, 255, 0).astype(np.uint8)
    else:
        warnings.warn(f"Mask at {path} has unsupported shape {mask.shape}.")
        m = None

    return m


def mask_to_polys(
    mask: np.ndarray,
    *,
    min_area_frac: float = 1e-4,
    epsilon_frac: float = 0.002,
) -> List[np.ndarray]:
    """Convert a binary mask (alpha or grayscale) to a list of polygons.
    Handles:
      - Auto-detection of foreground (inverts if background is bright)
      - Otsu binarization for robustness
    Returns: list of np.ndarray with shape (N, 2), dtype float32.
    """
    if mask is None:
        return []

    m = mask.copy()
    H, W = m.shape
    img_area = H * W

    # Auto-detect foreground: if mean > 127 the background is bright → invert
    if np.mean(m) > 127:
        m = cv2.bitwise_not(m)

    # Robust binarization (Otsu)
    _, bw = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find external contours only
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys: List[np.ndarray] = []
    min_area = max(1.0, min_area_frac * img_area)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        eps = epsilon_frac * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if approx is None or len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2).astype(np.float32)
        polys.append(pts)

    return polys


def polys_to_mask(polys: List[PolyClass], img_shape: Tuple[int, int]) -> np.ndarray:
    """Rasterize a list of PolyClass annotations into a binary mask (H, W) uint8."""
    H, W = img_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = [p.poly.astype(np.int32) for p in polys]
    cv2.fillPoly(mask, pts, color=255)
    return mask


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def rect_to_poly_xyxy(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Convert an axis-aligned box (x1, y1, x2, y2) to a 4-point polygon."""
    return np.array([[x1, y1],
                     [x2, y1],
                     [x2, y2],
                     [x1, y2]], dtype=np.float32)


def find_orthogonal_projection(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> np.ndarray:
    """Given a line segment p1→p2 and a point p3, find the two points that
    complete an oriented rectangle: project p1 and p2 onto the line parallel
    to p1→p2 passing through p3.

    Returns: np.ndarray shape (2, 2) — the two projected corners [proj_p2, proj_p1].
    """
    d = np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)

    # Vector from p1 to p3
    v = p3 - p1

    # Component of v orthogonal to d (the shift from the original line)
    d_norm_sq = np.dot(d, d)
    if d_norm_sq < 1e-12:
        # p1 and p2 are the same point; degenerate case
        return np.array([p3, p3], dtype=np.float32)

    # Orthogonal offset = v - proj_d(v)
    ortho = v - (np.dot(v, d) / d_norm_sq) * d

    # The two new corners are p1 and p2 shifted by the orthogonal offset
    proj_p1 = p1 + ortho
    proj_p2 = p1 + d + ortho  # = p2 + ortho

    return np.array([proj_p2, proj_p1], dtype=np.float32)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_annotations(
    img_bgr: np.ndarray,
    annots: List[PolyClass],
    conf_threshold: float,
    class_names: Dict[int, str] | List[str] | None,
    selected_idx: Optional[int] = None,
    show_label: bool = False,
    show_conf: bool = False,
) -> np.ndarray:
    """Draw verified / unverified / selected annotations on an image copy."""
    out = img_bgr.copy()
    for i, b in enumerate(annots):
        if b.deleted or b.conf < conf_threshold:
            continue

        pts = b.poly.reshape(-1, 2).astype(int)

        # Color scheme: green=verified, orange=unverified, magenta=selected
        if selected_idx is not None and i == selected_idx:
            color = (255, 0, 255)          # magenta highlight
            thick = 2
        elif b.verified:
            color = (0, 255, 0)            # green
            thick = 2
        else:
            color = (0, 200, 255)          # orange
            thick = 2

        # Draw polygon outline
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thick)

        # Label only for unverified annotations (when requested)
        if not b.verified and (show_label or show_conf):
            parts = []
            if show_label:
                name = class_names[int(b.cls_id)] if class_names is not None else str(int(b.cls_id))
                parts.append(name)
            if show_conf:
                parts.append(f"{b.conf:.2f}")
            label = " ".join(parts)

            if label:
                (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                x0, y0 = int(pts[0, 0]), int(pts[0, 1])
                cv2.rectangle(out, (x0, y0), (x0 + tw + 6, y0 + th + base + 6), color, -1)
                cv2.putText(out, label, (x0 + 3, y0 + th + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out