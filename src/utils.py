from typing import Optional, List, Dict
from dataclasses import dataclass

import cv2
import numpy as np
from PyQt6 import QtGui

def cvimg_to_qimage(img_bgr: np.ndarray) -> QtGui.QImage:
    if img_bgr is None:
        return QtGui.QImage()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    return QtGui.QImage(img_rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)

@dataclass
class PolyClass:
    """Abstract Annotation container for generic object classes."""
    poly: np.ndarray           # shape (n,2) float32 in image coords
    cls_id: int
    conf: float
    verified: bool = False
    deleted: bool = False

    def to_json(self):
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
    poly: np.ndarray           # shape (4,2) float32 in image coords




def rect_to_poly_xyxy(x1, y1, x2, y2) -> np.ndarray:
    return np.array([[x1, y1],
                     [x2, y1],
                     [x2, y2],
                     [x1, y2]], dtype=np.float32)


def draw_annotations(img_bgr: np.ndarray,
                     annots: List[PolyClass],
                     conf_threshold: float,
                     class_names: Dict[int, str] | List[str] | None,
                     selected_idx: Optional[int] = None) -> np.ndarray:
    """Draw verified/unverified/selected annotations."""
    out = img_bgr.copy()
    for i, b in enumerate(annots):
        if b.deleted:
            continue
        if b.conf < conf_threshold:
            continue

        pts = b.poly.reshape(-1, 2).astype(int)

        # Choose color/thickness
        if b.verified:
            color = (0, 255, 0)            # green for verified
            thick = 2
        else:
            color = (0, 200, 255)          # orange for unverified
            thick = 2

        # Selection overrides color
        if selected_idx is not None and i == selected_idx:
            color = (255, 0, 255)          # magenta highlight
            thick = 3

        # Polygon
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thick)

        # Label only if UNVERIFIED
        if not b.verified:
            label = f"{class_names[int(b.cls_id)] if class_names is not None else int(b.cls_id)} {b.conf:.2f}"
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            x1, y1 = int(pts[0, 0]), int(pts[0, 1])
            cv2.rectangle(out, (x1, y1), (x1 + tw + 6, y1 + th + base + 6), color, -1)
            cv2.putText(out, label, (x1 + 3, y1 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out