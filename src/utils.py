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

def find_parallel(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
    """Find the parallel line of p1 and p2 passing through p3."""
    # Get the line coefficients (Ax + B = y) for line p1p2
    A = (p2[1] - p1[1]) / (p2[0] - p1[0])
    if A == 0:
        A = 1e-6  # avoid division by zero
    B = p3[1] - A * p3[0]
    return A, B

def find_orthogonal_projection(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    A,B = find_parallel(p1, p2, p3)

    A_ortho = -1 / A
    B1_ortho = p1[1] - A_ortho * p1[0]
    B2_ortho = p2[1] - A_ortho * p2[0]

    # Solve for intersection
    x1_proj = (B1_ortho - B) / (A - A_ortho)
    y1_proj = A * x1_proj + B

    x2_proj = (B2_ortho - B) / (A - A_ortho)
    y2_proj = A * x2_proj + B
    return np.array([[x2_proj, y2_proj], [x1_proj, y1_proj]], dtype=np.float32)

import matplotlib.pyplot as plt

def test_plot_find_parallel():
    p1 = np.array([1.0, 1.0])
    p2 = np.array([4.0, 4.0])
    p3 = np.array([1.0, 4.0])


    primes = find_orthogonal_projection(p1, p2, p3)
    p1_prime = primes[2:4]
    p2_prime = primes[0:2]

    x_vals = np.array([0, 5])
    y_vals_line1 = (p2[1] - p1[1]) / (p2[0] - p1[0]) * (x_vals - p1[0]) + p1[1]
    # y_vals_line2 = A * x_vals + B

    plt.figure()
    plt.plot(x_vals, y_vals_line1, label='Original Line (p1 to p2)')
    # plt.plot(x_vals, y_vals_line2, label='Parallel Line through p3', linestyle='--')
    plt.scatter(*p1, color='red', label='p1')
    plt.scatter(*p2, color='blue', label='p2')
    plt.scatter(*p3, color='green', label='p3')
    plt.scatter(*p1_prime, color='orange', label="Projection of p1")
    plt.scatter(*p2_prime, color='purple', label="Projection of p2")
    plt.legend()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Parallel Line Test')
    plt.grid()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    test_plot_find_parallel()