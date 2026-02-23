from typing import Optional, List, Dict
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6 import QtGui

import warnings


def ensure_bgr_u8(img: np.ndarray) -> np.ndarray:
    """Convertit une image (8/16 bits, mono/RGBA) en BGR uint8 pour affichage + traitement.
       - 16 bits -> mise à l'échelle 0..255 (normalisation)
       - 1 canal -> BGR
       - 4 canaux (BGRA) -> BGR
    """
    if img is None:
        return img
    # 16-bit -> 8-bit
    if img.dtype == np.uint16:
        # normalise plein-écart pour l'affichage fiable
        i_min, i_max = int(img.min()), int(img.max())
        if i_max > i_min:
            img8 = ((img - i_min) * 255.0 / (i_max - i_min)).astype(np.uint8)
        else:
            img8 = (img / 256).astype(np.uint8)
        img = img8
    elif img.dtype != np.uint8:
        # fallback simple
        img = cv2.convertScaleAbs(img)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

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

def load_mask_png(path: str) -> np.ndarray:
    """Charge un masque PNG (RGBA ou grayscale) en ndarray."""
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if mask.ndim == 3 and mask.shape[2] == 4:
        # RGBA ou BGRA -> alpha
        m = mask[..., 3].astype(np.uint8)
    elif  mask.ndim == 2:
        m = np.where(mask < 50, 255, 0).astype(np.uint8)
    else:
        # warning
        warnings.warn(f"Mask image at {path} has unsupported format with shape {mask.shape}.")
        m = None
    
    return m
def mask_to_polys(mask: np.ndarray,
                  *,
                  min_area_frac: float = 1e-4,
                  epsilon_frac: float = 0.002) -> List[np.ndarray]:
    """
    Convertit un masque binaire (alpha ou grayscale) en polygones.
    Gère :
      - PNG RGBA (prend le canal alpha)
      - Masques noir/blanc (inversion automatique si fond clair)
    Retour : liste de np.ndarray (N,2) float32.
    """
    if mask is None:
        return []
    
    m = mask.copy()

    H, W = m.shape
    img_area = H * W

    # --- Auto-détection du foreground ---
    mean_val = np.mean(m)
    # si fond clair (>127) => les objets sont sombres -> inverser
    if mean_val > 127:
        m = cv2.bitwise_not(m)

    # --- Binarisation robuste (Otsu) ---
    _, bw = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Contours extérieurs ---
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

def polys_to_mask(polys: List[PolyClass], img_shape: tuple[int, int]) -> np.ndarray:
    """Convertit une liste de polygone (M,N,2) en masque binaire (H,W) uint8."""
    H, W = img_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = [p.poly.astype(np.int32) for p in polys]
    cv2.fillPoly(mask, pts, color=255)
    return mask

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
                     selected_idx: Optional[int] = None,
                     show_label: bool = False,
                     show_conf: bool = False) -> np.ndarray:
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
            thick = 2

        # Polygon
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thick)

        # Label only if UNVERIFIED
        label = ""
        if not b.verified:
            if show_label:
                label += f"{class_names[int(b.cls_id)] if class_names is not None else int(b.cls_id)}"
            if show_conf:
                label += f" {b.conf:.2f}"

            if label != "":
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

    if (p2[0] - p1[0]) ==0:

        A = 10e6
    elif (p2[1] - p1[1])==0:
        A = 1e-6
    
    else : 
        A = (p2[1] - p1[1]) / (p2[0] - p1[0])
    
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