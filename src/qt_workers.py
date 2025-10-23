import os
from typing import List

import numpy as np
from PyQt6 import QtCore

# -------- YOLO (Ultralytics) --------
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ------ Local imports from utils.py ------
from .utils import OBBOX, rect_to_poly_xyxy


MODEL_PATH = "./best.pt"  # your OBB checkpoint

class DetectionWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, object, object)  # (frame_idx, class_names, List[AnnotBox])
    error = QtCore.pyqtSignal(str)

    def __init__(self, frame_idx: int, frame_bgr: np.ndarray, conf: float = 0.01, model_path: str = MODEL_PATH):
        super().__init__()
        self.frame_idx = frame_idx
        self.frame_bgr = frame_bgr
        self.conf = conf
        self.model_path = model_path

    @QtCore.pyqtSlot()
    def run(self):
        try:
            if YOLO is None:
                raise RuntimeError("Ultralytics is not installed. `pip install ultralytics`")

            if not hasattr(DetectionWorker, "_model"):
                if not os.path.isfile(self.model_path):
                    raise FileNotFoundError(f"Model not found: {self.model_path}")
                DetectionWorker._model = YOLO(self.model_path)

            model = DetectionWorker._model
            res = model.predict(source=self.frame_bgr[..., ::-1], conf=self.conf, verbose=False)[0]
            names = getattr(model, "names", None)

            boxes: List[OBBOX] = []

            # Preferred: OBB polys directly
            if hasattr(res, "obb") and (res.obb is not None) and (len(res.obb) > 0):
                obb = res.obb
                polys = getattr(obb, "xyxyxyxy", None)
                cls = getattr(obb, "cls", None)
                conf = getattr(obb, "conf", None)

                if polys is not None and len(polys) > 0:
                    P = polys.cpu().numpy() if hasattr(polys, "cpu") else np.asarray(polys)
                    C = cls.cpu().numpy() if hasattr(cls, "cpu") else np.asarray(cls) if cls is not None else np.zeros((len(P),))
                    S = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf) if conf is not None else np.ones((len(P),))
                    for p, c, s in zip(P, C, S):
                        boxes.append(OBBOX(poly=p.reshape(4, 2).astype(np.float32), cls_id=int(c), conf=float(s)))

                else:
                    # Fallback: xywhr -> polygon
                    xywhr = getattr(obb, "xywhr", None)
                    if xywhr is not None and len(xywhr) > 0:
                        X = xywhr.cpu().numpy() if hasattr(xywhr, "cpu") else np.asarray(xywhr)
                        C = cls.cpu().numpy() if hasattr(cls, "cpu") else np.asarray(cls) if cls is not None else np.zeros((len(X),))
                        S = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf) if conf is not None else np.ones((len(X),))
                        for (cx, cy, w, h, rad), c, s in zip(X, C, S):
                            rect = np.array([[-w/2, -h/2],
                                             [ w/2, -h/2],
                                             [ w/2,  h/2],
                                             [-w/2,  h/2]], dtype=np.float32)
                            c0, s0 = np.cos(rad), np.sin(rad)
                            R = np.array([[c0, -s0], [s0, c0]], dtype=np.float32)
                            pts = rect @ R.T + np.array([cx, cy], dtype=np.float32)
                            boxes.append(OBBOX(poly=pts.astype(np.float32), cls_id=int(c), conf=float(s)))
            else:
                # Axis-aligned fallback
                if res.boxes is not None and len(res.boxes) > 0:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    C = res.boxes.cls.cpu().numpy() if hasattr(res.boxes, "cls") else np.zeros((len(xyxy),))
                    S = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, "conf") else np.ones((len(xyxy),))
                    for (x1, y1, x2, y2), c, s in zip(xyxy, C, S):
                        boxes.append(OBBOX(poly=rect_to_poly_xyxy(x1, y1, x2, y2), cls_id=int(c), conf=float(s)))

            self.finished.emit(self.frame_idx, names, boxes)

        except Exception as e:
            self.error.emit(str(e))