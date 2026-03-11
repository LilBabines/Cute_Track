import os
import sys
import io
from typing import Dict, List, Optional
import time

import numpy as np
from PySide6 import QtCore

# -------- YOLO (Ultralytics) --------
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# -------- SAM2-UNet --------
from src.deep_learning.models.SAMUNET import SAM2UNet, LitBinarySeg
import torch
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from lightning.pytorch.callbacks import ModelCheckpoint
from src.deep_learning.dataset.dataset import DataModule512Mask
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
import cv2

# ------ Local imports ------
from .utils import OBBOX, PolyClass, rect_to_poly_xyxy, mask_to_polys, polys_to_mask


YOLO_MODEL_PATH = "./models/best26n-obb.pt"
SAM2_UNET_MODEL_PATH = "./models/tiny_last.ckpt"
SAM2_CHECKPOINT_PATH = "src/sam2/checkpoints/sam2.1_hiera_tiny.pt"


# ---------------------------------------------------------------------------
# Stdout capture helper — thread-safe relay to a Qt signal
# ---------------------------------------------------------------------------

class _StdoutCapture(io.TextIOBase):
    """Captures writes to stdout and relays each line via a Qt signal,
    while still forwarding to the original stdout."""

    def __init__(self, signal: QtCore.SignalInstance, original_stdout):
        super().__init__()
        self._signal = signal
        self._original = original_stdout

    def write(self, text: str):
        if self._original:
            self._original.write(text)
        if text and text.strip():
            self._signal.emit(text.rstrip("\n"))
        return len(text) if text else 0

    def flush(self):
        if self._original:
            self._original.flush()

    def isatty(self):
        return False


# ---------------------------------------------------------------------------
# Detection (YOLO-OBB)
# ---------------------------------------------------------------------------
class DetectionWorker(QtCore.QObject):
    """Run oriented-bounding-box detection on a single frame using YOLO-OBB.

    Accepts EITHER:
      - source_path (str): path to an image file → passed directly to YOLO
      - frame_bgr (np.ndarray): BGR uint8 array (e.g. from video capture)

    When source_path is given, it takes priority (YOLO handles its own I/O).
    """
    finished = QtCore.Signal(object, object, object)
    error = QtCore.Signal(str)

    def __init__(
        self,
        frame_idx: int,
        frame_bgr: np.ndarray = None,
        conf: float = 0.5,
        imgsz: int = 1024,
        model_path: str = YOLO_MODEL_PATH,
        source_path: str = None,
    ):
        super().__init__()
        self.frame_idx = frame_idx
        self.frame_bgr = frame_bgr
        self.conf = conf
        self.model_path = model_path
        self.imgsz = imgsz
        self.source_path = source_path

    @classmethod
    def _get_model(cls, model_path: str):
        """Lazy-load model, reload if path changed."""
        if not hasattr(cls, "_model") or cls._model_path != model_path:
            print(f"[DetectionWorker] Loading model: {model_path}")
            cls._model = YOLO(model_path)
            cls._model_path = model_path
        return cls._model

    @QtCore.Slot()
    def run(self):
        try:

            model = self._get_model(self.model_path)

            # --- Choose source: file path preferred, numpy fallback ---
            if self.source_path and os.path.isfile(self.source_path):
                source = self.source_path
            elif self.frame_bgr is not None:
                # YOLO expects RGB uint8 when given a numpy array
                bgr = self.frame_bgr
                # Safety: ensure uint8 (don't use ensure_bgr_u8, just basic conversion)
                if bgr.dtype != np.uint8:
                    if bgr.dtype == np.uint16:
                        bgr = (bgr / 256).astype(np.uint8)
                    else:
                        bgr = bgr.astype(np.uint8)
                # YOLO's internal pipeline expects BGR (it does its own conversion)
                # Passing BGR directly — do NOT convert to RGB here
                source = bgr
            else:
                raise RuntimeError("No source_path and no frame_bgr provided.")

            # --- Predict ---
            results = model.predict(
                source=source,
                imgsz=self.imgsz,
                conf=self.conf,
                verbose=False,
            )
            res = results[0]
            names = getattr(model, "names", None)

            # --- Debug ---
            has_obb = hasattr(res, "obb") and res.obb is not None and len(res.obb) > 0
            has_boxes = res.boxes is not None and len(res.boxes) > 0

            boxes: List[OBBOX] = []

            # --- OBB path ---
            if has_obb:
                obb = res.obb
                polys = getattr(obb, "xyxyxyxy", None)
                cls = getattr(obb, "cls", None)
                conf_vals = getattr(obb, "conf", None)

                if polys is not None and len(polys) > 0:
                    P = polys.cpu().numpy() if hasattr(polys, "cpu") else np.asarray(polys)
                    C = cls.cpu().numpy() if hasattr(cls, "cpu") else np.zeros(len(P))
                    S = conf_vals.cpu().numpy() if hasattr(conf_vals, "cpu") else np.ones(len(P))
                    for i, (p, c, s) in enumerate(zip(P, C, S)):
                        boxes.append(OBBOX(
                            poly=p.reshape(4, 2).astype(np.float32),
                            cls_id=int(c), conf=float(s),
                        ))
                else:
                    xywhr = getattr(obb, "xywhr", None)
                    if xywhr is not None and len(xywhr) > 0:
                        X = xywhr.cpu().numpy() if hasattr(xywhr, "cpu") else np.asarray(xywhr)
                        C = cls.cpu().numpy() if hasattr(cls, "cpu") else np.zeros(len(X))
                        S = conf_vals.cpu().numpy() if hasattr(conf_vals, "cpu") else np.ones(len(X))
                        for (cx, cy, w, h, rad), c, s in zip(X, C, S):
                            rect = np.array([[-w/2, -h/2], [w/2, -h/2],
                                             [w/2, h/2], [-w/2, h/2]], dtype=np.float32)
                            cos_r, sin_r = np.cos(rad), np.sin(rad)
                            R = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float32)
                            pts = rect @ R.T + np.array([cx, cy], dtype=np.float32)
                            boxes.append(OBBOX(poly=pts, cls_id=int(c), conf=float(s)))

            # --- AABB fallback ---
            elif has_boxes:
                xyxy = res.boxes.xyxy.cpu().numpy()
                C = res.boxes.cls.cpu().numpy()
                S = res.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), c, s in zip(xyxy, C, S):
                    boxes.append(OBBOX(
                        poly=rect_to_poly_xyxy(x1, y1, x2, y2),
                        cls_id=int(c), conf=float(s),
                    ))
                

            print(f"[DetectionWorker] Emitting {len(boxes)} boxes")
            self.finished.emit(self.frame_idx, names, boxes)

        except Exception as e:
            import traceback
            print(f"[DetectionWorker] EXCEPTION:\n{traceback.format_exc()}")
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Detection fine-tuning
# ---------------------------------------------------------------------------

class DetectFinetuneWorker(QtCore.QObject):
    """Build a YOLO-OBB dataset from verified polygons and fine-tune the model.

    Signals:
        progress(str, float)   — message + progress in [0, 1]
        epoch_metrics(int, int, dict) — current_epoch, total_epochs, metrics dict
        log_line(str)          — a line of console output
        finished(str)          — path to best.pt
        error(str)
    """
    progress = QtCore.Signal(str, float)
    epoch_metrics = QtCore.Signal(int, int, object)   # epoch, total, {metric: value}
    log_line = QtCore.Signal(str)
    finished = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(
        self,
        class_names: List[str],
        base_model_path: str,
        out_root: Optional[str] = None,
        epochs: int = 20,
        imgsz: int = 1024,
        batch: int = 8,
        val_split: float = 0.1,
        seed: int = 1337,
        data_yaml: str = "datasets/datasets_build/dataset.yaml",
    ):
        super().__init__()
        self.class_names = class_names
        self.base_model_path = base_model_path
        self.out_root = out_root or os.path.join(os.getcwd(), "finetune_runs")
        self.epochs = int(epochs)
        self.imgsz = int(imgsz)
        self.batch = int(batch)
        self.val_split = float(val_split)
        self.seed = int(seed)
        self.data_yaml = data_yaml

    @QtCore.Slot()
    def run(self):
        # Capture stdout so ultralytics console output goes to the GUI
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _StdoutCapture(self.log_line, original_stdout)
        sys.stderr = _StdoutCapture(self.log_line, original_stderr)

        try:
            if YOLO is None:
                raise RuntimeError("Ultralytics is not installed. `pip install ultralytics`")
            if not os.path.isfile(self.base_model_path):
                raise FileNotFoundError(f"Base model not found: {self.base_model_path}")
            if not self.class_names:
                raise ValueError("class_names is empty; cannot write dataset.yaml.")

            ts = time.strftime("%Y%m%d-%H%M%S")
            run_dir = os.path.join(self.out_root, f"run-{ts}")

            model = YOLO(self.base_model_path)

            # --- Register ultralytics callbacks for per-epoch progress ---
            total_epochs = self.epochs
            worker_ref = self   # prevent garbage-collection issues in closure

            def _on_fit_epoch_end(trainer):
                """Called by ultralytics at the end of each epoch (after val)."""
                epoch = trainer.epoch + 1
                metrics = {}

                # Collect available metrics from the trainer
                if hasattr(trainer, "metrics") and trainer.metrics:
                    for k, v in trainer.metrics.items():
                        try:
                            metrics[k] = float(v)
                        except (TypeError, ValueError):
                            pass

                # Also grab the last training loss values
                if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
                    loss_names = getattr(trainer, "loss_names", None)
                    loss_vals = trainer.loss_items
                    if hasattr(loss_vals, "cpu"):
                        loss_vals = loss_vals.cpu().numpy()
                    if loss_names and len(loss_names) == len(loss_vals):
                        for name, val in zip(loss_names, loss_vals):
                            metrics[f"train/{name}"] = float(val)

                frac = epoch / total_epochs
                worker_ref.progress.emit(f"Epoch {epoch}/{total_epochs}", frac)
                worker_ref.epoch_metrics.emit(epoch, total_epochs, metrics)

            model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

            self.progress.emit("Starting training...", 0.0)
            self.log_line.emit(f"=== Training started: {total_epochs} epochs, "
                               f"imgsz={self.imgsz}, batch={self.batch} ===")

            model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                imgsz=self.imgsz,
                batch=self.batch,
                project=run_dir,
                name="finetune",
                exist_ok=True,
                verbose=True,
                flipud=0.5,
                fliplr=0.5,
            )

            # Locate best weights
            weights_dir = os.path.join(run_dir, "finetune", "weights")
            best_pt = os.path.join(weights_dir, "best.pt")
            if not os.path.isfile(best_pt):
                last_pt = os.path.join(weights_dir, "last.pt")
                if os.path.isfile(last_pt):
                    best_pt = last_pt
                else:
                    raise RuntimeError("Training finished but no weights found.")

            self.progress.emit("Training complete!", 1.0)
            self.log_line.emit(f"=== Training complete — weights: {best_pt} ===")
            self.finished.emit(best_pt)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


# ---------------------------------------------------------------------------
# Segmentation inference
# ---------------------------------------------------------------------------

class SegWorker(QtCore.QObject):
    finished = QtCore.Signal(object, object, object)
    error = QtCore.Signal(str)

    def __init__(
        self,
        frame_idx: int,
        frame_bgr: np.ndarray,
        conf: float = 0.5,
        imgsz: int = 512,
        model_path: str = SAM2_UNET_MODEL_PATH,
    ):
        super().__init__()
        self.frame_idx = frame_idx
        self.frame_bgr = frame_bgr
        self.conf = conf
        self.imgsz = imgsz
        self.model_path = model_path

    @QtCore.Slot()
    def run(self):
        try:
            if not hasattr(SegWorker, "_model"):
                net = SAM2UNet(
                    config="tiny",
                    sam_checkpoint_path=SAM2_CHECKPOINT_PATH,
                    freeze_encorder=True,
                ).to("cuda")
                SegWorker._model = LitBinarySeg.load_from_checkpoint(
                    self.model_path, net=net,
                    deep_supervision=False, dice_use_all_outputs=False, pos_weight=200,
                ).eval().to("cuda")

            model = SegWorker._model
            transform = Compose([
                Resize((self.imgsz, self.imgsz)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            from PIL import Image
            pil_img = Image.fromarray(cv2.cvtColor(self.frame_bgr, cv2.COLOR_BGR2RGB))
            orig_w, orig_h = pil_img.size
            inp = transform(pil_img).unsqueeze(0).to("cuda")

            with torch.no_grad():
                out = model(inp)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                prob = torch.sigmoid(out).squeeze().cpu().numpy()

            mask = (prob > self.conf).astype(np.uint8) * 255
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            polys_np = mask_to_polys(mask)
            poly_objs = [
                PolyClass(poly=p.astype(np.float32), cls_id=0, conf=1.0)
                for p in polys_np
            ]
            names = ["object"]
            self.finished.emit(self.frame_idx, names, poly_objs)

        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Segmentation fine-tuning
# ---------------------------------------------------------------------------

class SegFinetuneWorker(QtCore.QObject):
    progress = QtCore.Signal(str, float)
    finished = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(
        self,
        dataset: Dict[int, List[PolyClass]],
        dataset_images_names: Dict[int, str],
        base_model_path: str,
        out_root: Optional[str] = None,
        epochs: int = 20,
        batch: int = 8,
        val_split: float = 0.1,
    ):
        super().__init__()
        self.dataset = dataset
        self.dataset_images_names = dataset_images_names
        self.base_model_path = base_model_path
        self.out_root = out_root or os.path.join(os.getcwd(), "seg_finetune_runs")
        self.epochs = int(epochs)
        self.batch = int(batch)
        self.val_split = float(val_split)

    @QtCore.Slot()
    def run(self):
        try:
            import random
            ts = time.strftime("%Y%m%d-%H%M%S")
            run_dir = os.path.join(self.out_root, f"seg-run-{ts}")
            img_dir = os.path.join(run_dir, "images")
            mask_dir = os.path.join(run_dir, "masks")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            self.progress.emit("Building segmentation dataset...", 0.0)
            items = list(self.dataset.items())
            for i, (frame_idx, polys) in enumerate(items):
                img_path = self.dataset_images_names.get(frame_idx)
                if img_path is None or not os.path.isfile(img_path):
                    continue
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                h, w = img.shape[:2]
                mask = polys_to_mask(polys, w, h)
                stem = f"frame{frame_idx:06d}"
                cv2.imwrite(os.path.join(img_dir, f"{stem}.jpg"), img)
                cv2.imwrite(os.path.join(mask_dir, f"{stem}.png"), mask)

            self.progress.emit("Training segmentation model...", 0.1)

            net = SAM2UNet(
                config="tiny",
                sam_checkpoint_path=SAM2_CHECKPOINT_PATH,
                freeze_encorder=True,
            )
            lit = LitBinarySeg.load_from_checkpoint(
                self.base_model_path, net=net,
                deep_supervision=False, dice_use_all_outputs=False, pos_weight=200,
            )
            dm = DataModule512Mask(img_dir=img_dir, mask_dir=mask_dir, batch_size=self.batch)
            ckpt_cb = ModelCheckpoint(dirpath=run_dir, filename="best", monitor="val_loss", save_top_k=1)
            logger = TensorBoardLogger(save_dir=run_dir, name="logs")
            trainer = Trainer(
                max_epochs=self.epochs,
                accelerator="gpu",
                devices=1,
                callbacks=[ckpt_cb],
                logger=logger,
                enable_progress_bar=True,
            )
            trainer.fit(lit, dm)
            best_path = ckpt_cb.best_model_path or os.path.join(run_dir, "best.ckpt")
            self.progress.emit("Segmentation training complete!", 1.0)
            self.finished.emit(best_path)

        except Exception as e:
            self.error.emit(str(e))