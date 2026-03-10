import os
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
# Detection (YOLO-OBB)
# ---------------------------------------------------------------------------

class DetectionWorker(QtCore.QObject):
    """Run oriented-bounding-box detection on a single frame using YOLO-OBB.

    Signals:
        finished(frame_idx, class_names, List[OBBOX])
        error(str)
    """
    finished = QtCore.Signal(object, object, object)
    error = QtCore.Signal(str)

    def __init__(
        self,
        frame_idx: int,
        frame_bgr: np.ndarray,
        conf: float = 0.5,
        imgsz: int = 1024,
        model_path: str = YOLO_MODEL_PATH,
    ):
        super().__init__()
        self.frame_idx = frame_idx
        self.frame_bgr = frame_bgr
        self.conf = conf
        self.model_path = model_path
        self.imgsz = imgsz

    @QtCore.Slot()
    def run(self):
        try:
            if YOLO is None:
                raise RuntimeError("Ultralytics is not installed. `pip install ultralytics`")

            # Lazy-load: cache model on the class to avoid reloading each frame
            if not hasattr(DetectionWorker, "_model"):
                DetectionWorker._model = YOLO(self.model_path)

            model = DetectionWorker._model
            # BGR→RGB for ultralytics
            res = model.predict(
                source=self.frame_bgr[..., ::-1],
                imgsz=self.imgsz,
                conf=self.conf,
                verbose=False,
            )[0]
            names = getattr(model, "names", None)

            boxes: List[OBBOX] = []

            # --- Preferred path: OBB polygons directly ---
            if hasattr(res, "obb") and res.obb is not None and len(res.obb) > 0:
                obb = res.obb
                polys = getattr(obb, "xyxyxyxy", None)
                cls = getattr(obb, "cls", None)
                conf = getattr(obb, "conf", None)

                if polys is not None and len(polys) > 0:
                    P = polys.cpu().numpy() if hasattr(polys, "cpu") else np.asarray(polys)
                    C = cls.cpu().numpy() if hasattr(cls, "cpu") else np.asarray(cls) if cls is not None else np.zeros(len(P))
                    S = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf) if conf is not None else np.ones(len(P))
                    for p, c, s in zip(P, C, S):
                        boxes.append(OBBOX(poly=p.reshape(4, 2).astype(np.float32), cls_id=int(c), conf=float(s)))
                else:
                    # Fallback: xywhr → polygon via rotation matrix
                    xywhr = getattr(obb, "xywhr", None)
                    if xywhr is not None and len(xywhr) > 0:
                        X = xywhr.cpu().numpy() if hasattr(xywhr, "cpu") else np.asarray(xywhr)
                        C = cls.cpu().numpy() if hasattr(cls, "cpu") else np.asarray(cls) if cls is not None else np.zeros(len(X))
                        S = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf) if conf is not None else np.ones(len(X))
                        for (cx, cy, w, h, rad), c, s in zip(X, C, S):
                            rect = np.array([[-w/2, -h/2],
                                             [ w/2, -h/2],
                                             [ w/2,  h/2],
                                             [-w/2,  h/2]], dtype=np.float32)
                            cos_r, sin_r = np.cos(rad), np.sin(rad)
                            R = np.array([[cos_r, -sin_r],
                                          [sin_r,  cos_r]], dtype=np.float32)
                            pts = rect @ R.T + np.array([cx, cy], dtype=np.float32)
                            boxes.append(OBBOX(poly=pts, cls_id=int(c), conf=float(s)))

            # --- Axis-aligned fallback ---
            elif res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                C = res.boxes.cls.cpu().numpy() if hasattr(res.boxes, "cls") else np.zeros(len(xyxy))
                S = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, "conf") else np.ones(len(xyxy))
                for (x1, y1, x2, y2), c, s in zip(xyxy, C, S):
                    boxes.append(OBBOX(poly=rect_to_poly_xyxy(x1, y1, x2, y2), cls_id=int(c), conf=float(s)))

            self.finished.emit(self.frame_idx, names, boxes)

        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Detection fine-tuning
# ---------------------------------------------------------------------------

class DetectFinetuneWorker(QtCore.QObject):
    """Build a YOLO-OBB dataset from verified polygons and fine-tune the model.

    Signals:
        progress(str, float)   — message + progress in [0, 1]
        finished(str)          — path to best.pt
        error(str)
    """
    progress = QtCore.Signal(str, float)
    finished = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(
        self,
        video_path: str,
        dataset: Dict[int, List[OBBOX]],
        class_names: List[str],
        base_model_path: str,
        out_root: Optional[str] = None,
        epochs: int = 20,
        imgsz: int = 1024,
        batch: int = 8,
        val_split: float = 0.1,
        seed: int = 1337,
    ):
        super().__init__()
        self.video_path = video_path
        self.dataset = dataset
        self.class_names = class_names
        self.base_model_path = base_model_path
        self.out_root = out_root or os.path.join(os.getcwd(), "finetune_runs")
        self.epochs = int(epochs)
        self.imgsz = int(imgsz)
        self.batch = int(batch)
        self.val_split = float(val_split)
        self.seed = int(seed)

    @QtCore.Slot()
    def run(self):
        try:
            # --- Sanity checks ---
            if YOLO is None:
                raise RuntimeError("Ultralytics is not installed. `pip install ultralytics`")
            if not os.path.isfile(self.base_model_path):
                raise FileNotFoundError(f"Base model not found: {self.base_model_path}")
            if not os.path.isfile(self.video_path):
                raise FileNotFoundError(f"Video not found: {self.video_path}")
            if not self.class_names:
                raise ValueError("class_names is empty; cannot write dataset.yaml.")

            # --- Prepare run directory tree ---
            ts = time.strftime("%Y%m%d-%H%M%S")
            run_dir = os.path.join(self.out_root, f"run-{ts}")
            img_tr = os.path.join(run_dir, "images", "train")
            img_va = os.path.join(run_dir, "images", "val")
            lb_tr = os.path.join(run_dir, "labels", "train")
            lb_va = os.path.join(run_dir, "labels", "val")
            for p in (img_tr, img_va, lb_tr, lb_va):
                os.makedirs(p, exist_ok=True)

            self.progress.emit("Preparing frames and labels...", 0.02)

            # --- Open video and read dimensions for normalization ---
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {self.video_path}")
            img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if img_w <= 0 or img_h <= 0:
                raise RuntimeError("Could not read video dimensions.")

            # Deterministic train/val split
            rng = np.random.default_rng(self.seed)
            frame_indices = sorted(k for k in self.dataset if self.dataset[k])
            rng.shuffle(frame_indices)
            n_val = max(1, int(len(frame_indices) * self.val_split))
            val_set = set(frame_indices[:n_val])

            def _seek_read(idx: int) -> Optional[np.ndarray]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                return frame if ok else None

            def _normalize_poly(poly: np.ndarray) -> np.ndarray:
                """Normalize a (4, 2) polygon from pixel coords to [0, 1]."""
                pts = poly.astype(np.float32).copy()
                pts[:, 0] = np.clip(pts[:, 0] / max(img_w, 1), 0.0, 1.0)
                pts[:, 1] = np.clip(pts[:, 1] / max(img_h, 1), 0.0, 1.0)
                return pts

            # --- Write images and YOLO-OBB label files ---
            written = 0
            for i, fidx in enumerate(frame_indices):
                frame = _seek_read(fidx)
                if frame is None:
                    continue

                is_val = fidx in val_set
                img_dir = img_va if is_val else img_tr
                lb_dir = lb_va if is_val else lb_tr

                stem = f"frame_{fidx:06d}"
                cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), frame)

                # YOLO-OBB format: cls x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
                lines = []
                for ann in self.dataset.get(fidx, []):
                    poly = getattr(ann, "poly", None)
                    cls_id = int(getattr(ann, "cls_id", -1))
                    if poly is None or np.asarray(poly).shape != (4, 2) or cls_id < 0:
                        continue
                    pts = _normalize_poly(np.asarray(poly))
                    flat = " ".join(f"{v:.6f}" for v in pts.reshape(-1))
                    lines.append(f"{cls_id} {flat}")

                # YOLO expects a .txt per image even if empty
                with open(os.path.join(lb_dir, stem + ".txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))

                written += 1
                self.progress.emit(
                    f"Wrote {written}/{len(frame_indices)} samples...",
                    0.05 + 0.6 * (i + 1) / max(len(frame_indices), 1),
                )

            cap.release()

            if written == 0:
                raise RuntimeError("No samples written. Check that the dataset has verified annotations.")

            # --- Write dataset.yaml ---
            data_yaml = os.path.join(run_dir, "dataset.yaml")
            with open(data_yaml, "w", encoding="utf-8") as f:
                f.write(
                    "path: .\n"
                    f"train: finetune_runs/run-{ts}/images/train\n"
                    f"val: finetune_runs/run-{ts}/images/val\n"
                    "names:\n"
                )
                for idx, name in enumerate(self.class_names):
                    f.write(f"  {idx}: {name}\n")

            self.progress.emit("Launching training...", 0.70)

            # --- Train ---
            model = YOLO(self.base_model_path)
            model.train(
                data=data_yaml,
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

            # --- Locate best weights ---
            weights_dir = os.path.join(run_dir, "finetune", "weights")
            best_pt = os.path.join(weights_dir, "best.pt")
            if not os.path.isfile(best_pt):
                last_pt = os.path.join(weights_dir, "last.pt")
                if os.path.isfile(last_pt):
                    best_pt = last_pt
                else:
                    raise RuntimeError("Training finished but no weights found.")

            self.progress.emit("Training complete.", 0.99)
            self.finished.emit(best_pt)

        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Segmentation (SAM2-UNet)
# ---------------------------------------------------------------------------

class SegWorker(QtCore.QObject):
    """Run binary segmentation on a single frame using SAM2-UNet.

    The model operates at 512×512; output polygons are rescaled back
    to the original image dimensions.

    Signals:
        finished(frame_idx, class_names, List[OBBOX])
        error(str)
    """
    finished = QtCore.Signal(object, object, object)
    error = QtCore.Signal(str)

    # Input resolution expected by the model
    MODEL_SIZE = 512

    def __init__(
        self,
        frame_idx: int,
        frame_bgr: np.ndarray,
        conf: float = 0.5,
        model_path: str = SAM2_UNET_MODEL_PATH,
        **kwargs,                       # accept (and ignore) extra keys like imgsz
    ):
        super().__init__()
        self.frame_idx = frame_idx
        self.frame_bgr = frame_bgr
        self.conf = conf
        self.model_path = model_path

    def preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """BGR uint8 → normalized (1, 3, 512, 512) tensor."""
        x = frame_bgr[..., ::-1].copy()          # BGR → RGB
        transform = Compose([
            ToTensor(),
            Resize((self.MODEL_SIZE, self.MODEL_SIZE)),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
        ])
        return transform(x).unsqueeze(0)          # add batch dim

    @QtCore.Slot()
    def run(self):
        try:
            # Lazy-load model (cached on the class)
            if not hasattr(SegWorker, "_model"):
                if not os.path.isfile(self.model_path):
                    raise FileNotFoundError(f"Model not found: {self.model_path}")
                net = SAM2UNet(
                    config="tiny",
                    sam_checkpoint_path=SAM2_CHECKPOINT_PATH,
                    freeze_encorder=True,
                ).to("cuda")
                lit = LitBinarySeg.load_from_checkpoint(
                    self.model_path,
                    net=net,
                    deep_supervision=False,
                    dice_use_all_outputs=False,
                    pos_weight=200,
                )
                lit.eval()
                SegWorker._model = lit.to("cuda")

            model = SegWorker._model
            input_tensor = self.preprocess(self.frame_bgr).to("cuda")

            with torch.no_grad():
                res = model(input_tensor).sigmoid()

            # Binary mask at model resolution
            mask = (res[0, 0].cpu().numpy() > self.conf).astype(np.uint8) * 255

            # Extract polygons at 512×512, then rescale to original image size
            orig_h, orig_w = self.frame_bgr.shape[:2]
            polys_512 = mask_to_polys(mask)

            scale_x = orig_w / self.MODEL_SIZE
            scale_y = orig_h / self.MODEL_SIZE

            annot_polys = []
            for p in polys_512:
                scaled = p.copy()
                scaled[:, 0] *= scale_x
                scaled[:, 1] *= scale_y
                annot_polys.append(OBBOX(poly=scaled.astype(np.float32), cls_id=0, conf=1.0))

            names = ["Building"] * len(annot_polys)
            self.finished.emit(self.frame_idx, names, annot_polys)

        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Segmentation fine-tuning
# ---------------------------------------------------------------------------

class SegFinetuneWorker(QtCore.QObject):
    """Fine-tune the SAM2-UNet model on verified segmentation annotations.

    Signals:
        progress(str, float)   — message + progress in [0, 1]
        finished(str)          — path to best checkpoint
        error(str)
    """
    progress = QtCore.Signal(str, float)
    finished = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(
        self,
        dataset: Dict[int, List[PolyClass]],
        dataset_images_names: Dict[int, str],
        base_model_path: str,
        out_root: Optional[str] = None,
        epochs: int = 10,
        batch: int = 4,
        val_split: float = 0.1,
        seed: int = 1337,
    ):
        super().__init__()
        self.dataset = dataset
        self.dataset_images_names = dataset_images_names
        self.base_model_path = base_model_path
        self.out_root = out_root or os.path.join(os.getcwd(), "seg_finetune_runs")
        self.epochs = int(epochs)
        self.batch = int(batch)
        self.val_split = float(val_split)
        self.seed = int(seed)

    @QtCore.Slot()
    def run(self):
        try:
            # --- Sanity checks ---
            if not os.path.isfile(self.base_model_path):
                raise FileNotFoundError(f"Base model not found: {self.base_model_path}")
            if not self.dataset:
                raise ValueError("Dataset is empty; verify annotations first.")

            self.progress.emit("Preparing dataset...", 0.02)

            # --- Prepare run directory ---
            ts = time.strftime("%Y%m%d-%H%M%S")
            exp_name = "finetune_seg"
            run_dir = f"logs/{exp_name}/{ts}"
            os.makedirs(run_dir, exist_ok=True)

            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

            # --- Write images and masks to a dataset folder ---
            self.progress.emit("Preparing dataset...", 0.05)
            data_dir = f"datasets/seg_finetune/data_{ts}"
            data_dir_images = os.path.join(data_dir, "images")
            data_dir_masks = os.path.join(data_dir, "GT_Object")
            os.makedirs(data_dir_images, exist_ok=True)
            os.makedirs(data_dir_masks, exist_ok=True)

            for frame_idx, polys in self.dataset.items():
                img_src_path = self.dataset_images_names.get(frame_idx)
                if img_src_path is None or not os.path.isfile(img_src_path):
                    continue

                img_dst_path = os.path.join(data_dir_images, os.path.basename(img_src_path))
                cv2.imwrite(img_dst_path, cv2.imread(img_src_path))

                # Rasterize polygons into a binary mask
                mask = polys_to_mask(polys, (512, 512))
                mask_dst_path = os.path.join(data_dir_masks, f"frame_{frame_idx:06d}_mask.png")
                cv2.imwrite(mask_dst_path, mask)

            # --- Callbacks ---
            cb_checkpoint = ModelCheckpoint(
                monitor="val/dice",
                dirpath=run_dir,
                save_top_k=1,
                mode="max",
            )

            # --- Initialize model and trainer ---
            net = SAM2UNet(
                config="tiny",
                sam_checkpoint_path=SAM2_CHECKPOINT_PATH,
                freeze_encorder=True,
            ).to("cuda")

            lit = LitBinarySeg.load_from_checkpoint(
                self.base_model_path,
                net=net,
                deep_supervision=False,
                dice_use_all_outputs=False,
                pos_weight=200,
            )

            trainer = Trainer(
                accelerator="auto",
                devices=1,
                max_epochs=self.epochs,         # was hardcoded to 10 — now uses self.epochs
                logger=TensorBoardLogger(
                    save_dir=run_dir,
                    name=exp_name,
                    version=f"{ts}/tensorboard",
                ),
                callbacks=[cb_checkpoint],
            )

            data_module = DataModule512Mask(
                dataset_path=data_dir,
                batch_size=self.batch,
                val_split=self.val_split,
            )

            self.progress.emit("Launching training...", 0.70)
            trainer.fit(lit, datamodule=data_module)

            self.progress.emit("Training complete.", 0.99)
            self.finished.emit(cb_checkpoint.best_model_path)

        except Exception as e:
            self.error.emit(str(e))