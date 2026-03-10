import os
import threading
from typing import Optional, List, Dict, Any
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

# ------ Local imports ------
from .utils import (
    PolyClass, OBBOX, cvimg_to_qimage, draw_annotations,
    find_orthogonal_projection, ensure_bgr_u8, mask_to_polys, load_mask_png,
)
from .qt_workers import (
    SegWorker, SegFinetuneWorker, DetectionWorker, DetectFinetuneWorker,
    SAM2_UNET_MODEL_PATH, YOLO_MODEL_PATH, SAM2_CHECKPOINT_PATH,
)

from ultralytics import YOLO
from src.deep_learning.models.SAMUNET import LitBinarySeg, SAM2UNet


# ---------------------------------------------------------------------------
# Frame sources — abstract over video files and image folders
# ---------------------------------------------------------------------------

class FrameSource:
    """Abstract interface for sequential frame access."""
    def count(self) -> int: ...
    def read(self, idx: int) -> Optional[np.ndarray]: ...
    def fps(self) -> float: return 25.0
    def close(self): pass
    def name(self) -> str: return ""


class VideoSource(FrameSource):
    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self._count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)

    def count(self) -> int:
        return self._count

    def read(self, idx: int) -> Optional[np.ndarray]:
        idx = max(0, min(idx, self._count - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        return frame if ok else None

    def fps(self) -> float:
        return self._fps

    def close(self):
        if self.cap:
            self.cap.release()

    def name(self) -> str:
        return os.path.basename(self.path)


class ImageFolderSource(FrameSource):
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, folder: str):
        self.path = folder
        files = [f for f in os.listdir(self.path)
                 if os.path.splitext(f)[1].lower() in self.IMAGE_EXTS]
        if not files:
            raise RuntimeError("No images found in folder.")

        # Natural sort (numeric substrings sorted numerically)
        import re
        def _key(s):
            return [int(t) if t.isdigit() else t.lower()
                    for t in re.findall(r'\d+|\D+', s)]
        files.sort(key=_key)
        self.paths = [os.path.join(self.path, f) for f in files]

    def count(self) -> int:
        return len(self.paths)

    def read(self, idx: int) -> Optional[np.ndarray]:
        idx = max(0, min(idx, len(self.paths) - 1))
        img = cv2.imread(self.paths[idx], cv2.IMREAD_UNCHANGED)
        return ensure_bgr_u8(img) if img is not None else None

    def fps(self) -> float:
        return 10.0               # arbitrary for image-by-image playback

    def name(self) -> str:
        return os.path.basename(self.path)

    def path_at(self, idx: int) -> str:
        idx = max(0, min(idx, len(self.paths) - 1))
        return self.paths[idx]


# ---------------------------------------------------------------------------
# Signal bridge for finetuning (lives on main thread, avoids QThread timers)
# ---------------------------------------------------------------------------

class _FinetuneSignalBridge(QtCore.QObject):
    """Thin relay that lives on the main thread.
    Worker signals (emitted from a plain Python thread) are connected to these
    signals, which Qt automatically queues to the main-thread event loop."""
    progress = QtCore.Signal(str, float)
    finished = QtCore.Signal(str)
    error = QtCore.Signal(str)


# ---------------------------------------------------------------------------
# Base annotation window
# ---------------------------------------------------------------------------

class Base(QtWidgets.QMainWindow):
    """Shared base for video/image annotation with model inference and HITL editing."""

    def __init__(self):
        super().__init__()

        self.baseTitle = "Annotation Tool"
        self.setWindowTitle(self.baseTitle)
        self.resize(1200, 760)

        # --- Source / playback state ---
        self.source: Optional[FrameSource] = None
        self.total_frames: int = 0
        self.current_idx: int = 0
        self.current_frame_bgr: Optional[np.ndarray] = None
        self.src_path: Optional[str] = None
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._on_play_tick)
        self.playing = False

        # --- Zoom & pan ---
        self.zoom = 1.0                    # relative to "fit to window"
        self.min_zoom = 0.25
        self.max_zoom = 8.0
        self.pan_img = np.array([0.0, 0.0], dtype=np.float32)   # pan in image pixels

        # --- Annotations ---
        self.pred_cache: Dict[int, List[PolyClass]] = {}    # frame_idx → predictions
        self.class_names = None
        self.selected_idx: Optional[int] = None

        # --- Verified dataset ---
        self.dataset: Dict[int, List[PolyClass]] = {}
        self.dataset_images_names: Dict[int, str] = {}

        # --- Display ↔ image coordinate mapping ---
        self.draw_map: Dict[str, float] = {"scale": 1.0, "xoff": 0, "yoff": 0}

        # --- Interaction state ---
        self.space_held = False
        self.mode = "select"               # "select" | "add" | "edit"
        self.temp_poly_pts: List[List[float]] = []
        self.dragging = False
        self.drag_start_img: Optional[tuple] = None
        self.orig_poly: Optional[np.ndarray] = None
        self.vertex_drag_idx: Optional[int] = None

        # ====================== UI SETUP ======================
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        # Canvas
        self.video_label = QtWidgets.QLabel()
        self.video_label.setMouseTracking(True)
        self.video_label.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background:#111; border:1px solid #333;")
        self.video_label.setMinimumSize(720, 405)
        self.installEventFilter(self)
        self.centralWidget().installEventFilter(self)
        self.video_label.installEventFilter(self)

        # --- Action buttons ---
        self.add_btn = QtWidgets.QPushButton("Add (N)")
        self.add_btn.setToolTip("Start add-box mode.\nClick points to make a polygon.\nShortcut: N")
        self.add_btn.clicked.connect(self.start_add_mode)

        self.edit_btn = QtWidgets.QPushButton("Edit (E)")
        self.edit_btn.setToolTip("Toggle edit mode.\nDrag to move; Ctrl+drag a corner to reshape.\nShortcut: E")
        self.edit_btn.clicked.connect(self.toggle_edit_mode)

        self.verify_btn = QtWidgets.QPushButton("Verify (V)")
        self.verify_btn.setToolTip("Toggle verified state on selected box.\nShortcut: V")
        self.verify_btn.clicked.connect(self.verify_selected_toggle)

        self.delete_btn = QtWidgets.QPushButton("Delete (Del)")
        self.delete_btn.setToolTip("Delete selected box.\nShortcut: Delete")
        self.delete_btn.clicked.connect(self.delete_selected)

        # --- Zoom buttons ---
        self.zoom_in_btn = QtWidgets.QPushButton("Zoom +")
        self.zoom_in_btn.setToolTip("Zoom in ( + / mouse wheel up )")
        self.zoom_out_btn = QtWidgets.QPushButton("Zoom −")
        self.zoom_out_btn.setToolTip("Zoom out ( - / mouse wheel down )")
        self.zoom_fit_btn = QtWidgets.QPushButton("Fit")
        self.zoom_fit_btn.setToolTip("Reset zoom & pan to fit ( 0 )")

        self.zoom_in_btn.clicked.connect(lambda: self.zoom_step(+1))
        self.zoom_out_btn.clicked.connect(lambda: self.zoom_step(-1))
        self.zoom_fit_btn.clicked.connect(self.zoom_fit)

        # --- Transport / file buttons ---
        self.open_video_btn = QtWidgets.QPushButton("Open video")
        self.open_images_btn = QtWidgets.QPushButton("Open image folder")
        self.prev_btn = QtWidgets.QPushButton("⟸ Prev (←)")
        self.next_btn = QtWidgets.QPushButton("Next (→) ⟹")
        self.run_btn = QtWidgets.QPushButton("Run Model")
        self.finetune_btn = QtWidgets.QPushButton("Finetune Model")
        self.play_btn = QtWidgets.QPushButton("Play ▶")
        self.pause_btn = QtWidgets.QPushButton("Pause ⏸")
        self.save_btn = QtWidgets.QPushButton("Save dataset (JSON)")

        self.inference_conf_tresh = QtWidgets.QDoubleSpinBox()
        self.inference_conf_tresh.setRange(0.01, 0.99)
        self.inference_conf_tresh.setSingleStep(0.05)
        self.inference_conf_tresh.setValue(0.5)
        self.inference_conf_tresh.setPrefix("conf=")

        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.sliderReleased.connect(self._on_slider_released)

        # --- Layout ---
        # Left: canvas + slider
        left_stack = QtWidgets.QWidget()
        left_v = QtWidgets.QVBoxLayout(left_stack)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(6)
        left_v.addWidget(self.video_label, stretch=1)
        left_v.addWidget(self.frame_slider)

        # Main row: left stack + right side panel
        content_row = QtWidgets.QHBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(10)
        content_row.addWidget(left_stack, stretch=1)
        content_row.addWidget(self._build_side_panel(), stretch=0)

        self._build_menu_bar()

        # Overall page
        page = QtWidgets.QVBoxLayout(self.centralWidget())
        page.setContentsMargins(8, 8, 8, 8)
        page.setSpacing(8)
        page.addLayout(content_row, stretch=1)
        page.addWidget(self._build_transport_bar())

        # --- Signals ---
        self.open_video_btn.clicked.connect(self.open_video)
        self.open_images_btn.clicked.connect(self.open_folder)
        self.prev_btn.clicked.connect(self.prev_frame)
        self.next_btn.clicked.connect(self.next_frame)
        self.run_btn.clicked.connect(self.run_model_cached)
        self.finetune_btn.clicked.connect(self.finetune_model)
        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.save_btn.clicked.connect(self.save_dataset_json)

        # --- Keyboard shortcuts ---
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Left), self, activated=self.prev_frame)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Right), self, activated=self.next_frame)
        QtGui.QShortcut(QtGui.QKeySequence("V"), self, activated=self.verify_selected_toggle)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Delete), self, activated=self.delete_selected)
        QtGui.QShortcut(QtGui.QKeySequence("N"), self, activated=self.start_add_mode)
        QtGui.QShortcut(QtGui.QKeySequence("E"), self, activated=self.toggle_edit_mode)
        QtGui.QShortcut(QtGui.QKeySequence("Esc"), self, activated=self.cancel_add_mode)
        QtGui.QShortcut(QtGui.QKeySequence("S"), self, activated=self.save_dataset_json)
        QtGui.QShortcut(QtGui.QKeySequence("+"), self, activated=lambda: self.zoom_step(+1))
        QtGui.QShortcut(QtGui.QKeySequence("-"), self, activated=lambda: self.zoom_step(-1))
        QtGui.QShortcut(QtGui.QKeySequence("0"), self, activated=self.zoom_fit)

        # Worker placeholders (set by subclass)
        self.model_worker = None
        self.model_path = None

    # ==================== GUI builders ====================

    def _build_transport_bar(self) -> QtWidgets.QWidget:
        """Bottom-centered toolbar: prev/next, play/pause, zoom."""
        bar = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(10)
        h.addStretch(1)
        h.addWidget(self.prev_btn)
        h.addWidget(self.play_btn)
        h.addWidget(self.pause_btn)
        h.addWidget(self.next_btn)
        h.addSpacing(20)
        h.addWidget(self.zoom_out_btn)
        h.addWidget(self.zoom_in_btn)
        h.addWidget(self.zoom_fit_btn)
        h.addStretch(1)
        return bar

    def _build_side_panel(self) -> QtWidgets.QWidget:
        """Right-side panel for inference and annotation actions."""
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)
        v.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        # Inference group
        infer_box = QtWidgets.QGroupBox("Inference")
        infer_l = QtWidgets.QVBoxLayout(infer_box)
        infer_l.addWidget(self.run_btn)
        infer_l.addWidget(self.inference_conf_tresh)
        infer_l.addWidget(self.finetune_btn)

        # Annotation group
        anno_box = QtWidgets.QGroupBox("Annotation")
        anno_l = QtWidgets.QVBoxLayout(anno_box)
        anno_l.addWidget(self.add_btn)
        anno_l.addWidget(self.edit_btn)
        anno_l.addWidget(self.verify_btn)
        anno_l.addWidget(self.delete_btn)

        v.addWidget(infer_box)
        v.addWidget(anno_box)
        v.addStretch(1)

        panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        return panel

    def _build_menu_bar(self):
        """Create the top menu bar (File, Help)."""
        menubar = self.menuBar()

        # --- File menu ---
        file_menu = menubar.addMenu("&File")

        open_video_act = QtGui.QAction("Open Video...", self)
        open_video_act.setShortcut("Ctrl+O")
        open_video_act.triggered.connect(self.open_video)

        open_images_act = QtGui.QAction("Open Image Folder...", self)
        open_images_act.setShortcut("Ctrl+I")
        open_images_act.triggered.connect(self.open_folder)

        open_menu = QtWidgets.QMenu("Open", self)
        open_menu.addAction(open_video_act)
        open_menu.addAction(open_images_act)
        file_menu.addMenu(open_menu)

        load_masks_act = QtGui.QAction("Load Mask Folder...", self)
        load_masks_act.setShortcut("Ctrl+M")
        load_masks_act.triggered.connect(self.load_masks_folder)
        file_menu.addAction(load_masks_act)

        save_act = QtGui.QAction("Export Verified (JSON)", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self.save_dataset_json)
        file_menu.addAction(save_act)

        file_menu.addSeparator()

        exit_act = QtGui.QAction("Exit", self)
        exit_act.setShortcut("Ctrl+Q")
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        # --- Help menu ---
        help_menu = menubar.addMenu("&Help")
        about_act = QtGui.QAction("About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _show_about(self):
        QtWidgets.QMessageBox.information(
            self,
            "About",
            "Video / Image Annotation Tool\n"
            "Active-learning with YOLO-OBB and SAM2-UNet\n"
            "Built with PySide6",
        )

    # ==================== Source I/O ====================

    def open_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if path:
            self.load_video(path)

    def open_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Image Folder", "")
        if folder:
            self.load_folder(folder)

    def _set_source(self, src: FrameSource):
        """Replace the current source, reset all annotation state."""
        if self.source:
            try:
                self.source.close()
            except Exception:
                pass

        self.pred_cache.clear()
        self.dataset.clear()
        self.dataset_images_names.clear()
        self.selected_idx = None
        self.mode = "select"
        self.temp_poly_pts.clear()
        self.source = src
        self.total_frames = src.count()
        self.src_path = getattr(src, "path", None)
        self.current_idx = 0
        self.frame_slider.setRange(0, max(0, self.total_frames - 1))
        self.frame_slider.setValue(0)
        self.info(f"Loaded: {src.name()} | frames={self.total_frames} | fps={src.fps():.2f}")
        self.read_frame(self.current_idx)

    def load_video(self, path: str):
        try:
            src = VideoSource(path)
        except Exception as e:
            self.info(f"Failed to open video: {e}")
            return
        self._set_source(src)

    def load_folder(self, folder: str):
        try:
            src = ImageFolderSource(folder)
        except Exception as e:
            self.info(f"Failed to open folder: {e}")
            return
        self._set_source(src)

    def load_masks_folder(self):
        """Import a folder of PNG masks and populate pred_cache with polygons.
        Matches masks to images by filename stem.
        """
        if self.source is None or not hasattr(self.source, "path_at"):
            self.info("Load an image folder first (not a video).")
            return

        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Mask Folder (PNG)", "")
        if not folder:
            return

        # Index all .png files in the mask folder for fast lookup by stem
        mask_dir = Path(folder)
        png_files = {p.stem: str(p) for p in mask_dir.glob("*.png")}
        if not png_files:
            self.info("No .png masks found in this folder.")
            return

        self.pred_cache.clear()
        self.selected_idx = None

        prog = QtWidgets.QProgressDialog("Importing masks...", "Cancel", 0, self.total_frames, self)
        prog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(400)

        imported = 0
        for i in range(self.total_frames):
            if prog.wasCanceled():
                break
            prog.setValue(i)

            img_path = Path(self.source.path_at(i))
            mask_path = png_files.get(img_path.stem)
            if not mask_path:
                self.pred_cache[i] = []
                continue

            mask = load_mask_png(mask_path)
            if mask is None:
                self.pred_cache[i] = []
                continue

            polys_np = mask_to_polys(mask)
            poly_objs = [PolyClass(poly=p.astype(np.float32), cls_id=0, conf=1.0)
                         for p in polys_np]
            self.pred_cache[i] = poly_objs
            imported += len(poly_objs)

        prog.setValue(self.total_frames)
        self.info(f"Masks loaded: {imported} polygons across {self.total_frames} images.")
        self.redraw_current()

    # ==================== Frame reading & display ====================

    def read_frame(self, idx: int) -> bool:
        if not self.source:
            return False
        idx = max(0, min(idx, self.total_frames - 1))
        frame = self.source.read(idx)
        if frame is None:
            self.info("Failed to read frame.")
            return False
        self.current_idx = idx
        self.current_frame_bgr = frame
        # Update slider without triggering its signal
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(idx)
        self.frame_slider.blockSignals(False)
        self.update_title()
        self.redraw_current()
        return True

    def show_frame(self, frame_bgr: np.ndarray):
        """Render the frame onto the canvas with zoom & pan; update draw_map."""
        qimg = cvimg_to_qimage(frame_bgr)
        img_w, img_h = qimg.width(), qimg.height()
        lbl_w, lbl_h = self.video_label.width(), self.video_label.height()

        # Compute base scale (fit-to-window) then apply user zoom
        base = min(lbl_w / img_w, lbl_h / img_h) if img_w and img_h else 1.0
        scale = base * float(self.zoom)
        disp_w, disp_h = int(img_w * scale), int(img_h * scale)
        xoff = (lbl_w - disp_w) // 2
        yoff = (lbl_h - disp_h) // 2

        # Paint onto a canvas pixmap
        canvas = QtGui.QPixmap(lbl_w, lbl_h)
        canvas.fill(QtGui.QColor(17, 17, 17))
        painter = QtGui.QPainter(canvas)
        scaled = QtGui.QPixmap.fromImage(qimg).scaled(
            disp_w, disp_h,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        draw_x = int(xoff - self.pan_img[0] * scale)
        draw_y = int(yoff - self.pan_img[1] * scale)
        painter.drawPixmap(draw_x, draw_y, scaled)
        painter.end()

        # Store mapping for display ↔ image coordinate conversion
        self.draw_map = {
            "scale": scale, "xoff": xoff, "yoff": yoff,
            "img_w": img_w, "img_h": img_h,
            "panx": float(self.pan_img[0]), "pany": float(self.pan_img[1]),
            "base": base, "lbl_w": lbl_w, "lbl_h": lbl_h,
        }
        self.video_label.setPixmap(canvas)

    def redraw_current(self):
        """Re-render the current frame with annotations and any in-progress add ghost."""
        if self.current_frame_bgr is None:
            return
        base = self.current_frame_bgr
        annots = self.pred_cache.get(self.current_idx, [])
        annotated = draw_annotations(
            base, annots, self.inference_conf_tresh.value(),
            self.class_names, self.selected_idx,
            show_conf=False, show_label=False,
        )

        # Draw ghost polygon while in ADD mode
        if self.mode == "add" and self.temp_poly_pts:
            ghost = np.array(self.temp_poly_pts, dtype=np.int32)
            cv2.polylines(annotated, [ghost], isClosed=False,
                          color=(200, 200, 200), thickness=1, lineType=cv2.LINE_AA)
            for (gx, gy) in ghost:
                cv2.circle(annotated, (int(gx), int(gy)), 3,
                           (200, 200, 200), -1, lineType=cv2.LINE_AA)

        self.show_frame(annotated)

    # ==================== Playback controls ====================

    def prev_frame(self):
        if not self.source:
            return
        self.pause()
        self.read_frame(self.current_idx - 1)

    def next_frame(self):
        if not self.source:
            return
        self.pause()
        self.read_frame(self.current_idx + 1)

    def _on_slider_released(self):
        if not self.source:
            return
        self.pause()
        self.read_frame(self.frame_slider.value())

    def play(self):
        if not self.source or self.playing:
            return
        fps = self.source.fps() or 25
        self.play_timer.start(max(15, int(1000 / fps)))
        self.playing = True
        self.update_title()

    def pause(self):
        if self.playing:
            self.play_timer.stop()
            self.playing = False
            self.update_title()

    def toggle_play_pause(self):
        if self.playing:
            self.pause()
        else:
            self.play()

    def _on_play_tick(self):
        if self.current_idx + 1 >= self.total_frames:
            self.pause()
            return
        self.read_frame(self.current_idx + 1)

    # ==================== Model inference ====================

    def run_model_cached(self):
        """Launch model inference on the current frame in a background thread."""
        idx = self.current_idx
        if self.current_frame_bgr is None:
            return

        self.run_btn.setEnabled(False)
        self.run_btn.setText("Inference running...")
        conf = float(self.inference_conf_tresh.value())

        self.worker_thread = QtCore.QThread(self)
        self.worker = self.model_worker(
            idx, self.current_frame_bgr,
            conf=conf, imgsz=1024, model_path=self.model_path,
        )
        self.worker.moveToThread(self.worker_thread)

        # Wire signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_inference_done)
        self.worker.error.connect(self._on_inference_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def init_model(self, model_path: str):
        raise NotImplementedError("Subclass must implement init_model.")

    def launch_finetune_worker(self, base_model: str):
        raise NotImplementedError("Subclass must implement launch_finetune_worker.")

    def finetune_model(self):
        """Validate preconditions and launch fine-tuning in a plain Python thread.

        We use threading.Thread instead of QThread because Lightning's Trainer
        and ultralytics' model.train() internally create timers and progress
        bars that conflict with QThread's event loop, producing warnings like
        "Timers cannot be started/stopped from another thread".
        A small _SignalBridge QObject living on the main thread relays
        progress/finished/error signals safely back to the GUI.
        """
        if not self.src_path:
            QtWidgets.QMessageBox.warning(self, "Fine-tune", "Load a source first.")
            return
        if not self.dataset:
            QtWidgets.QMessageBox.warning(self, "Fine-tune", "No verified annotations to train on.")
            return
        if not self.class_names:
            QtWidgets.QMessageBox.warning(self, "Fine-tune", "No class names defined.")
            return

        self.finetune_btn.setEnabled(False)
        self.finetune_btn.setText("Finetuning...")

        base_model = self.model_path
        worker = self.launch_finetune_worker(base_model)

        # Signal bridge lives on the main thread — safe for GUI updates
        bridge = _FinetuneSignalBridge(self)
        bridge.progress.connect(lambda msg, p: self.info(f"{msg} ({int(p * 100)}%)"))
        bridge.error.connect(self._on_finetune_error)
        bridge.finished.connect(self._on_finetune_done)
        # prevent garbage collection while the thread is alive
        self._finetune_bridge = bridge

        def _run():
            # Forward worker signals through the bridge (thread-safe via Qt signals)
            worker.progress.connect(bridge.progress)
            worker.error.connect(bridge.error)
            worker.finished.connect(bridge.finished)
            worker.run()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        self._finetune_thread = t

    def _on_finetune_error(self, msg: str):
        QtWidgets.QMessageBox.critical(self, "Fine-tune Error", msg)
        self.info(f"Error: {msg}")
        self.finetune_btn.setEnabled(True)
        self.finetune_btn.setText("Finetune Model")

    def _on_finetune_done(self, best_pt_path: str):
        self.info(f"Fine-tune complete: {best_pt_path}")
        # Load the new weights immediately
        try:
            self.model_worker._model = self.init_model(best_pt_path)
            self.model_path = best_pt_path
            self.info(f"Loaded fine-tuned model: {os.path.basename(best_pt_path)}")
        except Exception as e:
            self.info(f"Model saved, but failed to load: {e}")
        self.finetune_btn.setEnabled(True)
        self.finetune_btn.setText("Finetune Model")

    def _on_inference_done(self, frame_idx: int, class_names, annots: List[PolyClass]):
        self.class_names = class_names
        self.pred_cache[frame_idx] = annots
        self.selected_idx = None
        if frame_idx == self.current_idx:
            self.redraw_current()
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Model")
        self.info(f"Predictions cached for frame {frame_idx + 1}.")

    def _on_inference_error(self, msg: str):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Model")
        self.info(f"Inference error: {msg}")

    # ==================== Mouse / keyboard interaction ====================

    def display_to_image_coords(self, xd: int, yd: int):
        """Convert display (widget) coordinates to image coordinates.
        Returns (None, None) if the point falls outside the image area.
        """
        m = self.draw_map
        if not m:
            return None, None
        s = m["scale"]
        xoff, yoff = m["xoff"], m["yoff"]
        panx, pany = m["panx"], m["pany"]
        img_w, img_h = m["img_w"], m["img_h"]

        xi = (xd - (xoff - panx * s)) / s
        yi = (yd - (yoff - pany * s)) / s
        if xi < 0 or yi < 0 or xi >= img_w or yi >= img_h:
            return None, None
        return float(xi), float(yi)

    def _check_canvas_mouse_event(self, event):
        """Extract display and image coordinates from a mouse event on the canvas.
        Returns None if the frame is not loaded or the cursor is outside the image.
        """
        if self.current_frame_bgr is None:
            return None
        if not hasattr(event, "position"):
            return None
        pos = event.position()
        x_disp, y_disp = int(pos.x()), int(pos.y())
        x_img, y_img = self.display_to_image_coords(x_disp, y_disp)
        if x_img is None:
            return None
        return x_disp, y_disp, x_img, y_img

    def eventFilter(self, obj, event):
        # Clear spinbox focus on any click so shortcuts work
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            if self.inference_conf_tresh.hasFocus():
                self.inference_conf_tresh.clearFocus()
            self.video_label.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)

        # Only handle canvas events from here
        if obj is not self.video_label:
            return super().eventFilter(obj, event)

        # --- Mouse wheel → zoom ---
        if event.type() == QtCore.QEvent.Type.Wheel:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_step(+1, anchor_disp=event.position())
            elif delta < 0:
                self.zoom_step(-1, anchor_disp=event.position())
            return True

        # --- Map coordinates for press/move/release ---
        if event.type() in (
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QEvent.Type.MouseMove,
            QtCore.QEvent.Type.MouseButtonRelease,
        ):
            coords = self._check_canvas_mouse_event(event)
            if coords is None:
                return False
            x_disp, y_disp, x_img, y_img = coords

        # --- PAN with Space + left drag ---
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            if event.button() == QtCore.Qt.MouseButton.LeftButton and self.space_held:
                self._pan_dragging = True
                self._pan_last_disp = (x_disp, y_disp)
                return True

        elif event.type() == QtCore.QEvent.Type.MouseMove:
            if getattr(self, "_pan_dragging", False):
                dx = x_disp - self._pan_last_disp[0]
                dy = y_disp - self._pan_last_disp[1]
                s = self.draw_map.get("scale", 1.0)
                self.pan_img[0] -= dx / s
                self.pan_img[1] -= dy / s
                self._pan_last_disp = (x_disp, y_disp)
                self.redraw_current()
                return True

        elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
            if getattr(self, "_pan_dragging", False):
                self._pan_dragging = False
                return True

        # --- LEFT CLICK: add point / select / start drag ---
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                if self.mode == "add":
                    self.add_click_point(x_img, y_img)
                    return True

                hit_idx = self.pick_annot(x_img, y_img)
                if hit_idx is not None:
                    self.selected_idx = hit_idx
                    self.redraw_current()
                    # Ctrl+drag a vertex in edit mode
                    if self.mode == "edit" and (event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
                        v = self.pick_vertex(x_img, y_img)
                        if v is not None:
                            self.vertex_drag_idx = v
                            self.dragging = True
                            return True
                    # Start translation drag
                    boxes = self.pred_cache.get(self.current_idx, [])
                    if self.selected_idx is not None and self.selected_idx < len(boxes):
                        self.dragging = True
                        self.drag_start_img = (x_img, y_img)
                        self.orig_poly = boxes[self.selected_idx].poly.copy()
                    return True
                else:
                    if self.mode != "add":
                        self.selected_idx = None
                        self.redraw_current()
                    return True

            # --- RIGHT CLICK: quick verify toggle ---
            elif event.button() == QtCore.Qt.MouseButton.RightButton:
                hit_idx = self.pick_annot(x_img, y_img)
                if hit_idx is not None:
                    self.selected_idx = hit_idx
                    self.verify_selected_toggle()
                    self.redraw_current()
                return True

        # --- DRAG (move or vertex edit) ---
        elif event.type() == QtCore.QEvent.Type.MouseMove:
            if self.dragging:
                if self.vertex_drag_idx is not None:
                    self._set_vertex_selected(self.vertex_drag_idx, x_img, y_img)
                elif self.drag_start_img is not None:
                    dx = x_img - self.drag_start_img[0]
                    dy = y_img - self.drag_start_img[1]
                    self._translate_selected(dx, dy)
                self.redraw_current()
                return True

        # --- RELEASE: end drag ---
        elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
            if self.dragging:
                self.dragging = False
                self.vertex_drag_idx = None
                self.drag_start_img = None
                self.orig_poly = None
                self.update_dataset_for_frame(self.current_idx)
                self.redraw_current()
                return True

        return super().eventFilter(obj, event)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key.Key_Delete:
            self.delete_selected()
        elif event.key() == QtCore.Qt.Key.Key_V:
            self.verify_selected_toggle()
        elif event.key() == QtCore.Qt.Key.Key_Space:
            self.space_held = True
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key.Key_Space:
            self.space_held = False
            return
        super().keyReleaseEvent(event)

    # ==================== Picking / hit-testing ====================

    def pick_annot(self, x: float, y: float) -> Optional[int]:
        """Return the index of the smallest annotation polygon containing (x, y)."""
        annots = self.pred_cache.get(self.current_idx, [])
        if not annots:
            return None
        best, best_area = None, None
        for i, b in enumerate(annots):
            if b.deleted:
                continue
            pts = b.poly.reshape(-1, 2).astype(np.float32)
            if cv2.pointPolygonTest(pts, (x, y), measureDist=False) >= 0:
                area = cv2.contourArea(pts.astype(np.int32))
                if best is None or area < best_area:
                    best, best_area = i, area
        return best

    def pick_vertex(self, x: float, y: float, tol_px: int = 10) -> Optional[int]:
        """Return the index of the nearest corner of the selected annotation, or None."""
        if self.selected_idx is None:
            return None
        annots = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx >= len(annots):
            return None
        a = annots[self.selected_idx]
        if a.deleted:
            return None
        pts = a.poly.reshape(-1, 2)
        for i in range(pts.shape[0]):
            if np.hypot(pts[i, 0] - x, pts[i, 1] - y) <= tol_px:
                return i
        return None

    # ==================== Annotation actions ====================

    def verify_selected_toggle(self):
        boxes = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(boxes):
            return
        box = boxes[self.selected_idx]
        if box.deleted:
            return
        box.verified = not box.verified
        self.update_dataset_for_frame(self.current_idx)
        state = "verified" if box.verified else "unverified"
        self.info(f"Box #{self.selected_idx} → {state}")
        self.selected_idx = None
        self.redraw_current()

    def delete_selected(self):
        boxes = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(boxes):
            return
        box = boxes[self.selected_idx]
        box.deleted = True
        box.verified = False
        self.update_dataset_for_frame(self.current_idx)
        self.info(f"Box #{self.selected_idx} deleted.")
        self.selected_idx = None
        self.redraw_current()

    def update_dataset_for_frame(self, frame_idx: int):
        """Sync the verified dataset dict: keep only verified & non-deleted boxes."""
        all_boxes = self.pred_cache.get(frame_idx, [])
        self.dataset[frame_idx] = [b for b in all_boxes if b.verified and not b.deleted]
        if isinstance(self.source, ImageFolderSource):
            self.dataset_images_names[frame_idx] = self.source.path_at(frame_idx)

    # ==================== Polygon editing ====================

    def _translate_selected(self, dx: float, dy: float):
        """Move the selected annotation by (dx, dy) from its original position."""
        annots = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(annots):
            return
        b = annots[self.selected_idx]
        b.poly = (self.orig_poly + np.array([dx, dy], dtype=np.float32)).astype(np.float32)
        self.update_dataset_for_frame(self.current_idx)

    def _set_vertex_selected(self, idx: int, x: float, y: float):
        """Move vertex `idx` of the selected annotation to (x, y)."""
        annots = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(annots):
            return
        b = annots[self.selected_idx]
        p = b.poly.copy()
        p[idx] = [x, y]
        b.poly = p.astype(np.float32)
        self.update_dataset_for_frame(self.current_idx)

    # ==================== Mode management ====================

    def set_mode(self, mode: str):
        self.mode = mode
        if mode != "add":
            self.temp_poly_pts.clear()
        self.info(f"Mode: {mode}")

    def start_add_mode(self):
        self.set_mode("add")
        self.selected_idx = None
        self.redraw_current()

    def cancel_add_mode(self):
        if self.mode == "add":
            self.temp_poly_pts.clear()
            self.set_mode("select")
            self.redraw_current()

    def toggle_edit_mode(self):
        self.set_mode("edit" if self.mode != "edit" else "select")

    # ==================== Add-polygon pipeline (n clicks) ====================

    def add_click_point(self, x: float, y: float):
        """Base implementation: close polygon when clicking near the first point."""
        self.temp_poly_pts.append([x, y])
        first_point = self.temp_poly_pts[0]

        # Close polygon if the new point is near the first point (and we have >= 3 pts)
        if len(self.temp_poly_pts) > 2 and np.linalg.norm(
            np.array([x, y]) - np.array(first_point)
        ) < 8.0:
            self.temp_poly_pts[-1] = first_point
            pts = np.array(self.temp_poly_pts, dtype=np.float32)
            new_box = PolyClass(poly=pts, cls_id=0, conf=1.0, verified=False)
            self.pred_cache.setdefault(self.current_idx, []).append(new_box)
            self.selected_idx = len(self.pred_cache[self.current_idx]) - 1
            self.temp_poly_pts.clear()
            self.set_mode("select")
            self.update_dataset_for_frame(self.current_idx)
        self.redraw_current()

    # ==================== Zoom ====================

    def zoom_fit(self):
        self.zoom = 1.0
        self.pan_img[:] = 0.0
        self.redraw_current()

    def _clamp_pan(self):
        """Clamp pan so at least 10% of the image remains visible."""
        m = self.draw_map
        if not m or "base" not in m:
            return
        s = m["base"] * self.zoom
        img_w, img_h = m["img_w"], m["img_h"]
        lbl_w, lbl_h = m["lbl_w"], m["lbl_h"]
        margin = 0.1
        max_pan_x = (img_w * s - (1.0 - margin) * lbl_w) / s / 2.0
        max_pan_y = (img_h * s - (1.0 - margin) * lbl_h) / s / 2.0
        self.pan_img[0] = float(np.clip(self.pan_img[0], -max_pan_x, max_pan_x))
        self.pan_img[1] = float(np.clip(self.pan_img[1], -max_pan_y, max_pan_y))

    def zoom_step(self, direction: int, anchor_disp: QtCore.QPointF | None = None):
        """Zoom in (+1) or out (-1), anchored around the cursor position."""
        if anchor_disp is None:
            w, h = self.video_label.width(), self.video_label.height()
            anchor_disp = QtCore.QPointF(w / 2.0, h / 2.0)

        m = self.draw_map
        if not m or self.current_frame_bgr is None:
            return

        step = 1.25 if direction > 0 else 0.8
        new_zoom = float(np.clip(self.zoom * step, self.min_zoom, self.max_zoom))
        if abs(new_zoom - self.zoom) < 1e-6:
            return

        xd, yd = float(anchor_disp.x()), float(anchor_disp.y())
        # Image coord at anchor under the old zoom
        xi, yi = self.display_to_image_coords(int(xd), int(yd))
        if xi is None:
            self.zoom = new_zoom
            self.redraw_current()
            return

        # Recompute offsets so the anchor stays under the cursor after zoom
        self.zoom = new_zoom
        base = float(m["base"])
        img_w, img_h = m["img_w"], m["img_h"]
        lbl_w, lbl_h = m["lbl_w"], m["lbl_h"]
        new_scale = base * self.zoom
        xoff = (lbl_w - img_w * new_scale) / 2.0
        yoff = (lbl_h - img_h * new_scale) / 2.0
        self.pan_img[0] = (xoff + xi * new_scale - xd) / new_scale
        self.pan_img[1] = (yoff + yi * new_scale - yd) / new_scale
        self._clamp_pan()
        self.redraw_current()

    # ==================== Save / export ====================

    def save_dataset_json(self):
        if not self.dataset:
            self.info("Dataset empty (no verified boxes).")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save dataset (JSON)", "dataset.json",
            "JSON (*.json);;All files (*)",
        )
        if not path:
            return
        data: Dict[str, Any] = {
            "video_path": self.src_path,
            "total_frames": self.total_frames,
            "frames": [],
        }
        for frame_idx, boxes in sorted(self.dataset.items()):
            if not boxes:
                continue
            data["frames"].append({
                "frame_index": int(frame_idx),
                "boxes": [b.to_json() for b in boxes],
            })
        try:
            import json
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.info(f"Saved dataset to {os.path.basename(path)}")
        except Exception as e:
            self.info(f"Save failed: {e}")

    # ==================== Utilities ====================

    def info(self, text: str):
        """Show a transient message in the status bar (no layout impact)."""
        self.statusBar().showMessage(text, 5000)

    def update_title(self):
        title = "Annotation Tool"
        if self.source:
            title += f" | {self.source.name()}"
            title += f" | frame {self.current_idx + 1}/{self.total_frames}"
        self.setWindowTitle(title)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self.redraw_current()


# ---------------------------------------------------------------------------
# Subclass: Oriented Bounding Box detection (YOLO-OBB)
# ---------------------------------------------------------------------------

class OBB_VideoPlayer(Base):
    def __init__(self, model_path: str = YOLO_MODEL_PATH):
        super().__init__()
        self.model_path = model_path
        self.model_worker = DetectionWorker

    def add_click_point(self, x: float, y: float):
        """OBB-specific: 3 clicks define a rectangle (2 edge points + width)."""
        if len(self.temp_poly_pts) == 2:
            # Third click: project to form an oriented rectangle
            primes = find_orthogonal_projection(
                self.temp_poly_pts[0], self.temp_poly_pts[1], [x, y],
            )
            pts = np.concatenate(
                (self.temp_poly_pts, primes), axis=0, dtype=np.float32,
            )
            new_box = OBBOX(poly=pts, cls_id=0, conf=1.0, verified=False)
            self.pred_cache.setdefault(self.current_idx, []).append(new_box)
            self.selected_idx = len(self.pred_cache[self.current_idx]) - 1
            self.temp_poly_pts.clear()
            self.set_mode("select")
            self.update_dataset_for_frame(self.current_idx)
        else:
            self.temp_poly_pts.append([x, y])
        self.redraw_current()

    def init_model(self, model_path: str):
        return YOLO(model_path)

    def launch_finetune_worker(self, base_model: str):
        return DetectFinetuneWorker(
            video_path=self.src_path,
            dataset=self.dataset,
            class_names=self.class_names,
            base_model_path=base_model,
            out_root=os.path.join(os.getcwd(), "finetune_runs"),
            epochs=20,
            imgsz=1024,
            batch=16,
            val_split=0.1,
        )


# ---------------------------------------------------------------------------
# Subclass: Binary segmentation (SAM2-UNet)
# ---------------------------------------------------------------------------

class Seg_VideoPlayer(Base):
    def __init__(self, model_path: str = SAM2_UNET_MODEL_PATH):
        super().__init__()
        self.model_path = model_path
        self.model_worker = SegWorker

    def init_model(self, model_path: str):
        """Load SAM2-UNet from the given checkpoint."""
        net = SAM2UNet(
            config="tiny",
            sam_checkpoint_path=SAM2_CHECKPOINT_PATH,
            freeze_encorder=True,
        ).to("cuda")
        lit = LitBinarySeg.load_from_checkpoint(
            model_path,
            net=net,
            deep_supervision=False,
            dice_use_all_outputs=False,
            pos_weight=200,
        ).eval().to("cuda")
        return lit

    def launch_finetune_worker(self, base_model: str):
        return SegFinetuneWorker(
            dataset=self.dataset,
            dataset_images_names=self.dataset_images_names,
            base_model_path=base_model,
            out_root=os.path.join(os.getcwd(), "seg_finetune_runs"),
            epochs=20,
            batch=8,
            val_split=0.1,
        )