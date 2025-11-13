
import os
from typing import Optional, List, Dict, Any
from pathlib import Path
import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

# ------ Local imports from utils.py ------
from .utils import PolyClass,OBBOX, cvimg_to_qimage, draw_annotations, find_orthogonal_projection, ensure_bgr_u8, mask_to_polys,load_mask_png
from .qt_workers import SegWorker, SegFinetuneWorker, DetectionWorker, DetectFinetuneWorker, SAM2_UNET_MODEL_PATH, YOLO_MODEL_PATH

from ultralytics import YOLO

from src.deep_learning.models.SAMUNET import LitBinarySeg, SAM2UNet


class FrameSource:
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
        if not ok or frame is None:
            return None
        return frame

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
        files = [f for f in os.listdir(self.path)]
        files = [f for f in files if os.path.splitext(f)[1].lower() in self.IMAGE_EXTS]
        if not files:
            raise RuntimeError("No images found in folder.")
        # tri naturel simple (numérique si possible)
        def _key(s):
            import re
            return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]
        files.sort(key=_key)
        self.paths = [os.path.join(self.path, f) for f in files]

    def count(self) -> int:
        return len(self.paths)

    def read(self, idx: int) -> Optional[np.ndarray]:
        idx = max(0, min(idx, len(self.paths) - 1))
        p = self.paths[idx]
        # cv2.imread lit TIFF (première page). Si multi-page, on prend la page 0.
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        return ensure_bgr_u8(img)

    def fps(self) -> float:
        # lecture image par image: fps arbitraire (pour Play)
        return 10.0

    def name(self) -> str:
        return os.path.basename(self.path)
    
    def path_at(self, idx: int) -> str:
        idx = max(0, min(idx, len(self.paths) - 1))
        return self.paths[idx]

class Base(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.baseTitle = "Video Player"
        self.setWindowTitle(self.baseTitle)
        self.resize(1200, 760)

        # Video state
        # self.cap: Optional[cv2.VideoCapture] = None
        self.source: Optional[FrameSource] = None
        self.total_frames: int = 0
        self.current_idx: int = 0
        self.current_frame_bgr: Optional[np.ndarray] = None
        self.src_path: Optional[str] = None
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._on_play_tick)
        self.playing = False

        # --- Zoom & Pan state ---
        self.zoom = 1.0              # relative to "fit to window" scale
        self.min_zoom = 0.25
        self.max_zoom = 8.0
        self.pan_img = np.array([0.0, 0.0], dtype=np.float32)   # pan in *image* pixels


        # Annotations state
        self.pred_cache: Dict[int, List[PolyClass]] = {}   # frame_idx -> boxes
        self.class_names = None
        self.selected_idx: Optional[int] = None           # index in current boxes list

        # Dataset (verified only)
        self.dataset: Dict[int, List[PolyClass]] = {}      # frame_idx -> verified boxes
        self.dataset_images_names: Dict[int, str] = {}  # frame_idx -> image filename

        # For click->image coord mapping
        self.draw_map = {"scale": 1.0, "xoff": 0, "yoff": 0}

        self.space_held = False  # track Space key for pan

        #modes
        self.mode = "select"            # "select" | "add" | "edit"
        self.temp_poly_pts = []         # for ADD mode: up to 4 (x,y) in image coords
        self.dragging = False
        self.drag_start_img = None      # (xi, yi) at mouse-down
        self.orig_poly = None           # copy of polygon at drag start
        self.vertex_drag_idx = None     # index 0..3 if dragging a vertex

        # UI
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setMouseTracking(True)              # get MouseMove events
        self.video_label.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background:#111; border:1px solid #333;")
        self.video_label.setMinimumSize(720, 405)
        self.installEventFilter(self)
        self.centralWidget().installEventFilter(self)
        self.video_label.installEventFilter(self)


        # --- new action buttons with tooltips showing shortcuts ---
        self.add_btn = QtWidgets.QPushButton("Add (N)")
        self.add_btn.setToolTip("Start add-box mode.\nClick 4 points to make an oriented box.\nShortcut: N")
        self.add_btn.clicked.connect(self.start_add_mode)

        self.edit_btn = QtWidgets.QPushButton("Edit (E)")
        self.edit_btn.setToolTip("Toggle edit mode.\nDrag box to move; CTRL+drag a corner to bend.\nShortcut: E")
        self.edit_btn.clicked.connect(self.toggle_edit_mode)

        # (verify/delete already exist via right-click/V/Delete, but buttons help)
        self.verify_btn = QtWidgets.QPushButton("Verify (V)")
        self.verify_btn.setToolTip("Verify/Unverify selected box.\nShortcut: V")
        self.verify_btn.clicked.connect(self.verify_selected_toggle)

        self.delete_btn = QtWidgets.QPushButton("Delete (Del)")
        self.delete_btn.setToolTip("Delete selected box.\nShortcut: Delete")
        self.delete_btn.clicked.connect(self.delete_selected)

        # --- Zoom buttons ---
        self.zoom_in_btn = QtWidgets.QPushButton("Zoom +")
        self.zoom_in_btn.setToolTip("Zoom in ( + / mouse wheel ↑ )")
        self.zoom_out_btn = QtWidgets.QPushButton("Zoom −")
        self.zoom_out_btn.setToolTip("Zoom out ( - / mouse wheel ↓ )")
        self.zoom_fit_btn = QtWidgets.QPushButton("Fit")
        self.zoom_fit_btn.setToolTip("Reset zoom & pan to fit ( 0 )")

        self.zoom_in_btn.clicked.connect(lambda: self.zoom_step(+1))
        self.zoom_out_btn.clicked.connect(lambda: self.zoom_step(-1))
        self.zoom_fit_btn.clicked.connect(self.zoom_fit)

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

        self.info_label = QtWidgets.QLabel(
            "Open a video → Run YOLO once per frame (cached). "
            "Left-click: select | Right-click or V: verify (turns green, hides label) | Delete: remove."
        )
        self.info_label.setStyleSheet("color:#aaa;")

        # 1) Left side: video + slider stacked vertically
        left_stack = QtWidgets.QWidget()
        left_v = QtWidgets.QVBoxLayout(left_stack)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(6)
        left_v.addWidget(self.video_label, stretch=1)
        left_v.addWidget(self.frame_slider)

        # 2) Main content row: left stack + right column
        content_row = QtWidgets.QHBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(10)
        content_row.addWidget(left_stack, stretch=1)
        content_row.addWidget(self._build_side_panel(), stretch=0)

        self._build_menu_bar()

        # 3) Overall page: content row, bottom transport bar (centered), info label
        page = QtWidgets.QVBoxLayout(self.centralWidget())
        page.setContentsMargins(8, 8, 8, 8)
        page.setSpacing(8)
        page.addLayout(content_row, stretch=1)
        page.addWidget(self._build_transport_bar())
        page.addWidget(self.info_label)

        # Signals
        self.open_video_btn.clicked.connect(self.open_video)
        self.open_images_btn.clicked.connect(self.open_folder)
        self.prev_btn.clicked.connect(self.prev_frame)
        self.next_btn.clicked.connect(self.next_frame)
        self.run_btn.clicked.connect(self.run_model_cached)
        self.finetune_btn.clicked.connect(self.finetune_model)
        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.save_btn.clicked.connect(self.save_dataset_json)

        # Shortcuts
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Left), self, activated=self.prev_frame)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Right), self, activated=self.next_frame)
        # QtGui.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.toggle_play_pause)
        QtGui.QShortcut(QtGui.QKeySequence("V"), self, activated=self.verify_selected_toggle)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Delete), self, activated=self.delete_selected)
        QtGui.QShortcut(QtGui.QKeySequence("N"), self, activated=self.start_add_mode)
        QtGui.QShortcut(QtGui.QKeySequence("E"), self, activated=self.toggle_edit_mode)
        QtGui.QShortcut(QtGui.QKeySequence("Esc"), self, activated=self.cancel_add_mode)
        QtGui.QShortcut(QtGui.QKeySequence("S"), self, activated=self.save_dataset_json)
        QtGui.QShortcut(QtGui.QKeySequence("+"), self, activated=lambda: self.zoom_step(+1))
        QtGui.QShortcut(QtGui.QKeySequence("-"), self, activated=lambda: self.zoom_step(-1))
        QtGui.QShortcut(QtGui.QKeySequence("0"), self, activated=self.zoom_fit)


        self.model_worker = None
        self.model_path = None


    # ---------- Build GUI ----------
    def _build_transport_bar(self) -> QtWidgets.QWidget:
        """
        Bottom-centered toolbar for reading controls: prev/next, play/pause, zoom.
        """
        bar = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(10)

        # center the controls
        h.addStretch(1)

        # reading controls
        h.addWidget(self.prev_btn)
        h.addWidget(self.play_btn)
        h.addWidget(self.pause_btn)
        h.addWidget(self.next_btn)

        # a little gap then zoom controls
        h.addSpacing(20)
        h.addWidget(self.zoom_out_btn)
        h.addWidget(self.zoom_in_btn)
        h.addWidget(self.zoom_fit_btn)

        h.addStretch(1)

        return bar

    def _build_side_panel(self) -> QtWidgets.QWidget:
        """
        Right-side vertical panel for all other actions.
        """
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        # Make the column hug the top
        v.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        # Group: Inference
        infer_box = QtWidgets.QGroupBox("Inference")
        infer_l = QtWidgets.QVBoxLayout(infer_box)
        infer_l.addWidget(self.run_btn)
        infer_l.addWidget(self.inference_conf_tresh)
        infer_l.addWidget(self.finetune_btn)

        # Group: Annotation
        anno_box = QtWidgets.QGroupBox("Annotation")
        anno_l = QtWidgets.QVBoxLayout(anno_box)
        anno_l.addWidget(self.add_btn)
        anno_l.addWidget(self.edit_btn)
        anno_l.addWidget(self.verify_btn)
        anno_l.addWidget(self.delete_btn)

        # Add groups to the column
        # v.addWidget(file_box)
        v.addWidget(infer_box)
        v.addWidget(anno_box)
        v.addStretch(1)

        # keep the column narrow
        panel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding)
        return panel
    
    def _build_menu_bar(self):
        """
        Create a top menu bar with File and Help menus.
        """
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

        # --- Optional Help/About menu ---
        help_menu = menubar.addMenu("&Help")
        about_act = QtGui.QAction("About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)  

    def _show_about(self):
        QtWidgets.QMessageBox.information(
            self,
            "About",
            "Video Annotation Tool\n© 2025 Your Name / Lab\nBuilt with PyQt6"
        )
        
    # ---------- Video I/O ----------
    def open_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if not path:
            return
        self.load_video(path)

    def open_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Image Folder", "")
        if not folder:
            return
        self.load_folder(folder)

    def _set_source(self, src: FrameSource):
        # ferme l’ancienne source
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
        """
        Ouvre un dossier de masques .png, et pré-remplit self.pred_cache:
        self.pred_cache[frame_idx] = [PolyClass(...), ...] pour chaque image.
        - On matche par nom de fichier (stem identique, extension .png).
        - cls_id = 0, conf = 1.0, verified/deleted = False.
        """
        # Vérification source
        if self.source is None or not hasattr(self.source, "path_at"):
            self.info("Charge d'abord un dossier d'images (pas une vidéo).")
            return

        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Mask Folder (PNG)", "")
        if not folder:
            return

        # Indexation des .png existants dans le dossier de masques (pour lookup rapide)
        mask_dir = Path(folder)
        png_files = {p.stem: str(p) for p in mask_dir.glob("*.png")}
        if not png_files:
            self.info("Aucun masque .png trouvé dans ce dossier.")
            return

        # Reset (on remplit depuis zéro)
        self.pred_cache.clear()
        self.selected_idx = None

        # Progression (optionnel mais utile pour gros dossiers)
        prog = QtWidgets.QProgressDialog("Importing masks…", "Cancel", 0, self.total_frames, self)
        prog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(400)

        imported = 0
        for i in range(self.total_frames):
            if prog.wasCanceled():
                break
            prog.setValue(i)

            img_path = Path(self.source.path_at(i))
            stem = img_path.stem
            mask_path = png_files.get(stem)
            if not mask_path:
                # pas de masque pour cette frame → liste vide
                self.pred_cache[i] = []
                continue

            # Lecture du masque (8/16 bits OK)
            mask = load_mask_png(mask_path)
            if mask is None:
                self.pred_cache[i] = []
                continue

            # Conversion en polygones
            polys_np = mask_to_polys(mask)  # List[np.ndarray (n,2)]
            poly_objs = [PolyClass(poly=p.astype(np.float32), cls_id=0, conf=1.0) for p in polys_np]
            self.pred_cache[i] = poly_objs
            imported += len(poly_objs)

        prog.setValue(self.total_frames)
        self.info(f"Masques chargés: {imported} polygones sur {self.total_frames} images.")
        self.redraw_current()
            
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
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(idx)
        self.frame_slider.blockSignals(False)
        self.update_title()
        self.redraw_current()
        # self.frame_changed.emit(idx)
        return True

    # ---------- Display & mapping ----------
    def show_frame(self, frame_bgr: np.ndarray):
        """Draw frame with zoom & pan; keep accurate mapping for picking/editing."""
        qimg = cvimg_to_qimage(frame_bgr)
        img_w, img_h = qimg.width(), qimg.height()
        lbl_w, lbl_h = self.video_label.width(), self.video_label.height()

        # base fit scale
        base = min(lbl_w / img_w, lbl_h / img_h) if img_w and img_h else 1.0
        scale = base * float(self.zoom)
        disp_w, disp_h = int(img_w * scale), int(img_h * scale)
        xoff = (lbl_w - disp_w) // 2
        yoff = (lbl_h - disp_h) // 2

        # Render to a canvas; apply pan by shifting the draw origin by -pan*scale
        canvas = QtGui.QPixmap(lbl_w, lbl_h)
        canvas.fill(QtGui.QColor(17, 17, 17))
        painter = QtGui.QPainter(canvas)

        # Scale the source image
        scaled = QtGui.QPixmap.fromImage(qimg).scaled(
            disp_w, disp_h,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )

        draw_x = int(xoff - self.pan_img[0] * scale)
        draw_y = int(yoff - self.pan_img[1] * scale)
        painter.drawPixmap(draw_x, draw_y, scaled)
        painter.end()

        # keep mapping for picking math
        self.draw_map = {
            "scale": scale, "xoff": xoff, "yoff": yoff,
            "img_w": img_w, "img_h": img_h,
            "panx": float(self.pan_img[0]), "pany": float(self.pan_img[1]),
            "base": base, "lbl_w": lbl_w, "lbl_h": lbl_h
        }
        self.video_label.setPixmap(canvas)


    def redraw_current(self):
        if self.current_frame_bgr is None:
            return
        base = self.current_frame_bgr
        annots = self.pred_cache.get(self.current_idx, [])
        annotated = draw_annotations(base, annots, self.inference_conf_tresh.value(), self.class_names, self.selected_idx, show_conf=False, show_label=False)

        # --- ghost for ADD mode ---
        if self.mode == "add" and len(self.temp_poly_pts) > 0:
            ghost = np.array(self.temp_poly_pts, dtype=np.int32)
            cv2.polylines(annotated, [ghost], isClosed=False, color=(200, 200, 200), thickness=1, lineType=cv2.LINE_AA)
            # draw points
            for (gx, gy) in ghost:
                cv2.circle(annotated, (int(gx), int(gy)), 3, (200, 200, 200), -1, lineType=cv2.LINE_AA)

        self.show_frame(annotated)

    # ---------- Controls ----------
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

    # ---------- Inference with cache ----------
    def run_model_cached(self):
        # """Only run model if no cache exists for this frame"""
        idx = self.current_idx
        # if idx in self.pred_cache and len(self.pred_cache[idx]) > 0 :
        #     self.info("Using cached predictions for this frame.")
        #     self.redraw_current()
        #     return

        if self.current_frame_bgr is None:
            return

        self.run_btn.setEnabled(False)
        self.run_btn.setText("Inference running...")
        conf = float(self.inference_conf_tresh.value())

        self.worker_thread = QtCore.QThread(self)

        print("Launching inference worker...", self.model_path)
        self.worker = self.model_worker(idx, self.current_frame_bgr, conf=conf, model_path=self.model_path)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_inference_done)
        self.worker.error.connect(self._on_inference_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def init_model(self, model_path: str):
        raise NotImplementedError("init_model must be implemented in subclass.")
    
    def launch_finetune_worker(self, base_model: str):
        raise NotImplementedError("launch_finetune_worker must be implemented in subclass.")
    
    def finetune_model(self):

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

            # choose a base OBB model, e.g., 'yolo11n-obb.pt'
        base_model = self.model_path  # or a file dialog / settings

        
        self.finetune_thread = QtCore.QThread(self)
        self.finetune_worker = self.launch_finetune_worker(base_model)
        self.finetune_worker.moveToThread(self.finetune_thread)

        # connect signals
        self.finetune_thread.started.connect(self.finetune_worker.run)
        self.finetune_worker.progress.connect(lambda msg, p: self.info_label.setText(f"{msg} ({int(p*100)}%)"))
        self.finetune_worker.error.connect(self._on_finetune_error)
        self.finetune_worker.finished.connect(self._on_finetune_done)

        # cleanup
        self.finetune_worker.finished.connect(self.finetune_thread.quit)
        self.finetune_worker.finished.connect(self.finetune_worker.deleteLater)
        self.finetune_thread.finished.connect(self.finetune_thread.deleteLater)
        self.finetune_worker.error.connect(self.finetune_thread.quit)
        self.finetune_worker.error.connect(self.finetune_worker.deleteLater)

        self.finetune_thread.start()

    def _on_finetune_error(self, msg: str):
        QtWidgets.QMessageBox.critical(self, "Fine-tune Error", msg)
        self.info_label.setText(f"Error: {msg}")

    def _on_finetune_done(self, best_pt_path: str):
        self.info_label.setText(f"Fine-tune complete: {best_pt_path}")
        # Optionally load the new weights right away:
        try:
            self.model_worker._model = self.init_model(best_pt_path)
            self.model_path = best_pt_path
            self.info_label.setText(f"Loaded fine-tuned model: {os.path.basename(best_pt_path)}")
        except Exception as e:
            self.info_label.setText(f"Model saved, but failed to load: {e}")
        self.finetune_btn.setEnabled(True)
        self.finetune_btn.setText("Finetune Model")

    def _on_inference_done(self, frame_idx: int, class_names, annots: List[PolyClass]):
        self.class_names = class_names
        self.pred_cache[frame_idx] = annots
        # keep selected clear
        self.selected_idx = None
        if frame_idx == self.current_idx:
            self.redraw_current()
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Model")
        self.info(f"Predictions cached for frame {frame_idx+1}.")

    def _on_inference_error(self, msg: str):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Model (cached)")
        self.info(f"Inference error: {msg}")

    # ---------- Mouse / Keyboard HITL ----------
    def clear_text_focus_if_needed(self):
        """Unfocus the conf spinbox if it currently has focus."""
        if self.inference_conf_tresh.hasFocus():
            self.inference_conf_tresh.clearFocus()
        # You can add more widgets later (e.g., line edits) to also clear here.

    def give_canvas_focus(self):
        """Move focus to the canvas so keyboard shortcuts feel natural."""
        self.video_label.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)

    def eventWheel(self, event):
        # Zoom around mouse cursor
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_step(+1, anchor_disp=event.position())
        elif delta < 0:
            self.zoom_step(-1, anchor_disp=event.position())
        return True
    
    def check_canvas_mouse_event(self, event):
        if self.current_frame_bgr is None:
            return None
        if hasattr(event, "position"):
            pos = event.position()
            x_disp, y_disp = int(pos.x()), int(pos.y())
        else:
            return None
        x_img, y_img = self.display_to_image_coords(x_disp, y_disp)
        if x_img is None:
            return None
        
        return x_disp, y_disp, x_img, y_img

    def unfocus_conf_tresh_on_click(self, event):
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            if self.inference_conf_tresh.hasFocus():
                self.inference_conf_tresh.clearFocus()
            self.video_label.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)

    def eventFilter(self, obj, event):

        # Clear focus from spinbox on any click
        self.unfocus_conf_tresh_on_click(event)
    
        # --- Mouse on video canvas ---
        if obj is self.video_label:

            # Mouse wheel zoom on canvas
            if event.type() == QtCore.QEvent.Type.Wheel:
                return self.eventWheel(event)
            
            # Map coords
            if event.type() in (QtCore.QEvent.Type.MouseButtonPress,
                                QtCore.QEvent.Type.MouseMove,
                                QtCore.QEvent.Type.MouseButtonRelease):
                
                coords =  self.check_canvas_mouse_event(event)
                if coords is None:
                    return False
                else :
                    x_disp, y_disp, x_img, y_img = coords

            # --- PAN with Space+Left drag ---
            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                if event.button() == QtCore.Qt.MouseButton.LeftButton and self.space_held:
                    self._pan_dragging = True
                    self._pan_last_disp = (x_disp, y_disp)
                    return True

            elif event.type() == QtCore.QEvent.Type.MouseMove:
                if getattr(self, "_pan_dragging", False):
                    dx = x_disp - self._pan_last_disp[0]
                    dy = y_disp - self._pan_last_disp[1]
                    s = self.draw_map["scale"] if self.draw_map else 1.0
                    # move pan in image pixels
                    self.pan_img[0] -= dx / s
                    self.pan_img[1] -= dy / s
                    self._pan_last_disp = (x_disp, y_disp)
                    self.redraw_current()
                    return True

            elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
                if getattr(self, "_pan_dragging", False):
                    self._pan_dragging = False
                    return True


            # PRESS
            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                if event.button() == QtCore.Qt.MouseButton.LeftButton:
                    if self.mode == "add":
                        # Add point
                        self.add_click_point(x_img, y_img)
                        return True

                    # select/ edit / move
                    hit_idx = self.pick_annot(x_img, y_img)
                    if hit_idx is not None:
                        self.selected_idx = hit_idx
                        self.redraw_current()
                        # CTRL + drag a vertex?
                        if self.mode == "edit" and event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
                            v = self.pick_vertex(x_img, y_img)
                            if v is not None:
                                self.vertex_drag_idx = v
                                self.dragging = True
                                return True
                        # else start translation drag if clicked inside selected
                        boxes = self.pred_cache.get(self.current_idx, [])
                        if self.selected_idx is not None and self.selected_idx < len(boxes):
                            self.dragging = True
                            self.drag_start_img = (x_img, y_img)
                            self.orig_poly = boxes[self.selected_idx].poly.copy()
                        return True
                    else:
                        # clicked empty – clear selection (unless in add)
                        if self.mode != "add":
                            self.selected_idx = None
                            self.redraw_current()
                        return True

                elif event.button() == QtCore.Qt.MouseButton.RightButton:
                    # keep your previous right-click verify toggle
                    hit_idx = self.pick_annot(x_img, y_img)
                    if hit_idx is not None:
                        self.selected_idx = hit_idx
                        self.verify_selected_toggle()
                        self.redraw_current()
                        return True
                    return True

            # MOVE
            elif event.type() == QtCore.QEvent.Type.MouseMove:
                if self.dragging:
                    if self.vertex_drag_idx is not None:
                        # vertex editing
                        self.set_vertex_selected(self.vertex_drag_idx, x_img, y_img)
                    elif self.drag_start_img is not None:
                        dx = x_img - self.drag_start_img[0]
                        dy = y_img - self.drag_start_img[1]
                        self.translate_selected(dx, dy)
                    self.redraw_current()
                    return True

            # RELEASE
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


    def display_to_image_coords(self, xd: int, yd: int):
        m = self.draw_map
        if not m:
            return None, None
        s   = m["scale"]
        xoff, yoff = m["xoff"], m["yoff"]
        panx, pany = m["panx"], m["pany"]
        img_w, img_h = m["img_w"], m["img_h"]

        # subtract frame origin incl. pan shift
        xi = (xd - (xoff - panx * s)) / s
        yi = (yd - (yoff - pany * s)) / s
        if xi < 0 or yi < 0 or xi >= img_w or yi >= img_h:
            return None, None
        return float(xi), float(yi)

    def pick_annot(self, x: float, y: float) -> Optional[int]:
        annot = self.pred_cache.get(self.current_idx, [])
        if not annot:
            return None
        best = None
        best_area = None
        p = (x, y)
        for i, b in enumerate(annot):
            if b.deleted:
                continue
            pts = b.poly.reshape(-1, 2).astype(np.float32)
            inside = cv2.pointPolygonTest(pts, p, measureDist=False)
            if inside >= 0:  # inside or on edge
                # prefer the smallest area box hit (in case of overlap)
                area = cv2.contourArea(pts.astype(np.int32))
                if best is None or area < best_area:
                    best = i
                    best_area = area
        return best

    def verify_selected_toggle(self):
        boxes = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(boxes):
            return
        box = boxes[self.selected_idx]
        if box.deleted:
            return
        box.verified = not box.verified
        self.update_dataset_for_frame(self.current_idx)
        state = "verified (green, no label)" if box.verified else "unverified (orange, with label)"
        self.info(f"Box #{self.selected_idx} -> {state}")
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
        """Keep dataset = verified & not deleted boxes only."""
        all_boxes = self.pred_cache.get(frame_idx, [])
        self.dataset[frame_idx] = [b for b in all_boxes if (b.verified and not b.deleted)]
        if isinstance(self.source, ImageFolderSource):
            self.dataset_images_names[frame_idx] = self.source.path_at(frame_idx)

    # ---------- Save dataset ----------
    def save_dataset_json(self):
        if not self.dataset:
            self.info("Dataset empty (no verified boxes).")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save dataset (JSON)", "dataset.json", "JSON (*.json);;All files (*)"
        )
        if not path:
            return
        data: Dict[str, Any] = {
            "video_path": self.src_path,
            "total_frames": self.total_frames,
            "frames": []
        }
        for frame_idx, boxes in sorted(self.dataset.items()):
            if not boxes:
                continue
            data["frames"].append({
                "frame_index": int(frame_idx),
                "boxes": [b.to_json() for b in boxes]
            })

        try:
            import json
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.info(f"Saved dataset to {os.path.basename(path)}")
        except Exception as e:
            self.info(f"Save failed: {e}")

    # ---------- Mode helpers ----------
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
    
    # ---------- Picking helpers ----------
    def pick_vertex(self, x: float, y: float, tol_px: int = 10) -> int | None:
        """Return corner index 0..n if near selected box corner, else None."""
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
            if np.hypot(pts[i,0]-x, pts[i,1]-y) <= tol_px:
                return i
        return None

    # ----------Zoom Helpers ----------
    def zoom_fit(self):
        self.zoom = 1.0
        self.pan_img[:] = 0.0
        self.redraw_current()

    # optional clamp to keep at least 10% of the image visible
    def _clamp_pan(self):
        m = self.draw_map
        if not m:
            return
        s = m["base"] * self.zoom
        img_w, img_h = m["img_w"], m["img_h"]
        lbl_w, lbl_h = m["lbl_w"], m["lbl_h"]
        disp_w, disp_h = img_w * s, img_h * s
        # max pan so that some part stays visible
        margin = 0.1
        max_pan_x = (disp_w - (1.0 - margin) * lbl_w) / s / 2.0
        max_pan_y = (disp_h - (1.0 - margin) * lbl_h) / s / 2.0
        self.pan_img[0] = float(np.clip(self.pan_img[0], -max_pan_x, max_pan_x))
        self.pan_img[1] = float(np.clip(self.pan_img[1], -max_pan_y, max_pan_y))
        
    def zoom_step(self, direction: int, anchor_disp: QtCore.QPointF | None = None):
        """direction: +1 or -1; anchor around cursor if provided, else center."""
        # pick anchor on screen
        if anchor_disp is None:
            # center of canvas
            w, h = self.video_label.width(), self.video_label.height()
            anchor_disp = QtCore.QPointF(w / 2.0, h / 2.0)

        # compute current base/scale and anchor image coord BEFORE zoom
        m = self.draw_map
        if not m or self.current_frame_bgr is None:
            return
        old_scale = float(m["scale"])
        base = float(m["base"])
        # zoom factor per step
        step = 1.25 if direction > 0 else 0.8
        new_zoom = float(np.clip(self.zoom * step, self.min_zoom, self.max_zoom))
        if abs(new_zoom - self.zoom) < 1e-6:
            return

        xd, yd = float(anchor_disp.x()), float(anchor_disp.y())
        # image coord at anchor under old zoom/pan
        xi, yi = self.display_to_image_coords(int(xd), int(yd))
        if xi is None:
            # if anchor outside image, just change zoom around center without preserving anchor
            self.zoom = new_zoom
            self.redraw_current()
            return

        # update zoom
        self.zoom = new_zoom
        # recompute new scale and offsets
        img_w, img_h = m["img_w"], m["img_h"]
        lbl_w, lbl_h = m["lbl_w"], m["lbl_h"]
        new_scale = base * self.zoom
        disp_w, disp_h = img_w * new_scale, img_h * new_scale
        xoff = (lbl_w - disp_w) / 2.0
        yoff = (lbl_h - disp_h) / 2.0

        # choose new pan so (xi,yi) stays under (xd,yd):
        # xd = xoff - panx*new_scale + xi*new_scale  ->  panx = (xoff + xi*new_scale - xd)/new_scale
        self.pan_img[0] = (xoff + xi * new_scale - xd) / new_scale
        self.pan_img[1] = (yoff + yi * new_scale - yd) / new_scale
        self._clamp_pan()

        self.redraw_current()

    # ---------- Poly editing ----------   
    def translate_selected(self, dx: float, dy: float):
        annots = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(annots):
            return
        b = annots[self.selected_idx]
        b.poly = (self.orig_poly + np.array([dx, dy], dtype=np.float32)).astype(np.float32)
        self.update_dataset_for_frame(self.current_idx)

    def set_vertex_selected(self, idx: int, x: float, y: float):
        annots = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(annots):
            return
        b = annots[self.selected_idx]
        p = b.poly.copy()
        p[idx] = [x, y]
        b.poly = p.astype(np.float32)
        self.update_dataset_for_frame(self.current_idx)
        
    # ---------- Utils ----------
    def info(self, text: str):
        self.info_label.setText(text)

    def update_title(self):
        base = "Base Video Player"
        if self.source:
            base += f" | {self.source.name()}"
        if self.source:
            base += f" | frame {self.current_idx+1}/{self.total_frames}"
        self.setWindowTitle(base)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self.redraw_current()

    # ---------- Add-poloy pipeline (n clicks) ----------
    def add_click_point(self, x: float, y: float):
        self.temp_poly_pts.append([x, y])
        first_point = self.temp_poly_pts[0]

        if len(self.temp_poly_pts) > 2 and np.linalg.norm(np.array([x, y]) - np.array(first_point)) < 8.0:
            # close polygon if near first point
            self.temp_poly_pts[-1] = first_point

            pts = np.array(self.temp_poly_pts, dtype=np.float32)
            new_box = PolyClass(poly=pts, cls_id=0, conf=1.0, verified=False)
            self.pred_cache.setdefault(self.current_idx, []).append(new_box)
            self.selected_idx = len(self.pred_cache[self.current_idx]) - 1
            self.temp_poly_pts.clear()
            self.set_mode("select")
            self.update_dataset_for_frame(self.current_idx)
        self.redraw_current()


class OBB_VideoPlayer(Base):
    def __init__(self, model_path: str = YOLO_MODEL_PATH):
        super().__init__()
        self.model_path = model_path
        self.model_worker = DetectionWorker

    # ---------- Add-box pipeline (4 clicks) ----------
    def add_click_point(self, x: float, y: float):
        
        if len(self.temp_poly_pts) == 2:

            primes = find_orthogonal_projection(self.temp_poly_pts[0], self.temp_poly_pts[1], [x, y])
            pts = np.concatenate((self.temp_poly_pts, primes), axis=0, dtype=np.float32)
            
            # pts = np.array(self.temp_poly_pts, dtype=np.float32)
            new_box = OBBOX(poly=pts, cls_id=0, conf=1.0, verified=False)
            self.pred_cache.setdefault(self.current_idx, []).append(new_box)
            self.selected_idx = len(self.pred_cache[self.current_idx]) - 1
            self.temp_poly_pts.clear()
            self.set_mode("select")
            self.update_dataset_for_frame(self.current_idx)
        else : 
            self.temp_poly_pts.append([x, y])

        self.redraw_current()
    

    def init_model(self, model_path: str):
        # load OBB model
        model = YOLO(model_path)
        return model

    def launch_finetune_worker(self, base_model: str):

        return DetectFinetuneWorker(
            video_path=self.src_path,
            dataset=self.dataset,
            class_names=self.class_names,
            base_model_path=base_model,
            out_root=os.path.join(os.getcwd(), "finetune_runs"),
            epochs=20,
            imgsz=640,
            batch=16,
            val_split=0.1,
        )
    
class Seg_VideoPlayer(Base):
    def __init__(self, model_path: str = SAM2_UNET_MODEL_PATH):
        super().__init__()
        self.model_path = model_path
        self.model_worker = SegWorker

    def init_model(self, model_path: str):

        net = SAM2UNet(config="tiny", sam_checkpoint_path="src/sam2/checkpoints/sam2.1_hiera_tiny.pt", freeze_encorder = True).to("cuda")
        lit = LitBinarySeg.load_from_checkpoint("logs/tiny_freeze_for_noisy_detect/epoch=94-step=3515.ckpt", 
                                            net=net,
                                            deep_supervision=False,
                                            dice_use_all_outputs=False,      # mets True si tu veux la Dice moyenne (out,out1,out2)
                                            pos_weight=200).eval().to("cuda")
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

    