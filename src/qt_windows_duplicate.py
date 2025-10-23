
import os
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

# ------ Local imports from utils.py ------
from .utils import AnnotBox, cvimg_to_qimage, draw_obb_annotations
from .qt_workers import DetectionWorker, MODEL_PATH


class BaseVideoPlayer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.baseTitle = "Video Player"
        self.setWindowTitle(self.baseTitle)
        self.resize(1200, 760)

        # Video state
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames: int = 0
        self.current_idx: int = 0
        self.current_frame_bgr: Optional[np.ndarray] = None
        self.video_path: Optional[str] = None
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._on_play_tick)
        self.playing = False

        # --- Zoom & Pan state ---
        self.zoom = 1.0              # relative to "fit to window" scale
        self.min_zoom = 0.25
        self.max_zoom = 8.0
        self.pan_img = np.array([0.0, 0.0], dtype=np.float32)   # pan in *image* pixels

        # Annotations state
        self.pred_cache: Dict[int, List[AnnotBox]] = {}   # frame_idx -> boxes
        self.class_names = None
        self.selected_idx: Optional[int] = None           # index in current boxes list

        # Dataset (verified only)
        self.dataset: Dict[int, List[AnnotBox]] = {}      # frame_idx -> verified boxes

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
        self.add_btn = QtWidgets.QPushButton("Add Box (N)")
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

        self.open_btn = QtWidgets.QPushButton("Open video")
        self.prev_btn = QtWidgets.QPushButton("⟸ Prev (←)")
        self.next_btn = QtWidgets.QPushButton("Next (→) ⟹")
        self.detect_btn = QtWidgets.QPushButton("Run YOLO (cached)")
        self.play_btn = QtWidgets.QPushButton("Play ▶")
        self.pause_btn = QtWidgets.QPushButton("Pause ⏸")
        self.save_btn = QtWidgets.QPushButton("Save dataset (JSON)")
        self.detect_conf = QtWidgets.QDoubleSpinBox()
        self.detect_conf.setRange(0.01, 0.99)
        self.detect_conf.setSingleStep(0.05)
        self.detect_conf.setValue(0.05)
        self.detect_conf.setPrefix("conf=")

        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.sliderReleased.connect(self._on_slider_released)

        self.info_label = QtWidgets.QLabel(
            "Open a video → Run YOLO once per frame (cached). "
            "Left-click: select | Right-click or V: verify (turns green, hides label) | Delete: remove."
        )
        self.info_label.setStyleSheet("color:#aaa;")

        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.addWidget(self.open_btn)
        ctrl_row.addSpacing(10)
        ctrl_row.addWidget(self.prev_btn)
        ctrl_row.addWidget(self.next_btn)
        ctrl_row.addSpacing(10)
        ctrl_row.addWidget(self.play_btn)
        ctrl_row.addWidget(self.pause_btn)
        ctrl_row.addSpacing(20)
        ctrl_row.addWidget(self.detect_btn)
        ctrl_row.addWidget(self.detect_conf)
        ctrl_row.addStretch(1)
        ctrl_row.insertWidget(6, self.add_btn)
        ctrl_row.insertWidget(7, self.edit_btn)
        ctrl_row.insertWidget(8, self.verify_btn)
        ctrl_row.insertWidget(9, self.delete_btn)
        ctrl_row.insertWidget(0, self.zoom_in_btn)
        ctrl_row.insertWidget(1, self.zoom_out_btn)
        ctrl_row.insertWidget(2, self.zoom_fit_btn)
        ctrl_row.addWidget(self.save_btn)

        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(self.video_label, stretch=1)
        layout.addWidget(self.frame_slider)
        layout.addLayout(ctrl_row)
        layout.addWidget(self.info_label)

        # Signals
        self.open_btn.clicked.connect(self.open_video)
        self.prev_btn.clicked.connect(self.prev_frame)
        self.next_btn.clicked.connect(self.next_frame)
        self.detect_btn.clicked.connect(self.run_detection_cached)
        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.save_btn.clicked.connect(self.save_dataset_json)

        # Shortcuts
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Left), self, activated=self.prev_frame)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Right), self, activated=self.next_frame)
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.toggle_play_pause)
        QtGui.QShortcut(QtGui.QKeySequence("V"), self, activated=self.verify_selected_toggle)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Delete), self, activated=self.delete_selected)
        QtGui.QShortcut(QtGui.QKeySequence("N"), self, activated=self.start_add_mode)
        QtGui.QShortcut(QtGui.QKeySequence("E"), self, activated=self.toggle_edit_mode)
        QtGui.QShortcut(QtGui.QKeySequence("Esc"), self, activated=self.cancel_add_mode)
        QtGui.QShortcut(QtGui.QKeySequence("S"), self, activated=self.save_dataset_json)
        QtGui.QShortcut(QtGui.QKeySequence("+"), self, activated=lambda: self.zoom_step(+1))
        QtGui.QShortcut(QtGui.QKeySequence("-"), self, activated=lambda: self.zoom_step(-1))
        QtGui.QShortcut(QtGui.QKeySequence("0"), self, activated=self.zoom_fit)

    # ---------- Video I/O ----------
    def open_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if not path:
            return
        self.load_video(path)

    def load_video(self, path: str):
        if self.cap:
            self.cap.release()
            self.cap = None

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.info("Failed to open video.")
            return

        self.cap = cap
        self.video_path = path
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        self.current_idx = 0
        self.pred_cache.clear()
        self.dataset.clear()
        self.selected_idx = None

        self.frame_slider.setRange(0, max(0, self.total_frames - 1))
        self.frame_slider.setValue(0)
        self.info(f"Loaded: {os.path.basename(path)} | frames={self.total_frames} | fps={fps:.2f}")
        self.read_frame(self.current_idx)

    def read_frame(self, idx: int) -> bool:
        if not self.cap:
            return False
        idx = max(0, min(idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.info("Failed to read frame.")
            return False
        self.current_idx = idx
        self.current_frame_bgr = frame
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(idx)
        self.frame_slider.blockSignals(False)
        self.update_title()

        self.redraw_current()
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
        boxes = self.pred_cache.get(self.current_idx, [])
        boxes_to_draw = [b for b in boxes if (not b.deleted and b.conf > self.detect_conf.value())]
        annotated = draw_obb_annotations(base, boxes_to_draw, self.class_names, self.selected_idx)

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
        if self.cap:
            self.pause()
            self.selected_idx = None
            self.read_frame(self.current_idx - 1)

    def next_frame(self):
        if self.cap:
            self.pause()
            self.selected_idx = None
            self.read_frame(self.current_idx + 1)

    def _on_slider_released(self):
        if self.cap:
            self.pause()
            self.selected_idx = None
            self.read_frame(self.frame_slider.value())

    def play(self):
        if not self.cap or self.playing:
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        interval_ms = int(1000 / fps)
        self.play_timer.start(max(15, interval_ms))
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
        self.selected_idx = None
        self.read_frame(self.current_idx + 1)

    # ---------- Detection with cache ----------
    def run_detection_cached(self):
        """Only run YOLO if no cache exists for this frame"""
        idx = self.current_idx
        if idx in self.pred_cache and len(self.pred_cache[idx]) > 0:
            self.info("Using cached predictions for this frame.")
            self.redraw_current()
            return

        if self.current_frame_bgr is None:
            return

        self.detect_btn.setEnabled(False)
        self.detect_btn.setText("Detecting…")
        # conf = float(self.detect_conf.value())

        self.worker_thread = QtCore.QThread(self)
        self.worker = DetectionWorker(idx, self.current_frame_bgr, conf=0.01, model_path=MODEL_PATH)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_detection_done)
        self.worker.error.connect(self._on_detection_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def _on_detection_done(self, frame_idx: int, class_names, boxes: List[AnnotBox]):
        self.class_names = class_names
        self.pred_cache[frame_idx] = boxes
        # keep selected clear
        self.selected_idx = None
        if frame_idx == self.current_idx:
            self.redraw_current()
        self.detect_btn.setEnabled(True)
        self.detect_btn.setText("Run YOLO (cached)")
        self.info(f"Predictions cached for frame {frame_idx+1}.")

    def _on_detection_error(self, msg: str):
        self.detect_btn.setEnabled(True)
        self.detect_btn.setText("Run YOLO (cached)")
        self.info(f"Detection error: {msg}")

    # ---------- Mouse / Keyboard HITL ----------
    def clear_text_focus_if_needed(self):
        """Unfocus the conf spinbox if it currently has focus."""
        if self.detect_conf.hasFocus():
            self.detect_conf.clearFocus()
        # You can add more widgets later (e.g., line edits) to also clear here.

    def give_canvas_focus(self):
        """Move focus to the canvas so keyboard shortcuts feel natural."""
        self.video_label.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)

    def eventFilter(self, obj, event):
        # Clear focus from spinbox on any click
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            if self.detect_conf.hasFocus():
                self.detect_conf.clearFocus()
            self.video_label.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)

        # Mouse wheel zoom on canvas
        if obj is self.video_label and event.type() == QtCore.QEvent.Type.Wheel:
            # Zoom around mouse cursor
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_step(+1, anchor_disp=event.position())
            elif delta < 0:
                self.zoom_step(-1, anchor_disp=event.position())
            return True
    
        # --- Mouse on video canvas ---
        if obj is self.video_label:
            # Map coords
            if event.type() in (QtCore.QEvent.Type.MouseButtonPress,
                                QtCore.QEvent.Type.MouseMove,
                                QtCore.QEvent.Type.MouseButtonRelease):
                if self.current_frame_bgr is None:
                    return False
                if hasattr(event, "position"):
                    pos = event.position()
                    x_disp, y_disp = int(pos.x()), int(pos.y())
                else:
                    return False
                x_img, y_img = self.display_to_image_coords(x_disp, y_disp)
                if x_img is None:
                    return False

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
                    hit_idx = self.pick_box(x_img, y_img)
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
                    hit_idx = self.pick_box(x_img, y_img)
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

    def pick_box(self, x: float, y: float) -> Optional[int]:
        boxes = self.pred_cache.get(self.current_idx, [])
        if not boxes:
            return None
        best = None
        best_area = None
        p = (x, y)
        for i, b in enumerate(boxes):
            if b.deleted:
                continue
            pts = b.poly.reshape(4, 2).astype(np.float32)
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
            "video_path": self.video_path,
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
        """Return corner index 0..3 if near selected box corner, else None."""
        if self.selected_idx is None:
            return None
        boxes = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx >= len(boxes):
            return None
        b = boxes[self.selected_idx]
        if b.deleted:
            return None
        pts = b.poly.reshape(4, 2)
        for i in range(4):
            if np.hypot(pts[i,0]-x, pts[i,1]-y) <= tol_px:
                return i
        return None

    def translate_selected(self, dx: float, dy: float):
        boxes = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(boxes):
            return
        b = boxes[self.selected_idx]
        b.poly = (self.orig_poly + np.array([dx, dy], dtype=np.float32)).astype(np.float32)
        self.update_dataset_for_frame(self.current_idx)

    def set_vertex_selected(self, idx: int, x: float, y: float):
        boxes = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(boxes):
            return
        b = boxes[self.selected_idx]
        p = b.poly.copy()
        p[idx] = [x, y]
        b.poly = p.astype(np.float32)
        self.update_dataset_for_frame(self.current_idx)

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
        
    # ---------- Add-box pipeline (4 clicks) ----------
    def add_click_point(self, x: float, y: float):
        self.temp_poly_pts.append([x, y])
        if len(self.temp_poly_pts) == 4:
            # finalize new box
            pts = np.array(self.temp_poly_pts, dtype=np.float32)
            new_box = AnnotBox(poly=pts, cls_id=0, conf=1.0, verified=False)
            self.pred_cache.setdefault(self.current_idx, []).append(new_box)
            self.selected_idx = len(self.pred_cache[self.current_idx]) - 1
            self.temp_poly_pts.clear()
            self.set_mode("select")
            self.update_dataset_for_frame(self.current_idx)
        self.redraw_current()
        
    # ---------- Utils ----------
    def info(self, text: str):
        self.info_label.setText(text)

    def update_title(self):
        base = self.baseTitle
        if self.video_path:
            base += f" | {os.path.basename(self.video_path)}"
        if self.cap:
            base += f" | frame {self.current_idx+1}/{self.total_frames}"
        self.setWindowTitle(base)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self.redraw_current()







class OBB_VideoPlayer(BaseVideoPlayer):
    def __init__(self):
        super().__init__()