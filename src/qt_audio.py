"""
Audio Spectrogram Player using PySide6, matplotlib and librosa.

Requirements (install via pip):
    pip install PySide6 matplotlib librosa numpy

This app lets you:
    - Load a folder (or nested folders) of audio files.
    - Select an audio file and display its spectrogram.
    - Play / pause the audio using QMediaPlayer.
    - Open a settings dialog to change spectrogram parameters.
"""

import csv
from dataclasses import dataclass
from typing import List, Optional
import sys
import os

import numpy as np
import librosa

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QPushButton, QFileDialog, QMessageBox, QDialog,
    QFormLayout, QDialogButtonBox, QSpinBox, QComboBox, QLabel, 
    QProgressDialog, QSplitter, QToolButton, QMenu, QDoubleSpinBox,
)
from PySide6 import QtGui
from PySide6.QtCore import Qt, QUrl, Signal, QRectF, QTimer
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput


import pyqtgraph as pg
import matplotlib.cm as cm  # reuse matplotlib colormaps for LUT




@dataclass
class SpectrogramSettings:
    """Container for spectrogram parameters."""
    n_fft: int = 256
    win_length: int = 128
    hop_ms: float = 2.0         # hop duration in milliseconds (time resolution)
    cmap: str = "turbo"
    max_freq: Optional[int] = 20000  # Hz, None means full range
    max_cache_mb: int = 256        # max memory for full audio cache


@dataclass
class Annotation:
    file: str
    start: float
    end: float
    label: str
    description: str


class SpectrogramSettingsDialog(QDialog):
    """
    Dialog window to configure spectrogram parameters.
    """

    def __init__(self, settings: SpectrogramSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spectrogram Settings")
        # Copy initial settings so we do not modify them until the user presses OK
        self._settings = SpectrogramSettings(
            n_fft=settings.n_fft,
            win_length=settings.win_length,
            hop_ms=settings.hop_ms,
            cmap=settings.cmap,
            max_freq=settings.max_freq,
            max_cache_mb=settings.max_cache_mb,
        )
        self._build_ui()

    def _build_ui(self):
        """Create and arrange widgets inside the dialog."""
        layout = QVBoxLayout(self)
        form = QFormLayout()

        # FFT size spin box
        self.n_fft_spin = QSpinBox()
        self.n_fft_spin.setRange(256, 16384)
        self.n_fft_spin.setSingleStep(256)
        self.n_fft_spin.setValue(self._settings.n_fft)
        form.addRow("FFT size (n_fft):", self.n_fft_spin)

        # Window length spin box
        self.win_length_spin = QSpinBox()
        self.win_length_spin.setRange(128, 16384)
        self.win_length_spin.setSingleStep(128)
        self.win_length_spin.setValue(self._settings.win_length)
        form.addRow("Window length (win_length):", self.win_length_spin)

        # Hop duration in milliseconds (time resolution)
        self.hop_ms_spin = QDoubleSpinBox()
        self.hop_ms_spin.setRange(1.0, 500.0)      # 1 ms to 500 ms
        self.hop_ms_spin.setSingleStep(1.0)
        self.hop_ms_spin.setDecimals(1)
        self.hop_ms_spin.setValue(self._settings.hop_ms)
        form.addRow("Hop duration (ms):", self.hop_ms_spin)

        # Max frequency spin box
        self.max_freq_spin = QSpinBox()
        self.max_freq_spin.setRange(0, 48000)
        self.max_freq_spin.setValue(20000)
        form.addRow("Max frequency (Hz):", self.max_freq_spin)

        # Colormap combo box
        self.cmap_combo = QComboBox()
        cmaps = ["magma","turbo", "viridis", "plasma", "inferno", "cividis"]
        self.cmap_combo.addItems(cmaps)
        if self._settings.cmap in cmaps:
            self.cmap_combo.setCurrentText(self._settings.cmap)
        form.addRow("Colormap:", self.cmap_combo)

        # Max cache size in MB
        self.cache_spin = QSpinBox()
        self.cache_spin.setRange(32, 4096)
        self.cache_spin.setSingleStep(32)
        self.cache_spin.setValue(self._settings.max_cache_mb)
        form.addRow("Max cache size (MB):", self.cache_spin)

        layout.addLayout(form)

        # OK / Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_settings(self) -> SpectrogramSettings:
        """Return updated settings based on the dialog inputs."""
        self._settings.n_fft = int(self.n_fft_spin.value())
        self._settings.win_length = int(self.win_length_spin.value())
        self._settings.hop_ms = float(self.hop_ms_spin.value())
        maxf = int(self.max_freq_spin.value())
        self._settings.max_freq = maxf if maxf > 0 else None
        self._settings.cmap = self.cmap_combo.currentText()
        self._settings.max_cache_mb = int(self.cache_spin.value())
        return self._settings

class SpectroViewBox(pg.ViewBox):
    """
    Custom ViewBox:
        - Left drag: built-in rect-zoom (RectMode)
        - Left click (no drag): emit time click
        - Right click: reset zoom
    """
    sigTimeClicked = Signal(float)   # x (seconds)
    sigZoomReset = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, enableMenu=False, **kwargs)
        # Use rectangle zoom, not panning
        self.setMouseMode(self.RectMode)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            pos = self.mapToView(ev.pos())
            self.sigTimeClicked.emit(float(pos.x()))
            ev.accept()
        elif ev.button() == Qt.RightButton:
            # Reset zoom
            self.autoRange()
            self.sigZoomReset.emit()
            ev.accept()
        else:
            super().mouseClickEvent(ev)

    # We keep default RectMode behaviour for drags -> rectangle zoom
    # so we don't override mouseDragEvent.

class SpectrogramCanvas(QWidget):
    """
    High-performance spectrogram canvas using PyQtGraph.

    - Dark DAW-style look (no frame, minimal axes)
    - Very fast image rendering (ImageItem)
    - Annotations as LinearRegionItem (blue / green when selected)
    - Playhead as InfiniteLine (updated frequently from a QTimer or positionChanged)
    - Left-click on spectrogram -> move playhead
    - Left-click on annotated zone -> select annotation + emit annotation_clicked
    - Right-click -> reset zoom
    """

    time_clicked = Signal(float)       # time in seconds
    annotation_clicked = Signal(object)  # emits Annotation object

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Use our custom ViewBox
        self.vb = SpectroViewBox()
        self.plot = pg.PlotWidget(parent=self, viewBox=self.vb)
        layout.addWidget(self.plot)

        # Dark background
        self.plot.setBackground("#232323")
        # self.plot.invertY(True)

        # Axes styling
        for name in ("bottom", "left"):
            axis = self.plot.getAxis(name)
            axis.setPen(pg.mkPen("#ffffff", width=1))
            axis.setTextPen(pg.mkPen("#ffffff"))

            # Only tickLength is valid
            axis.setStyle(tickLength=3)

            # Tick font
            font = QtGui.QFont("Sans", 8)
            axis.setTickFont(font)

        # Remove default title, grid, etc.
        self.plot.showGrid(x=False, y=False)
        self.plot.setMouseEnabled(x=False, y=False)  # delete pan/zoom behavior

        # Connect our custom viewbox signals
        self.vb.sigTimeClicked.connect(self._on_time_clicked_from_viewbox)
        self.vb.sigZoomReset.connect(self._on_zoom_reset)

        # Image item for the spectrogram
        self.img_item = pg.ImageItem()
        self.plot.addItem(self.img_item)

        # Playhead line
        self.playhead_line = pg.InfiniteLine(
            angle=90,
            pos=0.0,
            pen=pg.mkPen("#ff4444", width=2),
            movable=False,
        )
        self.plot.addItem(self.playhead_line)

        # Annotation handling
        self.annotations = []              # list[Annotation]
        self.annotation_regions = []       # list[pg.LinearRegionItem]
        self.selected_annotation = None

        # Data extents for checks
        self.times = None
        self.freqs = None

        # # Mouse click handling through the scene
        # self.plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        # self.plot.invertY(True)

    # ------------------------------------------------------------------
    # Spectrogram drawing
    # ------------------------------------------------------------------
    def plot_spectrogram(self, audio: np.ndarray, sr: int, settings):
        """
        Compute and display the spectrogram of the given audio signal.
        """
        # Reset any previous image / data references
        self.times = None
        self.freqs = None

        if audio is None or len(audio) == 0:
            # Clear and show message
            self.img_item.clear()
            self.plot.clear()
            self.plot.addItem(self.playhead_line)
            txt = pg.TextItem("No audio loaded", color="w", anchor=(0.5, 0.5))
            txt.setPos(0, 0)
            self.plot.addItem(txt)
            return

        # --- Compute STFT using librosa ---
        n_fft = settings.n_fft
        win_length = settings.win_length
        hop_length = max(1, int(sr * (settings.hop_ms / 1000.0)))
        print(f"Computing spectrogram: n_fft={n_fft}, win_length={win_length}, hop_length={hop_length}")

        S = librosa.stft(audio, n_fft=n_fft, win_length = win_length, hop_length=hop_length) #
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        print(f"Spectrogram shape: {S_db.shape}")
        # Build time and frequency axes
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        times = librosa.frames_to_time(
            np.arange(S_db.shape[1]),
            sr=sr,
            hop_length=hop_length,
        )

        # Limit frequency range if requested
        if settings.max_freq is not None:
            max_idx = np.searchsorted(freqs, settings.max_freq)
            freqs = freqs[:max_idx]
            S_db = S_db[:max_idx, :]

        self.times = times
        self.freqs = freqs

        # Use a matplotlib colormap as a LUT for pyqtgraph
        # S_db: shape (freqs, times)
        cmap = cm.get_cmap(settings.cmap)
        lut = (cmap(np.linspace(0.0, 1.0, 256)) * 255).astype(np.uint8)

        # We want time on X, frequency on Y.
        # pyqtgraph expects data array as (rows, cols) = (Y, X) -> already (freq, time),
        # so we can use it directly.
        S_db = S_db.astype(np.float32)
        db_min = float(S_db.min())
        db_max = float(S_db.max())

        self.img_item.setImage(
            S_db,
            lut=lut,
            levels=(db_min, db_max),
            autoLevels=False,
            axisOrder="row-major",
        )

        # Map image coordinates to real time/frequency
        t_min, t_max = times[0], times[-1]
        f_min, f_max = freqs[0], freqs[-1]
        rect = QRectF(
            t_min,
            f_min,
            (t_max - t_min),
            (f_max - f_min)
        )
        self.img_item.setRect(rect)

        # Reset view to full range
        self.plot.setXRange(t_min, t_max, padding=0.0)
        self.plot.setYRange(f_min, f_max, padding=0.0)
        self.plot.enableAutoRange(x=False, y=False)

        # Re-add playhead (if removed by clear)
        if self.playhead_line not in self.plot.items():
            self.plot.addItem(self.playhead_line)

    # ------------------------------------------------------------------
    # Playhead line
    # ------------------------------------------------------------------
    def update_playhead(self, time_sec: float):
        """
        Move the red playhead line. Very cheap operation.
        """
        self.playhead_line.setValue(time_sec)

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------
    def clear_annotations(self):
        """
        Remove existing annotation regions from the plot.
        """
        for region in self.annotation_regions:
            self.plot.removeItem(region)
        self.annotation_regions = []
        self.annotations = []
        self.selected_annotation = None

    def draw_annotations(self, annotations):
        """
        Draw annotation regions (vertical bands).
        Non-selected: blue, Selected: green.
        """
        # Remove old
        self.clear_annotations()

        if self.times is None or self.freqs is None:
            return

        self.annotations = list(annotations)
        self.annotation_regions = []

        for ann in self.annotations:
            region = pg.LinearRegionItem(
                values=[ann.start, ann.end],
                orientation="vertical",
                movable=False,
            )

            # # Color depending on selection
            # if ann is self.selected_annotation:
            #     pen = pg.mkPen("#00ff44", width=2)         # green
            #     brush = pg.mkBrush(0, 255, 68, 60)
            # else:
            #     pen = pg.mkPen("#0099ff", width=2)         # blue
            #     brush = pg.mkBrush(0, 153, 255, 60)
            # Color depending on class
            if ann.label =="Lamantin":
                pen = pg.mkPen("#00ff44", width=2)         # green
                brush = pg.mkBrush(0, 255, 68, 60)
            else:
                pen = pg.mkPen("#0099ff", width=2)         # blue
                brush = pg.mkBrush(0, 153, 255, 60)

            region.setBrush(brush)
            region.lines[0].setPen(pen)
            region.lines[1].setPen(pen)

            region.setZValue(10)

            self.plot.addItem(region)
            self.annotation_regions.append(region)

    def _update_annotation_colors(self):
        for ann, region in zip(self.annotations, self.annotation_regions):
            if ann is self.selected_annotation:
                pen = pg.mkPen("#00ff44", width=2)
                brush = pg.mkBrush(0, 255, 68, 60)
            else:
                pen = pg.mkPen("#0099ff", width=2)
                brush = pg.mkBrush(0, 153, 255, 60)
            pen.setCosmetic(True)
            region.setBrush(brush)
            region.lines[0].setPen(pen)
            region.lines[1].setPen(pen)

    # ------------------------------------------------------------------
    # Mouse interactions
    # ------------------------------------------------------------------
    def _on_mouse_clicked(self, event):
        """
        Handle left / right click:
        - Left click: if inside an annotation -> select annotation,
                      else -> move playhead (emit time_clicked)
        - Right click: reset zoom to full range.
        """
        if self.times is None or self.freqs is None:
            return

        pos = event.scenePos()
        vb = self.plot.getViewBox()
        if vb is None:
            return

        # Map scene position to data coordinates
        point = vb.mapSceneToView(pos)
        t = float(point.x())
        f = float(point.y())

        # Right click: reset zoom
        if event.button() == Qt.RightButton:
            self.plot.enableAutoRange(x=True, y=True)
            return

        if event.button() != Qt.LeftButton:
            return

        # Ignore clicks outside x-range
        if t < self.times[0] or t > self.times[-1]:
            return

        # Check if click hits an annotation
        hit_ann = None
        for ann in self.annotations:
            if ann.start <= t <= ann.end:
                hit_ann = ann
                break

        if hit_ann is not None:
            # Select annotation
            self.selected_annotation = hit_ann
            self._update_annotation_colors()
            self.annotation_clicked.emit(hit_ann)
        else:
            # Plain time click: move playhead
            self.time_clicked.emit(t)

    def _on_time_clicked_from_viewbox(self, t: float):
        """
        Called when the user left-clicks (without dragging) in the ViewBox.
        - If click is inside an annotation -> select annotation
        - Else -> emit time_clicked for playhead
        """
        if self.times is None:
            return

        hit_ann = None
        for ann in self.annotations:
            if ann.start <= t <= ann.end:
                hit_ann = ann
                break

        if hit_ann is not None:
            self.selected_annotation = hit_ann
            self._update_annotation_colors()
            self.annotation_clicked.emit(hit_ann)
        else:
            self.time_clicked.emit(float(t))
    def _on_zoom_reset(self):
        if self.times is None or self.freqs is None:
            return
        t_min, t_max = self.times[0], self.times[-1]
        f_min, f_max = self.freqs[0], self.freqs[-1]
        self.plot.setXRange(t_min, t_max, padding=0.0)
        self.plot.setYRange(f_min, f_max, padding=0.0)

class AudioSpectrogramPlayer(QMainWindow):
    """
    Main window of the application.

    Features:
        - Menu bar:
            File > Load audio (folder of audio files, recursive)
            Settings > Spectrogram Settings
        - Left: list of audio files found
        - Right: spectrogram of the selected file
        - Play / Pause buttons for playback
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Spectrogram Player")

        # Spectrogram configuration
        self.spectrogram_settings = SpectrogramSettings()

        # Audio file management
        self.audio_files: List[str] = []
        self.current_audio_path: Optional[str] = None
        self.current_audio_data: Optional[np.ndarray] = None
        self.current_sr: Optional[int] = None
        self.base_folder: Optional[str] = None

        # Build UI and audio player
        self._build_ui()
        self._setup_player()
        self._create_menu()

        self._playhead_time: float = 0.0


        self.annotations = []            # all annotations from all files
        self.annotations_by_file = {}    # mapping: file → list[Annotation]
        self.current_annotations = []    # annotations displayed for current audio
        self.current_annotation_patches = []  # rectangles drawn on canvas

        self.max_visible_duration = 10.0   # seconds, max zoom-out window
        self.window_start = 0.0            # start time of current window (seconds)
        self.full_duration = None          # full file duration in seconds
        self.cached_full_audio = False     # True if whole file is in RAM

        self.playhead_timer = QTimer(self)
        self.playhead_timer.setInterval(30)   # ~33 FPS
        self.playhead_timer.timeout.connect(self.on_playhead_timer)


    def _build_ui(self):
        """Create the central widgets and layout using horizontal and vertical splitters."""
        central = QWidget(self)
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        # Main horizontal splitter: left panel (files) / right panel (spectrogram + info)
        main_splitter = QSplitter(Qt.Horizontal, central)
        main_splitter.setChildrenCollapsible(False)
        main_layout.addWidget(main_splitter)

        # ---------- Left side container ----------
        left_container = QWidget(main_splitter)     # THIS replaces left_widget
        left_container_layout = QVBoxLayout(left_container)

        # Header row with "Select file" button
        header_layout = QHBoxLayout()
        self.file_select_button = QToolButton(left_container)
        self.file_select_button.setText("Select file")
        self.file_select_button.setPopupMode(QToolButton.InstantPopup)
        self.file_select_menu = QMenu(self.file_select_button)
        self.file_select_button.setMenu(self.file_select_menu)
        header_layout.addWidget(self.file_select_button)
        header_layout.addStretch()
        left_container_layout.addLayout(header_layout)

        # ---------- Split the left panel vertically ----------
        left_splitter = QSplitter(Qt.Vertical, left_container)
        left_splitter.setChildrenCollapsible(False)
        left_container_layout.addWidget(left_splitter)

        #
        # Upper part: file list
        #
        files_widget = QWidget(left_splitter)
        files_layout = QVBoxLayout(files_widget)

        files_header = QLabel("Audio files:")
        files_layout.addWidget(files_header)

        self.file_list = QListWidget(files_widget)
        self.file_list.currentRowChanged.connect(self.on_file_selected)
        files_layout.addWidget(self.file_list)

        #
        # Lower part: annotation list
        #
        annotations_widget = QWidget(left_splitter)
        annotations_layout = QVBoxLayout(annotations_widget)

        ann_header = QLabel("Annotations:")
        annotations_layout.addWidget(ann_header)

        self.annotation_list = QListWidget(annotations_widget)
        self.annotation_list.itemClicked.connect(self.on_annotation_item_clicked)
        annotations_layout.addWidget(self.annotation_list)

        left_splitter.addWidget(files_widget)
        left_splitter.addWidget(annotations_widget)
        left_splitter.setSizes([500, 300])

        # ---------- Playback controls BELOW the splitter ----------
        controls_layout = QHBoxLayout()
        self.play_button = QPushButton("Play", left_container)
        self.pause_button = QPushButton("Pause", left_container)
        self.play_button.clicked.connect(self.on_play_clicked)
        self.pause_button.clicked.connect(self.on_pause_clicked)
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.pause_button)

        left_container_layout.addLayout(controls_layout)

        # Add the LEFT CONTAINER to the main splitter
        main_splitter.addWidget(left_container)
        # ---------- Right side: vertical splitter (canvas + info label) ----------
        right_splitter = QSplitter(Qt.Vertical, main_splitter)
        right_splitter.setChildrenCollapsible(False)

        canvas_container = QWidget(right_splitter)
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = SpectrogramCanvas(canvas_container)
        canvas_layout.addWidget(self.canvas)

        self.canvas.time_clicked.connect(self.on_canvas_time_clicked)
        self.canvas.annotation_clicked.connect(self.on_annotation_clicked)

        info_container = QWidget(right_splitter)
        info_layout = QVBoxLayout(info_container)
        info_layout.setContentsMargins(4, 4, 4, 4)
        self.current_file_label = QLabel("No audio loaded", info_container)
        self.current_file_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.current_file_label.setWordWrap(True)
        self.current_file_label.setMinimumHeight(40)
        info_layout.addWidget(self.current_file_label)

        right_splitter.addWidget(canvas_container)
        right_splitter.addWidget(info_container)

        main_splitter.addWidget(right_splitter)

        main_splitter.setSizes([300, 900])
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 3)

        right_splitter.setSizes([600, 100])
        right_splitter.setStretchFactor(0, 4)
        right_splitter.setStretchFactor(1, 1)

    def _setup_player(self):
        """Initialize QMediaPlayer and audio output."""
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)

        # Connect playback audio output
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.8)

        # Lightweight position update (not for animation)
        self.player.positionChanged.connect(self.on_player_position_changed)

        # Start/stop smooth animation timer
        self.player.playbackStateChanged.connect(self.on_playback_state_changed)

        # Smooth playhead animation timer (~33 FPS)
        self.playhead_timer = QTimer(self)
        self.playhead_timer.setInterval(30)  # ~33 FPS
        self.playhead_timer.timeout.connect(self.on_playhead_timer)

    def _create_menu(self):
        """Create menu bar with File and Settings menus."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        load_action = file_menu.addAction("Load audio")
        load_action.triggered.connect(self.on_load_audio_folder)
        load_ann = file_menu.addAction("Load Annotation")
        load_ann.triggered.connect(self.on_load_annotations)

        # Settings menu
        settings_menu = menubar.addMenu("Settings")
        spectro_action = settings_menu.addAction("Spectrogram Settings")
        spectro_action.triggered.connect(self.on_open_spectrogram_settings)

    def on_load_annotations(self):
        """
        Load annotation CSV. Each row:
            file,start,end,class,description
        """
        csv_path, _ = QFileDialog.getOpenFileName(self, "Load annotation CSV", "", "CSV (*.csv)")
        if not csv_path:
            return

        self.annotations = []
        self.annotations_by_file.clear()

        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ann = Annotation(
                        file=row["file"],
                        start=float(row["start"]),
                        end=float(row["end"]),
                        label=row["class"],
                        description=row["description"],
                    )
                    self.annotations.append(ann)

                    # group by file
                    self.annotations_by_file.setdefault(ann.file, []).append(ann)

                # Populate annotation list (lower panel)
                self.annotation_list.clear()

                for ann in self.annotations:
                    text = f"{ann.file} | {ann.start:.2f}s - {ann.end:.2f}s | {ann.label}"
                    self.annotation_list.addItem(text)


        except Exception as e:
            QMessageBox.critical(self, "CSV Error", f"Failed to load CSV:\n{e}")
            return

        QMessageBox.information(self, "Annotation loaded", f"Loaded {len(self.annotations)} annotations.")
        

    def build_file_selection_menu(self, base_folder: str):
        """
        Build a hierarchical dropdown menu of audio files, grouped by folders and subfolders.
        """
        if not hasattr(self, "file_select_menu"):
            return

        self.file_select_menu.clear()

        # Map of folder path tuples to QMenu instances
        # Root (empty tuple) corresponds to the top-level menu
        menu_map: dict[tuple, QMenu] = {(): self.file_select_menu}

        for full_path in self.audio_files:
            # Relative path from base folder
            rel_path = os.path.relpath(full_path, base_folder)
            parts = rel_path.split(os.sep)

            if not parts:
                continue

            folder_parts = parts[:-1]
            filename = parts[-1]

            # Walk or create submenus for each folder level
            parent_key: tuple = ()
            parent_menu = self.file_select_menu

            for folder in folder_parts:
                key = parent_key + (folder,)
                if key not in menu_map:
                    submenu = parent_menu.addMenu(folder)
                    menu_map[key] = submenu
                parent_menu = menu_map[key]
                parent_key = key

            # Final action for the file
            action = parent_menu.addAction(filename)
            # Use default arg in lambda to capture current full_path
            action.triggered.connect(
                lambda checked=False, p=full_path: self.on_menu_file_selected(p)
            )


    def on_menu_file_selected(self, path: str):
        """
        Called when a file is selected from the dropdown menu.
        It synchronizes the QListWidget selection and loads the file.
        """
        if path in self.audio_files:
            index = self.audio_files.index(path)
            # This will trigger on_file_selected and load the audio
            self.file_list.setCurrentRow(index)
        else:
            # Fallback: load directly
            self.load_audio(path)

    # ---------- File loading and selection ----------

    def on_load_audio_folder(self):
        """Open a folder dialog and scan recursively for audio files with a progress dialog."""
        folder = QFileDialog.getExistingDirectory(self, "Select audio folder")
        if not folder:
            return

        audio_ext = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
        found_files = []

        # First pass: count files (for progress)
        total_files = 0
        for root, _, files in os.walk(folder):
            total_files += len(files)

        if total_files == 0:
            QMessageBox.warning(self, "No files", "This folder is empty.")
            return

        progress = QProgressDialog(
            "Scanning audio files...",
            None,
            0,
            total_files,
            self
        )
        progress.setWindowTitle("Loading folder")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        scanned = 0
        for root, _, files in os.walk(folder):
            for name in files:
                scanned += 1
                progress.setValue(scanned)
                QApplication.processEvents()

                if progress.wasCanceled():
                    return

                ext = os.path.splitext(name)[1].lower()
                if ext in audio_ext:
                    full_path = os.path.join(root, name)
                    found_files.append(full_path)

        progress.close()

        if not found_files:
            QMessageBox.warning(self, "No audio found", "No audio files were found in this folder.")
            return

        found_files.sort()

        self.audio_files = found_files
        self.base_folder = folder  # remember root folder for relative paths

        self.file_list.clear()
        base_len = len(folder.rstrip(os.sep)) + 1
        for path in self.audio_files:
            relative = path[base_len:]
            self.file_list.addItem(relative)

        # Build the hierarchical dropdown menu
        self.build_file_selection_menu(folder)

        # Select first file by default
        self.file_list.setCurrentRow(0)

    def on_file_selected(self, index: int):
        """Triggered when the user selects a file in the list."""
        if index < 0 or index >= len(self.audio_files):
            return

        path = self.audio_files[index]
        self.load_audio(path)

    def load_audio(self, path: str):
        """
        Load audio metadata and decide whether to cache the full file in memory
        or work in 10-second streaming windows. Never display more than
        max_visible_duration seconds at once.
        """
        self.current_audio_path = path

        # Get full duration (seconds) without loading entire file into RAM
        try:
            full_duration = librosa.get_duration(path=path)
        except Exception as exc:
            QMessageBox.critical(self, "Error reading audio", f"Could not read audio duration:\n{exc}")
            return

        # Load a small head chunk just to get the sample rate
        try:
            head_audio, sr = librosa.load(path, sr=None, mono=True, duration=min(self.max_visible_duration, 5.0))
        except Exception as exc:
            QMessageBox.critical(self, "Error loading audio", f"Could not load audio file:\n{exc}")
            return

        self.current_sr = sr
        self.full_duration = full_duration
        self._playhead_time = 0.0
        self.window_start = 0.0

        # Estimate full audio memory usage (mono, float32 ~ 4 bytes/sample)
        approx_bytes = int(full_duration * sr * 4)
        max_cache_bytes = self.spectrogram_settings.max_cache_mb * 1024 * 1024

        if approx_bytes <= max_cache_bytes:
            # Safe to cache entire file in memory
            try:
                audio_full, sr_full = librosa.load(path, sr=None, mono=True)
            except Exception as exc:
                QMessageBox.critical(self, "Error loading audio", f"Could not load full audio file:\n{exc}")
                return
            self.current_audio_data = audio_full
            self.current_sr = sr_full
            self.cached_full_audio = True
        else:
            # Streaming mode: keep only a small window in memory
            self.current_audio_data = head_audio
            self.cached_full_audio = False

        # Set initial visible window (at t=0)
        self.set_window_start(0.0)

        # Build info text using full_duration (not just the window length)
        num_samples = int(self.full_duration * self.current_sr)
        info_lines = [
            f"Current file: {os.path.basename(path)}",
            f"Sample rate: {self.current_sr} Hz  |  Duration: {self.full_duration:.2f} s  |  Samples: {num_samples}",
        ]

        db_min = getattr(self.canvas, "db_min", None)
        db_max = getattr(self.canvas, "db_max", None)
        if db_min is not None and db_max is not None:
            info_lines.append(f"Spectrogram dB range: min={db_min:.1f} dB  |  max={db_max:.1f} dB")

        self.current_file_label.setText("\n".join(info_lines))

        # Prepare QMediaPlayer for playback (it streams from disk by itself)
        self.player.setSource(QUrl.fromLocalFile(path))


    def set_window_start(self, start_time: float):
        """
        Set the visible window start time (in seconds), load the corresponding
        audio window (max 10s) and redraw the spectrogram.
        """
        if self.current_audio_path is None or self.current_sr is None or self.full_duration is None:
            return

        # Clamp start so that window stays inside [0, full_duration]
        max_start = max(0.0, self.full_duration - self.max_visible_duration)
        start_time = max(0.0, min(start_time, max_start))
        self.window_start = start_time

        if self.cached_full_audio and self.current_audio_data is not None:
            # Use slice from cached full audio
            sr = self.current_sr
            start_idx = int(start_time * sr)
            end_idx = int((start_time + self.max_visible_duration) * sr)
            audio_window = self.current_audio_data[start_idx:end_idx]
        else:
            # Streaming mode: only load this 10s window from disk
            audio_window, sr = librosa.load(
                self.current_audio_path,
                sr=None,
                mono=True,
                offset=start_time,
                duration=self.max_visible_duration,
            )
            self.current_sr = sr
            self.current_audio_data = audio_window

        # Redraw spectrogram for this window
        self.canvas.plot_spectrogram(audio_window, self.current_sr, self.spectrogram_settings)

        # Update playhead line position relative to this window
        self.update_playhead_visual()

    def on_play_clicked(self):
        """Play the current audio file."""
        if not self.current_audio_path:
            QMessageBox.information(self, "No audio", "Please load and select an audio file first.")
            return
        self.player.play()

    def on_pause_clicked(self):
        """Pause audio playback."""
        self.player.pause()


    def on_canvas_time_clicked(self, t_local: float):
        """
        t_local : time inside the current visible 10s window.
        Convert it to global audio time, move playhead, reposition window if needed.
        """
        if self.current_sr is None or self.full_duration is None:
            return

        # Convert local time -> global time
        global_t = self.window_start + t_local

        # Clamp to [0, full_duration]
        global_t = max(0.0, min(global_t, self.full_duration))

        # Move internal playhead
        self._playhead_time = global_t

        # Move QMediaPlayer (milliseconds)
        self.player.setPosition(int(global_t * 1000))

        # If click is outside current window, recenter window
        if not (self.window_start <= global_t <= self.window_start + self.max_visible_duration):
            self.set_window_start(global_t - self.max_visible_duration * 0.5)

        # Update visual red line
        self.update_playhead_visual()

    def on_player_position_changed(self, pos_ms: int):
        """
        Lightweight update; smooth movement is handled by the timer.
        """
        self._playhead_time = pos_ms / 1000.0

    def on_playback_state_changed(self, state):
        """
        Start or stop the playhead animation timer.
        """
        if state == QMediaPlayer.PlayingState:
            self.playhead_timer.start()
        else:
            self.playhead_timer.stop()


    def on_annotation_clicked(self, ann: Annotation):
        """
        Display annotation info in the bottom panel instead of file info.
        """
        text = (
            f"Annotation\n"
            f"Class: {ann.label}\n"
            f"Start: {ann.start:.2f} s\n"
            f"End: {ann.end:.2f} s\n"
            f"Description: {ann.description}"
        )
        self.current_file_label.setText(text)

    def on_annotation_item_clicked(self, item):
        """
        When clicking on an annotation in the lower-left list:
        - Find the corresponding annotation
        - Load its audio file
        - Select the file in the file list
        - Update annotation info in bottom panel
        """
        text = item.text()

        # retrieve the annotation object
        for ann in self.annotations:
            summary = f"{ann.file} | {ann.start:.2f}s - {ann.end:.2f}s | {ann.label}"
            if summary == text:
                # Found annotation
                # Step 1: locate audio file
                target_file = ann.file

                # Find full path in audio_files list
                for i, path in enumerate(self.audio_files):
                    if os.path.basename(path) == target_file:
                        # Select this file
                        self.file_list.setCurrentRow(i)
                        self.canvas.selected_annotation = ann
                        self.canvas.draw_annotations(self.current_annotations)

                        # Move playhead to annotation start
                        self._playhead_time = ann.start
                        self.update_playhead_visual()

                        # Print annotation info on bottom panel
                        info = (
                            f"Annotation\n"
                            f"Class: {ann.label}\n"
                            f"Start: {ann.start:.2f} s\n"
                            f"End:   {ann.end:.2f} s\n"
                            f"Description: {ann.description}"
                        )
                        self.current_file_label.setText(info)

                        return


    def update_playhead_visual(self):
        """
        Update the playhead line on the spectrogram canvas to reflect
        the current playhead time relative to the visible window.
        """
        if self.current_sr is None or self.full_duration is None:
            return

        # Global -> local time in current 10s window
        local_t = self._playhead_time - self.window_start
        if local_t < 0.0 or local_t > self.max_visible_duration:
            # Playhead is outside current window; optionally hide it
            return

        self.canvas.update_playhead(local_t)

    def on_playhead_timer(self):
        """
        Called periodically (e.g. every 30 ms) to smoothly update
        the playhead position and slide the visible 10s window if needed.
        """
        if self.player.playbackState() != QMediaPlayer.PlayingState:
            return
        if self.current_sr is None or self.full_duration is None:
            return

        # Get playback position
        pos_ms = self.player.position()
        t_sec = pos_ms / 1000.0
        self._playhead_time = t_sec

        # Auto-slide the visible 10s spectrogram window if file is long
        if self.full_duration > self.max_visible_duration:
            margin = 0.2 * self.max_visible_duration  # 20% margin on each side

            if (
                t_sec < self.window_start + margin
                or t_sec > self.window_start + self.max_visible_duration - margin
            ):
                # Recenter playhead in the window
                new_start = t_sec - 0.5 * self.max_visible_duration
                self.set_window_start(new_start)

        # Update red playhead position inside current window
        self.update_playhead_visual()



    # ---------- Spectrogram settings ----------

    def on_open_spectrogram_settings(self):
        """Open the settings dialog and redraw spectrogram if changed."""
        dialog = SpectrogramSettingsDialog(self.spectrogram_settings, self)
        if dialog.exec() == QDialog.Accepted:
            self.spectrogram_settings = dialog.get_settings()
            # Recompute spectrogram for the current audio if available
            if self.current_audio_data is not None and self.current_sr is not None:
                self.canvas.plot_spectrogram(
                    self.current_audio_data,
                    self.current_sr,
                    self.spectrogram_settings,
                )


