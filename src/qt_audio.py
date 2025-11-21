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
    QProgressDialog, QSplitter, QToolButton, QMenu, QDoubleSpinBox
)
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from mpl_toolkits.axes_grid1 import make_axes_locatable


@dataclass
class SpectrogramSettings:
    """Container for spectrogram parameters."""
    n_fft: int = 2048
    hop_ms: float = 10.0         # hop duration in milliseconds (time resolution)
    cmap: str = "magma"
    max_freq: Optional[int] = None  # Hz, None means full range


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
            hop_ms=settings.hop_ms,
            cmap=settings.cmap,
            max_freq=settings.max_freq,
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
        self.max_freq_spin.setValue(self._settings.max_freq or 0)
        self.max_freq_spin.setSpecialValueText("0 (full range)")
        form.addRow("Max frequency (Hz):", self.max_freq_spin)

        # Colormap combo box
        self.cmap_combo = QComboBox()
        cmaps = ["magma", "viridis", "plasma", "inferno", "cividis"]
        self.cmap_combo.addItems(cmaps)
        if self._settings.cmap in cmaps:
            self.cmap_combo.setCurrentText(self._settings.cmap)
        form.addRow("Colormap:", self.cmap_combo)

        layout.addLayout(form)

        # OK / Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_settings(self) -> SpectrogramSettings:
        """Return updated settings based on the dialog inputs."""
        self._settings.n_fft = int(self.n_fft_spin.value())
        self._settings.hop_ms = float(self.hop_ms_spin.value())
        maxf = int(self.max_freq_spin.value())
        self._settings.max_freq = maxf if maxf > 0 else None
        self._settings.cmap = self.cmap_combo.currentText()
        return self._settings

    
class SpectrogramCanvas(FigureCanvas):
    """
    Dark, borderless spectrogram canvas with:
        - DAW-style look (no frame, no title, minimal or no ticks)
        - zoom rectangle (left-drag)
        - right-click zoom reset
        - playhead (vertical red line) updated via update_playhead()
        - click vs drag detection (click moves playhead, drag zooms)
    """

    # Emitted when the user left-clicks (without dragging) on the spectrogram (time in seconds)
    time_clicked = Signal(float)

    def __init__(self, parent=None):
        # Create figure and a single axes
        fig = Figure(figsize=(6, 4))
        self.ax = fig.add_subplot(111)

        super().__init__(fig)
        self.setParent(parent)

        # Dark theme for the whole figure
        self.figure.patch.set_facecolor("#232323")
        self.ax.set_facecolor("#232323")

        # Remove all spines (no visible frame)
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        # No title
        self.ax.set_title("")

        # Optional: minimal ticks style (you can comment these out to remove ticks entirely)
        self.ax.tick_params(
            colors="white",
            which="both",
            direction="out",
            length=2,
            width=0.8,
            labelsize=7,
        )

        # If you prefer NO tick labels at all (pure borderless visual), uncomment:
        # self.ax.set_xticklabels([])
        # self.ax.set_yticklabels([])

        # Full extents for zoom reset
        self._full_xlim = None
        self._full_ylim = None

        # Playhead (vertical red line)
        self.playhead_line = None

        # Rectangle selector for zoom (left mouse button)
        self.selector = RectangleSelector(
            self.ax,
            self.on_select_rectangle,
            useblit=True,
            button=[1],  # left mouse button
            minspanx=0.01,
            minspany=0.01,
            spancoords="data",
            interactive=False,
            drag_from_anywhere=True,
        )

        # Variables to detect click vs drag
        self._press_event = None
        self._press_cid = self.mpl_connect("button_press_event", self._on_button_press)
        self._release_cid = self.mpl_connect("button_release_event", self._on_button_release)

    # ------------------------------------------------------------------
    # Spectrogram drawing
    # ------------------------------------------------------------------
    def plot_spectrogram(self, audio: np.ndarray, sr: int, settings):
        """
        Compute and draw the spectrogram of the given audio signal
        in a DAW-style, borderless, dark canvas.
        """
        # Clear figure and recreate axes to remove old colorbars cleanly
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        # Reapply dark theme and borderless style
        self.figure.patch.set_facecolor("#232323")
        self.ax.set_facecolor("#232323")

        for spine in self.ax.spines.values():
            spine.set_visible(False)

        self.ax.set_title("")

        self.ax.tick_params(
            colors="white",
            which="both",
            direction="out",
            length=2,
            width=0.8,
            labelsize=7,
        )

        # If you want completely borderless (no tick labels), uncomment:
        # self.ax.set_xticklabels([])
        # self.ax.set_yticklabels([])

        self.playhead_line = None  # will be recreated for new spectrogram

        if audio is None or len(audio) == 0:
            self.ax.text(
                0.5,
                0.5,
                "No audio loaded",
                color="white",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )
            self.draw()
            return

        # --- Compute STFT using librosa ---
        # Window size in samples
        n_fft = settings.n_fft

        # Time step (hop) in samples, computed from hop_ms so resolution is independent of file length
        hop_length = max(1, int(sr * (settings.hop_ms / 1000.0)))

        S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

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

        # --- Plot spectrogram image ---
        img = self.ax.imshow(
            S_db,
            origin="lower",
            aspect="auto",
            extent=[times.min(), times.max(), freqs.min(), freqs.max()],
            cmap=settings.cmap,
            interpolation="bilinear",  # slightly smoothed look
        )

        # Anchor axes to the left so we use as much width as possible
        self.ax.set_anchor("W")
        self.ax.margins(0)

        # Remove axis labels for a clean DAW look
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")

        # --- Colorbar tightly attached using axes_grid1 ---
        divider = make_axes_locatable(self.ax)
        # Thin, compact colorbar on the right, very close to the spectrogram
        cax = divider.append_axes("right", size="2%", pad=0.05)

        cbar = self.figure.colorbar(img, cax=cax, format="%+2.0f dB")
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(color="white", labelcolor="white", labelsize=7)
        cbar.ax.set_facecolor("#232323")

        # Save full limits for zoom reset
        self._full_xlim = (times.min(), times.max())
        self._full_ylim = (freqs.min(), freqs.max())

        # Recreate RectangleSelector on the new axis (because we recreated self.ax)
        self.selector = RectangleSelector(
            self.ax,
            self.on_select_rectangle,
            useblit=True,
            button=[1],
            minspanx=0.01,
            minspany=0.01,
            spancoords="data",
            interactive=False,
            drag_from_anywhere=True,
        )

        # Adjust subplot margins: enough for x/y ticks, still very clean
        self.figure.subplots_adjust(
            left=0.07,     # enough space for Y tick labels
            right=0.945,   # space reserved for right-side colorbar
            top=0.985,     # almost no top margin
            bottom=0.07,   # space for X tick labels
        )

        self.draw()

    # ------------------------------------------------------------------
    # Zoom handling
    # ------------------------------------------------------------------
    def on_select_rectangle(self, eclick, erelease):
        """
        Callback called when the user finishes drawing the rectangle.
        It zooms the axes to the selected region.
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        if None in (x1, y1, x2, y2):
            return

        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])

        if abs(xmax - xmin) < 1e-6 or abs(ymax - ymin) < 1e-6:
            return

        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.draw()

    def reset_zoom(self):
        """Reset axes limits to the full spectrogram view."""
        if self._full_xlim is not None and self._full_ylim is not None:
            self.ax.set_xlim(*self._full_xlim)
            self.ax.set_ylim(*self._full_ylim)
            self.draw()

    def mousePressEvent(self, event):
        """
        Allow right-click zoom reset on the Qt side.
        Left-click and drag are handled by Matplotlib events.
        """
        if event.button() == Qt.RightButton:
            self.reset_zoom()
        super().mousePressEvent(event)

    # ------------------------------------------------------------------
    # Click vs drag detection (for playhead vs zoom)
    # ------------------------------------------------------------------
    def _on_button_press(self, event):
        """Store the press event to later decide if it was a click or a drag."""
        if event.button != 1:
            return
        if event.inaxes != self.ax:
            return
        self._press_event = event

    def _on_button_release(self, event):
        """
        On release, if the mouse did not move much, treat this as a click
        and emit time_clicked. If there was a real drag, we consider it a
        zoom rectangle (handled by RectangleSelector).
        """
        if event.button != 1:
            return
        if event.inaxes != self.ax:
            return
        if self._press_event is None:
            return

        dx = event.x - self._press_event.x
        dy = event.y - self._press_event.y
        dist2 = dx * dx + dy * dy

        click_threshold = 5  # pixels
        if dist2 <= click_threshold * click_threshold:
            if event.xdata is not None:
                self.time_clicked.emit(float(event.xdata))

        self._press_event = None

    # ------------------------------------------------------------------
    # Playhead line
    # ------------------------------------------------------------------
    def update_playhead(self, time_sec: float):
        """
        Draw or move the vertical red playhead line at the given time (seconds).
        This is intended to be called frequently (e.g. from a QTimer).
        """
        if self.ax is None:
            return

        if self.playhead_line is None:
            self.playhead_line = self.ax.axvline(
                time_sec,
                color="#ff4444",
                linewidth=2.0,
            )
        else:
            self.playhead_line.set_xdata([time_sec, time_sec])

        # Light redraw, does not recompute the spectrogram
        self.draw_idle()

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

    def _build_ui(self):
        """Create the central widgets and layout using horizontal and vertical splitters."""
        central = QWidget(self)
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        # Main horizontal splitter: left panel (files) / right panel (spectrogram + info)
        main_splitter = QSplitter(Qt.Horizontal, central)
        main_splitter.setChildrenCollapsible(False)
        main_layout.addWidget(main_splitter)

        # ---------- Left side: file list + playback controls ----------
        left_widget = QWidget(main_splitter)
        left_layout = QVBoxLayout(left_widget)

        # Top row: label + dropdown button
        header_layout = QHBoxLayout()
        # header_label = QLabel("Audio files:", left_widget)
        # header_layout.addWidget(header_label)

        # Dropdown button with hierarchical menu for file selection
        self.file_select_button = QToolButton(left_widget)
        self.file_select_button.setText("Select file")
        self.file_select_button.setPopupMode(QToolButton.InstantPopup)
        self.file_select_menu = QMenu(self.file_select_button)
        self.file_select_button.setMenu(self.file_select_menu)
        header_layout.addWidget(self.file_select_button)

        header_layout.addStretch()
        left_layout.addLayout(header_layout)

        # File list widget
        self.file_list = QListWidget(left_widget)
        self.file_list.currentRowChanged.connect(self.on_file_selected)
        left_layout.addWidget(self.file_list)

        # Playback controls
        controls_layout = QHBoxLayout()
        self.play_button = QPushButton("Play", left_widget)
        self.pause_button = QPushButton("Pause", left_widget)
        self.play_button.clicked.connect(self.on_play_clicked)
        self.pause_button.clicked.connect(self.on_pause_clicked)
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.pause_button)
        left_layout.addLayout(controls_layout)

        main_splitter.addWidget(left_widget)

        # ---------- Right side: vertical splitter (canvas + info label) ----------
        right_splitter = QSplitter(Qt.Vertical, main_splitter)
        right_splitter.setChildrenCollapsible(False)

        canvas_container = QWidget(right_splitter)
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = SpectrogramCanvas(canvas_container)
        canvas_layout.addWidget(self.canvas)

        self.canvas.time_clicked.connect(self.on_canvas_time_clicked)

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

        self.player.positionChanged.connect(self.on_player_position_changed)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.8)

    def _create_menu(self):
        """Create menu bar with File and Settings menus."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        load_action = file_menu.addAction("Load audio")
        load_action.triggered.connect(self.on_load_audio_folder)

        # Settings menu
        settings_menu = menubar.addMenu("Settings")
        spectro_action = settings_menu.addAction("Spectrogram Settings")
        spectro_action.triggered.connect(self.on_open_spectrogram_settings)


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
        Load audio data from file path (for spectrogram) and
        also prepare QMediaPlayer for playback.
        """
        self.current_audio_path = path

        try:
            # Load audio samples using librosa (mono, original sample rate)
            audio, sr = librosa.load(path, sr=None, mono=True)
        except Exception as exc:
            QMessageBox.critical(self, "Error loading audio", f"Could not load audio file:\n{exc}")
            return

        self.current_audio_data = audio
        self.current_sr = sr

        # Compute basic audio characteristics
        duration = len(audio) / sr
        num_samples = len(audio)

        # Reset playhead to start of file
        self._playhead_time = 0.0

        # Compute spectrogram (this also updates db_min/db_max in canvas)
        self.canvas.plot_spectrogram(audio, sr, self.spectrogram_settings)

        # Retrieve dB values from canvas if available
        db_min = getattr(self.canvas, "db_min", None)
        db_max = getattr(self.canvas, "db_max", None)

        # Build info text
        info_lines = [
            f"Current file: {os.path.basename(path)}",
            f"Sample rate: {sr} Hz  |  Duration: {duration:.2f} s  |  Samples: {num_samples}",
        ]

        self.current_file_label.setText("\n".join(info_lines))

        # Update visual playhead line at time = 0
        self.update_playhead_visual()

        # Prepare QMediaPlayer for playback
        self.player.setSource(QUrl.fromLocalFile(path))

    def on_play_clicked(self):
        """Play the current audio file."""
        if not self.current_audio_path:
            QMessageBox.information(self, "No audio", "Please load and select an audio file first.")
            return
        self.player.play()

    def on_pause_clicked(self):
        """Pause audio playback."""
        self.player.pause()


    def on_canvas_time_clicked(self, t_sec: float):
        """
        Called when the user left-clicks on the spectrogram.
        Moves the audio playhead to the clicked time.
        """
        if self.current_sr is None or self.current_audio_data is None:
            return

        # Clamp time to valid range of the audio
        duration = len(self.current_audio_data) / self.current_sr
        t_sec = max(0.0, min(t_sec, duration))

        # Remember playhead time and update visual line
        self._playhead_time = t_sec
        self.update_playhead_visual()

        # If there is a loaded media, move QMediaPlayer's position
        if not self.player.source().isEmpty():
            position_ms = int(t_sec * 1000.0)
            self.player.setPosition(position_ms)
            # If player is already playing, it will continue from this new position.
            # If it is paused/stopped, the next Play will start from here.


    def on_player_position_changed(self, pos_ms: int):
        """
        Keep the playhead line in sync with the actual playback position.
        """
        if self.current_sr is None or self.current_audio_data is None:
            return

        t_sec = pos_ms / 1000.0
        duration = len(self.current_audio_data) / self.current_sr

        if t_sec < 0.0 or t_sec > duration + 0.1:
            return

        self._playhead_time = t_sec
        self.update_playhead_visual()


    def update_playhead_visual(self):
        """
        Update the playhead line on the spectrogram canvas
        to reflect the current playhead time.
        """
        if self.current_sr is None or self.current_audio_data is None:
            return
        self.canvas.update_playhead(self._playhead_time)


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


