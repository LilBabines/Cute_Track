import sys

from PySide6 import QtWidgets
from src.qt_audio import AudioSpectrogramPlayer
from src.qt_workers import DetectionWorker

from src.deep_learning.models.SAMUNET import SAM2UNet, LitBinarySeg

def main():
    """Application entry point."""
    app = QtWidgets.QApplication(sys.argv)
    window = AudioSpectrogramPlayer()
    window.resize(1200, 700)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()