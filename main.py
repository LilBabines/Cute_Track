import sys

from PySide6 import QtWidgets
from src.qt_windows import LauncherWindow
from src.qt_workers import DetectionWorker

from src.deep_learning.models.SAMUNET import SAM2UNet, LitBinarySeg

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = LauncherWindow()
    w.show()
    app.exec()

if __name__ == "__main__":

    main()
