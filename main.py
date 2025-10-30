import sys

from PyQt6 import QtWidgets
from src.qt_windows import OBB_VideoPlayer,BaseVideoPlayer

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = BaseVideoPlayer()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
