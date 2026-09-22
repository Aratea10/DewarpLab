import sys

from PySide6.QtGui import QImageReader
from PySide6.QtWidgets import QApplication

from dewarplab.ui.main_window import MainWindow


IMAGE_ALLOCATION_LIMIT_MB = 1024


def run() -> int:
    QImageReader.setAllocationLimit(IMAGE_ALLOCATION_LIMIT_MB)

    app = QApplication(sys.argv)

    app.setApplicationName("DewarpLab")

    app.setOrganizationName("DewarpLab")

    window = MainWindow()
    window.show()

    return app.exec()
