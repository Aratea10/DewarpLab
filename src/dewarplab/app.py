import sys

from PySide6.QtWidgets import QApplication

from dewarplab.ui.main_window import MainWindow


def run() -> int:
    app = QApplication(sys.argv)

    app.setApplicationName("DewarpLab")
    app.setOrganizationName("DewarpLab")

    window = MainWindow()
    window.show()

    return app.exec()
