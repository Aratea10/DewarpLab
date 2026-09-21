from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("DewarpLab")
        self.resize(1200, 800)

        welcome_label = QLabel("DewarpLab\n\n" "Abre un documento para comenzar.")

        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.setCentralWidget(welcome_label)

        self.statusBar().showMessage("DewarpLab listo.")
