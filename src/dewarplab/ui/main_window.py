from pathlib import Path

from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QDragEnterEvent,
    QDropEvent,
    QKeySequence,
)
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSpinBox,
    QToolBar,
)

from dewarplab.adapters.documents.document_loader import (
    DocumentLoadError,
    LoadedDocument,
    is_supported_document,
    load_document,
    render_page,
)
from dewarplab.ui.document_view import DocumentView


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self._document: LoadedDocument | None = None
        self._current_page_index = 0

        self.setWindowTitle("DewarpLab")
        self.resize(1300, 850)
        self.setAcceptDrops(True)

        self._view = DocumentView(self)

        self._view.file_dropped.connect(self.open_document)

        self.setCentralWidget(self._view)

        self._create_actions()
        self._create_toolbar()

        self.statusBar().showMessage("Arrastra un documento aquí o pulsa Abrir.")

    def _create_actions(self) -> None:
        self._open_action = QAction(
            "Abrir…",
            self,
        )

        self._open_action.setShortcut(QKeySequence.StandardKey.Open)

        self._open_action.triggered.connect(self._open_document_dialog)

        self._zoom_out_action = QAction(
            "Reducir",
            self,
        )

        self._zoom_out_action.triggered.connect(self._view.zoom_out)

        self._zoom_in_action = QAction(
            "Ampliar",
            self,
        )

        self._zoom_in_action.triggered.connect(self._view.zoom_in)

        self._fit_action = QAction(
            "Ajustar",
            self,
        )

        self._fit_action.triggered.connect(self._view.fit_document)

        self._set_document_actions_enabled(False)

    def _create_toolbar(self) -> None:
        toolbar = QToolBar(
            "Documento",
            self,
        )

        toolbar.setMovable(False)
        toolbar.setMinimumHeight(40)

        toolbar_font = toolbar.font()

        if toolbar_font.pointSizeF() > 0:
            toolbar_font.setPointSizeF(toolbar_font.pointSizeF() + 1.5)

        toolbar.setFont(toolbar_font)

        self.addToolBar(toolbar)

        toolbar.addAction(self._open_action)

        toolbar.addSeparator()

        toolbar.addAction(self._zoom_out_action)

        toolbar.addAction(self._zoom_in_action)

        toolbar.addAction(self._fit_action)

        toolbar.addSeparator()

        self._page_label = QLabel("Página: ")

        self._page_label.setFont(toolbar_font)

        toolbar.addWidget(self._page_label)

        self._page_spinbox = QSpinBox(self)

        self._page_spinbox.setFont(toolbar_font)

        self._page_spinbox.setMinimum(1)
        self._page_spinbox.setMaximum(1)
        self._page_spinbox.setValue(1)
        self._page_spinbox.setEnabled(False)

        self._page_spinbox.valueChanged.connect(self._page_changed)

        toolbar.addWidget(self._page_spinbox)

    def _set_document_actions_enabled(
        self,
        enabled: bool,
    ) -> None:
        self._zoom_out_action.setEnabled(enabled)

        self._zoom_in_action.setEnabled(enabled)

        self._fit_action.setEnabled(enabled)

    def _open_document_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Abrir documento",
            "",
            ("Documentos compatibles " "(*.png *.jpg *.jpeg *.tif *.tiff *.pdf)"),
        )

        if path:
            self.open_document(path)

    def open_document(
        self,
        path: str | Path,
    ) -> None:
        try:
            document = load_document(path)
        except DocumentLoadError as error:
            QMessageBox.critical(
                self,
                "No se pudo abrir el documento",
                str(error),
            )

            return

        previous_document = self._document

        self._document = document
        self._current_page_index = 0

        if previous_document is not None:
            previous_document.close()

        self._configure_page_selector()
        self._render_current_page()

        self._set_document_actions_enabled(True)

        self.setWindowTitle(f"DewarpLab — {document.path.name}")

    def _configure_page_selector(self) -> None:
        if self._document is None:
            self._page_spinbox.setEnabled(False)

            return

        self._page_spinbox.blockSignals(True)

        self._page_spinbox.setMinimum(1)

        self._page_spinbox.setMaximum(self._document.page_count)

        self._page_spinbox.setValue(1)

        self._page_spinbox.blockSignals(False)

        self._page_spinbox.setEnabled(self._document.page_count > 1)

    def _page_changed(
        self,
        page_number: int,
    ) -> None:
        self._current_page_index = page_number - 1

        self._render_current_page()

    def _render_current_page(self) -> None:
        if self._document is None:
            return

        try:
            image = render_page(
                self._document,
                self._current_page_index,
            )
        except DocumentLoadError as error:
            QMessageBox.critical(
                self,
                "No se pudo mostrar la página",
                str(error),
            )

            return

        self._view.set_image(image)

        if self._document.is_pdf:
            page_text = (
                f"Página "
                f"{self._current_page_index + 1} "
                f"de {self._document.page_count}"
            )
        else:
            page_text = "Imagen"

        self.statusBar().showMessage(
            f"{self._document.path.name} — "
            f"{page_text} — "
            f"{image.width()} × "
            f"{image.height()} px"
        )

    def dragEnterEvent(
        self,
        event: QDragEnterEvent,
    ) -> None:
        urls = event.mimeData().urls()

        if len(urls) != 1:
            event.ignore()
            return

        path = urls[0].toLocalFile()

        if path and is_supported_document(path):
            event.acceptProposedAction()
            return

        event.ignore()

    def dropEvent(
        self,
        event: QDropEvent,
    ) -> None:
        urls = event.mimeData().urls()

        if len(urls) != 1:
            event.ignore()
            return

        path = urls[0].toLocalFile()

        if not path:
            event.ignore()
            return

        self.open_document(path)

        event.acceptProposedAction()

    def closeEvent(
        self,
        event: QCloseEvent,
    ) -> None:
        if self._document is not None:
            self._document.close()

        super().closeEvent(event)
