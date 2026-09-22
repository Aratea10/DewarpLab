from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QIntValidator,
    QKeySequence,
)
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QToolBar,
    QToolButton,
)

from dewarplab.adapters.documents.document_loader import (
    DocumentLoadError,
    LoadedDocument,
    SUPPORTED_EXTENSIONS,
    is_supported_document,
    load_document,
    render_page,
)
from dewarplab.domain import Mesh
from dewarplab.ui.document_view import DocumentView


DEFAULT_MESH_ROWS = 8
DEFAULT_MESH_COLUMNS = 8


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self._document: LoadedDocument | None = None
        self._current_page_index = 0

        self._page_meshes: dict[
            int,
            Mesh,
        ] = {}

        self.setWindowTitle("DewarpLab")

        self.resize(
            1300,
            850,
        )

        self.setAcceptDrops(True)

        self._view = DocumentView(self)

        self._view.file_dropped.connect(self.open_document)

        self._view.browse_requested.connect(self._open_document_dialog)

        self._view.zoom_changed.connect(self._update_zoom_label)

        self.setCentralWidget(self._view)

        self._create_actions()
        self._create_toolbar()

        self._set_document_actions_enabled(False)

        self._update_page_controls()

        self.statusBar().hide()

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

        self._mesh_action = QAction(
            "Malla",
            self,
        )

        self._mesh_action.setCheckable(True)

        self._mesh_action.setChecked(True)

        self._mesh_action.toggled.connect(self._view.set_mesh_visible)

    def _create_toolbar(self) -> None:
        toolbar = QToolBar(
            "Documento",
            self,
        )

        toolbar.setMovable(False)

        toolbar.setMinimumHeight(40)

        base_font = toolbar.font()

        action_font = QFont(base_font)

        if action_font.pointSizeF() > 0:
            action_font.setPointSizeF(action_font.pointSizeF() + 0.5)

        toolbar.setFont(action_font)

        self.addToolBar(toolbar)

        toolbar.addAction(self._open_action)

        toolbar.addSeparator()

        toolbar.addAction(self._zoom_out_action)

        self._zoom_label = QLabel("—")

        self._zoom_label.setFont(action_font)

        self._zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._zoom_label.setMinimumWidth(54)

        toolbar.addWidget(self._zoom_label)

        toolbar.addAction(self._zoom_in_action)

        toolbar.addAction(self._fit_action)

        toolbar.addSeparator()

        toolbar.addAction(self._mesh_action)

        toolbar.addSeparator()

        self._page_label = QLabel("Página:")

        self._page_label.setFont(base_font)

        toolbar.addWidget(self._page_label)

        self._previous_page_button = QToolButton(self)

        self._previous_page_button.setText("‹")

        self._previous_page_button.setToolTip("Página anterior")

        self._previous_page_button.clicked.connect(self._go_to_previous_page)

        toolbar.addWidget(self._previous_page_button)

        page_number_font = QFont(base_font)

        if page_number_font.pointSizeF() > 0:
            page_number_font.setPointSizeF(page_number_font.pointSizeF() + 1.5)

        self._page_number_edit = QLineEdit(self)

        self._page_number_edit.setFont(page_number_font)

        self._page_number_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._page_number_edit.setFixedWidth(42)

        self._page_number_edit.setMinimumHeight(28)

        self._page_number_edit.setTextMargins(
            4,
            0,
            4,
            0,
        )

        self._page_number_validator = QIntValidator(
            1,
            1,
            self,
        )

        self._page_number_edit.setValidator(self._page_number_validator)

        self._page_number_edit.returnPressed.connect(self._commit_page_number)

        self._page_number_edit.editingFinished.connect(self._commit_page_number)

        toolbar.addWidget(self._page_number_edit)

        self._page_total_label = QLabel("/ 1")

        self._page_total_label.setFont(base_font)

        toolbar.addWidget(self._page_total_label)

        self._next_page_button = QToolButton(self)

        self._next_page_button.setText("›")

        self._next_page_button.setToolTip("Página siguiente")

        self._next_page_button.clicked.connect(self._go_to_next_page)

        toolbar.addWidget(self._next_page_button)

    def _set_document_actions_enabled(
        self,
        enabled: bool,
    ) -> None:
        self._zoom_out_action.setEnabled(enabled)

        self._zoom_in_action.setEnabled(enabled)

        self._fit_action.setEnabled(enabled)

        self._mesh_action.setEnabled(enabled)

        if not enabled:
            self._zoom_label.setText("—")

    def _update_zoom_label(
        self,
        percentage: int,
    ) -> None:
        self._zoom_label.setText(f"{percentage}%")

    def _open_document_dialog(
        self,
    ) -> None:
        patterns = " ".join(
            f"*{extension}" for extension in sorted(SUPPORTED_EXTENSIONS)
        )

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Abrir documento",
            "",
            ("Documentos compatibles " f"({patterns})"),
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

        self._page_meshes.clear()

        self._mesh_action.setChecked(True)

        if previous_document is not None:
            previous_document.close()

        self._configure_page_selector()

        self._render_current_page()

        self._set_document_actions_enabled(True)

        self.setWindowTitle(f"DewarpLab — " f"{document.path.name}")

    def _configure_page_selector(
        self,
    ) -> None:
        if self._document is None:
            self._page_number_validator.setRange(
                1,
                1,
            )

            self._update_page_controls()

            return

        self._page_number_validator.setRange(
            1,
            self._document.page_count,
        )

        self._update_page_controls()

    def _update_page_controls(
        self,
    ) -> None:
        if self._document is None:
            current_page = 1
            page_count = 1
            has_document = False
        else:
            current_page = self._current_page_index + 1

            page_count = self._document.page_count

            has_document = True

        self._page_number_edit.setText(str(current_page))

        self._page_total_label.setText(f"/ {page_count}")

        self._page_number_edit.setEnabled(has_document and page_count > 1)

        self._previous_page_button.setEnabled(has_document and current_page > 1)

        self._next_page_button.setEnabled(has_document and current_page < page_count)

    def _go_to_previous_page(
        self,
    ) -> None:
        if self._document is None:
            return

        if self._current_page_index <= 0:
            return

        self._current_page_index -= 1

        self._render_current_page()

        self._update_page_controls()

    def _go_to_next_page(
        self,
    ) -> None:
        if self._document is None:
            return

        if self._current_page_index >= self._document.page_count - 1:
            return

        self._current_page_index += 1

        self._render_current_page()

        self._update_page_controls()

    def _commit_page_number(
        self,
    ) -> None:
        if self._document is None:
            return

        text = self._page_number_edit.text().strip()

        if not text:
            self._update_page_controls()

            return

        page_number = int(text)

        page_number = max(
            1,
            min(
                self._document.page_count,
                page_number,
            ),
        )

        page_index = page_number - 1

        if page_index == self._current_page_index:
            self._update_page_controls()

            return

        self._current_page_index = page_index

        self._render_current_page()

        self._update_page_controls()

    def _mesh_for_current_page(
        self,
    ) -> Mesh:
        mesh = self._page_meshes.get(self._current_page_index)

        if mesh is None:
            mesh = Mesh.regular(
                rows=DEFAULT_MESH_ROWS,
                columns=DEFAULT_MESH_COLUMNS,
            )

            self._page_meshes[self._current_page_index] = mesh

        return mesh

    def _render_current_page(
        self,
    ) -> None:
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

        mesh = self._mesh_for_current_page()

        self._view.set_image(
            image=image,
            mesh=mesh,
        )

        self._view.set_mesh_visible(self._mesh_action.isChecked())

        self._update_page_controls()

        if self._document.is_pdf:
            page_text = (
                f"Página "
                f"{self._current_page_index + 1} "
                f"de "
                f"{self._document.page_count}"
            )
        else:
            page_text = "Imagen"

        self.statusBar().show()

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
