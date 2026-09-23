from pathlib import Path

from PySide6.QtCore import (
    QSettings,
    QStandardPaths,
    Qt,
)
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QFontMetrics,
    QImage,
    QIntValidator,
    QKeySequence,
)
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QToolBar,
    QToolButton,
)

from dewarplab.adapters.detection import (
    StructureAnalysisError,
    TextLineDetectionError,
    analyze_document_structure,
    detect_text_line_geometry,
)
from dewarplab.adapters.documents.document_loader import (
    DocumentLoadError,
    LoadedDocument,
    SUPPORTED_EXTENSIONS,
    is_supported_document,
    load_document,
    render_page,
)
from dewarplab.adapters.imaging import (
    MeshWarpError,
    QtImageBridgeError,
    qimage_to_rgba_array,
    warp_qimage_with_mesh,
)
from dewarplab.application import (
    StructureAnalysis,
    TextLineGeometry,
)
from dewarplab.domain import Mesh
from dewarplab.ui.document_view import DocumentView
from dewarplab.ui.mesh_controls import MeshControls


DEFAULT_MESH_ROWS = 8
DEFAULT_MESH_COLUMNS = 8


class PageNavigationField(QLineEdit):
    HORIZONTAL_PADDING = 8
    VERTICAL_PADDING = 2
    SEPARATOR_GAP = 6
    FRAME_EXTRA = 2
    MINIMUM_HEIGHT = 28

    def __init__(
        self,
        parent=None,
    ):
        super().__init__(parent)

        self._total_pages = 1

        self._slash_label = QLabel(
            "/",
            self,
        )

        self._total_label = QLabel(
            "1",
            self,
        )

        for label in (
            self._slash_label,
            self._total_label,
        ):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            label.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents,
                True,
            )

            label.setStyleSheet(
                "background: transparent;" "border: none;" "padding: 0;"
            )

        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._update_field_layout()

    def set_total_pages(
        self,
        total_pages: int,
    ) -> None:
        self._total_pages = max(
            1,
            total_pages,
        )

        self._slash_label.setFont(self.font())

        self._total_label.setFont(self.font())

        self._total_label.setText(str(self._total_pages))

        self._update_field_layout()

    def resizeEvent(
        self,
        event,
    ) -> None:
        super().resizeEvent(event)

        self._position_labels()

    def _update_field_layout(
        self,
    ) -> None:
        font_metrics = QFontMetrics(self.font())

        digit_count = len(str(self._total_pages))

        current_page_width = font_metrics.horizontalAdvance("8" * digit_count)

        slash_width = font_metrics.horizontalAdvance("/")

        total_width = font_metrics.horizontalAdvance(str(self._total_pages))

        text_height = font_metrics.height()

        field_width = (
            self.HORIZONTAL_PADDING
            + current_page_width
            + self.SEPARATOR_GAP
            + slash_width
            + self.SEPARATOR_GAP
            + total_width
            + self.HORIZONTAL_PADDING
            + self.FRAME_EXTRA
        )

        field_height = max(
            self.MINIMUM_HEIGHT,
            (text_height + self.VERTICAL_PADDING * 2 + self.FRAME_EXTRA),
        )

        right_margin = (
            self.SEPARATOR_GAP
            + slash_width
            + self.SEPARATOR_GAP
            + total_width
            + self.HORIZONTAL_PADDING
        )

        self.setTextMargins(
            self.HORIZONTAL_PADDING,
            self.VERTICAL_PADDING,
            right_margin,
            self.VERTICAL_PADDING,
        )

        self.setFixedSize(
            field_width,
            field_height,
        )

        self._slash_label.adjustSize()
        self._total_label.adjustSize()

        self._position_labels()

    def _position_labels(
        self,
    ) -> None:
        total_width = self._total_label.sizeHint().width()

        total_height = self._total_label.sizeHint().height()

        slash_width = self._slash_label.sizeHint().width()

        slash_height = self._slash_label.sizeHint().height()

        total_x = self.width() - self.HORIZONTAL_PADDING - total_width

        slash_x = total_x - self.SEPARATOR_GAP - slash_width

        total_y = (self.height() - total_height) // 2

        slash_y = (self.height() - slash_height) // 2

        self._slash_label.setGeometry(
            slash_x,
            slash_y,
            slash_width,
            slash_height,
        )

        self._total_label.setGeometry(
            total_x,
            total_y,
            total_width,
            total_height,
        )


class MainWindow(QMainWindow):
    SETTINGS_LAST_DOCUMENT_DIRECTORY = "documents/last_directory"

    def __init__(self):
        super().__init__()

        self._document: LoadedDocument | None = None
        self._current_page_index = 0
        self._current_original_image: QImage | None = None

        self._page_meshes: dict[
            int,
            Mesh,
        ] = {}

        self._page_density_modes: dict[
            int,
            str,
        ] = {}

        self._page_structure_analyses: dict[
            int,
            StructureAnalysis,
        ] = {}

        self._page_text_line_geometries: dict[
            int,
            TextLineGeometry,
        ] = {}

        self._page_detection_visibility: dict[
            int,
            bool,
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
        self._create_mesh_panel()

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

        self._original_view_action = QAction(
            "Original",
            self,
        )

        self._original_view_action.setCheckable(True)

        self._corrected_view_action = QAction(
            "Corregida",
            self,
        )

        self._corrected_view_action.setCheckable(True)

        self._view_mode_group = QActionGroup(self)

        self._view_mode_group.setExclusive(True)

        self._view_mode_group.addAction(self._original_view_action)

        self._view_mode_group.addAction(self._corrected_view_action)

        self._original_view_action.setChecked(True)

        self._original_view_action.triggered.connect(self._show_original_preview)

        self._corrected_view_action.triggered.connect(self._show_corrected_preview)

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

        view_label = QLabel("Vista:")

        view_label.setFont(base_font)

        toolbar.addWidget(view_label)

        toolbar.addAction(self._original_view_action)

        toolbar.addAction(self._corrected_view_action)

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

        self._page_number_edit = PageNavigationField(self)

        self._page_number_edit.setFont(page_number_font)

        self._page_number_edit.set_total_pages(1)

        self._page_number_validator = QIntValidator(
            1,
            1,
            self,
        )

        self._page_number_edit.setValidator(self._page_number_validator)

        self._page_number_edit.returnPressed.connect(self._commit_page_number)

        self._page_number_edit.editingFinished.connect(self._commit_page_number)

        toolbar.addWidget(self._page_number_edit)

        self._next_page_button = QToolButton(self)

        self._next_page_button.setText("›")

        self._next_page_button.setToolTip("Página siguiente")

        self._next_page_button.clicked.connect(self._go_to_next_page)

        toolbar.addWidget(self._next_page_button)

    def _create_mesh_panel(self) -> None:
        self._mesh_controls = MeshControls(self)

        self._mesh_controls.setMinimumWidth(260)

        self._mesh_controls.setMaximumWidth(320)

        self._mesh_controls.density_requested.connect(self._change_mesh_density)

        self._mesh_controls.density_mode_changed.connect(self._change_density_mode)

        self._mesh_controls.visibility_changed.connect(self._view.set_mesh_visible)

        self._mesh_controls.detection_visibility_changed.connect(
            self._change_detection_visibility
        )

        self._mesh_controls.reset_requested.connect(self._reset_current_mesh)

        self._mesh_dock = QDockWidget(
            "Malla",
            self,
        )

        self._mesh_dock.setObjectName("meshDock")

        self._mesh_dock.setMinimumWidth(280)

        self._mesh_dock.setMaximumWidth(340)

        self._mesh_dock.setWidget(self._mesh_controls)

        self._mesh_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )

        self._mesh_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self._mesh_dock,
        )

        self.resizeDocks(
            [
                self._mesh_dock,
            ],
            [
                300,
            ],
            Qt.Orientation.Horizontal,
        )

        self._mesh_dock.hide()

    def _set_document_actions_enabled(
        self,
        enabled: bool,
    ) -> None:
        self._zoom_out_action.setEnabled(enabled)

        self._zoom_in_action.setEnabled(enabled)

        self._fit_action.setEnabled(enabled)

        self._original_view_action.setEnabled(enabled)

        self._corrected_view_action.setEnabled(enabled)

        self._mesh_controls.set_controls_enabled(enabled)

        if not enabled:
            self._zoom_label.setText("—")

    def _update_zoom_label(
        self,
        percentage: int,
    ) -> None:
        self._zoom_label.setText(f"{percentage}%")

    def _initial_document_directory(
        self,
    ) -> str:
        settings = QSettings()

        saved_directory = settings.value(
            self.SETTINGS_LAST_DOCUMENT_DIRECTORY,
            "",
            type=str,
        )

        if saved_directory:
            directory = Path(saved_directory)

            if directory.is_dir():
                return str(directory)

        documents_locations = QStandardPaths.standardLocations(
            QStandardPaths.StandardLocation.DocumentsLocation
        )

        if documents_locations:
            return documents_locations[0]

        return str(Path.home())

    def _remember_document_directory(
        self,
        document_path: Path,
    ) -> None:
        settings = QSettings()

        settings.setValue(
            self.SETTINGS_LAST_DOCUMENT_DIRECTORY,
            str(document_path.parent),
        )

    def _open_document_dialog(
        self,
    ) -> None:
        patterns = " ".join(
            f"*{extension}" for extension in sorted(SUPPORTED_EXTENSIONS)
        )

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Abrir documento",
            self._initial_document_directory(),
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

        self._remember_document_directory(document.path)

        previous_document = self._document

        self._document = document
        self._current_page_index = 0
        self._current_original_image = None

        self._page_meshes.clear()
        self._page_density_modes.clear()
        self._page_structure_analyses.clear()
        self._page_text_line_geometries.clear()
        self._page_detection_visibility.clear()

        self._original_view_action.setChecked(True)

        if previous_document is not None:
            previous_document.close()

        self._configure_page_selector()

        self._set_document_actions_enabled(True)

        self._render_current_page()

        self._mesh_dock.show()

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

        self._page_number_edit.set_total_pages(page_count)

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

    def _go_to_next_page(
        self,
    ) -> None:
        if self._document is None:
            return

        if self._current_page_index >= self._document.page_count - 1:
            return

        self._current_page_index += 1

        self._render_current_page()

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

            self._page_density_modes[self._current_page_index] = (
                MeshControls.MODE_CUSTOM
            )

        return mesh

    def _density_mode_for_current_page(
        self,
    ) -> str:
        return self._page_density_modes.get(
            self._current_page_index,
            MeshControls.MODE_CUSTOM,
        )

    def _analysis_for_current_page(
        self,
    ) -> StructureAnalysis | None:
        return self._page_structure_analyses.get(self._current_page_index)

    def _text_geometry_for_current_page(
        self,
    ) -> TextLineGeometry | None:
        return self._page_text_line_geometries.get(self._current_page_index)

    def _detection_visible_for_current_page(
        self,
    ) -> bool:
        return self._page_detection_visibility.get(
            self._current_page_index,
            False,
        )

    def _change_density_mode(
        self,
        mode: str,
    ) -> None:
        if self._document is None:
            return

        if mode not in (
            MeshControls.MODE_AUTOMATIC,
            MeshControls.MODE_CUSTOM,
        ):
            return

        self._page_density_modes[self._current_page_index] = mode

        if mode == MeshControls.MODE_AUTOMATIC:
            self._analyze_current_page()

            return

        self._mesh_controls.set_analysis_summary(None)

    def _change_detection_visibility(
        self,
        visible: bool,
    ) -> None:
        if self._document is None:
            return

        self._page_detection_visibility[self._current_page_index] = visible

        self._view.set_text_line_geometry_visible(visible)

    def _analyze_current_page(
        self,
    ) -> None:
        if self._document is None or self._current_original_image is None:
            return

        self._ensure_original_view()

        self._mesh_controls.set_analysis_summary("Analizando…")

        QApplication.processEvents()

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            rgba = qimage_to_rgba_array(self._current_original_image)

            analysis = analyze_document_structure(rgba)

            geometry = detect_text_line_geometry(rgba)

            current_mesh = self._mesh_for_current_page()

            new_mesh = current_mesh.resampled(
                rows=analysis.suggested_rows,
                columns=analysis.suggested_columns,
            )

        except (
            QtImageBridgeError,
            StructureAnalysisError,
            TextLineDetectionError,
            ValueError,
        ) as error:
            self._mesh_controls.set_analysis_summary(None)

            QMessageBox.critical(
                self,
                "No se pudo analizar el documento",
                str(error),
            )

            return

        finally:
            QApplication.restoreOverrideCursor()

        self._page_structure_analyses[self._current_page_index] = analysis

        self._page_text_line_geometries[self._current_page_index] = geometry

        self._page_density_modes[self._current_page_index] = MeshControls.MODE_AUTOMATIC

        self._page_meshes[self._current_page_index] = new_mesh

        self._page_detection_visibility[self._current_page_index] = True

        self._mesh_controls.set_density_mode(MeshControls.MODE_AUTOMATIC)

        self._mesh_controls.set_mesh_shape(
            rows=new_mesh.rows,
            columns=new_mesh.columns,
        )

        self._mesh_controls.set_analysis_summary(
            self._analysis_summary(
                analysis,
                geometry,
            )
        )

        self._mesh_controls.set_detection_available(True)

        self._mesh_controls.set_detection_visible(True)

        self._view.set_mesh(new_mesh)

        self._view.set_mesh_visible(self._mesh_controls.is_mesh_visible())

        self._view.set_text_line_geometry(geometry)

        self._view.set_text_line_geometry_visible(True)

        self.statusBar().showMessage(
            (
                self._status_message(view_name="Original")
                + " — "
                + (f"Malla automática: " f"{new_mesh.rows} × " f"{new_mesh.columns}")
                + " — "
                + (f"Trazas detectadas: " f"{geometry.trace_count}")
            )
        )

    def _analysis_summary(
        self,
        analysis: StructureAnalysis,
        geometry: TextLineGeometry | None,
    ) -> str:
        summary = (
            "Malla sugerida: "
            f"{analysis.suggested_rows} × "
            f"{analysis.suggested_columns}"
        )

        if geometry is not None:
            summary += "\n" "Trazas detectadas: " f"{geometry.trace_count}"

        return summary

    def _change_mesh_density(
        self,
        rows: int,
        columns: int,
    ) -> None:
        if self._document is None:
            return

        self._ensure_original_view()

        current_mesh = self._mesh_for_current_page()

        if current_mesh.rows == rows and current_mesh.columns == columns:
            return

        try:
            new_mesh = current_mesh.resampled(
                rows=rows,
                columns=columns,
            )

        except ValueError:
            self._mesh_controls.set_mesh_shape(
                rows=current_mesh.rows,
                columns=current_mesh.columns,
            )

            QMessageBox.warning(
                self,
                "No se puede cambiar la densidad",
                (
                    "La malla resultante tendría "
                    "celdas cruzadas o invertidas.\n\n"
                    "Prueba con una densidad mayor "
                    "o restablece primero la malla."
                ),
            )

            return

        self._page_density_modes[self._current_page_index] = MeshControls.MODE_CUSTOM

        self._page_structure_analyses.pop(
            self._current_page_index,
            None,
        )

        self._page_meshes[self._current_page_index] = new_mesh

        self._mesh_controls.set_density_mode(MeshControls.MODE_CUSTOM)

        self._mesh_controls.set_analysis_summary(None)

        self._view.set_mesh(new_mesh)

        self._view.set_mesh_visible(self._mesh_controls.is_mesh_visible())

        self._mesh_controls.set_mesh_shape(
            rows=new_mesh.rows,
            columns=new_mesh.columns,
        )

        geometry = self._text_geometry_for_current_page()

        self._mesh_controls.set_detection_available(geometry is not None)

        if geometry is not None:
            self._view.set_text_line_geometry(geometry)

            self._view.set_text_line_geometry_visible(
                self._detection_visible_for_current_page()
            )

    def _reset_current_mesh(
        self,
    ) -> None:
        if self._document is None:
            return

        self._ensure_original_view()

        current_mesh = self._mesh_for_current_page()

        if current_mesh.is_regular():
            return

        message_box = QMessageBox(self)

        message_box.setIcon(QMessageBox.Icon.Question)

        message_box.setWindowTitle("Restablecer malla")

        message_box.setText("¿Quieres restablecer la malla " "de esta página?")

        message_box.setInformativeText(
            "Se conservará la densidad actual, "
            "pero se perderán los ajustes "
            "manuales de los nodos."
        )

        reset_button = message_box.addButton(
            "Restablecer",
            QMessageBox.ButtonRole.AcceptRole,
        )

        cancel_button = message_box.addButton(
            "Cancelar",
            QMessageBox.ButtonRole.RejectRole,
        )

        message_box.setDefaultButton(cancel_button)

        message_box.exec()

        if message_box.clickedButton() is not reset_button:
            return

        new_mesh = Mesh.regular(
            rows=current_mesh.rows,
            columns=current_mesh.columns,
        )

        self._page_meshes[self._current_page_index] = new_mesh

        self._view.set_mesh(new_mesh)

        self._view.set_mesh_visible(self._mesh_controls.is_mesh_visible())

        geometry = self._text_geometry_for_current_page()

        if geometry is not None:
            self._view.set_text_line_geometry(geometry)

            self._view.set_text_line_geometry_visible(
                self._detection_visible_for_current_page()
            )

    def _ensure_original_view(
        self,
    ) -> None:
        if self._original_view_action.isChecked():
            return

        self._original_view_action.setChecked(True)

        self._show_original_preview()

    def _show_original_preview(
        self,
    ) -> None:
        if self._document is None or self._current_original_image is None:
            return

        mesh = self._mesh_for_current_page()

        geometry = self._text_geometry_for_current_page()

        self._view.replace_image(
            image=self._current_original_image,
            mesh=mesh,
        )

        self._view.set_mesh_visible(self._mesh_controls.is_mesh_visible())

        if geometry is not None:
            self._view.set_text_line_geometry(geometry)

            self._view.set_text_line_geometry_visible(
                self._detection_visible_for_current_page()
            )

        self._mesh_controls.set_controls_enabled(True)

        self._mesh_controls.set_detection_available(geometry is not None)

        self._mesh_controls.set_detection_visible(
            geometry is not None and self._detection_visible_for_current_page()
        )

        self.statusBar().showMessage(self._status_message(view_name="Original"))

    def _show_corrected_preview(
        self,
    ) -> None:
        if self._document is None or self._current_original_image is None:
            return

        mesh = self._mesh_for_current_page()

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            corrected = warp_qimage_with_mesh(
                self._current_original_image,
                mesh,
            )

        except (
            MeshWarpError,
            QtImageBridgeError,
        ) as error:
            self._original_view_action.setChecked(True)

            QMessageBox.critical(
                self,
                "No se pudo generar la corrección",
                str(error),
            )

            return

        finally:
            QApplication.restoreOverrideCursor()

        self._view.replace_image(
            image=corrected,
            mesh=None,
        )

        self._mesh_controls.set_controls_enabled(False)

        self.statusBar().showMessage(self._status_message(view_name="Corregida"))

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

        self._current_original_image = image

        mesh = self._mesh_for_current_page()

        density_mode = self._density_mode_for_current_page()

        analysis = self._analysis_for_current_page()

        geometry = self._text_geometry_for_current_page()

        detection_visible = self._detection_visible_for_current_page()

        self._mesh_controls.set_mesh_shape(
            rows=mesh.rows,
            columns=mesh.columns,
        )

        self._mesh_controls.set_density_mode(density_mode)

        if analysis is None:
            self._mesh_controls.set_analysis_summary(None)
        else:
            self._mesh_controls.set_analysis_summary(
                self._analysis_summary(
                    analysis,
                    geometry,
                )
            )

        self._mesh_controls.set_detection_available(geometry is not None)

        self._mesh_controls.set_detection_visible(
            geometry is not None and detection_visible
        )

        self._view.set_image(
            image=image,
            mesh=mesh,
        )

        self._view.set_mesh_visible(self._mesh_controls.is_mesh_visible())

        if geometry is not None:
            self._view.set_text_line_geometry(geometry)

            self._view.set_text_line_geometry_visible(detection_visible)

        self._mesh_controls.set_controls_enabled(True)

        self._update_page_controls()

        if self._corrected_view_action.isChecked():
            self._show_corrected_preview()
        else:
            self.statusBar().show()

            self.statusBar().showMessage(self._status_message(view_name="Original"))

    def _status_message(
        self,
        view_name: str,
    ) -> str:
        if self._document is None or self._current_original_image is None:
            return ""

        if self._document.is_pdf:
            page_text = (
                f"Página "
                f"{self._current_page_index + 1} "
                f"de "
                f"{self._document.page_count}"
            )
        else:
            page_text = "Imagen"

        image = self._current_original_image

        return (
            f"{self._document.path.name} — "
            f"{page_text} — "
            f"{image.width()} × "
            f"{image.height()} px — "
            f"Vista {view_name}"
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
