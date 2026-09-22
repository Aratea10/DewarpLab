from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QButtonGroup,
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class MeshControls(QWidget):
    density_requested = Signal(
        int,
        int,
    )

    visibility_changed = Signal(bool)

    reset_requested = Signal()

    density_mode_changed = Signal(str)

    analysis_requested = Signal()

    MODE_AUTOMATIC = "automatic"
    MODE_CUSTOM = "custom"

    MIN_DENSITY = 2
    MAX_DENSITY = 30

    def __init__(
        self,
        parent=None,
    ):
        super().__init__(parent)

        self._analysis_summary: str | None = None

        self._automatic_radio = QRadioButton(
            "Automática",
            self,
        )

        self._custom_radio = QRadioButton(
            "Personalizada",
            self,
        )

        self._mode_group = QButtonGroup(self)

        self._mode_group.addButton(self._automatic_radio)

        self._mode_group.addButton(self._custom_radio)

        self._custom_radio.setChecked(True)

        self._automatic_radio.clicked.connect(self._automatic_mode_clicked)

        self._custom_radio.clicked.connect(self._custom_mode_clicked)

        mode_layout = QVBoxLayout()

        mode_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        mode_layout.setSpacing(6)

        mode_layout.addWidget(self._automatic_radio)

        mode_layout.addWidget(self._custom_radio)

        self._analysis_button = QPushButton(
            "Analizar documento",
            self,
        )

        self._analysis_button.clicked.connect(self.analysis_requested)

        self._analysis_result_label = QLabel(
            self,
        )

        self._analysis_result_label.setWordWrap(True)

        self._analysis_result_label.hide()

        self._rows_spinbox = self._create_density_spinbox(value=8)

        rows_control = self._create_stepper_control(
            spinbox=self._rows_spinbox,
            decrease_tooltip=("Reducir número de filas"),
            increase_tooltip=("Aumentar número de filas"),
        )

        self._columns_spinbox = self._create_density_spinbox(value=8)

        columns_control = self._create_stepper_control(
            spinbox=self._columns_spinbox,
            decrease_tooltip=("Reducir número de columnas"),
            increase_tooltip=("Aumentar número de columnas"),
        )

        form_layout = QFormLayout()

        form_layout.setHorizontalSpacing(12)

        form_layout.setVerticalSpacing(10)

        form_layout.addRow(
            "Filas:",
            rows_control,
        )

        form_layout.addRow(
            "Columnas:",
            columns_control,
        )

        self._apply_button = QPushButton(
            "Actualizar malla",
            self,
        )

        self._apply_button.clicked.connect(self._request_density_change)

        self._manual_controls = QWidget(self)

        manual_layout = QVBoxLayout(self._manual_controls)

        manual_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        manual_layout.setSpacing(10)

        manual_layout.addLayout(form_layout)

        manual_layout.addWidget(self._apply_button)

        self._reset_button = QPushButton(
            "Restablecer malla",
            self,
        )

        self._reset_button.clicked.connect(self.reset_requested)

        self._visibility_checkbox = QCheckBox(
            "Mostrar malla",
            self,
        )

        self._visibility_checkbox.setChecked(True)

        self._visibility_checkbox.toggled.connect(self.visibility_changed)

        description = QLabel(
            "En modo automático, DewarpLab analiza "
            "la estructura de la página y propone "
            "una densidad de malla.\n\n"
            "En modo personalizado puedes elegir "
            "manualmente el número de filas y columnas.",
            self,
        )

        description.setWordWrap(True)

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            16,
            16,
            16,
            16,
        )

        layout.setSpacing(12)

        density_label = QLabel(
            "Densidad",
            self,
        )

        layout.addWidget(density_label)

        layout.addLayout(mode_layout)

        layout.addWidget(self._analysis_button)

        layout.addWidget(self._analysis_result_label)

        layout.addSpacing(6)

        layout.addWidget(self._manual_controls)

        layout.addSpacing(4)

        layout.addWidget(self._reset_button)

        layout.addWidget(self._visibility_checkbox)

        layout.addSpacing(4)

        layout.addWidget(description)

        layout.addStretch()

        self._update_mode_controls()

    def _create_density_spinbox(
        self,
        value: int,
    ) -> QSpinBox:
        spinbox = QSpinBox(self)

        spinbox.setRange(
            self.MIN_DENSITY,
            self.MAX_DENSITY,
        )

        spinbox.setValue(value)

        spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        spinbox.setFixedWidth(40)

        spinbox.setMinimumHeight(28)

        spinbox.lineEdit().setTextMargins(
            6,
            0,
            6,
            0,
        )

        return spinbox

    def _create_stepper_control(
        self,
        spinbox: QSpinBox,
        decrease_tooltip: str,
        increase_tooltip: str,
    ) -> QWidget:
        container = QWidget(self)

        decrease_button = QToolButton(container)

        decrease_button.setText("−")

        decrease_button.setToolTip(decrease_tooltip)

        decrease_button.setFixedSize(
            28,
            28,
        )

        decrease_button.clicked.connect(spinbox.stepDown)

        increase_button = QToolButton(container)

        increase_button.setText("+")

        increase_button.setToolTip(increase_tooltip)

        increase_button.setFixedSize(
            28,
            28,
        )

        increase_button.clicked.connect(spinbox.stepUp)

        layout = QHBoxLayout(container)

        layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        layout.setSpacing(6)

        layout.addWidget(decrease_button)

        layout.addWidget(spinbox)

        layout.addWidget(increase_button)

        return container

    def set_mesh_shape(
        self,
        rows: int,
        columns: int,
    ) -> None:
        self._rows_spinbox.setValue(rows)

        self._columns_spinbox.setValue(columns)

    def set_density_mode(
        self,
        mode: str,
    ) -> None:
        if mode == self.MODE_AUTOMATIC:
            self._automatic_radio.setChecked(True)

        elif mode == self.MODE_CUSTOM:
            self._custom_radio.setChecked(True)

        else:
            raise ValueError(f"Unknown density mode: {mode}")

        self._update_mode_controls()

    def set_analysis_summary(
        self,
        summary: str | None,
    ) -> None:
        self._analysis_summary = summary

        if summary is None:
            self._analysis_result_label.clear()
        else:
            self._analysis_result_label.setText(summary)

        self._update_analysis_summary_visibility()

    def set_controls_enabled(
        self,
        enabled: bool,
    ) -> None:
        self.setEnabled(enabled)

        if enabled:
            self._update_mode_controls()

    def is_mesh_visible(
        self,
    ) -> bool:
        return self._visibility_checkbox.isChecked()

    def density_mode(
        self,
    ) -> str:
        if self._automatic_radio.isChecked():
            return self.MODE_AUTOMATIC

        return self.MODE_CUSTOM

    def _automatic_mode_clicked(
        self,
        checked: bool = False,
    ) -> None:
        del checked

        self._update_mode_controls()

        self.density_mode_changed.emit(self.MODE_AUTOMATIC)

    def _custom_mode_clicked(
        self,
        checked: bool = False,
    ) -> None:
        del checked

        self._update_mode_controls()

        self.density_mode_changed.emit(self.MODE_CUSTOM)

    def _update_mode_controls(
        self,
    ) -> None:
        automatic = self._automatic_radio.isChecked()

        self._analysis_button.setEnabled(automatic)

        self._manual_controls.setEnabled(not automatic)

        self._update_analysis_summary_visibility()

    def _update_analysis_summary_visibility(
        self,
    ) -> None:
        self._analysis_result_label.setVisible(
            self._automatic_radio.isChecked() and self._analysis_summary is not None
        )

    def _request_density_change(
        self,
    ) -> None:
        self.density_requested.emit(
            self._rows_spinbox.value(),
            self._columns_spinbox.value(),
        )
