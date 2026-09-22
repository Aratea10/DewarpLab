from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
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

    MIN_DENSITY = 2
    MAX_DENSITY = 30

    def __init__(
        self,
        parent=None,
    ):
        super().__init__(parent)

        self._rows_spinbox = self._create_density_spinbox(value=8)

        rows_control = self._create_stepper_control(
            spinbox=self._rows_spinbox,
            decrease_tooltip=("Reducir número de filas"),
            increase_tooltip=("Aumentar número de filas"),
        )

        self._columns_spinbox = self._create_density_spinbox(value=8)

        columns_control = self._create_stepper_control(
            spinbox=(self._columns_spinbox),
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
            "Al cambiar la densidad se conserva "
            "la forma actual de la malla.\n\n"
            "Los nodos no pueden moverse de forma "
            "que una celda se cruce o se invierta.",
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

        layout.addLayout(form_layout)

        layout.addWidget(self._apply_button)

        layout.addWidget(self._reset_button)

        layout.addSpacing(4)

        layout.addWidget(self._visibility_checkbox)

        layout.addSpacing(4)

        layout.addWidget(description)

        layout.addStretch()

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

    def set_controls_enabled(
        self,
        enabled: bool,
    ) -> None:
        self.setEnabled(enabled)

    def is_mesh_visible(
        self,
    ) -> bool:
        return self._visibility_checkbox.isChecked()

    def _request_density_change(
        self,
    ) -> None:
        self.density_requested.emit(
            self._rows_spinbox.value(),
            self._columns_spinbox.value(),
        )
