from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QGraphicsPathItem,
    QGraphicsScene,
)

from dewarplab.application import TextLineGeometry


class TextLineOverlay:
    def __init__(
        self,
        scene: QGraphicsScene,
        document_rect: QRectF,
        geometry: TextLineGeometry,
        color: QColor,
    ):
        self._scene = scene
        self._document_rect = QRectF(document_rect)

        self._geometry = geometry

        self._path_item = QGraphicsPathItem()

        line_color = QColor(color)

        line_color.setAlpha(220)

        pen = QPen(
            line_color,
            1.5,
        )

        pen.setCosmetic(True)

        self._path_item.setPen(pen)

        self._path_item.setBrush(Qt.BrushStyle.NoBrush)

        self._path_item.setZValue(15)

        self._path_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

        self._scene.addItem(self._path_item)

        self._update_path()

    def set_visible(
        self,
        visible: bool,
    ) -> None:
        self._path_item.setVisible(visible)

    def remove(self) -> None:
        if self._path_item.scene() is not None:
            self._scene.removeItem(self._path_item)

    def _update_path(self) -> None:
        path = QPainterPath()

        for trace in self._geometry.traces:
            if not trace.points:
                continue

            first_point = trace.points[0]

            path.moveTo(
                self._scene_x(first_point.x),
                self._scene_y(first_point.y),
            )

            for point in trace.points[1:]:
                path.lineTo(
                    self._scene_x(point.x),
                    self._scene_y(point.y),
                )

        self._path_item.setPath(path)

    def _scene_x(
        self,
        normalized_x: float,
    ) -> float:
        return self._document_rect.left() + normalized_x * self._document_rect.width()

    def _scene_y(
        self,
        normalized_y: float,
    ) -> float:
        return self._document_rect.top() + normalized_y * self._document_rect.height()
