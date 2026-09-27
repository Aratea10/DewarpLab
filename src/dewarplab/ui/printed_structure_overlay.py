from PySide6.QtCore import (
    QPointF,
    QRectF,
    Qt,
)
from PySide6.QtGui import (
    QColor,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QGraphicsPathItem,
    QGraphicsScene,
)

from dewarplab.application import (
    PrintedStructureGeometry,
    PrintedStructureOrientation,
)


STRUCTURE_LINE_WIDTH = 1.6
BOX_EDGE_LINE_WIDTH = 2.2


class PrintedStructureOverlay:
    def __init__(
        self,
        scene: QGraphicsScene,
        document_rect: QRectF,
        geometry: PrintedStructureGeometry,
        horizontal_color: QColor,
        vertical_color: QColor,
    ):
        self._scene = scene

        self._document_rect = QRectF(document_rect)

        self._geometry = geometry

        self._horizontal_color = QColor(horizontal_color)

        self._vertical_color = QColor(vertical_color)

        self._items: list[QGraphicsPathItem] = []

        self._visible = True

        self._create_items()

    def set_visible(
        self,
        visible: bool,
    ) -> None:
        self._visible = visible

        for item in self._items:
            item.setVisible(visible)

    def remove(
        self,
    ) -> None:
        for item in self._items:
            if item.scene() is not None:
                self._scene.removeItem(item)

        self._items.clear()

    def _create_items(
        self,
    ) -> None:
        for segment in self._geometry.segments:
            start = QPointF(
                self._scene_x(segment.start.x),
                self._scene_y(segment.start.y),
            )

            end = QPointF(
                self._scene_x(segment.end.x),
                self._scene_y(segment.end.y),
            )

            path = QPainterPath()

            path.moveTo(start)

            path.lineTo(end)

            item = QGraphicsPathItem(path)

            if segment.orientation == PrintedStructureOrientation.HORIZONTAL:
                color = QColor(self._horizontal_color)
            else:
                color = QColor(self._vertical_color)

            width = BOX_EDGE_LINE_WIDTH if segment.is_box_edge else STRUCTURE_LINE_WIDTH

            pen = QPen(
                color,
                width,
            )

            pen.setCosmetic(True)

            pen.setCapStyle(Qt.PenCapStyle.RoundCap)

            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

            item.setPen(pen)

            item.setBrush(Qt.BrushStyle.NoBrush)

            item.setZValue(14)

            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

            item.setVisible(self._visible)

            self._scene.addItem(item)

            self._items.append(item)

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
