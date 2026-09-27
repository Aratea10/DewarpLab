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
    PageBoundaryGeometry,
)


BOUNDARY_LINE_WIDTH = 2.4


class PageBoundaryOverlay:
    def __init__(
        self,
        scene: QGraphicsScene,
        document_rect: QRectF,
        geometry: PageBoundaryGeometry,
        color: QColor,
    ):
        self._scene = scene

        self._document_rect = QRectF(document_rect)

        self._geometry = geometry

        self._color = QColor(color)

        self._path_item = QGraphicsPathItem()

        self._create_path()

    def set_visible(
        self,
        visible: bool,
    ) -> None:
        self._path_item.setVisible(visible)

    def remove(
        self,
    ) -> None:
        if self._path_item.scene() is not None:
            self._scene.removeItem(self._path_item)

    def _create_path(
        self,
    ) -> None:
        if not self._geometry.points:
            return

        first = self._geometry.points[0]

        path = QPainterPath()

        path.moveTo(
            QPointF(
                self._scene_x(first.x),
                self._scene_y(first.y),
            )
        )

        for point in self._geometry.points[1:]:
            path.lineTo(
                QPointF(
                    self._scene_x(point.x),
                    self._scene_y(point.y),
                )
            )

        path.closeSubpath()

        pen = QPen(
            self._color,
            BOUNDARY_LINE_WIDTH,
        )

        pen.setCosmetic(True)

        pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

        self._path_item.setPath(path)

        self._path_item.setPen(pen)

        self._path_item.setBrush(Qt.BrushStyle.NoBrush)

        self._path_item.setZValue(13)

        self._path_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

        self._scene.addItem(self._path_item)

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
