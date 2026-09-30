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


BOUNDARY_LINE_WIDTH = 3.2
BOUNDARY_HALO_WIDTH = 5.5

BOUNDARY_HALO_COLOR = QColor(
    0,
    0,
    0,
    150,
)


class PageBoundaryOverlay:
    def __init__(
        self,
        scene: QGraphicsScene,
        document_rect: QRectF,
        geometry: PageBoundaryGeometry,
        color: QColor,
    ):
        self._scene = scene
        self._document_rect = QRectF(
            document_rect
        )
        self._geometry = geometry
        self._color = QColor(color)

        self._halo_item = (
            QGraphicsPathItem()
        )
        self._path_item = (
            QGraphicsPathItem()
        )

        self._create_path()

    def set_visible(
        self,
        visible: bool,
    ) -> None:
        self._halo_item.setVisible(
            visible
        )
        self._path_item.setVisible(
            visible
        )

    def remove(
        self,
    ) -> None:
        if (
            self._halo_item.scene()
            is not None
        ):
            self._scene.removeItem(
                self._halo_item
            )

        if (
            self._path_item.scene()
            is not None
        ):
            self._scene.removeItem(
                self._path_item
            )

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

        for (
            point
        ) in self._geometry.points[1:]:
            path.lineTo(
                QPointF(
                    self._scene_x(
                        point.x
                    ),
                    self._scene_y(
                        point.y
                    ),
                )
            )

        path.closeSubpath()

        halo_pen = QPen(
            BOUNDARY_HALO_COLOR,
            BOUNDARY_HALO_WIDTH,
        )
        halo_pen.setCosmetic(True)
        halo_pen.setCapStyle(
            Qt.PenCapStyle.RoundCap
        )
        halo_pen.setJoinStyle(
            Qt.PenJoinStyle.RoundJoin
        )

        self._halo_item.setPath(path)
        self._halo_item.setPen(halo_pen)
        self._halo_item.setBrush(
            Qt.BrushStyle.NoBrush
        )
        self._halo_item.setZValue(12)
        self._halo_item.setAcceptedMouseButtons(
            Qt.MouseButton.NoButton
        )

        self._scene.addItem(
            self._halo_item
        )

        boundary_pen = QPen(
            self._color,
            BOUNDARY_LINE_WIDTH,
        )
        boundary_pen.setCosmetic(True)
        boundary_pen.setCapStyle(
            Qt.PenCapStyle.RoundCap
        )
        boundary_pen.setJoinStyle(
            Qt.PenJoinStyle.RoundJoin
        )

        self._path_item.setPath(path)
        self._path_item.setPen(
            boundary_pen
        )
        self._path_item.setBrush(
            Qt.BrushStyle.NoBrush
        )
        self._path_item.setZValue(13)
        self._path_item.setAcceptedMouseButtons(
            Qt.MouseButton.NoButton
        )

        self._scene.addItem(
            self._path_item
        )

    def _scene_x(
        self,
        normalized_x: float,
    ) -> float:
        return (
            self._document_rect.left()
            + normalized_x
            * self._document_rect.width()
        )

    def _scene_y(
        self,
        normalized_y: float,
    ) -> float:
        return (
            self._document_rect.top()
            + normalized_y
            * self._document_rect.height()
        )
