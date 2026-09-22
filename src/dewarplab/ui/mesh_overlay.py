from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)

from dewarplab.domain import ControlPoint, Mesh


PointMovedCallback = Callable[
    [int, int, float, float],
    None,
]


class ControlPointItem(QGraphicsObject):
    NODE_RADIUS = 5.5
    HIT_MARGIN = 4.0

    def __init__(
        self,
        point: ControlPoint,
        document_rect: QRectF,
        color: QColor,
        on_moved: PointMovedCallback,
        parent: QGraphicsItem | None = None,
    ):
        super().__init__(parent)

        self._point = point
        self._document_rect = QRectF(document_rect)

        self._color = QColor(color)

        self._on_moved = on_moved
        self._syncing_position = False

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
            | QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
        )

        self.setCursor(Qt.CursorShape.OpenHandCursor)

        self.setZValue(20)

        self.setToolTip(f"Fila {point.row + 1}, " f"columna {point.column + 1}")

        self.sync_from_model()

    def boundingRect(self) -> QRectF:
        extent = self.NODE_RADIUS + self.HIT_MARGIN

        return QRectF(
            -extent,
            -extent,
            extent * 2,
            extent * 2,
        )

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        del option
        del widget

        color = QColor(self._color)

        if self.isSelected():
            color = color.lighter(135)

        fill_color = QColor(color)

        fill_color.setAlpha(220)

        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing,
            True,
        )

        painter.setPen(
            QPen(
                color,
                1.5,
            )
        )

        painter.setBrush(fill_color)

        radius = self.NODE_RADIUS

        painter.drawEllipse(
            QRectF(
                -radius,
                -radius,
                radius * 2,
                radius * 2,
            )
        )

    def sync_from_model(self) -> None:
        scene_position = QPointF(
            self._document_rect.left() + (self._point.x * self._document_rect.width()),
            self._document_rect.top() + (self._point.y * self._document_rect.height()),
        )

        self._syncing_position = True

        self.setPos(scene_position)

        self._syncing_position = False

    def itemChange(
        self,
        change: QGraphicsItem.GraphicsItemChange,
        value,
    ):
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemPositionChange
            and isinstance(value, QPointF)
        ):
            return self._clamp_position(value)

        if (
            change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged
            and not self._syncing_position
        ):
            self._update_model_position()

        return super().itemChange(
            change,
            value,
        )

    def mousePressEvent(
        self,
        event: QGraphicsSceneMouseEvent,
    ) -> None:
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

        super().mousePressEvent(event)

    def mouseReleaseEvent(
        self,
        event: QGraphicsSceneMouseEvent,
    ) -> None:
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        super().mouseReleaseEvent(event)

    def _clamp_position(
        self,
        position: QPointF,
    ) -> QPointF:
        x = min(
            max(
                position.x(),
                self._document_rect.left(),
            ),
            self._document_rect.right(),
        )

        y = min(
            max(
                position.y(),
                self._document_rect.top(),
            ),
            self._document_rect.bottom(),
        )

        return QPointF(
            x,
            y,
        )

    def _update_model_position(
        self,
    ) -> None:
        if self._document_rect.width() <= 0:
            return

        if self._document_rect.height() <= 0:
            return

        position = self.pos()

        normalized_x = (
            position.x() - self._document_rect.left()
        ) / self._document_rect.width()

        normalized_y = (
            position.y() - self._document_rect.top()
        ) / self._document_rect.height()

        self._on_moved(
            self._point.row,
            self._point.column,
            normalized_x,
            normalized_y,
        )


class MeshOverlay:
    def __init__(
        self,
        scene: QGraphicsScene,
        document_rect: QRectF,
        mesh: Mesh,
        color: QColor,
    ):
        self._scene = scene

        self._document_rect = QRectF(document_rect)

        self._mesh = mesh

        self._color = QColor(color)

        self._point_items: dict[
            tuple[int, int],
            ControlPointItem,
        ] = {}

        self._path_item = QGraphicsPathItem()

        line_color = QColor(self._color)

        line_color.setAlpha(175)

        line_pen = QPen(
            line_color,
            1.2,
        )

        line_pen.setCosmetic(True)

        self._path_item.setPen(line_pen)

        self._path_item.setZValue(10)

        self._path_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

        self._scene.addItem(self._path_item)

        self._create_point_items()
        self._update_path()

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    def set_visible(
        self,
        visible: bool,
    ) -> None:
        self._path_item.setVisible(visible)

        for item in self._point_items.values():
            item.setVisible(visible)

    def remove(self) -> None:
        if self._path_item.scene() is not None:
            self._scene.removeItem(self._path_item)

        for item in self._point_items.values():
            if item.scene() is not None:
                self._scene.removeItem(item)

        self._point_items.clear()

    def _create_point_items(self) -> None:
        for point in self._mesh.iter_points():
            item = ControlPointItem(
                point=point,
                document_rect=self._document_rect,
                color=self._color,
                on_moved=self._point_moved,
            )

            self._scene.addItem(item)

            self._point_items[
                (
                    point.row,
                    point.column,
                )
            ] = item

    def _point_moved(
        self,
        row: int,
        column: int,
        x: float,
        y: float,
    ) -> None:
        self._mesh.move_point(
            row=row,
            column=column,
            x=x,
            y=y,
        )

        self._update_path()

    def _update_path(self) -> None:
        path = QPainterPath()

        for row in range(self._mesh.rows):
            first_item = self._point_items[(row, 0)]

            path.moveTo(first_item.pos())

            for column in range(
                1,
                self._mesh.columns,
            ):
                path.lineTo(self._point_items[(row, column)].pos())

        for column in range(self._mesh.columns):
            first_item = self._point_items[(0, column)]

            path.moveTo(first_item.pos())

            for row in range(
                1,
                self._mesh.rows,
            ):
                path.lineTo(self._point_items[(row, column)].pos())

        self._path_item.setPath(path)
