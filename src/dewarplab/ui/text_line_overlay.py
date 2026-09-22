import math

from PySide6.QtCore import QPointF, QRectF, Qt
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
    TextLineGeometry,
)


ACTIVE_LINE_WIDTH = 1.5
IGNORED_LINE_WIDTH = 2.0
SELECTED_WIDTH_INCREASE = 2.5

IGNORED_CROSS_SIZE = 10.0


class TextLineOverlay:
    def __init__(
        self,
        scene: QGraphicsScene,
        document_rect: QRectF,
        geometry: TextLineGeometry,
        active_color: QColor,
        ignored_color: QColor,
        ignored_indices: set[int] | None = None,
    ):
        self._scene = scene

        self._document_rect = QRectF(document_rect)

        self._geometry = geometry

        self._active_color = QColor(active_color)

        self._ignored_color = QColor(ignored_color)

        self._ignored_indices = set(ignored_indices or set())

        self._selected_index: int | None = None

        self._trace_items: dict[
            int,
            QGraphicsPathItem,
        ] = {}

        self._ignored_marker_items: dict[
            int,
            QGraphicsPathItem,
        ] = {}

        self._trace_scene_points: dict[
            int,
            tuple[
                QPointF,
                ...,
            ],
        ] = {}

        self._visible = True

        self._create_items()

    def set_visible(
        self,
        visible: bool,
    ) -> None:
        self._visible = visible

        for item in self._trace_items.values():
            item.setVisible(visible)

        for item in self._ignored_marker_items.values():
            item.setVisible(visible)

    def remove(self) -> None:
        for item in self._trace_items.values():
            if item.scene() is not None:
                self._scene.removeItem(item)

        for item in self._ignored_marker_items.values():
            if item.scene() is not None:
                self._scene.removeItem(item)

        self._trace_items.clear()
        self._ignored_marker_items.clear()
        self._trace_scene_points.clear()

    def select_trace(
        self,
        index: int | None,
    ) -> None:
        if index is not None and index not in self._trace_items:
            index = None

        previous_index = self._selected_index

        self._selected_index = index

        if previous_index is not None:
            self._update_trace_style(previous_index)

        if index is not None:
            self._update_trace_style(index)

    def selected_index(
        self,
    ) -> int | None:
        return self._selected_index

    def is_ignored(
        self,
        index: int,
    ) -> bool:
        return index in self._ignored_indices

    def toggle_ignored(
        self,
        index: int,
    ) -> bool:
        if index not in self._trace_items:
            return False

        if index in self._ignored_indices:
            self._ignored_indices.remove(index)

            ignored = False
        else:
            self._ignored_indices.add(index)

            ignored = True

        self._update_trace_style(index)

        self._update_ignored_markers(index)

        return ignored

    def ignored_indices(
        self,
    ) -> set[int]:
        return set(self._ignored_indices)

    def trace_index_at(
        self,
        scene_position: QPointF,
        tolerance: float,
    ) -> int | None:
        if not self._visible:
            return None

        best_index: int | None = None
        best_distance = float("inf")

        for (
            index,
            points,
        ) in self._trace_scene_points.items():
            if len(points) < 2:
                continue

            distance = self._distance_to_polyline(
                scene_position,
                points,
            )

            if distance <= tolerance and distance < best_distance:
                best_distance = distance
                best_index = index

        return best_index

    def _create_items(
        self,
    ) -> None:
        for (
            index,
            trace,
        ) in enumerate(self._geometry.traces):
            scene_points = tuple(
                QPointF(
                    self._scene_x(point.x),
                    self._scene_y(point.y),
                )
                for point in trace.points
            )

            if len(scene_points) < 2:
                continue

            self._trace_scene_points[index] = scene_points

            path = QPainterPath()

            path.moveTo(scene_points[0])

            for point in scene_points[1:]:
                path.lineTo(point)

            item = QGraphicsPathItem(path)

            item.setBrush(Qt.BrushStyle.NoBrush)

            item.setZValue(15)

            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

            self._scene.addItem(item)

            self._trace_items[index] = item

            self._update_trace_style(index)

            self._update_ignored_markers(index)

    def _update_trace_style(
        self,
        index: int,
    ) -> None:
        item = self._trace_items.get(index)

        if item is None:
            return

        ignored = index in self._ignored_indices

        selected = index == self._selected_index

        if ignored:
            color = QColor(self._ignored_color)

            width = IGNORED_LINE_WIDTH

            style = Qt.PenStyle.DashLine
        else:
            color = QColor(self._active_color)

            width = ACTIVE_LINE_WIDTH

            style = Qt.PenStyle.SolidLine

        if selected:
            width += SELECTED_WIDTH_INCREASE

        pen = QPen(
            color,
            width,
            style,
        )

        pen.setCosmetic(True)

        pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

        item.setPen(pen)

    def _update_ignored_markers(
        self,
        index: int,
    ) -> None:
        previous_item = self._ignored_marker_items.pop(
            index,
            None,
        )

        if previous_item is not None and previous_item.scene() is not None:
            self._scene.removeItem(previous_item)

        if index not in self._ignored_indices:
            return

        points = self._trace_scene_points.get(index)

        if points is None or len(points) < 2:
            return

        marker_path = QPainterPath()

        marker_indices = self._marker_indices(len(points))

        half_size = IGNORED_CROSS_SIZE / 2.0

        for point_index in marker_indices:
            point = points[point_index]

            marker_path.moveTo(
                point.x() - half_size,
                point.y() - half_size,
            )

            marker_path.lineTo(
                point.x() + half_size,
                point.y() + half_size,
            )

            marker_path.moveTo(
                point.x() - half_size,
                point.y() + half_size,
            )

            marker_path.lineTo(
                point.x() + half_size,
                point.y() - half_size,
            )

        marker_item = QGraphicsPathItem(marker_path)

        pen = QPen(
            self._ignored_color,
            1.6,
        )

        pen.setCosmetic(True)

        pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        marker_item.setPen(pen)

        marker_item.setBrush(Qt.BrushStyle.NoBrush)

        marker_item.setZValue(16)

        marker_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

        marker_item.setVisible(self._visible)

        self._scene.addItem(marker_item)

        self._ignored_marker_items[index] = marker_item

    def _marker_indices(
        self,
        point_count: int,
    ) -> tuple[int, ...]:
        if point_count <= 2:
            return (point_count // 2,)

        if point_count <= 5:
            return (point_count // 2,)

        return (
            point_count // 4,
            point_count // 2,
            (point_count * 3) // 4,
        )

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

    def _distance_to_polyline(
        self,
        point: QPointF,
        polyline: tuple[
            QPointF,
            ...,
        ],
    ) -> float:
        best_distance = float("inf")

        for index in range(len(polyline) - 1):
            start = polyline[index]

            end = polyline[index + 1]

            distance = self._distance_to_segment(
                point=point,
                start=start,
                end=end,
            )

            best_distance = min(
                best_distance,
                distance,
            )

        return best_distance

    def _distance_to_segment(
        self,
        point: QPointF,
        start: QPointF,
        end: QPointF,
    ) -> float:
        segment_x = end.x() - start.x()

        segment_y = end.y() - start.y()

        length_squared = segment_x * segment_x + segment_y * segment_y

        if length_squared <= 0:
            return math.hypot(
                point.x() - start.x(),
                point.y() - start.y(),
            )

        projection = (
            (point.x() - start.x()) * segment_x + (point.y() - start.y()) * segment_y
        ) / length_squared

        projection = max(
            0.0,
            min(
                1.0,
                projection,
            ),
        )

        closest_x = start.x() + projection * segment_x

        closest_y = start.y() + projection * segment_y

        return math.hypot(
            point.x() - closest_x,
            point.y() - closest_y,
        )
