from PySide6.QtCore import (
    QEvent,
    QPointF,
    QRectF,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QDragEnterEvent,
    QDragLeaveEvent,
    QDragMoveEvent,
    QDropEvent,
    QFont,
    QFontMetricsF,
    QImage,
    QMouseEvent,
    QNativeGestureEvent,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

from dewarplab.domain import Mesh
from dewarplab.ui.mesh_overlay import MeshOverlay


class DocumentView(QGraphicsView):
    file_dropped = Signal(str)
    browse_requested = Signal()
    zoom_changed = Signal(int)

    ZOOM_FACTOR = 1.2
    MIN_ZOOM_SCALE = 0.02
    MAX_ZOOM_SCALE = 8.0

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._mesh_overlay: MeshOverlay | None = None

        self._drag_active = False
        self._browse_rect = QRectF()

        self.setScene(self._scene)

        self.setAcceptDrops(True)

        self.setMouseTracking(True)

        self.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform,
            True,
        )

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_image(
        self,
        image: QImage,
        mesh: Mesh | None = None,
    ) -> None:
        self._replace_scene_image(
            image=image,
            mesh=mesh,
        )

        self.fit_document()

    def replace_image(
        self,
        image: QImage,
        mesh: Mesh | None = None,
    ) -> None:
        if self._pixmap_item is None:
            self.set_image(
                image=image,
                mesh=mesh,
            )

            return

        viewport_center = self.viewport().rect().center()

        scene_center = self.mapToScene(viewport_center)

        current_transform = self.transform()

        self._replace_scene_image(
            image=image,
            mesh=mesh,
        )

        self.setTransform(current_transform)

        self.centerOn(scene_center)

        self._emit_zoom_changed()

    def _replace_scene_image(
        self,
        image: QImage,
        mesh: Mesh | None,
    ) -> None:
        self._scene.clear()

        self._pixmap_item = None
        self._mesh_overlay = None

        pixmap = QPixmap.fromImage(image)

        self._pixmap_item = self._scene.addPixmap(pixmap)

        self._scene.setSceneRect(self._pixmap_item.boundingRect())

        self._drag_active = False
        self._browse_rect = QRectF()

        if mesh is not None:
            self.set_mesh(mesh)

    def set_mesh(
        self,
        mesh: Mesh,
    ) -> None:
        if self._pixmap_item is None:
            return

        if self._mesh_overlay is not None:
            self._mesh_overlay.remove()

        mesh_color = self.palette().color(QPalette.ColorRole.Highlight)

        self._mesh_overlay = MeshOverlay(
            scene=self._scene,
            document_rect=(self._pixmap_item.sceneBoundingRect()),
            mesh=mesh,
            color=mesh_color,
        )

    def set_mesh_visible(
        self,
        visible: bool,
    ) -> None:
        if self._mesh_overlay is None:
            return

        self._mesh_overlay.set_visible(visible)

    def clear_document(self) -> None:
        self._scene.clear()

        self._pixmap_item = None
        self._mesh_overlay = None

        self._drag_active = False
        self._browse_rect = QRectF()

        self.resetTransform()

        self.zoom_changed.emit(100)

        self.viewport().update()

    def fit_document(self) -> None:
        if self._pixmap_item is None:
            return

        self.resetTransform()

        self.fitInView(
            self._pixmap_item,
            Qt.AspectRatioMode.KeepAspectRatio,
        )

        current_scale = self._current_zoom_scale()

        if current_scale > self.MAX_ZOOM_SCALE:
            self.resetTransform()

            self.scale(
                self.MAX_ZOOM_SCALE,
                self.MAX_ZOOM_SCALE,
            )

            self.centerOn(self._pixmap_item)

        self._emit_zoom_changed()

    def zoom_in(self) -> None:
        self._apply_zoom_factor(self.ZOOM_FACTOR)

    def zoom_out(self) -> None:
        self._apply_zoom_factor(1.0 / self.ZOOM_FACTOR)

    def _current_zoom_scale(
        self,
    ) -> float:
        return self.transform().m11()

    def _emit_zoom_changed(
        self,
    ) -> None:
        zoom_percentage = round(self._current_zoom_scale() * 100)

        self.zoom_changed.emit(zoom_percentage)

    def _apply_zoom_factor(
        self,
        factor: float,
        anchor_position: QPointF | None = None,
    ) -> None:
        if self._pixmap_item is None:
            return

        if factor <= 0:
            return

        current_scale = self._current_zoom_scale()

        if current_scale <= 0:
            return

        target_scale = current_scale * factor

        target_scale = max(
            self.MIN_ZOOM_SCALE,
            min(
                self.MAX_ZOOM_SCALE,
                target_scale,
            ),
        )

        actual_factor = target_scale / current_scale

        if abs(actual_factor - 1.0) < 0.000001:
            return

        if anchor_position is None:
            self.scale(
                actual_factor,
                actual_factor,
            )

            self._emit_zoom_changed()

            return

        previous_anchor = self.transformationAnchor()

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)

        scene_position_before = self.mapToScene(anchor_position.toPoint())

        self.scale(
            actual_factor,
            actual_factor,
        )

        scene_position_after = self.mapToScene(anchor_position.toPoint())

        position_delta = scene_position_after - scene_position_before

        self.translate(
            position_delta.x(),
            position_delta.y(),
        )

        self.setTransformationAnchor(previous_anchor)

        self._emit_zoom_changed()

    def viewportEvent(
        self,
        event: QEvent,
    ) -> bool:
        if (
            self._pixmap_item is not None
            and event.type() == QEvent.Type.NativeGesture
            and isinstance(
                event,
                QNativeGestureEvent,
            )
            and event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture
        ):
            zoom_factor = 1.0 + event.value()

            self._apply_zoom_factor(
                zoom_factor,
                event.position(),
            )

            event.accept()

            return True

        return super().viewportEvent(event)

    def paintEvent(
        self,
        event,
    ) -> None:
        super().paintEvent(event)

        if self._pixmap_item is not None:
            return

        painter = QPainter(self.viewport())

        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing,
            True,
        )

        self._paint_empty_state(painter)

    def _paint_empty_state(
        self,
        painter: QPainter,
    ) -> None:
        viewport_rect = QRectF(self.viewport().rect())

        center = viewport_rect.center()

        palette = self.palette()

        muted_color = palette.color(QPalette.ColorRole.PlaceholderText)

        text_color = palette.color(QPalette.ColorRole.Text)

        accent_color = palette.color(QPalette.ColorRole.Highlight)

        icon_color = accent_color if self._drag_active else muted_color

        icon_width = 76.0
        icon_height = 96.0
        fold_size = 22.0

        icon_left = center.x() - icon_width / 2

        icon_top = center.y() - 120

        document_path = QPainterPath()

        document_path.moveTo(
            icon_left,
            icon_top,
        )

        document_path.lineTo(
            icon_left + icon_width - fold_size,
            icon_top,
        )

        document_path.lineTo(
            icon_left + icon_width,
            icon_top + fold_size,
        )

        document_path.lineTo(
            icon_left + icon_width,
            icon_top + icon_height,
        )

        document_path.lineTo(
            icon_left,
            icon_top + icon_height,
        )

        document_path.closeSubpath()

        painter.setPen(
            QPen(
                icon_color,
                2.0,
            )
        )

        painter.setBrush(Qt.BrushStyle.NoBrush)

        painter.drawPath(document_path)

        fold_path = QPainterPath()

        fold_path.moveTo(
            icon_left + icon_width - fold_size,
            icon_top,
        )

        fold_path.lineTo(
            icon_left + icon_width - fold_size,
            icon_top + fold_size,
        )

        fold_path.lineTo(
            icon_left + icon_width,
            icon_top + fold_size,
        )

        painter.drawPath(fold_path)

        plus_center = QPointF(
            center.x(),
            icon_top + icon_height / 2 + 5,
        )

        plus_size = 14.0

        painter.drawLine(
            QPointF(
                plus_center.x() - plus_size,
                plus_center.y(),
            ),
            QPointF(
                plus_center.x() + plus_size,
                plus_center.y(),
            ),
        )

        painter.drawLine(
            QPointF(
                plus_center.x(),
                plus_center.y() - plus_size,
            ),
            QPointF(
                plus_center.x(),
                plus_center.y() + plus_size,
            ),
        )

        primary_font = QFont(self.font())

        if primary_font.pointSizeF() > 0:
            primary_font.setPointSizeF(primary_font.pointSizeF() + 1.0)

        painter.setFont(primary_font)

        painter.setPen(accent_color if self._drag_active else text_color)

        if self._drag_active:
            primary_text = "Suelta el archivo aquí"
        else:
            primary_text = "Arrastra y suelta un archivo"

        primary_rect = QRectF(
            viewport_rect.left(),
            icon_top + icon_height + 28,
            viewport_rect.width(),
            30,
        )

        painter.drawText(
            primary_rect,
            Qt.AlignmentFlag.AlignCenter,
            primary_text,
        )

        if self._drag_active:
            self._browse_rect = QRectF()

            return

        secondary_font = QFont(primary_font)

        browse_font = QFont(primary_font)

        browse_font.setUnderline(True)

        prefix = "o "
        browse_text = "examina"

        prefix_metrics = QFontMetricsF(secondary_font)

        browse_metrics = QFontMetricsF(browse_font)

        prefix_width = prefix_metrics.horizontalAdvance(prefix)

        browse_width = browse_metrics.horizontalAdvance(browse_text)

        total_width = prefix_width + browse_width

        secondary_top = primary_rect.bottom() + 4

        secondary_height = max(
            prefix_metrics.height(),
            browse_metrics.height(),
        )

        baseline_y = secondary_top + browse_metrics.ascent()

        start_x = center.x() - total_width / 2

        painter.setFont(secondary_font)

        painter.setPen(muted_color)

        painter.drawText(
            QPointF(
                start_x,
                baseline_y,
            ),
            prefix,
        )

        painter.setFont(browse_font)

        painter.setPen(accent_color)

        painter.drawText(
            QPointF(
                start_x + prefix_width,
                baseline_y,
            ),
            browse_text,
        )

        self._browse_rect = QRectF(
            start_x + prefix_width,
            secondary_top,
            browse_width,
            secondary_height,
        )

    def mouseMoveEvent(
        self,
        event: QMouseEvent,
    ) -> None:
        if self._pixmap_item is None and self._browse_rect.contains(event.position()):
            self.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.viewport().unsetCursor()

        super().mouseMoveEvent(event)

    def mousePressEvent(
        self,
        event: QMouseEvent,
    ) -> None:
        if (
            self._pixmap_item is None
            and event.button() == Qt.MouseButton.LeftButton
            and self._browse_rect.contains(event.position())
        ):
            self.browse_requested.emit()

            event.accept()

            return

        super().mousePressEvent(event)

    def dragEnterEvent(
        self,
        event: QDragEnterEvent,
    ) -> None:
        urls = event.mimeData().urls()

        if len(urls) == 1 and urls[0].isLocalFile():
            self._drag_active = True

            self.viewport().update()

            event.acceptProposedAction()

            return

        event.ignore()

    def dragMoveEvent(
        self,
        event: QDragMoveEvent,
    ) -> None:
        urls = event.mimeData().urls()

        if len(urls) == 1 and urls[0].isLocalFile():
            event.acceptProposedAction()

            return

        event.ignore()

    def dragLeaveEvent(
        self,
        event: QDragLeaveEvent,
    ) -> None:
        self._drag_active = False

        self.viewport().update()

        event.accept()

    def dropEvent(
        self,
        event: QDropEvent,
    ) -> None:
        urls = event.mimeData().urls()

        self._drag_active = False

        self.viewport().update()

        if len(urls) != 1:
            event.ignore()

            return

        path = urls[0].toLocalFile()

        if not path:
            event.ignore()

            return

        self.file_dropped.emit(path)

        event.acceptProposedAction()
