from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QDragEnterEvent,
    QDragLeaveEvent,
    QDragMoveEvent,
    QDropEvent,
    QFont,
    QFontMetricsF,
    QImage,
    QMouseEvent,
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


class DocumentView(QGraphicsView):
    file_dropped = Signal(str)
    browse_requested = Signal()

    ZOOM_FACTOR = 1.2

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self._pixmap_item: QGraphicsPixmapItem | None = None
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

    def set_image(self, image: QImage) -> None:
        self._scene.clear()

        pixmap = QPixmap.fromImage(image)

        self._pixmap_item = self._scene.addPixmap(pixmap)

        self._scene.setSceneRect(self._pixmap_item.boundingRect())

        self._drag_active = False
        self._browse_rect = QRectF()

        self.fit_document()

    def clear_document(self) -> None:
        self._scene.clear()
        self._pixmap_item = None
        self._drag_active = False
        self._browse_rect = QRectF()

        self.resetTransform()
        self.viewport().update()

    def fit_document(self) -> None:
        if self._pixmap_item is None:
            return

        self.resetTransform()

        self.fitInView(
            self._pixmap_item,
            Qt.AspectRatioMode.KeepAspectRatio,
        )

    def zoom_in(self) -> None:
        if self._pixmap_item is None:
            return

        self.scale(
            self.ZOOM_FACTOR,
            self.ZOOM_FACTOR,
        )

    def zoom_out(self) -> None:
        if self._pixmap_item is None:
            return

        self.scale(
            1.0 / self.ZOOM_FACTOR,
            1.0 / self.ZOOM_FACTOR,
        )

    def paintEvent(self, event) -> None:
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
