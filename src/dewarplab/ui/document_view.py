from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QImage,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)


class DocumentView(QGraphicsView):
    file_dropped = Signal(str)

    ZOOM_FACTOR = 1.2

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self._pixmap_item: QGraphicsPixmapItem | None = None

        self.setScene(self._scene)
        self.setAcceptDrops(True)

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

        self.fit_document()

    def clear_document(self) -> None:
        self._scene.clear()
        self._pixmap_item = None

        self.resetTransform()

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

    def dragEnterEvent(
        self,
        event: QDragEnterEvent,
    ) -> None:
        urls = event.mimeData().urls()

        if len(urls) == 1 and urls[0].isLocalFile():
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

    def dropEvent(
        self,
        event: QDropEvent,
    ) -> None:
        urls = event.mimeData().urls()

        if len(urls) != 1:
            event.ignore()
            return

        path = urls[0].toLocalFile()

        if not path:
            event.ignore()
            return

        self.file_dropped.emit(path)

        event.acceptProposedAction()
