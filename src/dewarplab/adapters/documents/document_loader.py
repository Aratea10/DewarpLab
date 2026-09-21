from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtGui import QImage, QImageReader
from PySide6.QtPdf import QPdfDocument


SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
}

SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | {
    ".pdf",
}

PDF_PREVIEW_DPI = 180
PREVIEW_MAX_DIMENSION = 3200


class DocumentLoadError(Exception):
    """Raised when a document cannot be loaded or rendered."""


@dataclass
class LoadedDocument:
    path: Path
    pdf: QPdfDocument | None = None

    @property
    def is_pdf(self) -> bool:
        return self.pdf is not None

    @property
    def page_count(self) -> int:
        if self.pdf is not None:
            return self.pdf.pageCount()

        return 1

    def close(self) -> None:
        if self.pdf is not None:
            self.pdf.close()


def is_supported_document(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def load_document(path: str | Path) -> LoadedDocument:
    document_path = Path(path).expanduser().resolve()

    if not document_path.exists():
        raise DocumentLoadError("El archivo no existe.")

    if not document_path.is_file():
        raise DocumentLoadError("La ruta seleccionada no es un archivo.")

    suffix = document_path.suffix.lower()

    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        _validate_image(document_path)

        return LoadedDocument(
            path=document_path,
        )

    if suffix == ".pdf":
        return _load_pdf(document_path)

    raise DocumentLoadError("Formato no compatible. Usa PNG, JPG, JPEG o PDF.")


def render_page(
    document: LoadedDocument,
    page_index: int = 0,
) -> QImage:
    if document.is_pdf:
        return _render_pdf_page(
            document,
            page_index,
        )

    if page_index != 0:
        raise DocumentLoadError("Una imagen solo contiene una página.")

    return _render_image_preview(document.path)


def _calculate_preview_size(
    width: float,
    height: float,
    max_dimension: int = PREVIEW_MAX_DIMENSION,
) -> QSize:
    if width <= 0 or height <= 0:
        raise DocumentLoadError("El documento tiene unas dimensiones no válidas.")

    largest_dimension = max(
        width,
        height,
    )

    scale = min(
        1.0,
        max_dimension / largest_dimension,
    )

    return QSize(
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )


def _validate_image(path: Path) -> None:
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)

    if not reader.canRead():
        message = reader.errorString() or "No se pudo leer la imagen."

        raise DocumentLoadError(message)


def _render_image_preview(path: Path) -> QImage:
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)

    original_size = reader.size()

    if original_size.isValid():
        preview_size = _calculate_preview_size(
            original_size.width(),
            original_size.height(),
        )

        if preview_size != original_size:
            reader.setScaledSize(preview_size)

    image = reader.read()

    if image.isNull():
        message = reader.errorString() or "No se pudo renderizar la imagen."

        raise DocumentLoadError(message)

    # Medida de seguridad por si el formato no proporcionó
    # correctamente sus dimensiones antes de leer la imagen.
    if image.width() > PREVIEW_MAX_DIMENSION or image.height() > PREVIEW_MAX_DIMENSION:
        preview_size = _calculate_preview_size(
            image.width(),
            image.height(),
        )

        image = image.scaled(
            preview_size,
        )

    return image


def _load_pdf(path: Path) -> LoadedDocument:
    pdf = QPdfDocument()

    error = pdf.load(str(path))

    if error != QPdfDocument.Error.None_:
        pdf.close()

        raise DocumentLoadError(f"No se pudo abrir el PDF: {error.name}")

    if pdf.pageCount() < 1:
        pdf.close()

        raise DocumentLoadError("El PDF no contiene ninguna página.")

    return LoadedDocument(
        path=path,
        pdf=pdf,
    )


def _render_pdf_page(
    document: LoadedDocument,
    page_index: int,
) -> QImage:
    pdf = document.pdf

    if pdf is None:
        raise DocumentLoadError("El documento no contiene un PDF.")

    if not 0 <= page_index < pdf.pageCount():
        raise DocumentLoadError("La página solicitada no existe.")

    point_size = pdf.pagePointSize(page_index)

    if point_size.width() <= 0 or point_size.height() <= 0:
        raise DocumentLoadError("La página del PDF tiene unas dimensiones no válidas.")

    dpi_scale = PDF_PREVIEW_DPI / 72.0

    target_width = point_size.width() * dpi_scale

    target_height = point_size.height() * dpi_scale

    pixel_size = _calculate_preview_size(
        target_width,
        target_height,
    )

    image = pdf.render(
        page_index,
        pixel_size,
    )

    if image.isNull():
        raise DocumentLoadError("No se pudo renderizar esta página del PDF.")

    return image
