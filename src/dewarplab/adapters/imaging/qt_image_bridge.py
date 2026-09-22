import numpy as np
from numpy.typing import NDArray
from PySide6.QtGui import QImage

from dewarplab.adapters.imaging.opencv_mesh_warper import (
    MeshWarpError,
    warp_image_with_mesh,
)
from dewarplab.domain import Mesh


ImageArray = NDArray[np.uint8]


class QtImageBridgeError(ValueError):
    pass


def qimage_to_rgba_array(
    image: QImage,
) -> ImageArray:
    if image.isNull():
        raise QtImageBridgeError("Cannot convert a null QImage")

    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)

    height = converted.height()
    width = converted.width()
    bytes_per_line = converted.bytesPerLine()

    buffer = converted.constBits()

    raw = np.frombuffer(
        buffer,
        dtype=np.uint8,
        count=converted.sizeInBytes(),
    )

    rows = raw.reshape(
        height,
        bytes_per_line,
    )

    rgba = rows[
        :,
        : width * 4,
    ].reshape(
        height,
        width,
        4,
    )

    return rgba.copy()


def rgba_array_to_qimage(
    image: ImageArray,
) -> QImage:
    if not isinstance(
        image,
        np.ndarray,
    ):
        raise QtImageBridgeError("Image must be a NumPy array")

    if image.dtype != np.uint8:
        raise QtImageBridgeError("Image must use uint8 pixels")

    if image.ndim != 3 or image.shape[2] != 4:
        raise QtImageBridgeError("Image must have RGBA channels")

    height, width, _ = image.shape

    if width < 1 or height < 1:
        raise QtImageBridgeError("Image cannot be empty")

    contiguous = np.ascontiguousarray(image)

    bytes_per_line = width * 4

    qimage = QImage(
        contiguous.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGBA8888,
    )

    if qimage.isNull():
        raise QtImageBridgeError("Could not create QImage")

    return qimage.copy()


def warp_qimage_with_mesh(
    image: QImage,
    mesh: Mesh,
) -> QImage:
    rgba = qimage_to_rgba_array(image)

    try:
        warped_rgba = warp_image_with_mesh(
            rgba,
            mesh,
        )
    except MeshWarpError:
        raise
    except Exception as error:
        raise QtImageBridgeError("Could not warp QImage") from error

    result = rgba_array_to_qimage(warped_rgba)

    color_space = image.colorSpace()

    if color_space.isValid():
        result.setColorSpace(color_space)

    return result
