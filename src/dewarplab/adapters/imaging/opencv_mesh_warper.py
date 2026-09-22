import cv2
import numpy as np
from numpy.typing import NDArray

from dewarplab.domain import Mesh


ImageArray = NDArray[np.uint8]


class MeshWarpError(ValueError):
    pass


def warp_image_with_mesh(
    image: ImageArray,
    mesh: Mesh,
) -> ImageArray:
    _validate_image(image)

    if not mesh.is_topologically_valid():
        raise MeshWarpError("The mesh contains invalid cells")

    if mesh.is_regular():
        return image.copy()

    height, width = image.shape[:2]

    map_x, map_y = _build_remap(
        width=width,
        height=height,
        mesh=mesh,
    )

    warped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return warped


def _build_remap(
    width: int,
    height: int,
    mesh: Mesh,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
]:
    source_x = np.empty(
        (
            mesh.rows,
            mesh.columns,
        ),
        dtype=np.float32,
    )

    source_y = np.empty(
        (
            mesh.rows,
            mesh.columns,
        ),
        dtype=np.float32,
    )

    for point in mesh.iter_points():
        source_x[
            point.row,
            point.column,
        ] = point.x

        source_y[
            point.row,
            point.column,
        ] = point.y

    horizontal_x = np.empty(
        (
            mesh.rows,
            width,
        ),
        dtype=np.float32,
    )

    horizontal_y = np.empty(
        (
            mesh.rows,
            width,
        ),
        dtype=np.float32,
    )

    x_position = np.linspace(
        0.0,
        mesh.columns - 1,
        width,
        dtype=np.float32,
    )

    x_cell = np.floor(x_position).astype(np.int32)

    x_cell = np.minimum(
        x_cell,
        mesh.columns - 2,
    )

    x_fraction = (x_position - x_cell).astype(np.float32)

    inverse_x_fraction = 1.0 - x_fraction

    for row in range(mesh.rows):
        horizontal_x[row] = (
            source_x[
                row,
                x_cell,
            ]
            * inverse_x_fraction
            + source_x[
                row,
                x_cell + 1,
            ]
            * x_fraction
        )

        horizontal_y[row] = (
            source_y[
                row,
                x_cell,
            ]
            * inverse_x_fraction
            + source_y[
                row,
                x_cell + 1,
            ]
            * x_fraction
        )

    y_position = np.linspace(
        0.0,
        mesh.rows - 1,
        height,
        dtype=np.float32,
    )

    y_cell = np.floor(y_position).astype(np.int32)

    y_cell = np.minimum(
        y_cell,
        mesh.rows - 2,
    )

    y_fraction = (y_position - y_cell).astype(np.float32)

    map_x = np.empty(
        (
            height,
            width,
        ),
        dtype=np.float32,
    )

    map_y = np.empty(
        (
            height,
            width,
        ),
        dtype=np.float32,
    )

    for destination_y in range(height):
        cell_row = y_cell[destination_y]

        vertical_fraction = y_fraction[destination_y]

        inverse_vertical_fraction = 1.0 - vertical_fraction

        normalized_source_x = (
            horizontal_x[cell_row] * inverse_vertical_fraction
            + horizontal_x[cell_row + 1] * vertical_fraction
        )

        normalized_source_y = (
            horizontal_y[cell_row] * inverse_vertical_fraction
            + horizontal_y[cell_row + 1] * vertical_fraction
        )

        map_x[destination_y] = normalized_source_x * (width - 1)

        map_y[destination_y] = normalized_source_y * (height - 1)

    return (
        map_x,
        map_y,
    )


def _validate_image(
    image: ImageArray,
) -> None:
    if not isinstance(
        image,
        np.ndarray,
    ):
        raise MeshWarpError("Image must be a NumPy array")

    if image.dtype != np.uint8:
        raise MeshWarpError("Image must use uint8 pixels")

    if image.ndim not in (
        2,
        3,
    ):
        raise MeshWarpError("Image must be grayscale or multi-channel")

    height, width = image.shape[:2]

    if width < 2 or height < 2:
        raise MeshWarpError("Image must be at least 2 × 2 pixels")

    if image.ndim == 3 and image.shape[2] not in (
        3,
        4,
    ):
        raise MeshWarpError("Image must have 3 or 4 channels")
