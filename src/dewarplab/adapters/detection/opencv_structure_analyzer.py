import cv2
import numpy as np
from numpy.typing import NDArray

from dewarplab.application import StructureAnalysis


ImageArray = NDArray[np.uint8]

MAX_ANALYSIS_DIMENSION = 1800

MIN_MESH_ROWS = 6
MAX_MESH_ROWS = 24

MIN_MESH_COLUMNS = 6
MAX_MESH_COLUMNS = 24


class StructureAnalysisError(ValueError):
    pass


def analyze_document_structure(
    image: ImageArray,
) -> StructureAnalysis:
    gray = _to_grayscale(image)

    gray = _resize_for_analysis(gray)

    height, width = gray.shape

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    block_size = _adaptive_block_size(
        width=width,
        height=height,
    )

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        15,
    )

    text_component_count = _count_text_components(binary)

    line_mask = _build_line_mask(binary)

    line_band_count = _count_line_bands(line_mask)

    vertical_structure_count = _count_vertical_structures(line_mask)

    suggested_rows, suggested_columns = _suggest_mesh_shape(
        width=width,
        height=height,
        text_component_count=(text_component_count),
        line_band_count=(line_band_count),
        vertical_structure_count=(vertical_structure_count),
    )

    return StructureAnalysis(
        text_component_count=(text_component_count),
        line_band_count=(line_band_count),
        vertical_structure_count=(vertical_structure_count),
        suggested_rows=(suggested_rows),
        suggested_columns=(suggested_columns),
    )


def _to_grayscale(
    image: ImageArray,
) -> NDArray[np.uint8]:
    if not isinstance(
        image,
        np.ndarray,
    ):
        raise StructureAnalysisError("Image must be a NumPy array")

    if image.dtype != np.uint8:
        raise StructureAnalysisError("Image must use uint8 pixels")

    if image.ndim == 2:
        gray = image

    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY,
        )

    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGBA2GRAY,
        )

    else:
        raise StructureAnalysisError("Image must be grayscale, RGB or RGBA")

    height, width = gray.shape

    if width < 32 or height < 32:
        raise StructureAnalysisError("Image is too small for structure analysis")

    return gray


def _resize_for_analysis(
    gray: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    height, width = gray.shape

    largest_dimension = max(
        width,
        height,
    )

    if largest_dimension <= MAX_ANALYSIS_DIMENSION:
        return gray

    scale = MAX_ANALYSIS_DIMENSION / largest_dimension

    target_width = max(
        1,
        round(width * scale),
    )

    target_height = max(
        1,
        round(height * scale),
    )

    return cv2.resize(
        gray,
        (
            target_width,
            target_height,
        ),
        interpolation=cv2.INTER_AREA,
    )


def _adaptive_block_size(
    width: int,
    height: int,
) -> int:
    smallest_dimension = min(
        width,
        height,
    )

    candidate = max(
        15,
        smallest_dimension // 25,
    )

    candidate = min(
        candidate,
        101,
    )

    maximum_allowed = (
        smallest_dimension if smallest_dimension % 2 == 1 else smallest_dimension - 1
    )

    candidate = min(
        candidate,
        maximum_allowed,
    )

    if candidate % 2 == 0:
        candidate -= 1

    return max(
        3,
        candidate,
    )


def _count_text_components(
    binary: NDArray[np.uint8],
) -> int:
    height, width = binary.shape

    image_area = width * height

    (
        component_count,
        _,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    valid_components = 0

    minimum_area = max(
        4,
        round(image_area * 0.000002),
    )

    maximum_area = image_area * 0.01

    for index in range(
        1,
        component_count,
    ):
        (
            _,
            _,
            component_width,
            component_height,
            component_area,
        ) = stats[index]

        if component_area < minimum_area:
            continue

        if component_area > maximum_area:
            continue

        if component_width < 1 or component_height < 2:
            continue

        if component_width > width * 0.25:
            continue

        if component_height > height * 0.08:
            continue

        valid_components += 1

    return valid_components


def _build_line_mask(
    binary: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    _, width = binary.shape

    kernel_width = max(
        3,
        round(width * 0.012),
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            kernel_width,
            1,
        ),
    )

    return cv2.dilate(
        binary,
        kernel,
        iterations=1,
    )


def _count_line_bands(
    line_mask: NDArray[np.uint8],
) -> int:
    height, width = line_mask.shape

    (
        component_count,
        _,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        line_mask,
        connectivity=8,
    )

    centers: list[float] = []
    heights: list[int] = []

    for index in range(
        1,
        component_count,
    ):
        (
            _,
            y,
            component_width,
            component_height,
            _,
        ) = stats[index]

        if component_width < max(
            20,
            width * 0.025,
        ):
            continue

        if component_height < 2:
            continue

        if component_height > height * 0.06:
            continue

        if component_width / component_height < 2.0:
            continue

        centers.append(y + component_height / 2)

        heights.append(component_height)

    if not centers:
        return 0

    median_height = float(np.median(heights))

    tolerance = max(
        3.0,
        median_height * 0.75,
    )

    return _group_centers(
        centers,
        tolerance,
    )


def _group_centers(
    centers: list[float],
    tolerance: float,
) -> int:
    if not centers:
        return 0

    sorted_centers = sorted(centers)

    groups: list[list[float]] = [[sorted_centers[0]]]

    for center in sorted_centers[1:]:
        current_group = groups[-1]

        group_center = sum(current_group) / len(current_group)

        if abs(center - group_center) <= tolerance:
            current_group.append(center)
        else:
            groups.append([center])

    return len(groups)


def _count_vertical_structures(
    line_mask: NDArray[np.uint8],
) -> int:
    height, width = line_mask.shape

    kernel_height = max(
        3,
        round(height * 0.018),
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            1,
            kernel_height,
        ),
    )

    block_mask = cv2.dilate(
        line_mask,
        kernel,
        iterations=1,
    )

    (
        component_count,
        _,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        block_mask,
        connectivity=8,
    )

    valid_structures = 0

    for index in range(
        1,
        component_count,
    ):
        (
            _,
            _,
            component_width,
            component_height,
            _,
        ) = stats[index]

        if component_height < height * 0.06:
            continue

        if component_width < width * 0.025:
            continue

        if component_width > width * 0.85 and component_height > height * 0.5:
            continue

        valid_structures += 1

    return valid_structures


def _suggest_mesh_shape(
    width: int,
    height: int,
    text_component_count: int,
    line_band_count: int,
    vertical_structure_count: int,
) -> tuple[int, int]:
    if text_component_count < 30 or line_band_count < 3:
        return (
            8,
            8,
        )

    rows = round(5 + line_band_count / 5)

    columns = round(
        7
        + min(
            vertical_structure_count,
            10,
        )
        * 1.2
    )

    if text_component_count >= 250:
        columns += min(
            4,
            text_component_count // 250,
        )

    aspect_ratio = width / height

    if aspect_ratio > 1.25:
        columns += 2

    if aspect_ratio < 0.625:
        rows += 1

    rows = max(
        MIN_MESH_ROWS,
        min(
            MAX_MESH_ROWS,
            rows,
        ),
    )

    columns = max(
        MIN_MESH_COLUMNS,
        min(
            MAX_MESH_COLUMNS,
            columns,
        ),
    )

    return (
        rows,
        columns,
    )
