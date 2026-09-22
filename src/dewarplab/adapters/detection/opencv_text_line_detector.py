import cv2
import numpy as np
from numpy.typing import NDArray

from dewarplab.application import (
    NormalizedPoint,
    TextLineGeometry,
    TextLineTrace,
)


ImageArray = NDArray[np.uint8]

MAX_ANALYSIS_DIMENSION = 1800

MIN_TRACE_SAMPLE_COUNT = 5
MAX_TRACE_SAMPLE_COUNT = 14

MIN_COMPONENT_ASPECT_RATIO = 2.2
MIN_COMPONENT_WIDTH_RATIO = 0.035
MAX_COMPONENT_HEIGHT_RATIO = 0.06


class TextLineDetectionError(ValueError):
    pass


def detect_text_line_geometry(
    image: ImageArray,
) -> TextLineGeometry:
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

    line_mask = _build_line_mask(binary)

    (
        component_count,
        labels,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        line_mask,
        connectivity=8,
    )

    traces: list[TextLineTrace] = []

    for component_index in range(
        1,
        component_count,
    ):
        left = int(
            stats[
                component_index,
                cv2.CC_STAT_LEFT,
            ]
        )

        top = int(
            stats[
                component_index,
                cv2.CC_STAT_TOP,
            ]
        )

        component_width = int(
            stats[
                component_index,
                cv2.CC_STAT_WIDTH,
            ]
        )

        component_height = int(
            stats[
                component_index,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        if not _is_candidate_component(
            component_width=component_width,
            component_height=component_height,
            image_width=width,
            image_height=height,
        ):
            continue

        component_labels = labels[
            top : top + component_height,
            left : left + component_width,
        ]

        component_mask = component_labels == component_index

        sampled_points = _sample_component_centerline(
            component_mask=component_mask,
            offset_x=left,
            offset_y=top,
        )

        if len(sampled_points) < 3:
            continue

        smoothed_points = _smooth_points(sampled_points)

        normalized_points = tuple(
            NormalizedPoint(
                x=_normalize_coordinate(
                    x,
                    width,
                ),
                y=_normalize_coordinate(
                    y,
                    height,
                ),
            )
            for x, y in smoothed_points
        )

        trace = TextLineTrace(points=normalized_points)

        if trace.horizontal_span < MIN_COMPONENT_WIDTH_RATIO:
            continue

        traces.append(trace)

    traces.sort(key=_trace_sort_key)

    return TextLineGeometry(traces=tuple(traces))


def _to_grayscale(
    image: ImageArray,
) -> NDArray[np.uint8]:
    if not isinstance(
        image,
        np.ndarray,
    ):
        raise TextLineDetectionError("Image must be a NumPy array")

    if image.dtype != np.uint8:
        raise TextLineDetectionError("Image must use uint8 pixels")

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
        raise TextLineDetectionError("Image must be grayscale, RGB or RGBA")

    height, width = gray.shape

    if width < 32 or height < 32:
        raise TextLineDetectionError("Image is too small for text line detection")

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


def _build_line_mask(
    binary: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    _, width = binary.shape

    join_width = max(
        7,
        round(width * 0.012),
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            join_width,
            1,
        ),
    )

    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        horizontal_kernel,
    )

    reinforcement_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            3,
            1,
        ),
    )

    return cv2.dilate(
        closed,
        reinforcement_kernel,
        iterations=1,
    )


def _is_candidate_component(
    component_width: int,
    component_height: int,
    image_width: int,
    image_height: int,
) -> bool:
    minimum_width = max(
        20,
        round(image_width * MIN_COMPONENT_WIDTH_RATIO),
    )

    maximum_height = max(
        8,
        round(image_height * MAX_COMPONENT_HEIGHT_RATIO),
    )

    if component_width < minimum_width:
        return False

    if component_height < 3 or component_height > maximum_height:
        return False

    aspect_ratio = component_width / component_height

    if aspect_ratio < MIN_COMPONENT_ASPECT_RATIO:
        return False

    return True


def _sample_component_centerline(
    component_mask: NDArray[np.bool_],
    offset_x: int,
    offset_y: int,
) -> list[tuple[float, float]]:
    component_height, component_width = component_mask.shape

    del component_height

    sample_count = round(component_width / 45)

    sample_count = max(
        MIN_TRACE_SAMPLE_COUNT,
        min(
            MAX_TRACE_SAMPLE_COUNT,
            sample_count,
        ),
    )

    edges = np.linspace(
        0,
        component_width,
        sample_count + 1,
        dtype=np.int32,
    )

    points: list[tuple[float, float]] = []

    for sample_index in range(sample_count):
        start_x = int(edges[sample_index])

        end_x = int(edges[sample_index + 1])

        if end_x <= start_x:
            continue

        sample_mask = component_mask[
            :,
            start_x:end_x,
        ]

        y_coordinates, x_coordinates = np.nonzero(sample_mask)

        if y_coordinates.size == 0:
            continue

        local_x = start_x + float(np.median(x_coordinates))

        local_y = float(np.median(y_coordinates))

        points.append(
            (
                offset_x + local_x,
                offset_y + local_y,
            )
        )

    return points


def _smooth_points(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if len(points) < 3:
        return points

    x_values = np.asarray(
        [point[0] for point in points],
        dtype=np.float64,
    )

    y_values = np.asarray(
        [point[1] for point in points],
        dtype=np.float64,
    )

    smoothed_y = y_values.copy()

    smoothed_y[1:-1] = (y_values[:-2] + 2.0 * y_values[1:-1] + y_values[2:]) / 4.0

    return [
        (
            float(x),
            float(y),
        )
        for x, y in zip(
            x_values,
            smoothed_y,
            strict=True,
        )
    ]


def _normalize_coordinate(
    value: float,
    dimension: int,
) -> float:
    if dimension <= 1:
        return 0.0

    normalized = value / (dimension - 1)

    return min(
        1.0,
        max(
            0.0,
            normalized,
        ),
    )


def _trace_sort_key(
    trace: TextLineTrace,
) -> tuple[
    float,
    float,
]:
    first_point = trace.points[0]

    return (
        first_point.y,
        first_point.x,
    )
