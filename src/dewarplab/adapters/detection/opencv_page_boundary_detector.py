import cv2
import numpy as np
from numpy.typing import NDArray

from dewarplab.application import (
    NormalizedPoint,
    PageBoundaryGeometry,
)


ImageArray = NDArray[np.uint8]

MAX_DETECTION_DIMENSION = 1800

MIN_PAGE_AREA_RATIO = 0.15
MAX_PAGE_AREA_RATIO = 0.965

MIN_PAGE_WIDTH_RATIO = 0.25
MIN_PAGE_HEIGHT_RATIO = 0.25

MORPHOLOGY_KERNEL_RATIO = 0.008

CONTOUR_APPROXIMATION_RATIO = 0.0015

MAX_BOUNDARY_POINTS = 120


class PageBoundaryDetectionError(ValueError):
    pass


def detect_page_boundary_geometry(
    image: ImageArray,
) -> PageBoundaryGeometry:
    gray = _to_grayscale(image)

    gray = _resize_for_detection(gray)

    height, width = gray.shape

    candidate_masks = _candidate_masks(gray)

    best_contour = None
    best_score = float("-inf")

    for mask in candidate_masks:
        prepared_mask = _prepare_candidate_mask(mask)

        contour, score = _best_page_contour(
            mask=prepared_mask,
            width=width,
            height=height,
        )

        if contour is not None and score > best_score:
            best_contour = contour
            best_score = score

    if best_contour is None:
        return _full_frame_boundary()

    contour_area = float(cv2.contourArea(best_contour))

    image_area = float(width * height)

    if image_area <= 0:
        return _full_frame_boundary()

    area_ratio = contour_area / image_area

    if area_ratio >= MAX_PAGE_AREA_RATIO:
        return _full_frame_boundary()

    points = _simplify_contour(
        contour=best_contour,
        width=width,
        height=height,
    )

    if len(points) < 4:
        return _full_frame_boundary()

    return PageBoundaryGeometry(
        points=points,
        is_full_frame=False,
    )


def mask_image_to_page_boundary(
    image: ImageArray,
    geometry: PageBoundaryGeometry,
) -> ImageArray:
    if not isinstance(
        image,
        np.ndarray,
    ):
        raise PageBoundaryDetectionError("Image must be a NumPy array")

    if image.dtype != np.uint8:
        raise PageBoundaryDetectionError("Image must use uint8 pixels")

    result = image.copy()

    if geometry.is_full_frame:
        return result

    height, width = image.shape[:2]

    if width < 1 or height < 1:
        return result

    polygon = np.array(
        [
            [
                _pixel_coordinate(
                    point.x,
                    width,
                ),
                _pixel_coordinate(
                    point.y,
                    height,
                ),
            ]
            for point in geometry.points
        ],
        dtype=np.int32,
    )

    mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint8,
    )

    cv2.fillPoly(
        mask,
        [
            polygon,
        ],
        255,
    )

    outside = mask == 0

    if result.ndim == 2:
        result[outside] = 255

        return result

    if result.ndim == 3 and result.shape[2] in (
        3,
        4,
    ):
        result[
            outside,
            :3,
        ] = 255

        if result.shape[2] == 4:
            result[
                outside,
                3,
            ] = 255

        return result

    raise PageBoundaryDetectionError("Image must be grayscale, RGB or RGBA")


def _to_grayscale(
    image: ImageArray,
) -> NDArray[np.uint8]:
    if not isinstance(
        image,
        np.ndarray,
    ):
        raise PageBoundaryDetectionError("Image must be a NumPy array")

    if image.dtype != np.uint8:
        raise PageBoundaryDetectionError("Image must use uint8 pixels")

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
        raise PageBoundaryDetectionError("Image must be grayscale, RGB or RGBA")

    height, width = gray.shape

    if width < 32 or height < 32:
        raise PageBoundaryDetectionError(
            "Image is too small for page boundary detection"
        )

    return gray


def _resize_for_detection(
    gray: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    height, width = gray.shape

    largest_dimension = max(
        width,
        height,
    )

    if largest_dimension <= MAX_DETECTION_DIMENSION:
        return gray

    scale = MAX_DETECTION_DIMENSION / largest_dimension

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


def _candidate_masks(
    gray: NDArray[np.uint8],
) -> tuple[
    NDArray[np.uint8],
    ...,
]:
    _, light_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )

    dark_mask = cv2.bitwise_not(light_mask)

    background_level = _estimate_border_level(gray)

    difference = np.abs(gray.astype(np.int16) - int(background_level)).astype(np.uint8)

    _, difference_mask = cv2.threshold(
        difference,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )

    inverse_difference_mask = cv2.bitwise_not(difference_mask)

    return (
        light_mask,
        dark_mask,
        difference_mask,
        inverse_difference_mask,
    )


def _estimate_border_level(
    gray: NDArray[np.uint8],
) -> float:
    height, width = gray.shape

    band = max(
        2,
        round(
            min(
                width,
                height,
            )
            * 0.025
        ),
    )

    top = gray[
        :band,
        :,
    ].reshape(-1)

    bottom = gray[
        height - band :,
        :,
    ].reshape(-1)

    left = gray[
        :,
        :band,
    ].reshape(-1)

    right = gray[
        :,
        width - band :,
    ].reshape(-1)

    border_pixels = np.concatenate(
        (
            top,
            bottom,
            left,
            right,
        )
    )

    return float(np.median(border_pixels))


def _prepare_candidate_mask(
    mask: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    height, width = mask.shape

    kernel_size = max(
        3,
        round(
            min(
                width,
                height,
            )
            * MORPHOLOGY_KERNEL_RATIO
        ),
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size,
        ),
    )

    closed = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    small_kernel_size = max(
        3,
        kernel_size // 3,
    )

    if small_kernel_size % 2 == 0:
        small_kernel_size += 1

    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            small_kernel_size,
            small_kernel_size,
        ),
    )

    return cv2.morphologyEx(
        closed,
        cv2.MORPH_OPEN,
        small_kernel,
    )


def _best_page_contour(
    mask: NDArray[np.uint8],
    width: int,
    height: int,
) -> tuple[
    NDArray[np.int32] | None,
    float,
]:
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    best_contour = None
    best_score = float("-inf")

    image_area = float(width * height)

    center = (
        width / 2.0,
        height / 2.0,
    )

    for contour in contours:
        area = float(cv2.contourArea(contour))

        if area <= 0:
            continue

        area_ratio = area / image_area

        if area_ratio < MIN_PAGE_AREA_RATIO:
            continue

        x, y, box_width, box_height = cv2.boundingRect(contour)

        del x
        del y

        width_ratio = box_width / width

        height_ratio = box_height / height

        if width_ratio < MIN_PAGE_WIDTH_RATIO or height_ratio < MIN_PAGE_HEIGHT_RATIO:
            continue

        hull = cv2.convexHull(contour)

        hull_area = float(cv2.contourArea(hull))

        if hull_area <= 0:
            continue

        solidity = min(
            1.0,
            area / hull_area,
        )

        box_area = float(box_width * box_height)

        rectangularity = area / box_area if box_area > 0 else 0.0

        center_inside = (
            cv2.pointPolygonTest(
                contour,
                center,
                False,
            )
            >= 0
        )

        score = area_ratio * 2.0 + solidity * 0.55 + rectangularity * 0.45

        if center_inside:
            score += 1.0

        if area_ratio > MAX_PAGE_AREA_RATIO:
            score -= 1.4

        if score > best_score:
            best_contour = contour
            best_score = score

    return (
        best_contour,
        best_score,
    )


def _simplify_contour(
    contour: NDArray[np.int32],
    width: int,
    height: int,
) -> tuple[
    NormalizedPoint,
    ...,
]:
    perimeter = float(
        cv2.arcLength(
            contour,
            True,
        )
    )

    epsilon = max(
        1.0,
        perimeter * CONTOUR_APPROXIMATION_RATIO,
    )

    approximated = cv2.approxPolyDP(
        contour,
        epsilon,
        True,
    )

    raw_points = [
        (
            int(point[0][0]),
            int(point[0][1]),
        )
        for point in approximated
    ]

    if len(raw_points) > MAX_BOUNDARY_POINTS:
        step = len(raw_points) / MAX_BOUNDARY_POINTS

        raw_points = [
            raw_points[
                min(
                    len(raw_points) - 1,
                    int(index * step),
                )
            ]
            for index in range(MAX_BOUNDARY_POINTS)
        ]

    normalized_points: list[NormalizedPoint] = []

    for x, y in raw_points:
        point = NormalizedPoint(
            x=_normalized_coordinate(
                x,
                width,
            ),
            y=_normalized_coordinate(
                y,
                height,
            ),
        )

        if normalized_points and point == normalized_points[-1]:
            continue

        normalized_points.append(point)

    if len(normalized_points) >= 2 and normalized_points[0] == normalized_points[-1]:
        normalized_points.pop()

    return tuple(normalized_points)


def _full_frame_boundary() -> PageBoundaryGeometry:
    return PageBoundaryGeometry(
        points=(
            NormalizedPoint(
                0.0,
                0.0,
            ),
            NormalizedPoint(
                1.0,
                0.0,
            ),
            NormalizedPoint(
                1.0,
                1.0,
            ),
            NormalizedPoint(
                0.0,
                1.0,
            ),
        ),
        is_full_frame=True,
    )


def _normalized_coordinate(
    value: float,
    dimension: int,
) -> float:
    maximum = max(
        1,
        dimension - 1,
    )

    normalized = float(value) / maximum

    return max(
        0.0,
        min(
            1.0,
            normalized,
        ),
    )


def _pixel_coordinate(
    value: float,
    dimension: int,
) -> int:
    maximum = max(
        0,
        dimension - 1,
    )

    return max(
        0,
        min(
            maximum,
            round(value * maximum),
        ),
    )
