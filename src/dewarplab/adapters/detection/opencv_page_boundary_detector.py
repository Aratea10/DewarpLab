import cv2
import numpy as np
from numpy.typing import NDArray
from dewarplab.application import (
    NormalizedPoint,
    PageBoundaryCandidate,
    PageBoundaryCandidateSource,
    PageBoundaryGeometry,
)

ImageArray = NDArray[np.uint8]
MAX_DETECTION_DIMENSION = 1800
FULL_FRAME_BORDER_RATIO = 0.025
FULL_FRAME_INTERIOR_MARGIN_RATIO = 0.12
FULL_FRAME_CORNER_RATIO = 0.08
FULL_FRAME_MIN_PAPER_LEVEL = 95.0
FULL_FRAME_TONE_TOLERANCE_MIN = 24.0
FULL_FRAME_TONE_TOLERANCE_MAX = 42.0
FULL_FRAME_MIN_SIDE_SIMILARITY = 0.56
FULL_FRAME_MIN_CORNER_SIMILARITY = 0.52
FULL_FRAME_MIN_AVERAGE_SIMILARITY = 0.68
FULL_FRAME_MAX_BORDER_DARK_RATIO = 0.24
FULL_FRAME_DARK_OFFSET = 55.0
MIN_PAGE_AREA_RATIO = 0.15
MAX_PAGE_AREA_RATIO = 0.985
MIN_PAGE_WIDTH_RATIO = 0.25
MIN_PAGE_HEIGHT_RATIO = 0.25
MORPHOLOGY_KERNEL_RATIO = 0.008
CONTOUR_APPROXIMATION_RATIO = 0.0015
MIN_SPREAD_ASPECT_RATIO = 1.1
SPINE_SEARCH_START_RATIO = 0.22
SPINE_SEARCH_END_RATIO = 0.78
SPINE_MIN_SIDE_RATIO = 0.18
SPINE_VERTICAL_MARGIN_RATIO = 0.06
SPINE_DARK_OPEN_RATIO = 0.1
SPINE_EDGE_OPEN_RATIO = 0.12
SPINE_GAP_CLOSE_RATIO = 0.025
SPINE_HORIZONTAL_BAND_RATIO = 0.004
SPINE_MIN_COVERAGE_RATIO = 0.12
SPINE_MIN_LONG_RUN_RATIO = 0.1
SPINE_MIN_SCORE = 0.16
SPINE_CENTER_WEIGHT = 0.08
CONTENT_MARGIN_RATIO = 0.035
CONTENT_SPINE_EXCLUSION_RATIO = 0.025
BOUNDARY_SEARCH_BAND_RATIO = 0.12
BOUNDARY_OUTER_MARGIN_RATIO = 0.018
BOUNDARY_SAMPLE_TARGET = 180
BOUNDARY_CURVE_MIN_ERROR_RATIO = 0.008
BOUNDARY_CURVE_MAD_MULTIPLIER = 3.5
BOUNDARY_CURVE_MAX_ITERATIONS = 4
BOUNDARY_CURVE_MIN_INLIER_RATIO = 0.55
BOUNDARY_SPIKE_MEDIAN_WINDOW = 11
BOUNDARY_SPIKE_MIN_DEVIATION_RATIO = (
    0.007
)
BOUNDARY_SPIKE_MAD_MULTIPLIER = 4.0
BOUNDARY_SPIKE_MAX_RUN_RATIO = 0.08
BOUNDARY_SPIKE_MAX_RUN_POINTS = 14
EDGE_TANGENT_SMOOTHING_RATIO = 0.018
EDGE_NORMAL_SMOOTHING_SIZE = 3
EDGE_MIN_RESPONSE = 8.0
EDGE_RELATIVE_THRESHOLD = 0.25
EDGE_MAX_CANDIDATES = 8
EDGE_INNER_PRIORITY_WEIGHT = 0.35
EDGE_COARSE_FALLBACK_SCORE = 0.03
EDGE_CONTINUITY_WEIGHT = 30.0
EDGE_MAX_STEP_RATIO = 0.055
EDGE_MEDIAN_WINDOW = 5
MAX_BOUNDARY_POINTS = 140


class PageBoundaryDetectionError(
    ValueError
):
    pass


def detect_page_boundary_geometry(
    image: ImageArray,
) -> PageBoundaryGeometry:
    return (
        _detect_page_boundary_candidate(
            image
        ).geometry
    )


def _detect_page_boundary_candidate(
    image: ImageArray,
) -> PageBoundaryCandidate:
    gray = _resize_for_detection(
        _to_grayscale(image)
    )

    full_frame_candidate = (
        _generate_full_frame_candidate(
            gray
        )
    )
    if full_frame_candidate is not None:
        return full_frame_candidate

    best_contour = (
        _select_best_page_contour(gray)
    )
    if best_contour is None:
        return (
            _fallback_full_frame_candidate()
        )

    spread_candidate = (
        _generate_spread_candidate(
            gray,
            best_contour,
        )
    )
    if spread_candidate is not None:
        return spread_candidate

    clipped_page_candidate = _generate_clipped_page_candidate(
        gray,
        best_contour,
    )
    if (
        clipped_page_candidate
        is not None
    ):
        return clipped_page_candidate

    single_page_candidate = (
        _generate_single_page_candidate(
            best_contour,
            width=gray.shape[1],
            height=gray.shape[0],
        )
    )
    if (
        single_page_candidate
        is not None
    ):
        return single_page_candidate

    return (
        _fallback_full_frame_candidate()
    )


def _generate_full_frame_candidate(
    gray: NDArray[np.uint8],
) -> PageBoundaryCandidate | None:
    if not _looks_like_full_frame_page(
        gray
    ):
        return None

    return _full_frame_candidate(
        PageBoundaryCandidateSource.FULL_FRAME
    )


def _select_best_page_contour(
    gray: NDArray[np.uint8],
) -> NDArray[np.int32] | None:
    height, width = gray.shape
    best_contour: (
        NDArray[np.int32] | None
    ) = None
    best_score = float("-inf")

    for mask in _candidate_masks(gray):
        prepared = (
            _prepare_candidate_mask(
                mask
            )
        )
        contour, score = (
            _best_page_contour(
                prepared,
                width,
                height,
            )
        )
        if (
            contour is not None
            and score > best_score
        ):
            best_contour = contour
            best_score = score

    return best_contour


def _generate_spread_candidate(
    gray: NDArray[np.uint8],
    contour: NDArray[np.int32],
) -> PageBoundaryCandidate | None:
    geometry = (
        _detect_primary_page_in_spread(
            gray,
            contour,
        )
    )
    if geometry is None:
        return None

    return PageBoundaryCandidate(
        geometry=geometry,
        source=PageBoundaryCandidateSource.SPREAD,
    )


def _generate_clipped_page_candidate(
    gray: NDArray[np.uint8],
    contour: NDArray[np.int32],
) -> PageBoundaryCandidate | None:
    # Reserved for a dedicated clipped-page hypothesis. Keeping it
    # inactive here makes this refactor behavior-preserving while
    # giving clipped pages an independent place to evolve.
    del gray, contour
    return None


def _generate_single_page_candidate(
    contour: NDArray[np.int32],
    *,
    width: int,
    height: int,
) -> PageBoundaryCandidate | None:
    image_area = float(width * height)
    if image_area <= 0:
        return None

    area_ratio = (
        float(cv2.contourArea(contour))
        / image_area
    )
    if (
        area_ratio
        >= MAX_PAGE_AREA_RATIO
    ):
        return None

    points = _simplify_contour(
        contour,
        width,
        height,
    )
    if len(points) < 4:
        return None

    return PageBoundaryCandidate(
        geometry=PageBoundaryGeometry(
            points=points,
            is_full_frame=False,
        ),
        source=(
            PageBoundaryCandidateSource.SINGLE_PAGE
        ),
    )


def _fallback_full_frame_candidate() -> (
    PageBoundaryCandidate
):
    return _full_frame_candidate(
        PageBoundaryCandidateSource.FALLBACK_FULL_FRAME
    )


def _full_frame_candidate(
    source: PageBoundaryCandidateSource,
) -> PageBoundaryCandidate:
    return PageBoundaryCandidate(
        geometry=_full_frame_boundary(),
        source=source,
    )


def mask_image_to_page_boundary(
    image: ImageArray,
    geometry: PageBoundaryGeometry,
) -> ImageArray:
    if not isinstance(
        image, np.ndarray
    ):
        raise PageBoundaryDetectionError(
            "Image must be a NumPy array"
        )
    if image.dtype != np.uint8:
        raise PageBoundaryDetectionError(
            "Image must use uint8 pixels"
        )
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
                    point.x, width
                ),
                _pixel_coordinate(
                    point.y, height
                ),
            ]
            for point in geometry.points
        ],
        dtype=np.int32,
    )
    mask = np.zeros(
        (height, width), dtype=np.uint8
    )
    cv2.fillPoly(mask, [polygon], 255)
    outside = mask == 0
    if result.ndim == 2:
        result[outside] = 255
        return result
    if (
        result.ndim == 3
        and result.shape[2] in (3, 4)
    ):
        result[outside, :3] = 255
        if result.shape[2] == 4:
            result[outside, 3] = 255
        return result
    raise PageBoundaryDetectionError(
        "Image must be grayscale, RGB or RGBA"
    )


def _detect_primary_page_in_spread(
    gray: NDArray[np.uint8],
    contour: NDArray[np.int32],
) -> PageBoundaryGeometry | None:
    height, width = gray.shape
    (
        box_x,
        box_y,
        box_width,
        box_height,
    ) = cv2.boundingRect(contour)
    if (
        box_width < 32
        or box_height < 32
    ):
        return None
    aspect_ratio = box_width / max(
        1, box_height
    )
    if (
        aspect_ratio
        < MIN_SPREAD_ASPECT_RATIO
    ):
        return None
    contour_mask = np.zeros(
        (height, width), dtype=np.uint8
    )
    cv2.drawContours(
        contour_mask,
        [contour],
        -1,
        255,
        thickness=cv2.FILLED,
    )
    spine_x = _detect_spine_x(
        gray,
        contour_mask,
        box_x,
        box_y,
        box_width,
        box_height,
    )
    if spine_x is None:
        return None
    y_end = box_y + box_height - 1
    left_score = _page_content_score(
        gray,
        contour_mask,
        box_x,
        spine_x,
        box_y,
        y_end,
        spine_x,
        "left",
    )
    right_score = _page_content_score(
        gray,
        contour_mask,
        spine_x,
        box_x + box_width - 1,
        box_y,
        y_end,
        spine_x,
        "right",
    )
    if (
        left_score <= 0
        and right_score <= 0
    ):
        return None
    side = (
        "right"
        if right_score >= left_score
        else "left"
    )
    side_mask = _build_side_mask(
        contour_mask, spine_x, side
    )
    points = (
        _trace_selected_page_boundary(
            gray,
            side_mask,
            spine_x,
            side,
        )
    )
    if len(points) < 4:
        return None
    polygon = np.asarray(
        [
            [
                _pixel_coordinate(
                    point.x, width
                ),
                _pixel_coordinate(
                    point.y, height
                ),
            ]
            for point in points
        ],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    area = abs(
        float(cv2.contourArea(polygon))
    )
    if area < width * height * 0.08:
        return None
    return PageBoundaryGeometry(
        points=points,
        is_full_frame=False,
    )


def _detect_spine_x(
    gray: NDArray[np.uint8],
    contour_mask: NDArray[np.uint8],
    box_x: int,
    box_y: int,
    box_width: int,
    box_height: int,
) -> int | None:
    height, width = gray.shape
    search_start = round(
        box_x
        + box_width
        * SPINE_SEARCH_START_RATIO
    )
    search_end = round(
        box_x
        + box_width
        * SPINE_SEARCH_END_RATIO
    )
    minimum_side_width = round(
        box_width * SPINE_MIN_SIDE_RATIO
    )
    search_start = max(
        search_start,
        box_x + minimum_side_width,
        0,
    )
    search_end = min(
        search_end,
        box_x
        + box_width
        - 1
        - minimum_side_width,
        width - 1,
    )
    if search_end <= search_start:
        return None
    y_margin = max(
        2,
        round(
            box_height
            * SPINE_VERTICAL_MARGIN_RATIO
        ),
    )
    y_start = max(0, box_y + y_margin)
    y_end = min(
        height,
        box_y + box_height - y_margin,
    )
    if y_end - y_start < 32:
        return None
    roi = gray[
        y_start:y_end,
        search_start : search_end + 1,
    ]
    roi_mask = contour_mask[
        y_start:y_end,
        search_start : search_end + 1,
    ]
    if roi.size == 0:
        return None
    dark_mask = (
        _persistent_vertical_dark_mask(
            roi=roi, roi_mask=roi_mask
        )
    )
    edge_mask = (
        _persistent_vertical_edge_mask(
            roi=roi, roi_mask=roi_mask
        )
    )
    valid_mask = roi_mask > 0
    _, roi_width = roi.shape
    band_width = max(
        1,
        round(
            box_width
            * SPINE_HORIZONTAL_BAND_RATIO
        ),
    )
    if band_width % 2 == 0:
        band_width += 1
    dark_coverage = (
        _vertical_coverage_profile(
            dark_mask, valid_mask
        )
    )
    edge_coverage = (
        _vertical_coverage_profile(
            edge_mask, valid_mask
        )
    )
    combined = (
        (dark_mask > 0)
        | (edge_mask > 0)
    ) & valid_mask
    longest_runs = np.array(
        [
            _longest_true_run_ratio(
                combined[:, index]
            )
            for index in range(
                roi_width
            )
        ],
        dtype=np.float32,
    )
    if band_width > 1:
        dark_coverage = cv2.blur(
            dark_coverage.reshape(
                1, -1
            ),
            (band_width, 1),
        ).reshape(-1)
        edge_coverage = cv2.blur(
            edge_coverage.reshape(
                1, -1
            ),
            (band_width, 1),
        ).reshape(-1)
        longest_runs = cv2.blur(
            longest_runs.reshape(1, -1),
            (band_width, 1),
        ).reshape(-1)
    center = (roi_width - 1) / 2.0
    if center > 0:
        center_proximity = (
            1.0
            - np.abs(
                np.arange(
                    roi_width,
                    dtype=np.float32,
                )
                - center
            )
            / center
        )
    else:
        center_proximity = np.ones(
            roi_width, dtype=np.float32
        )
    scores = (
        dark_coverage * 0.45
        + edge_coverage * 0.3
        + longest_runs * 0.35
        + center_proximity
        * SPINE_CENTER_WEIGHT
    )
    plausible = (
        (
            dark_coverage
            >= SPINE_MIN_COVERAGE_RATIO
        )
        | (
            edge_coverage
            >= SPINE_MIN_COVERAGE_RATIO
        )
    ) & (
        longest_runs
        >= SPINE_MIN_LONG_RUN_RATIO
    )
    if not np.any(plausible):
        return None
    masked_scores = np.where(
        plausible, scores, -np.inf
    )
    local_index = int(
        np.argmax(masked_scores)
    )
    best_score = float(
        masked_scores[local_index]
    )
    if (
        not np.isfinite(best_score)
        or best_score < SPINE_MIN_SCORE
    ):
        return None
    return search_start + local_index


def _persistent_vertical_dark_mask(
    roi: NDArray[np.uint8],
    roi_mask: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    masked_roi = roi.copy()
    masked_roi[roi_mask == 0] = 255
    _, dark_mask = cv2.threshold(
        masked_roi,
        0,
        255,
        cv2.THRESH_BINARY_INV
        | cv2.THRESH_OTSU,
    )
    return _keep_persistent_vertical_structure(
        dark_mask,
        open_ratio=SPINE_DARK_OPEN_RATIO,
    )


def _persistent_vertical_edge_mask(
    roi: NDArray[np.uint8],
    roi_mask: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    gradient_x = np.abs(
        cv2.Sobel(
            roi.astype(np.float32),
            cv2.CV_32F,
            1,
            0,
            ksize=3,
        )
    )
    valid_values = gradient_x[
        roi_mask > 0
    ]
    if valid_values.size == 0:
        return np.zeros_like(
            roi, dtype=np.uint8
        )
    threshold = max(
        8.0,
        float(
            np.percentile(
                valid_values, 82
            )
        ),
    )
    edge_mask = np.zeros_like(
        roi, dtype=np.uint8
    )
    edge_mask[
        (gradient_x >= threshold)
        & (roi_mask > 0)
    ] = 255
    return _keep_persistent_vertical_structure(
        edge_mask,
        open_ratio=SPINE_EDGE_OPEN_RATIO,
    )


def _keep_persistent_vertical_structure(
    binary: NDArray[np.uint8],
    open_ratio: float,
) -> NDArray[np.uint8]:
    height, _ = binary.shape
    open_length = max(
        9, round(height * open_ratio)
    )
    close_length = max(
        3,
        round(
            height
            * SPINE_GAP_CLOSE_RATIO
        ),
    )
    open_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, open_length),
        )
    )
    close_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, close_length),
        )
    )
    persistent = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        open_kernel,
    )
    return cv2.morphologyEx(
        persistent,
        cv2.MORPH_CLOSE,
        close_kernel,
    )


def _vertical_coverage_profile(
    binary: NDArray[np.uint8],
    valid_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    active = (binary > 0) & valid_mask
    valid_counts = np.count_nonzero(
        valid_mask, axis=0
    )
    active_counts = np.count_nonzero(
        active, axis=0
    )
    coverage = np.zeros(
        binary.shape[1],
        dtype=np.float32,
    )
    valid_columns = valid_counts > 0
    coverage[valid_columns] = (
        active_counts[valid_columns]
        / valid_counts[valid_columns]
    )
    return coverage


def _longest_true_run_ratio(
    values: NDArray[np.bool_],
) -> float:
    if values.size == 0:
        return 0.0
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(
                longest, current
            )
        else:
            current = 0
    return longest / len(values)


def _page_content_score(
    gray: NDArray[np.uint8],
    contour_mask: NDArray[np.uint8],
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
    spine_x: int,
    side: str,
) -> float:
    height, width = gray.shape
    x_start = max(
        0, min(width - 1, x_start)
    )
    x_end = max(
        0, min(width - 1, x_end)
    )
    y_start = max(
        0, min(height - 1, y_start)
    )
    y_end = max(
        0, min(height - 1, y_end)
    )
    if (
        x_end <= x_start
        or y_end <= y_start
    ):
        return 0.0
    side_width = x_end - x_start + 1
    side_height = y_end - y_start + 1
    x_margin = max(
        2,
        round(
            side_width
            * CONTENT_MARGIN_RATIO
        ),
    )
    y_margin = max(
        2,
        round(
            side_height
            * CONTENT_MARGIN_RATIO
        ),
    )
    spine_margin = max(
        2,
        round(
            side_width
            * CONTENT_SPINE_EXCLUSION_RATIO
        ),
    )
    inner_x_start = x_start + x_margin
    inner_x_end = x_end - x_margin
    if side == "left":
        inner_x_end = min(
            inner_x_end,
            spine_x - spine_margin,
        )
    else:
        inner_x_start = max(
            inner_x_start,
            spine_x + spine_margin,
        )
    inner_y_start = y_start + y_margin
    inner_y_end = y_end - y_margin
    if (
        inner_x_end <= inner_x_start
        or inner_y_end <= inner_y_start
    ):
        return 0.0
    roi = gray[
        inner_y_start : inner_y_end + 1,
        inner_x_start : inner_x_end + 1,
    ]
    roi_mask = (
        contour_mask[
            inner_y_start : inner_y_end
            + 1,
            inner_x_start : inner_x_end
            + 1,
        ]
        > 0
    )
    if np.count_nonzero(roi_mask) < 100:
        return 0.0
    paper_values = roi[roi_mask]
    paper_level = float(
        np.percentile(paper_values, 72)
    )
    ink_threshold = max(
        0.0, paper_level - 22.0
    )
    ink_pixels = (
        roi.astype(np.float32)
        < ink_threshold
    ) & roi_mask
    ink_ratio = np.count_nonzero(
        ink_pixels
    ) / max(
        1, np.count_nonzero(roi_mask)
    )
    roi_float = roi.astype(np.float32)
    gradient_x = cv2.Sobel(
        roi_float,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )
    gradient_y = cv2.Sobel(
        roi_float,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )
    gradient = np.hypot(
        gradient_x, gradient_y
    )
    gradient_score = float(
        np.percentile(
            gradient[roi_mask], 72
        )
    )
    occupancy = (
        np.count_nonzero(roi_mask)
        / roi_mask.size
    )
    return (
        ink_ratio * 8.0
        + gradient_score / 255.0
        + occupancy * 0.25
    )


def _build_side_mask(
    contour_mask: NDArray[np.uint8],
    spine_x: int,
    side: str,
) -> NDArray[np.uint8]:
    _, width = contour_mask.shape
    result = contour_mask.copy()
    spine_x = max(
        0, min(width - 1, spine_x)
    )
    if side == "left":
        result[:, spine_x + 1 :] = 0
    elif side == "right":
        result[:, :spine_x] = 0
    else:
        raise ValueError(
            f"Unknown page side: {side}"
        )
    return result


def _trace_selected_page_boundary(
    gray: NDArray[np.uint8],
    page_mask: NDArray[np.uint8],
    spine_x: int,
    side: str,
) -> tuple[NormalizedPoint, ...]:
    height, width = gray.shape
    ys, xs = np.nonzero(page_mask)
    if xs.size == 0 or ys.size == 0:
        return ()
    x_start = int(np.min(xs))
    x_end = int(np.max(xs))
    y_start = int(np.min(ys))
    y_end = int(np.max(ys))
    if (
        x_end - x_start < 12
        or y_end - y_start < 12
    ):
        return ()
    top, bottom, left, right = (
        _extract_coarse_profiles(
            page_mask,
            x_start,
            x_end,
            y_start,
            y_end,
        )
    )
    (
        horizontal_response,
        vertical_response,
    ) = _edge_responses(gray)
    top_points = _trace_horizontal_side(
        horizontal_response,
        top,
        x_start,
        x_end,
        height,
        "top",
    )
    bottom_points = (
        _trace_horizontal_side(
            horizontal_response,
            bottom,
            x_start,
            x_end,
            height,
            "bottom",
        )
    )
    if side == "right":
        left_points = (
            _spine_side_points(
                page_mask,
                spine_x,
                y_start,
                y_end,
            )
        )
        right_points = (
            _trace_vertical_side(
                vertical_response,
                right,
                y_start,
                y_end,
                width,
                "right",
            )
        )
    else:
        left_points = (
            _trace_vertical_side(
                vertical_response,
                left,
                y_start,
                y_end,
                width,
                "left",
            )
        )
        right_points = (
            _spine_side_points(
                page_mask,
                spine_x,
                y_start,
                y_end,
            )
        )
    top_points = _repair_boundary_curve(
        points=top_points,
        dependent_dimension=height,
        horizontal=True,
    )
    bottom_points = (
        _repair_boundary_curve(
            points=bottom_points,
            dependent_dimension=height,
            horizontal=True,
        )
    )
    left_points = (
        _repair_boundary_curve(
            points=left_points,
            dependent_dimension=width,
            horizontal=False,
        )
    )
    right_points = (
        _repair_boundary_curve(
            points=right_points,
            dependent_dimension=width,
            horizontal=False,
        )
    )
    top_points = _remove_short_boundary_excursions(
        points=top_points,
        dependent_dimension=height,
        horizontal=True,
    )
    bottom_points = _remove_short_boundary_excursions(
        points=bottom_points,
        dependent_dimension=height,
        horizontal=True,
    )
    left_points = _remove_short_boundary_excursions(
        points=left_points,
        dependent_dimension=width,
        horizontal=False,
    )
    right_points = _remove_short_boundary_excursions(
        points=right_points,
        dependent_dimension=width,
        horizontal=False,
    )
    (
        top_points,
        bottom_points,
        left_points,
        right_points,
    ) = _reconcile_boundary_corners(
        top_points=top_points,
        bottom_points=bottom_points,
        left_points=left_points,
        right_points=right_points,
        side=side,
    )
    if (
        min(
            len(top_points),
            len(bottom_points),
            len(left_points),
            len(right_points),
        )
        < 2
    ):
        return ()
    polygon_points = (
        top_points
        + right_points[1:]
        + list(reversed(bottom_points))[
            1:
        ]
        + list(reversed(left_points))[
            1:
        ]
    )
    return _simplify_pixel_polygon(
        polygon_points, width, height
    )


def _repair_boundary_curve(
    points: list[tuple[int, int]],
    dependent_dimension: int,
    horizontal: bool,
) -> list[tuple[int, int]]:
    if len(points) < 8:
        return points
    if horizontal:
        independent = np.asarray(
            [x for x, _ in points],
            dtype=np.float64,
        )
        dependent = np.asarray(
            [y for _, y in points],
            dtype=np.float64,
        )
    else:
        independent = np.asarray(
            [y for _, y in points],
            dtype=np.float64,
        )
        dependent = np.asarray(
            [x for x, _ in points],
            dtype=np.float64,
        )
    independent_span = float(
        np.ptp(independent)
    )
    if independent_span <= 0:
        return points
    normalized_independent = (
        independent - independent[0]
    ) / independent_span
    inliers = np.ones(
        len(points), dtype=bool
    )
    fitted = dependent.copy()
    for _ in range(
        BOUNDARY_CURVE_MAX_ITERATIONS
    ):
        inlier_count = int(
            np.count_nonzero(inliers)
        )
        if inlier_count < 4:
            break
        degree = (
            2
            if inlier_count >= 6
            else 1
        )
        coefficients = np.polyfit(
            normalized_independent[
                inliers
            ],
            dependent[inliers],
            degree,
        )
        fitted = np.polyval(
            coefficients,
            normalized_independent,
        )
        residuals = np.abs(
            dependent - fitted
        )
        inlier_residuals = residuals[
            inliers
        ]
        median_residual = float(
            np.median(inlier_residuals)
        )
        mad = float(
            np.median(
                np.abs(
                    inlier_residuals
                    - median_residual
                )
            )
        )
        robust_scale = max(
            1.0, mad * 1.4826
        )
        minimum_error = max(
            2.0,
            dependent_dimension
            * BOUNDARY_CURVE_MIN_ERROR_RATIO,
        )
        threshold = max(
            minimum_error,
            robust_scale
            * BOUNDARY_CURVE_MAD_MULTIPLIER,
        )
        new_inliers = (
            residuals <= threshold
        )
        minimum_inliers = max(
            4,
            round(
                len(points)
                * BOUNDARY_CURVE_MIN_INLIER_RATIO
            ),
        )
        if (
            np.count_nonzero(
                new_inliers
            )
            < minimum_inliers
        ):
            break
        if np.array_equal(
            new_inliers, inliers
        ):
            inliers = new_inliers
            break
        inliers = new_inliers
    if np.all(inliers):
        return points
    repaired = dependent.copy()
    repaired[~inliers] = fitted[
        ~inliers
    ]
    repaired = _smooth_repaired_curve(
        repaired
    )
    if horizontal:
        return [
            (
                int(
                    round(
                        independent[
                            index
                        ]
                    )
                ),
                int(
                    round(
                        repaired[index]
                    )
                ),
            )
            for index in range(
                len(points)
            )
        ]
    return [
        (
            int(round(repaired[index])),
            int(
                round(
                    independent[index]
                )
            ),
        )
        for index in range(len(points))
    ]


def _smooth_repaired_curve(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    if len(values) < 5:
        return values
    result = values.copy()
    padded = np.pad(
        values, (2, 2), mode="edge"
    )
    for index in range(len(values)):
        window = padded[
            index : index + 5
        ]
        result[index] = (
            values[index] * 0.5
            + np.median(window) * 0.5
        )
    return result


def _remove_short_boundary_excursions(
    points: list[tuple[int, int]],
    dependent_dimension: int,
    horizontal: bool,
) -> list[tuple[int, int]]:
    if (
        len(points)
        < BOUNDARY_SPIKE_MEDIAN_WINDOW
    ):
        return points
    if horizontal:
        independent = np.asarray(
            [x for x, _ in points],
            dtype=np.float64,
        )
        dependent = np.asarray(
            [y for _, y in points],
            dtype=np.float64,
        )
    else:
        independent = np.asarray(
            [y for _, y in points],
            dtype=np.float64,
        )
        dependent = np.asarray(
            [x for x, _ in points],
            dtype=np.float64,
        )
    window = (
        BOUNDARY_SPIKE_MEDIAN_WINDOW
    )
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(
        dependent,
        (radius, radius),
        mode="edge",
    )
    local_baseline = np.empty_like(
        dependent
    )
    for index in range(len(dependent)):
        local_baseline[index] = (
            np.median(
                padded[
                    index : index
                    + window
                ]
            )
        )
    residuals = np.abs(
        dependent - local_baseline
    )
    median_residual = float(
        np.median(residuals)
    )
    mad = float(
        np.median(
            np.abs(
                residuals
                - median_residual
            )
        )
    )
    robust_scale = max(
        1.0, mad * 1.4826
    )
    minimum_deviation = max(
        2.0,
        dependent_dimension
        * BOUNDARY_SPIKE_MIN_DEVIATION_RATIO,
    )
    threshold = max(
        minimum_deviation,
        robust_scale
        * BOUNDARY_SPIKE_MAD_MULTIPLIER,
    )
    excursions = residuals > threshold
    runs = _true_runs(excursions)
    maximum_run = min(
        BOUNDARY_SPIKE_MAX_RUN_POINTS,
        max(
            1,
            round(
                len(points)
                * BOUNDARY_SPIKE_MAX_RUN_RATIO
            ),
        ),
    )
    repaired = dependent.copy()
    for run_start, run_end in runs:
        run_length = (
            run_end - run_start + 1
        )
        if run_length > maximum_run:
            continue
        if (
            run_start == 0
            or run_end
            >= len(points) - 1
        ):
            continue
        left_index = run_start - 1
        right_index = run_end + 1
        independent_span = (
            independent[right_index]
            - independent[left_index]
        )
        for index in range(
            run_start, run_end + 1
        ):
            if (
                abs(independent_span)
                > 1e-09
            ):
                position = (
                    independent[index]
                    - independent[
                        left_index
                    ]
                ) / independent_span
            else:
                position = (
                    index - left_index
                ) / (
                    right_index
                    - left_index
                )
            repaired[index] = dependent[
                left_index
            ] + position * (
                dependent[right_index]
                - dependent[left_index]
            )
    if horizontal:
        return [
            (
                int(
                    round(
                        independent[
                            index
                        ]
                    )
                ),
                int(
                    round(
                        repaired[index]
                    )
                ),
            )
            for index in range(
                len(points)
            )
        ]
    return [
        (
            int(round(repaired[index])),
            int(
                round(
                    independent[index]
                )
            ),
        )
        for index in range(len(points))
    ]


def _reconcile_boundary_corners(
    top_points: list[tuple[int, int]],
    bottom_points: list[
        tuple[int, int]
    ],
    left_points: list[tuple[int, int]],
    right_points: list[tuple[int, int]],
    side: str,
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[int, int]],
]:
    if not (
        top_points
        and bottom_points
        and left_points
        and right_points
    ):
        return (
            top_points,
            bottom_points,
            left_points,
            right_points,
        )
    top = list(top_points)
    bottom = list(bottom_points)
    left = list(left_points)
    right = list(right_points)
    top_left = _average_corner(
        top[0], left[0]
    )
    top_right = _average_corner(
        top[-1], right[0]
    )
    bottom_left = _average_corner(
        bottom[0], left[-1]
    )
    bottom_right = _average_corner(
        bottom[-1], right[-1]
    )
    if side == "right":
        top_left = (
            left[0][0],
            top_left[1],
        )
        bottom_left = (
            left[-1][0],
            bottom_left[1],
        )
    elif side == "left":
        top_right = (
            right[0][0],
            top_right[1],
        )
        bottom_right = (
            right[-1][0],
            bottom_right[1],
        )
    top[0] = top_left
    left[0] = top_left
    top[-1] = top_right
    right[0] = top_right
    bottom[0] = bottom_left
    left[-1] = bottom_left
    bottom[-1] = bottom_right
    right[-1] = bottom_right
    return (top, bottom, left, right)


def _average_corner(
    first: tuple[int, int],
    second: tuple[int, int],
) -> tuple[int, int]:
    return (
        round(
            (first[0] + second[0]) / 2
        ),
        round(
            (first[1] + second[1]) / 2
        ),
    )


def _spine_side_points(
    page_mask: NDArray[np.uint8],
    spine_x: int,
    y_start: int,
    y_end: int,
) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for y in _sample_coordinates(
        y_start, y_end
    ):
        xs = np.flatnonzero(
            page_mask[y, :]
        )
        if xs.size == 0:
            points.append((spine_x, y))
            continue
        index = int(
            np.argmin(
                np.abs(xs - spine_x)
            )
        )
        points.append(
            (int(xs[index]), y)
        )
    return points


def _extract_coarse_profiles(
    page_mask: NDArray[np.uint8],
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    height, width = page_mask.shape
    top = np.full(
        width, np.nan, dtype=np.float32
    )
    bottom = np.full(
        width, np.nan, dtype=np.float32
    )
    for x in range(x_start, x_end + 1):
        ys = np.flatnonzero(
            page_mask[:, x]
        )
        if ys.size:
            top[x] = float(ys[0])
            bottom[x] = float(ys[-1])
    left = np.full(
        height, np.nan, dtype=np.float32
    )
    right = np.full(
        height, np.nan, dtype=np.float32
    )
    for y in range(y_start, y_end + 1):
        xs = np.flatnonzero(
            page_mask[y, :]
        )
        if xs.size:
            left[y] = float(xs[0])
            right[y] = float(xs[-1])
    return (
        _fill_profile_gaps(top),
        _fill_profile_gaps(bottom),
        _fill_profile_gaps(left),
        _fill_profile_gaps(right),
    )


def _fill_profile_gaps(
    profile: NDArray[np.float32],
) -> NDArray[np.float32]:
    result = profile.copy()
    valid = np.flatnonzero(
        ~np.isnan(result)
    )
    if valid.size == 0:
        return result
    missing = np.flatnonzero(
        np.isnan(result)
    )
    if missing.size:
        result[missing] = np.interp(
            missing,
            valid,
            result[valid],
        )
    return result


def _edge_responses(
    gray: NDArray[np.uint8],
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
]:
    gray_float = gray.astype(np.float32)
    gradient_x = np.abs(
        cv2.Sobel(
            gray_float,
            cv2.CV_32F,
            1,
            0,
            ksize=3,
        )
    )
    gradient_y = np.abs(
        cv2.Sobel(
            gray_float,
            cv2.CV_32F,
            0,
            1,
            ksize=3,
        )
    )
    height, width = gray.shape
    horizontal_kernel = _odd_kernel_size(
        width,
        EDGE_TANGENT_SMOOTHING_RATIO,
    )
    vertical_kernel = _odd_kernel_size(
        height,
        EDGE_TANGENT_SMOOTHING_RATIO,
    )
    horizontal = cv2.blur(
        gradient_y,
        (
            horizontal_kernel,
            EDGE_NORMAL_SMOOTHING_SIZE,
        ),
    )
    vertical = cv2.blur(
        gradient_x,
        (
            EDGE_NORMAL_SMOOTHING_SIZE,
            vertical_kernel,
        ),
    )
    return (horizontal, vertical)


def _trace_horizontal_side(
    response: NDArray[np.float32],
    coarse_profile: NDArray[np.float32],
    coordinate_start: int,
    coordinate_end: int,
    normal_dimension: int,
    side: str,
) -> list[tuple[int, int]]:
    coordinates = _sample_coordinates(
        coordinate_start, coordinate_end
    )
    candidate_sets: list[
        list[tuple[int, float]]
    ] = []
    for x in coordinates:
        coarse_value = coarse_profile[x]
        if np.isnan(coarse_value):
            candidate_sets.append([])
            continue
        candidate_sets.append(
            _edge_candidates_from_inside(
                response[:, x],
                int(
                    round(
                        float(
                            coarse_value
                        )
                    )
                ),
                normal_dimension,
                side,
            )
        )
    positions = _best_continuous_path(
        candidate_sets, normal_dimension
    )
    return list(
        zip(
            coordinates,
            positions,
            strict=True,
        )
    )


def _trace_vertical_side(
    response: NDArray[np.float32],
    coarse_profile: NDArray[np.float32],
    coordinate_start: int,
    coordinate_end: int,
    normal_dimension: int,
    side: str,
) -> list[tuple[int, int]]:
    coordinates = _sample_coordinates(
        coordinate_start, coordinate_end
    )
    candidate_sets: list[
        list[tuple[int, float]]
    ] = []
    for y in coordinates:
        coarse_value = coarse_profile[y]
        if np.isnan(coarse_value):
            candidate_sets.append([])
            continue
        candidate_sets.append(
            _edge_candidates_from_inside(
                response[y, :],
                int(
                    round(
                        float(
                            coarse_value
                        )
                    )
                ),
                normal_dimension,
                side,
            )
        )
    positions = _best_continuous_path(
        candidate_sets, normal_dimension
    )
    return [
        (position, y)
        for y, position in zip(
            coordinates,
            positions,
            strict=True,
        )
    ]


def _edge_candidates_from_inside(
    values: NDArray[np.float32],
    coarse_position: int,
    dimension: int,
    side: str,
) -> list[tuple[int, float]]:
    search_band = max(
        8,
        round(
            dimension
            * BOUNDARY_SEARCH_BAND_RATIO
        ),
    )
    outer_margin = max(
        2,
        round(
            dimension
            * BOUNDARY_OUTER_MARGIN_RATIO
        ),
    )
    if side in ("bottom", "right"):
        start = max(
            0,
            coarse_position
            - search_band,
        )
        end = min(
            dimension - 1,
            coarse_position
            + outer_margin,
        )
        positions = np.arange(
            start,
            end + 1,
            dtype=np.int32,
        )
    elif side in ("top", "left"):
        start = min(
            dimension - 1,
            coarse_position
            + search_band,
        )
        end = max(
            0,
            coarse_position
            - outer_margin,
        )
        positions = np.arange(
            start,
            end - 1,
            -1,
            dtype=np.int32,
        )
    else:
        raise ValueError(
            f"Unknown boundary side: {side}"
        )
    if positions.size == 0:
        return [
            (
                coarse_position,
                EDGE_COARSE_FALLBACK_SCORE,
            )
        ]
    strengths = values[positions]
    finite = strengths[
        np.isfinite(strengths)
    ]
    if finite.size == 0:
        return [
            (
                coarse_position,
                EDGE_COARSE_FALLBACK_SCORE,
            )
        ]
    maximum = float(np.max(finite))
    if maximum <= 0:
        return [
            (
                coarse_position,
                EDGE_COARSE_FALLBACK_SCORE,
            )
        ]
    threshold = max(
        EDGE_MIN_RESPONSE,
        maximum
        * EDGE_RELATIVE_THRESHOLD,
    )
    runs = _true_runs(
        strengths >= threshold
    )
    candidates: list[
        tuple[int, float]
    ] = []
    denominator = max(
        1, len(positions) - 1
    )
    for run_start, run_end in runs:
        run = strengths[
            run_start : run_end + 1
        ]
        index = run_start + int(
            np.argmax(run)
        )
        position = int(positions[index])
        strength_score = (
            float(strengths[index])
            / maximum
        )
        inside_priority = (
            1.0 - index / denominator
        )
        score = (
            strength_score
            + inside_priority
            * EDGE_INNER_PRIORITY_WEIGHT
        )
        candidates.append(
            (position, score)
        )
        if (
            len(candidates)
            >= EDGE_MAX_CANDIDATES
        ):
            break
    if not candidates:
        return [
            (
                coarse_position,
                EDGE_COARSE_FALLBACK_SCORE,
            )
        ]
    if all(
        (
            position != coarse_position
            for position, _ in candidates
        )
    ):
        candidates.append(
            (
                coarse_position,
                EDGE_COARSE_FALLBACK_SCORE,
            )
        )
    return candidates


def _true_runs(
    values: NDArray[np.bool_],
) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index, value in enumerate(
        values
    ):
        if (
            bool(value)
            and run_start is None
        ):
            run_start = index
        elif (
            not bool(value)
            and run_start is not None
        ):
            runs.append(
                (run_start, index - 1)
            )
            run_start = None
    if run_start is not None:
        runs.append(
            (run_start, len(values) - 1)
        )
    return runs


def _best_continuous_path(
    candidate_sets: list[
        list[tuple[int, float]]
    ],
    dimension: int,
) -> list[int]:
    if not candidate_sets:
        return []
    normalized: list[
        list[tuple[int, float]]
    ] = []
    fallback_position = dimension // 2
    for candidates in candidate_sets:
        if candidates:
            normalized.append(
                candidates
            )
            fallback_position = (
                candidates[0][0]
            )
        else:
            normalized.append(
                [
                    (
                        fallback_position,
                        EDGE_COARSE_FALLBACK_SCORE,
                    )
                ]
            )
    scores: list[list[float]] = [
        [
            score
            for _, score in normalized[
                0
            ]
        ]
    ]
    parents: list[list[int]] = [
        [-1 for _ in normalized[0]]
    ]
    maximum_step = max(
        3.0,
        dimension * EDGE_MAX_STEP_RATIO,
    )
    for index in range(
        1, len(normalized)
    ):
        previous = normalized[index - 1]
        current = normalized[index]
        current_scores: list[float] = []
        current_parents: list[int] = []
        for (
            current_position,
            current_score,
        ) in current:
            best_score = float("-inf")
            best_parent = 0
            for previous_index, (
                previous_position,
                _,
            ) in enumerate(previous):
                delta = abs(
                    current_position
                    - previous_position
                )
                penalty = (
                    EDGE_CONTINUITY_WEIGHT
                    * delta
                    / max(1, dimension)
                )
                if delta > maximum_step:
                    penalty += 2.0
                score = (
                    scores[index - 1][
                        previous_index
                    ]
                    + current_score
                    - penalty
                )
                if score > best_score:
                    best_score = score
                    best_parent = (
                        previous_index
                    )
            current_scores.append(
                best_score
            )
            current_parents.append(
                best_parent
            )
        scores.append(current_scores)
        parents.append(current_parents)
    last_index = int(
        np.argmax(
            np.asarray(
                scores[-1],
                dtype=np.float32,
            )
        )
    )
    path = [0 for _ in normalized]
    for index in range(
        len(normalized) - 1, -1, -1
    ):
        path[index] = normalized[index][
            last_index
        ][0]
        parent_index = parents[index][
            last_index
        ]
        if parent_index < 0:
            break
        last_index = parent_index
    return _median_smooth_path(path)


def _median_smooth_path(
    path: list[int],
) -> list[int]:
    if len(path) < EDGE_MEDIAN_WINDOW:
        return path
    radius = EDGE_MEDIAN_WINDOW // 2
    values = np.asarray(
        path, dtype=np.float32
    )
    padded = np.pad(
        values,
        (radius, radius),
        mode="edge",
    )
    smoothed = np.empty_like(values)
    for index in range(len(values)):
        smoothed[index] = np.median(
            padded[
                index : index
                + EDGE_MEDIAN_WINDOW
            ]
        )
    return [
        int(round(value))
        for value in smoothed
    ]


def _sample_coordinates(
    start: int, end: int
) -> list[int]:
    length = max(1, end - start + 1)
    step = max(
        2,
        round(
            length
            / BOUNDARY_SAMPLE_TARGET
        ),
    )
    coordinates = list(
        range(start, end + 1, step)
    )
    if (
        not coordinates
        or coordinates[-1] != end
    ):
        coordinates.append(end)
    return coordinates


def _simplify_pixel_polygon(
    points: list[tuple[int, int]],
    width: int,
    height: int,
) -> tuple[NormalizedPoint, ...]:
    if len(points) < 4:
        return ()
    polygon = np.asarray(
        points, dtype=np.int32
    ).reshape(-1, 1, 2)
    perimeter = float(
        cv2.arcLength(polygon, True)
    )
    approximated = cv2.approxPolyDP(
        polygon,
        max(1.0, perimeter * 0.0009),
        True,
    )
    raw_points = [
        (
            int(point[0][0]),
            int(point[0][1]),
        )
        for point in approximated
    ]
    return _normalize_raw_points(
        _limit_raw_points(raw_points),
        width,
        height,
    )


def _to_grayscale(
    image: ImageArray,
) -> NDArray[np.uint8]:
    if not isinstance(
        image, np.ndarray
    ):
        raise PageBoundaryDetectionError(
            "Image must be a NumPy array"
        )
    if image.dtype != np.uint8:
        raise PageBoundaryDetectionError(
            "Image must use uint8 pixels"
        )
    if image.ndim == 2:
        gray = image
    elif (
        image.ndim == 3
        and image.shape[2] == 3
    ):
        gray = cv2.cvtColor(
            image, cv2.COLOR_RGB2GRAY
        )
    elif (
        image.ndim == 3
        and image.shape[2] == 4
    ):
        gray = cv2.cvtColor(
            image, cv2.COLOR_RGBA2GRAY
        )
    else:
        raise PageBoundaryDetectionError(
            "Image must be grayscale, RGB or RGBA"
        )
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
        width, height
    )
    if (
        largest_dimension
        <= MAX_DETECTION_DIMENSION
    ):
        return gray
    scale = (
        MAX_DETECTION_DIMENSION
        / largest_dimension
    )
    target_width = max(
        1, round(width * scale)
    )
    target_height = max(
        1, round(height * scale)
    )
    return cv2.resize(
        gray,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )


def _looks_like_full_frame_page(
    gray: NDArray[np.uint8],
) -> bool:
    height, width = gray.shape
    if width < 64 or height < 64:
        return False
    border_band = max(
        3,
        round(
            min(width, height)
            * FULL_FRAME_BORDER_RATIO
        ),
    )
    x_margin = max(
        border_band * 2,
        round(
            width
            * FULL_FRAME_INTERIOR_MARGIN_RATIO
        ),
    )
    y_margin = max(
        border_band * 2,
        round(
            height
            * FULL_FRAME_INTERIOR_MARGIN_RATIO
        ),
    )
    if (
        width - 2 * x_margin < 16
        or height - 2 * y_margin < 16
    ):
        return False
    interior = gray[
        y_margin : height - y_margin,
        x_margin : width - x_margin,
    ]
    if interior.size == 0:
        return False
    paper_level = float(
        np.percentile(interior, 78)
    )
    if (
        paper_level
        < FULL_FRAME_MIN_PAPER_LEVEL
    ):
        return False
    interior_mid = float(
        np.percentile(interior, 55)
    )
    interior_high = float(
        np.percentile(interior, 90)
    )
    tolerance = (
        18.0
        + max(
            0.0,
            interior_high
            - interior_mid,
        )
        * 1.8
    )
    tolerance = max(
        FULL_FRAME_TONE_TOLERANCE_MIN,
        min(
            FULL_FRAME_TONE_TOLERANCE_MAX,
            tolerance,
        ),
    )
    top = gray[:border_band, :]
    bottom = gray[
        height - border_band :, :
    ]
    left = gray[:, :border_band]
    right = gray[
        :, width - border_band :
    ]
    side_scores = [
        _paper_similarity(
            top, paper_level, tolerance
        ),
        _paper_similarity(
            bottom,
            paper_level,
            tolerance,
        ),
        _paper_similarity(
            left, paper_level, tolerance
        ),
        _paper_similarity(
            right,
            paper_level,
            tolerance,
        ),
    ]
    corner_width = max(
        border_band,
        round(
            width
            * FULL_FRAME_CORNER_RATIO
        ),
    )
    corner_height = max(
        border_band,
        round(
            height
            * FULL_FRAME_CORNER_RATIO
        ),
    )
    corner_width = min(
        width, corner_width
    )
    corner_height = min(
        height, corner_height
    )
    corners = (
        gray[
            :corner_height,
            :corner_width,
        ],
        gray[
            :corner_height,
            width - corner_width :,
        ],
        gray[
            height - corner_height :,
            :corner_width,
        ],
        gray[
            height - corner_height :,
            width - corner_width :,
        ],
    )
    corner_scores = [
        _paper_similarity(
            corner,
            paper_level,
            tolerance,
        )
        for corner in corners
    ]
    border_pixels = np.concatenate(
        (
            top.reshape(-1),
            bottom.reshape(-1),
            left.reshape(-1),
            right.reshape(-1),
        )
    ).astype(np.float32)
    dark_threshold = max(
        0.0,
        paper_level
        - FULL_FRAME_DARK_OFFSET,
    )
    border_dark_ratio = float(
        np.count_nonzero(
            border_pixels
            < dark_threshold
        )
        / max(1, border_pixels.size)
    )
    good_sides = sum(
        (
            score
            >= FULL_FRAME_MIN_SIDE_SIMILARITY
            for score in side_scores
        )
    )
    good_corners = sum(
        (
            score
            >= FULL_FRAME_MIN_CORNER_SIMILARITY
            for score in corner_scores
        )
    )
    average_similarity = float(
        np.mean(
            side_scores + corner_scores
        )
    )
    return (
        good_sides >= 3
        and good_corners >= 3
        and (
            average_similarity
            >= FULL_FRAME_MIN_AVERAGE_SIMILARITY
        )
        and (
            border_dark_ratio
            <= FULL_FRAME_MAX_BORDER_DARK_RATIO
        )
    )


def _paper_similarity(
    region: NDArray[np.uint8],
    paper_level: float,
    tolerance: float,
) -> float:
    if region.size == 0:
        return 0.0
    values = region.astype(np.float32)
    similar = (
        np.abs(values - paper_level)
        <= tolerance
    )
    return float(
        np.count_nonzero(similar)
        / region.size
    )


def _candidate_masks(
    gray: NDArray[np.uint8],
) -> tuple[NDArray[np.uint8], ...]:
    _, light_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY
        | cv2.THRESH_OTSU,
    )
    dark_mask = cv2.bitwise_not(
        light_mask
    )
    background_level = (
        _estimate_border_level(gray)
    )
    difference = np.abs(
        gray.astype(np.int16)
        - int(background_level)
    ).astype(np.uint8)
    _, difference_mask = cv2.threshold(
        difference,
        0,
        255,
        cv2.THRESH_BINARY
        | cv2.THRESH_OTSU,
    )
    inverse_difference_mask = (
        cv2.bitwise_not(difference_mask)
    )
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
            min(width, height) * 0.025
        ),
    )
    border_pixels = np.concatenate(
        (
            gray[:band, :].reshape(-1),
            gray[
                height - band :, :
            ].reshape(-1),
            gray[:, :band].reshape(-1),
            gray[
                :, width - band :
            ].reshape(-1),
        )
    )
    return float(
        np.median(border_pixels)
    )


def _prepare_candidate_mask(
    mask: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    height, width = mask.shape
    kernel_size = max(
        3,
        round(
            min(width, height)
            * MORPHOLOGY_KERNEL_RATIO
        ),
    )
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    closed = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )
    small_kernel_size = max(
        3, kernel_size // 3
    )
    if small_kernel_size % 2 == 0:
        small_kernel_size += 1
    small_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                small_kernel_size,
                small_kernel_size,
            ),
        )
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
    NDArray[np.int32] | None, float
]:
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    best_contour = None
    best_score = float("-inf")
    image_area = float(width * height)
    center = (width / 2.0, height / 2.0)
    for contour in contours:
        area = float(
            cv2.contourArea(contour)
        )
        if area <= 0:
            continue
        area_ratio = area / image_area
        if (
            area_ratio
            < MIN_PAGE_AREA_RATIO
        ):
            continue
        _, _, box_width, box_height = (
            cv2.boundingRect(contour)
        )
        width_ratio = box_width / width
        height_ratio = (
            box_height / height
        )
        if (
            width_ratio
            < MIN_PAGE_WIDTH_RATIO
            or height_ratio
            < MIN_PAGE_HEIGHT_RATIO
        ):
            continue
        hull = cv2.convexHull(contour)
        hull_area = float(
            cv2.contourArea(hull)
        )
        if hull_area <= 0:
            continue
        solidity = min(
            1.0, area / hull_area
        )
        box_area = float(
            box_width * box_height
        )
        rectangularity = (
            area / box_area
            if box_area > 0
            else 0.0
        )
        center_inside = (
            cv2.pointPolygonTest(
                contour, center, False
            )
            >= 0
        )
        score = (
            area_ratio * 2.0
            + solidity * 0.55
            + rectangularity * 0.45
        )
        if center_inside:
            score += 1.0
        if (
            area_ratio
            > MAX_PAGE_AREA_RATIO
        ):
            score -= 1.4
        if score > best_score:
            best_contour = contour
            best_score = score
    return (best_contour, best_score)


def _simplify_contour(
    contour: NDArray[np.int32],
    width: int,
    height: int,
) -> tuple[NormalizedPoint, ...]:
    perimeter = float(
        cv2.arcLength(contour, True)
    )
    epsilon = max(
        1.0,
        perimeter
        * CONTOUR_APPROXIMATION_RATIO,
    )
    approximated = cv2.approxPolyDP(
        contour, epsilon, True
    )
    raw_points = [
        (
            int(point[0][0]),
            int(point[0][1]),
        )
        for point in approximated
    ]
    raw_points = _limit_raw_points(
        raw_points
    )
    return _normalize_raw_points(
        raw_points, width, height
    )


def _limit_raw_points(
    raw_points: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if (
        len(raw_points)
        <= MAX_BOUNDARY_POINTS
    ):
        return raw_points
    step = (
        len(raw_points)
        / MAX_BOUNDARY_POINTS
    )
    return [
        raw_points[
            min(
                len(raw_points) - 1,
                int(index * step),
            )
        ]
        for index in range(
            MAX_BOUNDARY_POINTS
        )
    ]


def _normalize_raw_points(
    raw_points: list[tuple[int, int]],
    width: int,
    height: int,
) -> tuple[NormalizedPoint, ...]:
    normalized_points: list[
        NormalizedPoint
    ] = []
    for x, y in raw_points:
        point = NormalizedPoint(
            x=_normalized_coordinate(
                x, width
            ),
            y=_normalized_coordinate(
                y, height
            ),
        )
        if (
            normalized_points
            and point
            == normalized_points[-1]
        ):
            continue
        normalized_points.append(point)
    if (
        len(normalized_points) >= 2
        and normalized_points[0]
        == normalized_points[-1]
    ):
        normalized_points.pop()
    return tuple(normalized_points)


def _odd_kernel_size(
    dimension: int, ratio: float
) -> int:
    kernel_size = max(
        3, round(dimension * ratio)
    )
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def _full_frame_boundary() -> (
    PageBoundaryGeometry
):
    return PageBoundaryGeometry(
        points=(
            NormalizedPoint(0.0, 0.0),
            NormalizedPoint(1.0, 0.0),
            NormalizedPoint(1.0, 1.0),
            NormalizedPoint(0.0, 1.0),
        ),
        is_full_frame=True,
    )


def _normalized_coordinate(
    value: float, dimension: int
) -> float:
    maximum = max(1, dimension - 1)
    normalized = float(value) / maximum
    return max(
        0.0, min(1.0, normalized)
    )


def _pixel_coordinate(
    value: float, dimension: int
) -> int:
    maximum = max(0, dimension - 1)
    return max(
        0,
        min(
            maximum,
            round(value * maximum),
        ),
    )
