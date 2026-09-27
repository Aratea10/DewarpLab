from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from dewarplab.application import (
    NormalizedPoint,
    PrintedStructureGeometry,
    PrintedStructureOrientation,
    PrintedStructureSegment,
)


ImageArray = NDArray[np.uint8]

MAX_DETECTION_DIMENSION = 1800

MIN_HORIZONTAL_SPAN_RATIO = 0.05
MIN_VERTICAL_SPAN_RATIO = 0.05

MIN_ASPECT_RATIO = 5.0

MAX_HORIZONTAL_THICKNESS_RATIO = 0.025
MAX_VERTICAL_THICKNESS_RATIO = 0.025

LINE_KERNEL_RATIO = 0.035
GAP_KERNEL_RATIO = 0.008

MERGE_POSITION_TOLERANCE_RATIO = 0.004
MERGE_GAP_RATIO = 0.012

BOX_CORNER_TOLERANCE_RATIO = 0.012


class PrintedStructureDetectionError(ValueError):
    pass


@dataclass(
    slots=True,
)
class _PixelSegment:
    orientation: PrintedStructureOrientation

    x1: float
    y1: float
    x2: float
    y2: float

    is_box_edge: bool = False

    @property
    def primary_start(self) -> float:
        if self.orientation == PrintedStructureOrientation.HORIZONTAL:
            return min(
                self.x1,
                self.x2,
            )

        return min(
            self.y1,
            self.y2,
        )

    @property
    def primary_end(self) -> float:
        if self.orientation == PrintedStructureOrientation.HORIZONTAL:
            return max(
                self.x1,
                self.x2,
            )

        return max(
            self.y1,
            self.y2,
        )

    @property
    def position(self) -> float:
        if self.orientation == PrintedStructureOrientation.HORIZONTAL:
            return (self.y1 + self.y2) / 2.0

        return (self.x1 + self.x2) / 2.0


def detect_printed_structure_geometry(
    image: ImageArray,
) -> PrintedStructureGeometry:
    gray = _to_grayscale(image)

    gray = _resize_for_detection(gray)

    height, width = gray.shape

    blurred = cv2.GaussianBlur(
        gray,
        (
            3,
            3,
        ),
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

    horizontal_mask = _build_directional_mask(
        binary=binary,
        orientation=(PrintedStructureOrientation.HORIZONTAL),
    )

    vertical_mask = _build_directional_mask(
        binary=binary,
        orientation=(PrintedStructureOrientation.VERTICAL),
    )

    horizontal_segments = _extract_segments(
        mask=horizontal_mask,
        orientation=(PrintedStructureOrientation.HORIZONTAL),
        width=width,
        height=height,
    )

    vertical_segments = _extract_segments(
        mask=vertical_mask,
        orientation=(PrintedStructureOrientation.VERTICAL),
        width=width,
        height=height,
    )

    horizontal_segments = _merge_collinear_segments(
        segments=horizontal_segments,
        width=width,
        height=height,
    )

    vertical_segments = _merge_collinear_segments(
        segments=vertical_segments,
        width=width,
        height=height,
    )

    _mark_box_edges(
        horizontal_segments=horizontal_segments,
        vertical_segments=vertical_segments,
        width=width,
        height=height,
    )

    pixel_segments = horizontal_segments + vertical_segments

    pixel_segments.sort(key=_segment_sort_key)

    segments = tuple(
        _normalize_segment(
            segment=segment,
            width=width,
            height=height,
        )
        for segment in pixel_segments
    )

    return PrintedStructureGeometry(segments=segments)


def _to_grayscale(
    image: ImageArray,
) -> NDArray[np.uint8]:
    if not isinstance(
        image,
        np.ndarray,
    ):
        raise PrintedStructureDetectionError("Image must be a NumPy array")

    if image.dtype != np.uint8:
        raise PrintedStructureDetectionError("Image must use uint8 pixels")

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
        raise PrintedStructureDetectionError("Image must be grayscale, RGB or RGBA")

    height, width = gray.shape

    if width < 32 or height < 32:
        raise PrintedStructureDetectionError(
            "Image is too small for printed structure detection"
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


def _build_directional_mask(
    binary: NDArray[np.uint8],
    orientation: PrintedStructureOrientation,
) -> NDArray[np.uint8]:
    height, width = binary.shape

    if orientation == PrintedStructureOrientation.HORIZONTAL:
        line_length = max(
            15,
            round(width * LINE_KERNEL_RATIO),
        )

        line_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                line_length,
                1,
            ),
        )

        gap_length = max(
            3,
            round(width * GAP_KERNEL_RATIO),
        )

        gap_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                gap_length,
                1,
            ),
        )

    else:
        line_length = max(
            15,
            round(height * LINE_KERNEL_RATIO),
        )

        line_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                1,
                line_length,
            ),
        )

        gap_length = max(
            3,
            round(height * GAP_KERNEL_RATIO),
        )

        gap_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                1,
                gap_length,
            ),
        )

    opened = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        line_kernel,
    )

    return cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        gap_kernel,
    )


def _extract_segments(
    mask: NDArray[np.uint8],
    orientation: PrintedStructureOrientation,
    width: int,
    height: int,
) -> list[_PixelSegment]:
    (
        component_count,
        _,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    segments: list[_PixelSegment] = []

    for index in range(
        1,
        component_count,
    ):
        (
            x,
            y,
            component_width,
            component_height,
            _,
        ) = stats[index]

        if orientation == PrintedStructureOrientation.HORIZONTAL:
            minimum_span = max(
                20,
                round(width * MIN_HORIZONTAL_SPAN_RATIO),
            )

            maximum_thickness = max(
                6,
                round(height * MAX_HORIZONTAL_THICKNESS_RATIO),
            )

            if component_width < minimum_span:
                continue

            if component_height > maximum_thickness:
                continue

            aspect_ratio = component_width / max(
                component_height,
                1,
            )

            if aspect_ratio < MIN_ASPECT_RATIO:
                continue

            center_y = y + (component_height - 1) / 2.0

            segments.append(
                _PixelSegment(
                    orientation=orientation,
                    x1=float(x),
                    y1=float(center_y),
                    x2=float(x + component_width - 1),
                    y2=float(center_y),
                )
            )

        else:
            minimum_span = max(
                20,
                round(height * MIN_VERTICAL_SPAN_RATIO),
            )

            maximum_thickness = max(
                6,
                round(width * MAX_VERTICAL_THICKNESS_RATIO),
            )

            if component_height < minimum_span:
                continue

            if component_width > maximum_thickness:
                continue

            aspect_ratio = component_height / max(
                component_width,
                1,
            )

            if aspect_ratio < MIN_ASPECT_RATIO:
                continue

            center_x = x + (component_width - 1) / 2.0

            segments.append(
                _PixelSegment(
                    orientation=orientation,
                    x1=float(center_x),
                    y1=float(y),
                    x2=float(center_x),
                    y2=float(y + component_height - 1),
                )
            )

    return segments


def _merge_collinear_segments(
    segments: list[_PixelSegment],
    width: int,
    height: int,
) -> list[_PixelSegment]:
    if not segments:
        return []

    orientation = segments[0].orientation

    if orientation == PrintedStructureOrientation.HORIZONTAL:
        position_tolerance = max(
            2.0,
            height * MERGE_POSITION_TOLERANCE_RATIO,
        )

        gap_tolerance = max(
            4.0,
            width * MERGE_GAP_RATIO,
        )

    else:
        position_tolerance = max(
            2.0,
            width * MERGE_POSITION_TOLERANCE_RATIO,
        )

        gap_tolerance = max(
            4.0,
            height * MERGE_GAP_RATIO,
        )

    ordered = sorted(
        segments,
        key=lambda segment: (
            segment.position,
            segment.primary_start,
        ),
    )

    merged: list[_PixelSegment] = []

    for segment in ordered:
        merged_index: int | None = None

        for index in range(
            len(merged) - 1,
            -1,
            -1,
        ):
            candidate = merged[index]

            if abs(segment.position - candidate.position) > position_tolerance:
                continue

            if (
                _interval_gap(
                    first_start=(candidate.primary_start),
                    first_end=(candidate.primary_end),
                    second_start=(segment.primary_start),
                    second_end=(segment.primary_end),
                )
                > gap_tolerance
            ):
                continue

            merged_index = index

            break

        if merged_index is None:
            merged.append(segment)

            continue

        merged[merged_index] = _merge_pair(
            merged[merged_index],
            segment,
        )

    return merged


def _interval_gap(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> float:
    left = max(
        first_start,
        second_start,
    )

    right = min(
        first_end,
        second_end,
    )

    return max(
        0.0,
        left - right,
    )


def _merge_pair(
    first: _PixelSegment,
    second: _PixelSegment,
) -> _PixelSegment:
    if first.orientation != second.orientation:
        raise ValueError("Cannot merge differently oriented segments")

    if first.orientation == PrintedStructureOrientation.HORIZONTAL:
        position = (first.position + second.position) / 2.0

        return _PixelSegment(
            orientation=(first.orientation),
            x1=min(
                first.primary_start,
                second.primary_start,
            ),
            y1=position,
            x2=max(
                first.primary_end,
                second.primary_end,
            ),
            y2=position,
        )

    position = (first.position + second.position) / 2.0

    return _PixelSegment(
        orientation=(first.orientation),
        x1=position,
        y1=min(
            first.primary_start,
            second.primary_start,
        ),
        x2=position,
        y2=max(
            first.primary_end,
            second.primary_end,
        ),
    )


def _mark_box_edges(
    horizontal_segments: list[_PixelSegment],
    vertical_segments: list[_PixelSegment],
    width: int,
    height: int,
) -> None:
    corner_tolerance = max(
        4.0,
        min(
            width,
            height,
        )
        * BOX_CORNER_TOLERANCE_RATIO,
    )

    for horizontal in horizontal_segments:
        for vertical in vertical_segments:
            if not _segments_share_corner(
                horizontal=horizontal,
                vertical=vertical,
                tolerance=corner_tolerance,
            ):
                continue

            horizontal.is_box_edge = True
            vertical.is_box_edge = True


def _segments_share_corner(
    horizontal: _PixelSegment,
    vertical: _PixelSegment,
    tolerance: float,
) -> bool:
    horizontal_endpoints = (
        (
            horizontal.x1,
            horizontal.y1,
        ),
        (
            horizontal.x2,
            horizontal.y2,
        ),
    )

    vertical_endpoints = (
        (
            vertical.x1,
            vertical.y1,
        ),
        (
            vertical.x2,
            vertical.y2,
        ),
    )

    for (
        horizontal_x,
        horizontal_y,
    ) in horizontal_endpoints:
        for (
            vertical_x,
            vertical_y,
        ) in vertical_endpoints:
            if (
                abs(horizontal_x - vertical_x) <= tolerance
                and abs(horizontal_y - vertical_y) <= tolerance
            ):
                return True

    return False


def _normalize_segment(
    segment: _PixelSegment,
    width: int,
    height: int,
) -> PrintedStructureSegment:
    return PrintedStructureSegment(
        start=NormalizedPoint(
            x=_normalized_coordinate(
                segment.x1,
                width,
            ),
            y=_normalized_coordinate(
                segment.y1,
                height,
            ),
        ),
        end=NormalizedPoint(
            x=_normalized_coordinate(
                segment.x2,
                width,
            ),
            y=_normalized_coordinate(
                segment.y2,
                height,
            ),
        ),
        orientation=(segment.orientation),
        is_box_edge=(segment.is_box_edge),
    )


def _normalized_coordinate(
    value: float,
    dimension: int,
) -> float:
    maximum = max(
        1,
        dimension - 1,
    )

    normalized = value / maximum

    return max(
        0.0,
        min(
            1.0,
            normalized,
        ),
    )


def _segment_sort_key(
    segment: _PixelSegment,
) -> tuple[
    str,
    float,
    float,
]:
    return (
        segment.orientation.value,
        segment.position,
        segment.primary_start,
    )
