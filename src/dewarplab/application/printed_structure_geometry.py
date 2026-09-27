from dataclasses import dataclass
from enum import Enum

from dewarplab.application.text_line_geometry import NormalizedPoint


class PrintedStructureOrientation(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


@dataclass(
    frozen=True,
    slots=True,
)
class PrintedStructureSegment:
    start: NormalizedPoint
    end: NormalizedPoint
    orientation: PrintedStructureOrientation
    is_box_edge: bool = False

    def __post_init__(self) -> None:
        if self.start == self.end:
            raise ValueError("A printed structure segment must have distinct endpoints")

    @property
    def span(self) -> float:
        if self.orientation == PrintedStructureOrientation.HORIZONTAL:
            return abs(self.end.x - self.start.x)

        return abs(self.end.y - self.start.y)


@dataclass(
    frozen=True,
    slots=True,
)
class PrintedStructureGeometry:
    segments: tuple[
        PrintedStructureSegment,
        ...,
    ]

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def horizontal_count(self) -> int:
        return sum(
            segment.orientation == PrintedStructureOrientation.HORIZONTAL
            for segment in self.segments
        )

    @property
    def vertical_count(self) -> int:
        return sum(
            segment.orientation == PrintedStructureOrientation.VERTICAL
            for segment in self.segments
        )

    @property
    def box_edge_count(self) -> int:
        return sum(segment.is_box_edge for segment in self.segments)
