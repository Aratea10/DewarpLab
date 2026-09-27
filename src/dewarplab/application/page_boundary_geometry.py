from dataclasses import dataclass

from dewarplab.application.text_line_geometry import NormalizedPoint


@dataclass(
    frozen=True,
    slots=True,
)
class PageBoundaryGeometry:
    points: tuple[
        NormalizedPoint,
        ...,
    ]

    is_full_frame: bool = False

    def __post_init__(
        self,
    ) -> None:
        if len(self.points) < 4:
            raise ValueError("A page boundary must contain at least 4 points")

        if len(set(self.points)) < 4:
            raise ValueError("A page boundary must contain at least 4 distinct points")

    @property
    def point_count(
        self,
    ) -> int:
        return len(self.points)
