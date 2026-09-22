from dataclasses import dataclass


@dataclass(
    frozen=True,
    slots=True,
)
class NormalizedPoint:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.x <= 1.0:
            raise ValueError("x must be between 0.0 and 1.0")

        if not 0.0 <= self.y <= 1.0:
            raise ValueError("y must be between 0.0 and 1.0")


@dataclass(
    frozen=True,
    slots=True,
)
class TextLineTrace:
    points: tuple[
        NormalizedPoint,
        ...,
    ]

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError("A text line trace must contain " "at least 3 points")

    @property
    def horizontal_span(self) -> float:
        x_values = [point.x for point in self.points]

        return max(x_values) - min(x_values)


@dataclass(
    frozen=True,
    slots=True,
)
class TextLineGeometry:
    traces: tuple[
        TextLineTrace,
        ...,
    ]

    @property
    def trace_count(self) -> int:
        return len(self.traces)
