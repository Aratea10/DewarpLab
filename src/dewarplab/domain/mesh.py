from dataclasses import dataclass, field
from typing import Iterator

from dewarplab.domain.control_point import ControlPoint


@dataclass(slots=True)
class Mesh:
    rows: int
    columns: int
    _points: list[list[ControlPoint]] = field(
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.rows < 2:
            raise ValueError("A mesh must have at least 2 rows")

        if self.columns < 2:
            raise ValueError("A mesh must have at least 2 columns")

        if len(self._points) != self.rows:
            raise ValueError("Point rows do not match mesh rows")

        for row in self._points:
            if len(row) != self.columns:
                raise ValueError("Point columns do not match mesh columns")

    @classmethod
    def regular(
        cls,
        rows: int,
        columns: int,
    ) -> "Mesh":
        if rows < 2:
            raise ValueError("A mesh must have at least 2 rows")

        if columns < 2:
            raise ValueError("A mesh must have at least 2 columns")

        points: list[list[ControlPoint]] = []

        for row in range(rows):
            row_points: list[ControlPoint] = []

            y = row / (rows - 1)

            for column in range(columns):
                x = column / (columns - 1)

                row_points.append(
                    ControlPoint(
                        row=row,
                        column=column,
                        x=x,
                        y=y,
                    )
                )

            points.append(row_points)

        return cls(
            rows=rows,
            columns=columns,
            _points=points,
        )

    def point(
        self,
        row: int,
        column: int,
    ) -> ControlPoint:
        self._validate_index(
            row,
            column,
        )

        return self._points[row][column]

    def move_point(
        self,
        row: int,
        column: int,
        x: float,
        y: float,
    ) -> None:
        point = self.point(
            row,
            column,
        )

        point.move_to(
            x,
            y,
        )

    def reference_position(
        self,
        row: int,
        column: int,
    ) -> tuple[float, float]:
        self._validate_index(
            row,
            column,
        )

        return (
            column / (self.columns - 1),
            row / (self.rows - 1),
        )

    def iter_points(
        self,
    ) -> Iterator[ControlPoint]:
        for row in self._points:
            yield from row

    def _validate_index(
        self,
        row: int,
        column: int,
    ) -> None:
        if not 0 <= row < self.rows:
            raise IndexError("Mesh row is out of range")

        if not 0 <= column < self.columns:
            raise IndexError("Mesh column is out of range")
