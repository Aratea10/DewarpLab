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
        self._validate_dimensions(
            self.rows,
            self.columns,
        )

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
        cls._validate_dimensions(
            rows,
            columns,
        )

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

    def resampled(
        self,
        rows: int,
        columns: int,
    ) -> "Mesh":
        self._validate_dimensions(
            rows,
            columns,
        )

        points: list[list[ControlPoint]] = []

        for target_row in range(rows):
            target_row_points: list[ControlPoint] = []

            source_y = target_row * (self.rows - 1) / (rows - 1)

            row_start = int(source_y)

            row_end = min(
                row_start + 1,
                self.rows - 1,
            )

            row_fraction = source_y - row_start

            for target_column in range(columns):
                source_x = target_column * (self.columns - 1) / (columns - 1)

                column_start = int(source_x)

                column_end = min(
                    column_start + 1,
                    self.columns - 1,
                )

                column_fraction = source_x - column_start

                top_left = self.point(
                    row_start,
                    column_start,
                )

                top_right = self.point(
                    row_start,
                    column_end,
                )

                bottom_left = self.point(
                    row_end,
                    column_start,
                )

                bottom_right = self.point(
                    row_end,
                    column_end,
                )

                top_x = self._lerp(
                    top_left.x,
                    top_right.x,
                    column_fraction,
                )

                top_y = self._lerp(
                    top_left.y,
                    top_right.y,
                    column_fraction,
                )

                bottom_x = self._lerp(
                    bottom_left.x,
                    bottom_right.x,
                    column_fraction,
                )

                bottom_y = self._lerp(
                    bottom_left.y,
                    bottom_right.y,
                    column_fraction,
                )

                x = self._lerp(
                    top_x,
                    bottom_x,
                    row_fraction,
                )

                y = self._lerp(
                    top_y,
                    bottom_y,
                    row_fraction,
                )

                target_row_points.append(
                    ControlPoint(
                        row=target_row,
                        column=target_column,
                        x=x,
                        y=y,
                    )
                )

            points.append(target_row_points)

        return Mesh(
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

    @staticmethod
    def _validate_dimensions(
        rows: int,
        columns: int,
    ) -> None:
        if rows < 2:
            raise ValueError("A mesh must have at least 2 rows")

        if columns < 2:
            raise ValueError("A mesh must have at least 2 columns")

    @staticmethod
    def _lerp(
        start: float,
        end: float,
        amount: float,
    ) -> float:
        return start + (end - start) * amount
