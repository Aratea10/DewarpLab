from dataclasses import dataclass, field
from typing import ClassVar, Iterator

from dewarplab.domain.control_point import ControlPoint


@dataclass(slots=True)
class Mesh:
    rows: int
    columns: int
    _points: list[list[ControlPoint]] = field(
        repr=False,
    )

    MINIMUM_CELL_CROSS: ClassVar[float] = 1e-9

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

        if not self.is_topologically_valid():
            raise ValueError("Mesh contains inverted or collapsed cells")

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
        self._validate_index(
            row,
            column,
        )

        if not 0.0 <= x <= 1.0:
            raise ValueError("x must be between 0.0 and 1.0")

        if not 0.0 <= y <= 1.0:
            raise ValueError("y must be between 0.0 and 1.0")

        if not self.can_move_point(
            row=row,
            column=column,
            x=x,
            y=y,
        ):
            raise ValueError("Move would invert or collapse a mesh cell")

        self._points[row][column].move_to(
            x,
            y,
        )

    def can_move_point(
        self,
        row: int,
        column: int,
        x: float,
        y: float,
    ) -> bool:
        self._validate_index(
            row,
            column,
        )

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return False

        first_cell_row = max(
            0,
            row - 1,
        )

        last_cell_row = min(
            self.rows - 2,
            row,
        )

        first_cell_column = max(
            0,
            column - 1,
        )

        last_cell_column = min(
            self.columns - 2,
            column,
        )

        for cell_row in range(
            first_cell_row,
            last_cell_row + 1,
        ):
            for cell_column in range(
                first_cell_column,
                last_cell_column + 1,
            ):
                if not self._cell_is_valid(
                    cell_row=cell_row,
                    cell_column=cell_column,
                    candidate=(
                        row,
                        column,
                        x,
                        y,
                    ),
                ):
                    return False

        return True

    def is_topologically_valid(
        self,
    ) -> bool:
        for row in range(self.rows - 1):
            for column in range(self.columns - 1):
                if not self._cell_is_valid(
                    cell_row=row,
                    cell_column=column,
                ):
                    return False

        return True

    def is_regular(
        self,
        tolerance: float = 1e-9,
    ) -> bool:
        for point in self.iter_points():
            reference_x, reference_y = self.reference_position(
                point.row,
                point.column,
            )

            if abs(point.x - reference_x) > tolerance:
                return False

            if abs(point.y - reference_y) > tolerance:
                return False

        return True

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

    def _cell_is_valid(
        self,
        cell_row: int,
        cell_column: int,
        candidate: (
            tuple[
                int,
                int,
                float,
                float,
            ]
            | None
        ) = None,
    ) -> bool:
        top_left = self._point_coordinates(
            cell_row,
            cell_column,
            candidate,
        )

        top_right = self._point_coordinates(
            cell_row,
            cell_column + 1,
            candidate,
        )

        bottom_right = self._point_coordinates(
            cell_row + 1,
            cell_column + 1,
            candidate,
        )

        bottom_left = self._point_coordinates(
            cell_row + 1,
            cell_column,
            candidate,
        )

        corners = (
            top_left,
            top_right,
            bottom_right,
            bottom_left,
        )

        for index in range(4):
            first = corners[index]
            second = corners[(index + 1) % 4]
            third = corners[(index + 2) % 4]

            if (
                self._cross_product(
                    first,
                    second,
                    third,
                )
                <= self.MINIMUM_CELL_CROSS
            ):
                return False

        return True

    def _point_coordinates(
        self,
        row: int,
        column: int,
        candidate: (
            tuple[
                int,
                int,
                float,
                float,
            ]
            | None
        ),
    ) -> tuple[float, float]:
        if candidate is not None:
            (
                candidate_row,
                candidate_column,
                candidate_x,
                candidate_y,
            ) = candidate

            if row == candidate_row and column == candidate_column:
                return (
                    candidate_x,
                    candidate_y,
                )

        point = self._points[row][column]

        return (
            point.x,
            point.y,
        )

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
    def _cross_product(
        first: tuple[float, float],
        second: tuple[float, float],
        third: tuple[float, float],
    ) -> float:
        return (second[0] - first[0]) * (third[1] - first[1]) - (
            second[1] - first[1]
        ) * (third[0] - first[0])

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
