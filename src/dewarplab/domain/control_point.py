from dataclasses import dataclass


@dataclass(slots=True)
class ControlPoint:
    row: int
    column: int
    x: float
    y: float

    def __post_init__(self) -> None:
        self._validate_coordinate(
            self.x,
            "x",
        )

        self._validate_coordinate(
            self.y,
            "y",
        )

        if self.row < 0:
            raise ValueError("row must be zero or greater")

        if self.column < 0:
            raise ValueError("column must be zero or greater")

    def move_to(
        self,
        x: float,
        y: float,
    ) -> None:
        self._validate_coordinate(
            x,
            "x",
        )

        self._validate_coordinate(
            y,
            "y",
        )

        self.x = x
        self.y = y

    @staticmethod
    def _validate_coordinate(
        value: float,
        name: str,
    ) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0")
