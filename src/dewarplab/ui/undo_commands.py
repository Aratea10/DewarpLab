from collections.abc import Callable

from PySide6.QtGui import QUndoCommand

from dewarplab.domain import Mesh


RefreshCallback = Callable[[], None]


class MoveMeshPointCommand(QUndoCommand):
    def __init__(
        self,
        mesh: Mesh,
        row: int,
        column: int,
        previous_x: float,
        previous_y: float,
        new_x: float,
        new_y: float,
        on_applied: RefreshCallback,
        parent: QUndoCommand | None = None,
    ):
        super().__init__(
            "Mover nodo de malla",
            parent,
        )

        self._mesh = mesh

        self._row = row
        self._column = column

        self._previous_x = previous_x
        self._previous_y = previous_y

        self._new_x = new_x
        self._new_y = new_y

        self._on_applied = on_applied
        
        self._initial_redo = True

    def undo(
        self,
    ) -> None:
        self._mesh.move_point(
            row=self._row,
            column=self._column,
            x=self._previous_x,
            y=self._previous_y,
        )

        self._on_applied()

    def redo(
        self,
    ) -> None:
        if self._initial_redo:
            self._initial_redo = False

            return

        self._mesh.move_point(
            row=self._row,
            column=self._column,
            x=self._new_x,
            y=self._new_y,
        )

        self._on_applied()
