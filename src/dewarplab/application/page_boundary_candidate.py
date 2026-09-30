from dataclasses import dataclass
from enum import StrEnum

from dewarplab.application.page_boundary_geometry import (
    PageBoundaryGeometry,
)


class PageBoundaryCandidateSource(
    StrEnum
):
    FULL_FRAME = "full_frame"
    SPREAD = "spread"
    SINGLE_PAGE = "single_page"
    CLIPPED_PAGE = "clipped_page"
    FALLBACK_FULL_FRAME = (
        "fallback_full_frame"
    )


@dataclass(
    frozen=True,
    slots=True,
)
class PageBoundaryCandidate:
    geometry: PageBoundaryGeometry
    source: PageBoundaryCandidateSource
    confidence: float | None = None

    def __post_init__(
        self,
    ) -> None:
        if self.confidence is None:
            return

        if (
            not 0.0
            <= self.confidence
            <= 1.0
        ):
            raise ValueError(
                "Page boundary candidate confidence must be between 0 and 1"
            )
