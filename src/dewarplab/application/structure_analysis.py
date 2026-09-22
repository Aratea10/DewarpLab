from dataclasses import dataclass


@dataclass(
    frozen=True,
    slots=True,
)
class StructureAnalysis:
    text_component_count: int
    line_band_count: int
    vertical_structure_count: int
    suggested_rows: int
    suggested_columns: int
