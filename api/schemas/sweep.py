"""Esquemas Pydantic para el Laboratorio Paramétrico y Barridos FDTD (Capítulo 11)."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class SweepDesignPointSchema(BaseModel):
    """Punto de diseño evaluado en la simulación paramétrica."""
    model_config = ConfigDict(extra="forbid")

    design_id: int
    a_mm: float
    d_mm: float
    w_mm: float
    periods: int
    family: str
    is_pec: bool
    relative_permittivity: float
    f0_ghz: float
    t_f0: float
    t_f0_db: float
    r_f0: float
    t_pass_mean: float
    rejection_depth_db: float
    f2_score: float
    rank: int = 0


class Sweep1DRequestSchema(BaseModel):
    """Solicitud de barrido unidimensional sobre un parámetro físico."""
    model_config = ConfigDict(extra="forbid")

    parameter_name: Literal["w_mm", "d_mm", "a_mm"] = Field(default="w_mm")
    param_values: list[float] | None = Field(default=None)
    base_a_mm: float = Field(default=10.0, ge=4.0, le=25.0)
    base_d_mm: float = Field(default=6.0, ge=1.0, le=20.0)
    base_w_mm: float = Field(default=1.0, ge=0.2, le=5.0)
    family: Literal["rod_array", "square_ring"] = Field(default="square_ring")
    periods: int = Field(default=3, ge=1, le=6)
    is_pec: bool = Field(default=True)
    relative_permittivity: float = Field(default=1.0, ge=1.0, le=100.0)
    dx_mm: float = Field(default=0.5, ge=0.2, le=1.0)
    f0_ghz: float = Field(default=10.0, ge=5.0, le=15.0)


class Sweep1DResponseSchema(BaseModel):
    """Respuesta con la familia de curvas y tabla de resultados del barrido 1D."""
    model_config = ConfigDict(extra="forbid")

    parameter_name: str
    parameter_values: list[float]
    frequencies_ghz: list[float]
    family_curves_t: dict[str, list[float]]
    family_curves_r: dict[str, list[float]]
    table_rows: list[SweepDesignPointSchema]
    best_design: SweepDesignPointSchema
    best_by_f1: SweepDesignPointSchema
    metadata: dict[str, Any]


class Sweep2DRequestSchema(BaseModel):
    """Solicitud de barrido bidimensional para generación de mapas de calor T(f, p)."""
    model_config = ConfigDict(extra="forbid")

    param_x_name: Literal["a_mm", "d_mm", "w_mm"] = Field(default="a_mm")
    param_x_values: list[float] | None = Field(default=None)
    param_y_name: Literal["d_mm", "w_mm", "a_mm"] = Field(default="d_mm")
    param_y_values: list[float] | None = Field(default=None)
    base_w_mm: float = Field(default=1.0, ge=0.2, le=5.0)
    family: Literal["rod_array", "square_ring"] = Field(default="square_ring")
    periods: int = Field(default=3, ge=1, le=6)
    is_pec: bool = Field(default=True)
    relative_permittivity: float = Field(default=1.0, ge=1.0, le=100.0)
    dx_mm: float = Field(default=0.5, ge=0.2, le=1.0)
    f0_ghz: float = Field(default=10.0, ge=5.0, le=15.0)


class Sweep2DResponseSchema(BaseModel):
    """Respuesta con matrices 2D para mapas de calor e información tabular."""
    model_config = ConfigDict(extra="forbid")

    param_x_name: str
    param_x_values: list[float]
    param_y_name: str
    param_y_values: list[float]
    frequencies_ghz: list[float]
    heatmap_t_f0: list[list[float]]
    heatmap_t_f0_db: list[list[float]]
    heatmap_f2: list[list[float]]
    table_rows: list[SweepDesignPointSchema]
    best_design: SweepDesignPointSchema
    metadata: dict[str, Any]


class RobustnessResponseSchema(BaseModel):
    """Respuesta de la prueba de robustez formal frente a resolución de malla."""
    model_config = ConfigDict(extra="forbid")

    num_designs_tested: int
    scores_coarse_dx_0_5mm: list[float]
    scores_fine_dx_0_25mm: list[float]
    spearman_rank_correlation: float
    threshold: float
    is_robust: bool
