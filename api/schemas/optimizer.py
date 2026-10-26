"""Esquemas Pydantic para el optimizador genético y búsqueda sistemática (Capítulo 12)."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field
from api.schemas.sweep import SweepDesignPointSchema


class OptimizerRequest(BaseModel):
    family: Literal["square_ring", "rod_array"] = Field("square_ring", description="Familia geométrica")
    pop_size: int = Field(16, ge=4, le=60, description="Tamaño de la población del GA")
    generations: int = Field(15, ge=2, le=50, description="Número de generaciones a evolucionar")
    seed: int = Field(42, description="Semilla aleatoria reproducible")
    f0_ghz: float = Field(10.0, ge=1.0, le=30.0, description="Frecuencia objetivo central (GHz)")
    periods: int = Field(3, ge=1, le=8, description="Número de celdas unitarias en x")
    is_pec: bool = Field(True, description="Si es conductor eléctrico perfecto")
    relative_permittivity: float = Field(1.0, ge=1.0, le=100.0, description="Permitividad dieléctrica")
    dx_mm: float = Field(0.5, ge=0.1, le=1.0, description="Resolución espacial de simulación")
    a_min: float = Field(6.0, ge=4.0, le=30.0)
    a_max: float = Field(16.0, ge=4.0, le=30.0)
    d_min: float = Field(2.0, ge=1.0, le=20.0)
    d_max: float = Field(8.0, ge=1.0, le=20.0)
    w_min: float = Field(0.5, ge=0.2, le=10.0)
    w_max: float = Field(2.5, ge=0.2, le=10.0)


class GenerationReportSchema(BaseModel):
    generation: int
    best_fitness: float
    mean_fitness: float
    best_individual: list[float]
    best_design_point: SweepDesignPointSchema
    population: list[list[float]]
    fitness_values: list[float]
    total_evaluations: int
    cache_hits: int


class OptimizerRunResponse(BaseModel):
    reports: list[GenerationReportSchema]
    best_design: SweepDesignPointSchema
    total_time_seconds: float
    cache_hit_rate_percent: float


class GridSearchRequest(BaseModel):
    family: Literal["square_ring", "rod_array"] = Field("square_ring")
    max_evaluations: int = Field(50, ge=5, le=300)
    stride: int = Field(2, ge=1, le=10)
    f0_ghz: float = Field(10.0, ge=1.0, le=30.0)
    periods: int = Field(3, ge=1, le=8)
    is_pec: bool = Field(True)
    relative_permittivity: float = Field(1.0)
    dx_mm: float = Field(0.5)


class GridSearchResponse(BaseModel):
    total_combinations_in_space: int
    feasible_combinations: int
    evaluations_executed: int
    best_fitness: float
    best_individual: list[float]
    best_design: SweepDesignPointSchema | None
    evaluations_history: list[int]
    best_fitness_history: list[float]


class RevalidationRequest(BaseModel):
    a_mm: float = Field(..., ge=2.0, le=40.0)
    d_mm: float = Field(..., ge=0.5, le=30.0)
    w_mm: float = Field(..., ge=0.2, le=20.0)
    periods: int = Field(3, ge=1, le=8)
    family: Literal["square_ring", "rod_array"] = Field("square_ring")
    is_pec: bool = Field(True)
    relative_permittivity: float = Field(1.0)
    f0_ghz: float = Field(10.0, ge=1.0, le=30.0)
    tolerance_db: float = Field(2.0, ge=0.1, le=10.0)


class RevalidationResponse(BaseModel):
    design_id: int
    a_mm: float
    d_mm: float
    w_mm: float
    periods: int
    family: str
    f0_ghz: float
    dx_fast_mm: float
    t_f0_fast: float
    t_f0_fast_db: float
    f2_fast: float
    time_fast_s: float
    dx_fine_mm: float
    t_f0_fine: float
    t_f0_fine_db: float
    f2_fine: float
    time_fine_s: float
    delta_t_db: float
    delta_t_linear: float
    passed_tolerance: bool
    tolerance_threshold_db: float
    freqs_ghz: list[float]
    t_fast_spectrum: list[float]
    t_fine_spectrum: list[float]
    r_fast_spectrum: list[float]
    r_fine_spectrum: list[float]
