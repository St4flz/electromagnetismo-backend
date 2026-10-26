"""Esquemas Pydantic para el Motor FDTD v2 y Capítulo 10."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class UnitCellParamsSchema(BaseModel):
    """Parámetros geométricos y constitutivos de la celda unitaria de metamaterial."""
    model_config = ConfigDict(extra="forbid")

    a_mm: float = Field(default=10.0, ge=2.0, le=50.0, description="Periodo de red transversal y longitudinal [mm]")
    d_mm: float = Field(default=3.0, ge=0.5, le=40.0, description="Dimensión exterior del elemento (varilla o anillo) [mm]")
    w_mm: float = Field(default=1.0, ge=0.2, le=20.0, description="Grosor de la pared o traza metálica/dieléctrica [mm]")
    periods: int = Field(default=3, ge=1, le=10, description="Número de repeticiones en x")
    family: Literal["rod_array", "square_ring"] = Field(default="rod_array", description="Familia topológica (B o C)")
    relative_permittivity: float = Field(default=9.8, ge=1.0, le=100.0, description="Permitividad relativa (si no es PEC)")
    conductivity: float = Field(default=0.0, ge=0.0, le=1e6, description="Conductividad [S/m]")
    is_pec: bool = Field(default=False, description="True si es conductor eléctrico perfecto")


class FDTDSimulationConfigSchema(BaseModel):
    """Configuración de la simulación FDTD en el dominio del tiempo."""
    model_config = ConfigDict(extra="forbid")

    dx_mm: float = Field(default=0.5, ge=0.1, le=2.0, description="Resolución espacial dx=dy [mm]")
    courant_factor: float = Field(default=0.70, ge=0.1, le=0.7071, description="Factor de Courant S <= 1/sqrt(2)")
    pml_thickness: int = Field(default=16, ge=8, le=32, description="Número de celdas PML en x")
    geometry: UnitCellParamsSchema = Field(default_factory=UnitCellParamsSchema)
    max_steps: int = Field(default=2200, ge=100, le=10000, description="Número máximo de pasos temporales")
    f_min_ghz: float = Field(default=5.0, ge=1.0, le=25.0)
    f_max_ghz: float = Field(default=15.0, ge=5.0, le=40.0)


class SpectrumResponseSchema(BaseModel):
    """Espectro electromagnético de transmisión, reflexión y absorción física."""
    model_config = ConfigDict(extra="forbid")

    frequencies_ghz: list[float]
    transmission: list[float]
    reflection: list[float]
    absorption: list[float]
    transmission_db: list[float]
    energy_conservation_max_dev: float
    is_lossless_conserved: bool
    is_passive: bool
    simulation_steps: int
    elapsed_seconds: float


class ValidationCaseResultSchema(BaseModel):
    """Resultado formal de un caso de la batería de validación física (V1 a V11)."""
    model_config = ConfigDict(extra="allow")

    case: str
    is_verified: bool
