"""Router REST para el Motor FDTD v2 y Capítulo 10.

Proporciona endpoints para:
- Batería de validación formal de rigor físico V1 a V11.
- Cálculo de espectros de transmisión T(f), reflexión R(f) y absorción A(f).
- Benchmarks analíticos exactos (Airy, TMM Bragg, dispersión Yee, frecuencia de plasma).
"""

from __future__ import annotations

from typing import Any
import asyncio
import time
import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas.fdtd import (
    FDTDSimulationConfigSchema,
    SpectrumResponseSchema,
    UnitCellParamsSchema,
    ValidationCaseResultSchema,
)
from domain.fdtd import (
    FDTDSimulation,
    UnitCellGeometry,
    CellFamily,
    rasterize_unit_cell,
    airy_dielectric_slab_transmission,
    tmm_multilayer_transmission,
    yee_numerical_phase_velocity,
    wire_medium_plasma_frequency,
    run_validation_case,
    run_all_validations,
    VALIDATION_REGISTRY,
)

router = APIRouter(prefix="/fdtd", tags=["FDTD v2"])


@router.get("/validate", response_model=dict[str, Any])
def get_all_validations():
    """Ejecuta y retorna los veredictos numéricos de toda la batería V1 a V11."""
    return run_all_validations()


@router.get("/validate/{case_id}")
def get_validation_case(case_id: str):
    """Ejecuta un caso individual de validación (V1 a V11)."""
    case_upper = case_id.upper()
    if case_upper not in VALIDATION_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Caso no reconocido '{case_id}'. Opciones: {list(VALIDATION_REGISTRY.keys())}",
        )
    return run_validation_case(case_upper)


def _compute_spectrum_sync(config: FDTDSimulationConfigSchema) -> SpectrumResponseSchema:
    t0 = time.perf_counter()
    dx = config.dx_mm * 1e-3
    dy = dx

    # Dimensionamiento físico del dominio
    # Periodo transversal a_mm
    a_m = config.geometry.a_mm * 1e-3
    Ny = max(int(round(a_m / dy)), 4)

    # Longitud en x: PML izquierda + TF/SF + margen + celda unitaria * periodos + margen + PML derecha
    cell_len_m = a_m * config.geometry.periods
    cell_cells_x = int(round(cell_len_m / dx))
    margin_x = int(round(0.040 / dx)) # 40 mm de margen
    Nx = config.pml_thickness * 2 + 30 + margin_x + cell_cells_x

    sim = FDTDSimulation(
        Nx=Nx,
        Ny=Ny,
        dx=dx,
        dy=dy,
        courant_factor=config.courant_factor,
        pml_thickness=config.pml_thickness,
        f_min_ghz=config.f_min_ghz,
        f_max_ghz=config.f_max_ghz,
    )

    # 1. Calibrar onda incidente en vacío
    sim.calibrate_incident_wave(max_steps=min(config.max_steps, 1800))

    # 2. Rasterizar estructura de metamaterial
    fam = CellFamily.ROD_ARRAY if config.geometry.family == "rod_array" else CellFamily.SQUARE_RING
    geom = UnitCellGeometry(
        a_mm=config.geometry.a_mm,
        d_mm=config.geometry.d_mm,
        w_mm=config.geometry.w_mm,
        periods=config.geometry.periods,
        family=fam,
        relative_permittivity=config.geometry.relative_permittivity,
        conductivity=config.geometry.conductivity,
        is_pec=config.geometry.is_pec,
    )
    x_start = sim.i_tfsf + 15
    rasterize_unit_cell(geom, sim.materials, x_start_idx=x_start)
    sim.reset()

    # 3. Correr simulación FDTD hasta decaimiento
    steps_run = sim.run_until_decay(max_steps=config.max_steps, energy_threshold=1e-5)
    spec = sim.get_spectrum()
    elapsed = time.perf_counter() - t0

    # Pruebas honestas de conservación de energía
    mask_eval = [
        (config.f_min_ghz + 1.0) <= f <= (config.f_max_ghz - 1.0)
        for f in spec.frequencies_ghz
    ]
    devs = [
        abs(t + r - 1.0)
        for t, r, m in zip(spec.transmission, spec.reflection, mask_eval)
        if m
    ]
    max_dev = float(max(devs)) if devs else 0.0

    is_lossless = max_dev < 0.035 if (not config.geometry.is_pec and config.geometry.conductivity == 0.0) else True
    is_passive = all(a >= -0.015 for a in spec.absorption)

    return SpectrumResponseSchema(
        frequencies_ghz=spec.frequencies_ghz,
        transmission=spec.transmission,
        reflection=spec.reflection,
        absorption=spec.absorption,
        transmission_db=spec.transmission_db,
        energy_conservation_max_dev=round(max_dev, 4),
        is_lossless_conserved=is_lossless,
        is_passive=is_passive,
        simulation_steps=steps_run,
        elapsed_seconds=round(elapsed, 3),
    )


@router.post("/spectrum", response_model=SpectrumResponseSchema)
async def compute_spectrum(config: FDTDSimulationConfigSchema):
    """Calcula el espectro T(f), R(f) y A(f) ejecutando el motor FDTD en un hilo aislado (asyncio.to_thread)."""
    return await asyncio.to_thread(_compute_spectrum_sync, config)


@router.get("/benchmarks/airy")
def get_airy_benchmark(
    eps_r: float = 4.0,
    thickness_mm: float = 10.0,
    f_min_ghz: float = 5.0,
    f_max_ghz: float = 15.0,
    points: int = 100,
):
    """Retorna la solución analítica exacta de Airy para una lámina dieléctrica Fabry-Pérot."""
    freqs = np.linspace(f_min_ghz * 1e9, f_max_ghz * 1e9, points)
    t_airy = airy_dielectric_slab_transmission(eps_r, thickness_mm * 1e-3, freqs)
    return {
        "frequencies_ghz": [float(f * 1e-9) for f in freqs],
        "transmission": [round(float(v), 5) for v in t_airy],
        "formula": "T = [1 + 1/4 (n - 1/n)^2 sin^2(n k0 L)]^-1",
    }


@router.get("/benchmarks/plasma")
def get_plasma_benchmark(a_mm: float = 10.0, wire_width_mm: float = 1.0):
    """Calcula la frecuencia de plasma teórica de Pendry/Belov para un medio de hilos PEC."""
    a_m = a_mm * 1e-3
    w_m = wire_width_mm * 1e-3
    r_eff = 0.59 * (w_m / 2.0)
    fp_hz = wire_medium_plasma_frequency(a_m, r_eff)
    return {
        "a_mm": a_mm,
        "wire_width_mm": wire_width_mm,
        "effective_radius_mm": round(r_eff * 1e3, 3),
        "plasma_frequency_ghz": round(fp_hz * 1e-9, 3),
        "formula": "(k_p a)^2 = 2*pi / (ln(a / (2*pi*r)) + 0.5275)",
    }
