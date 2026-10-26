"""Router REST para el Laboratorio Paramétrico y Barridos FDTD (Capítulo 11)."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from fastapi import APIRouter, Response

from api.schemas.sweep import (
    Sweep1DRequestSchema,
    Sweep1DResponseSchema,
    Sweep2DRequestSchema,
    Sweep2DResponseSchema,
    RobustnessResponseSchema,
)
from domain.sweep import (
    run_1d_sweep,
    run_2d_sweep,
    export_sweep_to_csv,
    evaluate_sweep_robustness,
)

router = APIRouter(prefix="/sweep", tags=["Parametric Lab"])


@router.post("/1d", response_model=Sweep1DResponseSchema)
async def post_1d_sweep(req: Sweep1DRequestSchema):
    """Ejecuta un barrido 1D en un hilo aislado y devuelve la familia de curvas T(f) y tabla clasificada."""
    result = await asyncio.to_thread(
        run_1d_sweep,
        parameter_name=req.parameter_name,
        param_values=req.param_values,
        base_a_mm=req.base_a_mm,
        base_d_mm=req.base_d_mm,
        base_w_mm=req.base_w_mm,
        family_str=req.family,
        periods=req.periods,
        is_pec=req.is_pec,
        relative_permittivity=req.relative_permittivity,
        dx_mm=req.dx_mm,
        f0_ghz=req.f0_ghz,
    )
    return asdict(result)


@router.post("/2d", response_model=Sweep2DResponseSchema)
async def post_2d_sweep(req: Sweep2DRequestSchema):
    """Ejecuta un barrido 2D en un hilo aislado y devuelve matrices para mapas de calor T(f0, p)."""
    result = await asyncio.to_thread(
        run_2d_sweep,
        param_x_name=req.param_x_name,
        param_x_values=req.param_x_values,
        param_y_name=req.param_y_name,
        param_y_values=req.param_y_values,
        base_w_mm=req.base_w_mm,
        family_str=req.family,
        periods=req.periods,
        is_pec=req.is_pec,
        relative_permittivity=req.relative_permittivity,
        dx_mm=req.dx_mm,
        f0_ghz=req.f0_ghz,
    )
    return asdict(result)


@router.post("/export/csv")
async def post_export_csv(req: Sweep1DRequestSchema):
    """Ejecuta el barrido y devuelve directamente el contenido en formato CSV descargable."""
    result = await asyncio.to_thread(
        run_1d_sweep,
        parameter_name=req.parameter_name,
        param_values=req.param_values,
        base_a_mm=req.base_a_mm,
        base_d_mm=req.base_d_mm,
        base_w_mm=req.base_w_mm,
        family_str=req.family,
        periods=req.periods,
        is_pec=req.is_pec,
        relative_permittivity=req.relative_permittivity,
        dx_mm=req.dx_mm,
        f0_ghz=req.f0_ghz,
    )
    csv_str = export_sweep_to_csv(result.table_rows)
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=sweep_{req.parameter_name}.csv"},
    )


@router.get("/robustness", response_model=RobustnessResponseSchema)
async def get_sweep_robustness():
    """Ejecuta la prueba de robustez de conservación de ranking con correlación de Spearman."""
    res = await asyncio.to_thread(evaluate_sweep_robustness)
    return res
