"""Router REST para el optimizador automático y re-validación (Capítulo 12)."""

from __future__ import annotations

import time
from fastapi import APIRouter, HTTPException

from api.schemas.optimizer import (
    OptimizerRequest,
    OptimizerRunResponse,
    GenerationReportSchema,
    GridSearchRequest,
    GridSearchResponse,
    RevalidationRequest,
    RevalidationResponse,
)
from api.schemas.sweep import SweepDesignPointSchema
from domain.optimization.search_space import SearchSpace, ParameterBounds
from domain.optimization.evaluator import FDTDEvaluator
from domain.optimization.genetic import evolve_optimizer
from domain.optimization.grid_search import run_grid_search
from domain.optimization.revalidation import revalidate_winner_design

router = APIRouter(prefix="/api/optimizer", tags=["Optimizer"])


@router.post("/run", response_model=OptimizerRunResponse)
async def run_optimizer_endpoint(req: OptimizerRequest):
    """Ejecuta el Algoritmo Genético de optimización de metamateriales de forma síncrona."""
    t0 = time.perf_counter()

    space = SearchSpace(
        family=req.family,
        a_bounds=ParameterBounds(min_val=req.a_min, max_val=req.a_max, step=1.0),
        d_bounds=ParameterBounds(min_val=req.d_min, max_val=req.d_max, step=0.5),
        w_bounds=ParameterBounds(min_val=req.w_min, max_val=req.w_max, step=0.25),
        dx_mm=req.dx_mm,
    )

    evaluator = FDTDEvaluator(
        family=req.family,
        periods=req.periods,
        is_pec=req.is_pec,
        relative_permittivity=req.relative_permittivity,
        dx_mm=req.dx_mm,
        f0_ghz=req.f0_ghz,
        max_steps=1400,
    )

    reports: list[GenerationReportSchema] = []
    best_overall_pt = None

    for rep in evolve_optimizer(
        evaluator=evaluator,
        space=space,
        pop_size=req.pop_size,
        generations=req.generations,
        seed=req.seed,
    ):
        pt_schema = SweepDesignPointSchema(**rep.best_design_point.__dict__)
        best_overall_pt = pt_schema
        reports.append(
            GenerationReportSchema(
                generation=rep.generation,
                best_fitness=rep.best_fitness,
                mean_fitness=rep.mean_fitness,
                best_individual=rep.best_individual,
                best_design_point=pt_schema,
                population=rep.population,
                fitness_values=rep.fitness_values,
                total_evaluations=rep.total_evaluations,
                cache_hits=rep.cache_hits,
            )
        )

    if not best_overall_pt:
        raise HTTPException(status_code=500, detail="La optimización no generó ningún diseño factible.")

    elapsed = time.perf_counter() - t0
    return OptimizerRunResponse(
        reports=reports,
        best_design=best_overall_pt,
        total_time_seconds=round(elapsed, 3),
        cache_hit_rate_percent=round(evaluator.stats.hit_rate_percent, 2),
    )


@router.post("/grid-search", response_model=GridSearchResponse)
async def grid_search_endpoint(req: GridSearchRequest):
    """Ejecuta una búsqueda sistemática exhaustiva en el espacio discreto."""
    space = SearchSpace(family=req.family, dx_mm=req.dx_mm)
    evaluator = FDTDEvaluator(
        family=req.family,
        periods=req.periods,
        is_pec=req.is_pec,
        relative_permittivity=req.relative_permittivity,
        dx_mm=req.dx_mm,
        f0_ghz=req.f0_ghz,
        max_steps=1400,
    )

    res = run_grid_search(
        evaluator=evaluator,
        space=space,
        max_evaluations=req.max_evaluations,
        stride=req.stride,
    )

    best_pt_schema = SweepDesignPointSchema(**res.best_design_point.__dict__) if res.best_design_point else None

    return GridSearchResponse(
        total_combinations_in_space=res.total_combinations_in_space,
        feasible_combinations=res.feasible_combinations,
        evaluations_executed=res.evaluations_executed,
        best_fitness=res.best_fitness,
        best_individual=res.best_individual,
        best_design=best_pt_schema,
        evaluations_history=res.evaluations_history,
        best_fitness_history=res.best_fitness_history,
    )


@router.post("/revalidate", response_model=RevalidationResponse)
async def revalidate_endpoint(req: RevalidationRequest):
    """Re-valida el diseño ganador en malla fina (Δx = 0.25 mm) y calcula la discrepancia espectral."""
    try:
        report = revalidate_winner_design(
            a_mm=req.a_mm,
            d_mm=req.d_mm,
            w_mm=req.w_mm,
            periods=req.periods,
            family_str=req.family,
            is_pec=req.is_pec,
            relative_permittivity=req.relative_permittivity,
            f0_ghz=req.f0_ghz,
            tolerance_db=req.tolerance_db,
        )
        return RevalidationResponse(**report.__dict__)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error durante la re-validación: {exc}")
