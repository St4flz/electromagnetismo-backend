"""Tests unitarios para el optimizador genético, búsqueda sistemática y re-validación (Capítulo 12)."""

import numpy as np
import pytest

from domain.optimization.search_space import SearchSpace, ParameterBounds
from domain.optimization.evaluator import FDTDEvaluator
from domain.optimization.genetic import (
    evolve_optimizer,
    tournament_selection,
    blx_alpha_crossover,
    adaptive_gaussian_mutation,
)
from domain.optimization.grid_search import (
    run_grid_search,
    get_all_feasible_combinations,
)
from domain.optimization.revalidation import revalidate_winner_design


def test_search_space_feasibility_and_repair():
    """Verifica las restricciones físicas de la celda unitaria de anillo cuadrado y su proyección."""
    space = SearchSpace(
        family="square_ring",
        a_bounds=ParameterBounds(min_val=6.0, max_val=16.0, step=1.0),
        d_bounds=ParameterBounds(min_val=2.0, max_val=8.0, step=0.5),
        w_bounds=ParameterBounds(min_val=0.5, max_val=2.5, step=0.25),
        dx_mm=0.5,
    )

    # Caso inválido: hueco interior (a - d) <= 2*w
    # a = 8.0, d = 6.0 => hueco disponible = 2.0. Con w = 1.5 => 2*w = 3.0 > 2.0 (imposible)
    assert not space.is_feasible([8.0, 6.0, 1.5])

    # Reparar el individuo debe proyectarlo a un estado geométrico válido
    repaired = space.repair_individual([8.0, 6.0, 1.5])
    assert space.is_feasible(repaired)
    # Comprobar cuantización a múltiplos del paso/malla
    assert repaired[0] % 0.5 == 0
    assert repaired[1] % 0.5 == 0
    assert repaired[2] % 0.25 == 0


def test_evaluator_cache_hits():
    """Comprueba que la evaluación FDTD reutilice en 0 ms individuos ya calculados."""
    evaluator = FDTDEvaluator(
        family="square_ring",
        periods=2,
        dx_mm=0.5,
        f0_ghz=10.0,
        max_steps=300,  # corrida rápida para prueba
    )

    ind = [10.0, 3.0, 1.0]

    # Primera evaluación -> simulación física
    score1, pt1 = evaluator.evaluate_individual(ind, design_id=1)
    assert evaluator.stats.total_requested == 1
    assert evaluator.stats.simulations_executed == 1
    assert evaluator.stats.cache_hits == 0

    # Segunda evaluación idéntica -> caché hit inmediato
    score2, pt2 = evaluator.evaluate_individual(ind, design_id=2)
    assert evaluator.stats.total_requested == 2
    assert evaluator.stats.simulations_executed == 1
    assert evaluator.stats.cache_hits == 1
    assert score1 == score2
    assert evaluator.stats.hit_rate_percent == 50.0


def test_genetic_algorithm_reproducibility():
    """Garantiza reproducibilidad estricta: dos corridas con la misma semilla son idénticas bit a bit."""
    space = SearchSpace(
        family="square_ring",
        a_bounds=ParameterBounds(min_val=8.0, max_val=12.0, step=1.0),
        d_bounds=ParameterBounds(min_val=2.0, max_val=4.0, step=0.5),
        w_bounds=ParameterBounds(min_val=0.5, max_val=1.5, step=0.25),
        dx_mm=0.5,
    )

    def run_with_seed(seed: int):
        evaluator = FDTDEvaluator(
            family="square_ring",
            periods=2,
            dx_mm=0.5,
            f0_ghz=10.0,
            max_steps=200,
        )
        reports = list(
            evolve_optimizer(
                evaluator=evaluator,
                space=space,
                pop_size=6,
                generations=3,
                seed=seed,
            )
        )
        return [r.best_fitness for r in reports], [r.best_individual for r in reports]

    fitness1, ind1 = run_with_seed(42)
    fitness2, ind2 = run_with_seed(42)

    assert fitness1 == fitness2
    assert ind1 == ind2


def test_grid_search_and_feasible_combinations():
    """Comprueba la generación sistemática exhaustiva en el espacio discreto."""
    space = SearchSpace(
        family="square_ring",
        a_bounds=ParameterBounds(min_val=10.0, max_val=12.0, step=1.0),
        d_bounds=ParameterBounds(min_val=2.0, max_val=3.0, step=0.5),
        w_bounds=ParameterBounds(min_val=0.5, max_val=1.0, step=0.5),
        dx_mm=0.5,
    )

    combos = get_all_feasible_combinations(space)
    assert len(combos) > 0
    for cand in combos:
        assert space.is_feasible(cand)

    evaluator = FDTDEvaluator(
        family="square_ring",
        periods=2,
        dx_mm=0.5,
        f0_ghz=10.0,
        max_steps=200,
    )

    report = run_grid_search(evaluator, space, max_evaluations=4, stride=1)
    assert report.evaluations_executed == 4
    assert len(report.best_fitness_history) == 4
    assert report.best_fitness < float("inf")


def test_winner_revalidation_tolerance():
    """Valida que la re-validación del diseño en malla fina (Δx = 0.25 mm) se ejecute y cumpla |ΔT(f0)| < 2 dB."""
    report = revalidate_winner_design(
        a_mm=10.0,
        d_mm=3.0,
        w_mm=1.0,
        periods=2,
        family_str="square_ring",
        is_pec=True,
        f0_ghz=10.0,
        tolerance_db=2.0,
    )

    assert report.dx_fast_mm == 0.50
    assert report.dx_fine_mm == 0.25
    assert len(report.freqs_ghz) > 0
    assert len(report.t_fast_spectrum) == len(report.t_fine_spectrum)
    assert report.delta_t_db >= 0.0
    # Criterio de rigor físico: el error entre mallas es menor a la tolerancia
    assert report.delta_t_db < 2.0
    assert report.passed_tolerance is True
