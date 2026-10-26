"""Búsqueda exhaustiva sistemática (Grid Search) en el espacio de parámetros cuantizado (Capítulo 12)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from domain.sweep import SweepDesignPoint
from domain.optimization.search_space import SearchSpace
from domain.optimization.evaluator import FDTDEvaluator


@dataclass
class GridSearchReport:
    """Reporte de la búsqueda sistemática para comparación con el Algoritmo Genético."""
    total_combinations_in_space: int
    feasible_combinations: int
    evaluations_executed: int
    best_fitness: float
    best_individual: list[float]  # [a, d, w]
    best_design_point: SweepDesignPoint | None
    evaluations_history: list[int]
    best_fitness_history: list[float]


def get_all_feasible_combinations(space: SearchSpace) -> list[list[float]]:
    """Genera todas las combinaciones discretas (a, d, w) físicamente válidas en el espacio de búsqueda."""
    a_step = max(space.a_bounds.step, space.dx_mm)
    d_step = max(space.d_bounds.step, space.dx_mm)
    w_step = max(space.w_bounds.step, space.dx_mm)

    a_vals = np.arange(space.a_bounds.min_val, space.a_bounds.max_val + 1e-5, a_step)
    d_vals = np.arange(space.d_bounds.min_val, space.d_bounds.max_val + 1e-5, d_step)
    w_vals = np.arange(space.w_bounds.min_val, space.w_bounds.max_val + 1e-5, w_step)

    feasible_pts: list[list[float]] = []
    for a in a_vals:
        for d in d_vals:
            for w in w_vals:
                cand = [round(float(a), 2), round(float(d), 2), round(float(w), 2)]
                if space.is_feasible(cand):
                    feasible_pts.append(cand)

    return feasible_pts


def run_grid_search(
    evaluator: FDTDEvaluator,
    space: SearchSpace,
    max_evaluations: int = 150,
    stride: int = 1,
) -> GridSearchReport:
    """Ejecuta una búsqueda sistemática evaluando combinaciones válidas.
    
    Permite establecer un stride o un límite de evaluaciones para benchmarks rápidos,
    registrando la curva de progreso (best_fitness vs evaluations).
    """
    all_feasible = get_all_feasible_combinations(space)
    total_feasible = len(all_feasible)

    # Subconjunto a evaluar según stride o límite
    candidates_to_eval = all_feasible[::stride][:max_evaluations]

    best_fitness = float("inf")
    best_ind: list[float] = [0.0, 0.0, 0.0]
    best_pt: SweepDesignPoint | None = None

    eval_history: list[int] = []
    fitness_history: list[float] = []

    for i, ind in enumerate(candidates_to_eval):
        score, pt = evaluator.evaluate_individual(ind, design_id=i + 1)
        if score < best_fitness:
            best_fitness = score
            best_ind = ind
            best_pt = pt

        eval_history.append(i + 1)
        fitness_history.append(round(best_fitness, 5))

    return GridSearchReport(
        total_combinations_in_space=len(all_feasible),
        feasible_combinations=total_feasible,
        evaluations_executed=len(candidates_to_eval),
        best_fitness=round(best_fitness, 5),
        best_individual=best_ind,
        best_design_point=best_pt,
        evaluations_history=eval_history,
        best_fitness_history=fitness_history,
    )
