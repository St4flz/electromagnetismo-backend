"""Módulo de Optimización Automática de Metamateriales (Capítulo 12)."""

from domain.optimization.search_space import ParameterBounds, SearchSpace
from domain.optimization.evaluator import FDTDEvaluator, EvaluationCacheStats
from domain.optimization.genetic import (
    GenerationReport,
    tournament_selection,
    blx_alpha_crossover,
    adaptive_gaussian_mutation,
    evolve_optimizer,
)
from domain.optimization.grid_search import GridSearchReport, run_grid_search, get_all_feasible_combinations
from domain.optimization.revalidation import RevalidationReport, revalidate_winner_design

__all__ = [
    "ParameterBounds",
    "SearchSpace",
    "FDTDEvaluator",
    "EvaluationCacheStats",
    "GenerationReport",
    "tournament_selection",
    "blx_alpha_crossover",
    "adaptive_gaussian_mutation",
    "evolve_optimizer",
    "GridSearchReport",
    "run_grid_search",
    "get_all_feasible_combinations",
    "RevalidationReport",
    "revalidate_winner_design",
]
