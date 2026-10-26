"""Algoritmo Genético con selección por torneo, cruce BLX-α, mutación gaussiana adaptativa y elitismo."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterator, Sequence
import numpy as np

from domain.sweep import SweepDesignPoint
from domain.optimization.search_space import SearchSpace
from domain.optimization.evaluator import FDTDEvaluator


@dataclass
class GenerationReport:
    """Informe detallado emitido en cada generación para monitoreo en vivo."""

    generation: int
    best_fitness: float
    mean_fitness: float
    best_individual: list[float]          # [a, d, w]
    best_design_point: SweepDesignPoint
    population: list[list[float]]         # Coordenadas (a, d, w) para scatter plot
    fitness_values: list[float]
    total_evaluations: int
    cache_hits: int


def tournament_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    k: int = 3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Selección por torneo: elige k individuos al azar y gana el de menor fitness (minimización)."""
    if rng is None:
        rng = np.random.default_rng()

    pop_size = len(population)
    parents = np.empty_like(population)

    for i in range(pop_size):
        # Elegir k candidatos al azar sin reemplazo
        candidates = rng.choice(pop_size, size=min(k, pop_size), replace=False)
        best_cand = candidates[np.argmin(fitness[candidates])]
        parents[i] = population[best_cand]

    return parents


def blx_alpha_crossover(
    parents: np.ndarray,
    alpha: float = 0.5,
    p_c: float = 0.90,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Cruce Blend-α (BLX-α) para variables continuas:

    c_i in U[min(p_i, q_i) - alpha * I_i, max(p_i, q_i) + alpha * I_i], donde I_i = |p_i - q_i|.
    """
    if rng is None:
        rng = np.random.default_rng()

    pop_size, num_genes = parents.shape
    children = np.empty_like(parents)

    # Procesar parejas (i, i+1)
    for i in range(0, pop_size, 2):
        p1 = parents[i]
        p2 = parents[(i + 1) % pop_size]

        if rng.random() < p_c:
            for g in range(num_genes):
                min_v = min(p1[g], p2[g])
                max_v = max(p1[g], p2[g])
                diff = max_v - min_v
                low = min_v - alpha * diff
                high = max_v + alpha * diff
                children[i, g] = rng.uniform(low, high)
                children[(i + 1) % pop_size, g] = rng.uniform(low, high)
        else:
            children[i] = p1
            children[(i + 1) % pop_size] = p2

    return children


def adaptive_gaussian_mutation(
    children: np.ndarray,
    span: np.ndarray,
    generation: int,
    max_generations: int,
    p_m: float = 0.25,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Mutación gaussiana con recocido: sigma decrece linealmente con el avance de las generaciones.

    sigma_g = 0.1 * span * (1 - gen / max_gens).
    """
    if rng is None:
        rng = np.random.default_rng()

    mutated = children.copy()
    annealing = max(0.05, 1.0 - (generation / max(max_generations, 1)))
    sigma = 0.15 * span * annealing

    pop_size, num_genes = children.shape
    for i in range(pop_size):
        for g in range(num_genes):
            if rng.random() < p_m:
                mutated[i, g] += rng.normal(0.0, sigma[g])

    return mutated


def evolve_optimizer(
    evaluator: FDTDEvaluator,
    space: SearchSpace,
    pop_size: int = 16,
    generations: int = 20,
    seed: int = 42,
    elite_count: int = 2,
    tournament_k: int = 3,
    blx_alpha: float = 0.5,
    mutation_prob: float = 0.25,
) -> Iterator[GenerationReport]:
    """Generador que ejecuta el ciclo evolutivo del algoritmo genético y emite un informe por generación."""
    rng = np.random.default_rng(seed)
    pop = space.sample(rng, pop_size)

    global_best_fitness = float("inf")
    global_best_pt: SweepDesignPoint | None = None

    for gen in range(generations):
        # 1. Evaluación de aptitud (fitness) con caché
        base_id = gen * pop_size
        fitness, points = evaluator.evaluate_batch(pop, base_id=base_id)

        # 2. Identificar el mejor de la generación y actualizar récord histórico
        best_idx = int(np.argmin(fitness))
        gen_best_fitness = float(fitness[best_idx])
        gen_mean_fitness = float(np.mean(fitness))

        if gen_best_fitness < global_best_fitness:
            global_best_fitness = gen_best_fitness
            global_best_pt = points[best_idx]

        # Emitir reporte para streaming
        report = GenerationReport(
            generation=gen + 1,
            best_fitness=round(gen_best_fitness, 5),
            mean_fitness=round(gen_mean_fitness, 5),
            best_individual=[round(float(v), 2) for v in pop[best_idx]],
            best_design_point=global_best_pt or points[best_idx],
            population=[[round(float(coord), 2) for coord in ind] for ind in pop],
            fitness_values=[round(float(f), 5) for f in fitness],
            total_evaluations=evaluator.stats.total_requested,
            cache_hits=evaluator.stats.cache_hits,
        )
        yield report

        # 3. Elitismo: conservar los mejores individuos intactos
        order = np.argsort(fitness)
        elites = pop[order[:elite_count]].copy()

        # 4. Selección por torneo
        parents = tournament_selection(pop, fitness, k=tournament_k, rng=rng)

        # 5. Cruce BLX-alpha
        children = blx_alpha_crossover(parents, alpha=blx_alpha, rng=rng)

        # 6. Mutación gaussiana adaptativa
        mutated = adaptive_gaussian_mutation(
            children,
            span=space.span,
            generation=gen,
            max_generations=generations,
            p_m=mutation_prob,
            rng=rng,
        )

        # 7. Reparación geométrica y reemplazo de población
        repaired = space.repair(mutated)
        pop = np.vstack([elites, repaired[: pop_size - elite_count]])
