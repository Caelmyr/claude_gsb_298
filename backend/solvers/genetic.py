"""Genetic algorithm solver.

Representation is a *random-key* vector (one priority in [0, 1) per task).  The
SGS decoder maps a priority vector to a precedence- and capacity-feasible
schedule deterministically, so any crossover or mutation remains valid without
repair -- the usual precedence-repair step for permutation encodings disappears.

Operators: uniform crossover, per-gene random reset mutation, k-tournament
selection, and elitism.  The objective is minimised.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple

from .. import models
from .base import INFEASIBILITY_PENALTY, Solver, evaluate_priorities, register
from . import schedule_builder


@register
class GeneticSolver(Solver):
    name = "genetic"

    def solve(self, problem: models.Problem,
              params: Dict[str, Any]) -> models.Solution:
        pop_size = int(params.get("population_size", 60))
        generations = int(params.get("generations", 150))
        mutation_rate = float(params.get("mutation_rate", 0.15))
        crossover_rate = float(params.get("crossover_rate", 0.8))
        elitism = int(params.get("elitism", 4))
        seed = int(params.get("seed", 42))
        time_limit = float(params.get("time_limit", 30))

        rng = random.Random(seed)
        t0 = time.time()
        task_ids = [t.id for t in problem.tasks]

        def random_individual() -> Dict[str, float]:
            return {tid: rng.random() for tid in task_ids}

        def fitness(ind: Dict[str, float]) -> Tuple[float, Dict[str, int]]:
            return evaluate_priorities(problem, ind)

        # initialise population
        population: List[Tuple[Dict[str, float], float, Dict[str, int]]] = []
        for _ in range(pop_size):
            ind = random_individual()
            starts, obj = fitness(ind)
            population.append((ind, obj, starts))

        best = min(population, key=lambda p: p[1])
        generations_run = 0
        timed_out = False

        for gen in range(generations):
            if time.time() - t0 > time_limit:
                timed_out = True
                break
            generations_run = gen + 1

            population.sort(key=lambda p: p[1])
            elites = population[:elitism]

            next_pop: List[Tuple[Dict[str, float], float, Dict[str, int]]] = list(elites)
            while len(next_pop) < pop_size:
                p1 = self._tournament(population, rng)
                p2 = self._tournament(population, rng)
                if rng.random() < crossover_rate:
                    child = self._uniform_crossover(p1[0], p2[0], rng)
                else:
                    child = dict(p1[0])
                child = self._mutate(child, mutation_rate, rng)
                starts, obj = fitness(child)
                next_pop.append((child, obj, starts))

            population = next_pop
            cand = min(population, key=lambda p: p[1])
            if cand[1] < best[1]:
                best = cand

        _, best_obj, best_starts = best
        feasible = len(best_starts) == len(problem.tasks)
        sol = self.make_solution(
            problem, best_starts,
            status="feasible" if feasible else "infeasible",
            solve_time=time.time() - t0,
            message=("time limit reached" if timed_out else
                     f"{generations_run} generations, pop {pop_size}"),
            params=params,
            extra_metrics={
                "generations": generations_run,
                "population_size": pop_size,
                "fitness": round(best_obj, 4),
                "feasible": feasible,
            })
        return sol

    # -- operators -------------------------------------------------------- #
    def _tournament(self, population, rng, k: int = 3):
        sample = [population[rng.randrange(len(population))] for _ in range(k)]
        return min(sample, key=lambda p: p[1])

    def _uniform_crossover(self, a: Dict[str, float], b: Dict[str, float],
                           rng) -> Dict[str, float]:
        return {tid: (a[tid] if rng.random() < 0.5 else b[tid]) for tid in a}

    def _mutate(self, ind: Dict[str, float], rate: float, rng) -> Dict[str, float]:
        out = dict(ind)
        for tid in out:
            if rng.random() < rate:
                out[tid] = rng.random()
        return out
