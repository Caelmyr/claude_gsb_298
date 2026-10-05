"""Linear-programming relaxation solver.

Solves the LP relaxation of the time-indexed formulation.  The LP optimum is a
*lower bound* on the true scheduling objective; the fractional solution is also
used to seed a heuristic schedule (tasks ordered by expected start time fed
through the SGS decoder), so the LP route yields both a bound and a usable,
feasible schedule.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from .. import models
from .base import Solver, register
from . import schedule_builder, simplex, time_indexed


@register
class LPSolver(Solver):
    name = "lp"

    def solve(self, problem: models.Problem,
              params: Dict[str, Any]) -> models.Solution:
        time_limit = float(params.get("time_limit", 60))
        t0 = time.time()

        try:
            model = time_indexed.build(problem)
        except ValueError as exc:
            return models.Solution(
                id=models.new_id("sol"), problem_id=problem.id, solver=self.name,
                status="error", message=str(exc), params=params,
                solve_time=round(time.time() - t0, 4))

        res = simplex.linprog(model.c, model.A_ub, model.b_ub,
                              model.A_eq, model.b_eq)

        elapsed = time.time() - t0
        if res.status == "infeasible":
            return models.Solution(
                id=models.new_id("sol"), problem_id=problem.id, solver=self.name,
                status="infeasible", message="LP relaxation is infeasible",
                lower_bound=res.objective, params=params,
                solve_time=round(elapsed, 4))

        expected = time_indexed.solution_from_x(problem, model, res.x)
        priorities = time_indexed.expected_starts_to_priorities(expected)
        starts = schedule_builder.decode(problem, priorities)

        sol = self.make_solution(
            problem, starts,
            status="optimal" if len(starts) == len(problem.tasks) else "feasible",
            solve_time=elapsed, params=params,
            message=model.objective_note or "LP relaxation solved",
            lower_bound=res.objective,
            extra_metrics={
                "lp_objective": round(res.objective, 4) if res.objective is not None else None,
                "lp_iterations": res.iterations,
                "n_variables": len(model.vars),
                "n_constraints": len(model.A_ub) + len(model.A_eq),
                "n_fractional": sum(1 for v in res.x if 1e-6 < v < 1 - 1e-6),
                "heuristic_recovered": len(starts) == len(problem.tasks),
            })
        # The "objective_value" of a relaxation is the bound; keep the recovered
        # schedule's own objective as the primary value, and the LP bound aside.
        return sol
