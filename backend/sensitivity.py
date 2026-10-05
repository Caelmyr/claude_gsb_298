"""
Sensitivity analysis.

One-at-a-time (OAT) perturbation: a single parameter is varied across a set of
levels while everything else is held fixed, and the chosen solver is re-run for
each level.  The result is a tornado-style table of objective responses that
reveals which inputs the optimum is most sensitive to.

Supported parameter kinds (see :func:`build_variations`):
  resource_capacity   -- scale a resource's capacity
  task_duration       -- shift a task's duration
  release_time        -- shift a task's release time
  due_date            -- shift a task's due date
  horizon             -- shift the planning horizon
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from . import models, storage


def clone_problem(problem: models.Problem) -> models.Problem:
    return models.Problem.from_dict(problem.to_dict())


def build_variations(problem: models.Problem,
                     spec: Dict[str, Any]) -> List[Tuple[str, models.Problem]]:
    """Return (label, modified_problem) pairs for a sensitivity spec.

    ``spec`` keys: ``kind`` plus the parameters documented above; deltas are
    supplied as ``multipliers`` (capacity) or ``offsets`` (everything else).
    """
    kind = spec.get("kind", "resource_capacity")
    out: List[Tuple[str, models.Problem]] = []

    if kind == "resource_capacity":
        rid = spec.get("resource")
        muls = spec.get("multipliers", [0.5, 0.75, 1.0, 1.25, 1.5])
        for m in muls:
            p = clone_problem(problem)
            for r in p.resources:
                if r.id == rid:
                    r.capacity = round(r.capacity * m, 4)
                    break
            out.append((f"{rid} capacity ×{m}", p))

    elif kind == "task_duration":
        tid = spec.get("task")
        offs = spec.get("offsets", [-2, -1, 0, 1, 2])
        for o in offs:
            p = clone_problem(problem)
            for t in p.tasks:
                if t.id == tid:
                    t.duration = max(1, t.duration + o)
                    break
            out.append((f"{tid} duration {o:+d}", p))

    elif kind == "release_time":
        tid = spec.get("task")
        offs = spec.get("offsets", [-2, -1, 0, 1, 2])
        for o in offs:
            p = clone_problem(problem)
            for t in p.tasks:
                if t.id == tid:
                    t.release_time = max(0, t.release_time + o)
                    break
            out.append((f"{tid} release {o:+d}", p))

    elif kind == "due_date":
        tid = spec.get("task")
        offs = spec.get("offsets", [-4, -2, 0, 2, 4])
        for o in offs:
            p = clone_problem(problem)
            for t in p.tasks:
                if t.id == tid and t.due_date is not None:
                    t.due_date = max(0, t.due_date + o)
                    break
            out.append((f"{tid} due {o:+d}", p))

    elif kind == "horizon":
        offs = spec.get("offsets", [-10, -5, 0, 5, 10])
        for o in offs:
            p = clone_problem(problem)
            p.horizon = max(1, p.horizon + o)
            out.append((f"horizon {o:+d}", p))

    else:
        raise ValueError(f"unknown sensitivity kind: {kind}")

    return out


def run_sensitivity(problem: models.Problem,
                    solver_name: str,
                    spec: Dict[str, Any],
                    solver_params: Optional[Dict[str, Any]] = None,
                    persist: bool = False) -> models.SensitivityResult:
    """Run OAT sensitivity with the given solver.  Persists when ``persist``."""
    from .solvers.base import get_solver, default_params

    solver = get_solver(solver_name)
    params = dict(solver_params or default_params(solver_name))

    base = solver.solve(problem, params)
    base_obj = base.objective_value if base.objective_value is not None else 0.0

    variations: List[Dict[str, Any]] = []
    for label, variant in build_variations(problem, spec):
        sol = solver.solve(variant, params)
        obj = sol.objective_value if sol.objective_value is not None else None
        variations.append({
            "label": label,
            "objective_value": obj,
            "makespan": sol.makespan,
            "status": sol.status,
            "delta": (round(obj - base_obj, 4) if obj is not None and
                      base_obj is not None else None),
            "solve_time": sol.solve_time,
        })

    result = models.SensitivityResult(
        id=models.new_id("sens"),
        problem_id=problem.id,
        solver=solver_name,
        base_objective=round(base_obj, 4) if base_obj is not None else None,
        parameter=f"{spec.get('kind')}:{spec.get('resource') or spec.get('task') or ''}",
        variations=variations,
    )
    if persist:
        storage.save_sensitivity(problem.id, result)
    return result
