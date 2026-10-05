"""
Objective evaluation and soft-constraint penalty computation.

A "schedule" here is a dict mapping task id -> start time (``starts``).  The
evaluation functions are pure and shared by every solver so that GA/SA/IP all
optimise exactly the same quantity.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from . import models


def completion_times(problem: models.Problem, starts: Dict[str, int]) -> Dict[str, int]:
    """Completion time C_j = start_j + duration_j."""
    return {tid: s + problem.task_map()[tid].duration for tid, s in starts.items()}


def makespan(problem: models.Problem, starts: Dict[str, int]) -> int:
    if not starts:
        return 0
    return max(completion_times(problem, starts).values())


def total_completion(problem: models.Problem, starts: Dict[str, int]) -> int:
    return sum(completion_times(problem, starts).values())


def weighted_completion(problem: models.Problem, starts: Dict[str, int]) -> float:
    tmap = problem.task_map()
    return sum(tmap[t].weight * (starts[t] + tmap[t].duration) for t in starts)


def total_tardiness(problem: models.Problem, starts: Dict[str, int]) -> float:
    tmap = problem.task_map()
    comp = completion_times(problem, starts)
    return sum(
        tmap[t].weight * max(0, comp[t] - tmap[t].due_date)
        for t in starts
        if tmap[t].due_date is not None
    )


def resource_cost(problem: models.Problem, starts: Dict[str, int]) -> float:
    rmap = problem.resource_map()
    tmap = problem.task_map()
    cost = 0.0
    for t, s in starts.items():
        task = tmap[t]
        for r, amount in task.resource_requirements.items():
            if r in rmap:
                cost += amount * task.duration * rmap[r].cost_per_unit
    return cost


# Primitive objective evaluators -------------------------------------------------- #

_PRIMITIVES = {
    "makespan": makespan,
    "total_completion": total_completion,
    "weighted_completion": weighted_completion,
    "tardiness": total_tardiness,
    "cost": resource_cost,
}


def evaluate_primitive(name: str, problem: models.Problem,
                       starts: Dict[str, int]) -> float:
    if name not in _PRIMITIVES:
        raise ValueError(f"unknown primitive objective: {name}")
    return float(_PRIMITIVES[name](problem, starts))


def evaluate_objective(problem: models.Problem,
                       starts: Dict[str, int],
                       soft_penalties: float = 0.0) -> float:
    """Total objective = base objective + soft-constraint penalty contributions.

    * ``type == custom`` -> weighted sum of the primitives given in ``weights``.
    * otherwise -> the single primitive, plus soft penalties.
    """
    o = problem.objective
    if o.type == "custom":
        value = sum(
            w * evaluate_primitive(k, problem, starts)
            for k, w in o.weights.items()
        )
    else:
        value = evaluate_primitive(o.type, problem, starts)
    return value + soft_penalties


# Soft constraint penalties ------------------------------------------------------- #

def soft_penalty(problem: models.Problem, starts: Dict[str, int]) -> float:
    """Sum of all soft-constraint penalties for a schedule."""
    total = 0.0
    comp = completion_times(problem, starts)
    tmap = problem.task_map()
    rmap = problem.resource_map()
    for c in problem.soft_constraints:
        p = c.params
        contrib = 0.0
        if c.type == "due_date":
            t = p.get("task")
            if t and t in starts:
                due = p.get("due", tmap[t].due_date)
                if due is not None:
                    contrib = max(0, comp[t] - due)
        elif c.type == "preferred_window":
            t = p.get("task")
            if t and t in starts:
                lo, hi = p.get("start"), p.get("end")
                s = starts[t]
                if s < lo:
                    contrib = lo - s
                elif s > hi:
                    contrib = s - hi
        elif c.type == "min_gap":
            a, b = p.get("a"), p.get("b")
            gap_req = p.get("gap", 0)
            if a in starts and b in starts:
                gap = max(starts[b] - comp[a], starts[a] - comp[b])
                contrib = max(0, gap_req - gap)
        elif c.type == "max_makespan":
            target = p.get("target", problem.horizon)
            contrib = max(0, makespan(problem, starts) - target)
        elif c.type == "setup_time":
            # penalise consecutive jobs on a shared resource with no setup slack
            resource = p.get("resource")
            setup = p.get("setup", 0)
            if resource in rmap:
                jobs = sorted(
                    (starts[t], comp[t]) for t in starts
                    if resource in tmap[t].resource_requirements
                )
                for (_, e1), (s2, _) in zip(jobs, jobs[1:]):
                    contrib += max(0, setup - (s2 - e1))
        elif c.type == "resource_balance":
            # penalise variance of per-resource utilisation
            resource = p.get("resource")
            if resource in rmap:
                load = sum(
                    tmap[t].resource_requirements.get(resource, 0) * tmap[t].duration
                    for t in starts
                )
                # simple deviation from 0 is meaningless; use idle-time proxy
                contrib = abs(load - p.get("target_load", load))
        total += c.penalty * c.factor * contrib
    return total


# Hard constraint checking -------------------------------------------------------- #

def hard_violations(problem: models.Problem, starts: Dict[str, int]) -> List[str]:
    """Return a list of hard-constraint violation descriptions (empty = feasible)."""
    violations: List[str] = []
    tmap = problem.task_map()
    comp = completion_times(problem, starts)

    # precedence
    for b, a in problem.precedence_edges():
        if b in starts and a in starts and starts[a] < comp[b]:
            violations.append(f"precedence {b} -> {a}")

    # time windows / fixed start / release times
    for t in problem.tasks:
        if t.id not in starts:
            violations.append(f"task {t.id} has no start time")
            continue
        if starts[t.id] < t.release_time:
            violations.append(f"task {t.id} before release {t.release_time}")
        if t.due_date is not None and comp[t.id] > t.due_date and _due_is_hard(problem):
            violations.append(f"task {t.id} exceeds due date {t.due_date}")

    for c in problem.hard_constraints:
        p = c.params
        if c.type == "time_window":
            t = p.get("task")
            if t in starts:
                if p.get("release") is not None and starts[t] < p["release"]:
                    violations.append(f"{c.id}: {t} before window release")
                if p.get("deadline") is not None and comp[t] > p["deadline"]:
                    violations.append(f"{c.id}: {t} after window deadline")
        elif c.type == "fixed_start":
            t = p.get("task")
            if t in starts and starts[t] != p.get("start"):
                violations.append(f"{c.id}: {t} not at fixed start {p.get('start')}")
        elif c.type == "non_overlap":
            tasks = p.get("tasks", [])
            spans = [(starts[t], comp[t]) for t in tasks if t in starts]
            for i in range(len(spans)):
                for j in range(i + 1, len(spans)):
                    s1, e1 = spans[i]
                    s2, e2 = spans[j]
                    if s1 < e2 and s2 < e1:
                        violations.append(f"{c.id}: overlap {tasks[i]}/{tasks[j]}")
        elif c.type == "max_concurrent":
            limit = p.get("limit", 1)
            if starts:
                tmin = min(starts.values())
                tmax = max(comp.values())
                for slot in range(tmin, tmax):
                    active = sum(1 for t in starts if starts[t] <= slot < comp[t])
                    if active > limit:
                        violations.append(f"{c.id}: {active} concurrent at t={slot} > {limit}")
                        break
        elif c.type == "resource_capacity":
            res = p.get("resource")
            cap = p.get("capacity", 1)
            if starts:
                tmin = min(starts.values())
                tmax = max(comp.values())
                for slot in range(tmin, tmax):
                    used = sum(
                        tmap[t].resource_requirements.get(res, 0)
                        for t in starts if starts[t] <= slot < comp[t]
                    )
                    if used > cap:
                        violations.append(f"{c.id}: resource {res} overloaded at t={slot}")
                        break
        elif c.type == "resource_assignment":
            t = p.get("task")
            required = set(p.get("resources", []))
            # ensure at least one of the required resources is engaged by the task
            if t in starts and required and required.isdisjoint(tmap[t].resource_requirements):
                violations.append(f"{c.id}: {t} missing required resources")

    # resource capacity implied by each resource
    for r in problem.resources:
        if starts:
            tmin = min(starts.values())
            tmax = max(comp.values())
            for slot in range(tmin, tmax):
                used = sum(
                    tmap[t].resource_requirements.get(r.id, 0)
                    for t in starts if starts[t] <= slot < comp[t]
                )
                if used > r.capacity:
                    violations.append(f"resource {r.id} capacity {r.capacity} exceeded at t={slot}")
                    break

    return violations


def is_feasible(problem: models.Problem, starts: Dict[str, int]) -> bool:
    return len(hard_violations(problem, starts)) == 0


def _due_is_hard(problem: models.Problem) -> bool:
    """Due dates are hard only when the problem declares them so via a hard
    ``time_window`` deadline; plain task.due_date is soft (tardiness)."""
    return any(
        c.type == "time_window" and c.params.get("deadline") is not None
        for c in problem.hard_constraints
    )


# Schedule metrics for reporting -------------------------------------------------- #

def compute_metrics(problem: models.Problem,
                    starts: Dict[str, int]) -> Dict[str, object]:
    """Aggregate metrics for a schedule (used by UI and reports)."""
    comp = completion_times(problem, starts)
    ms = makespan(problem, starts)
    n = len(problem.tasks)
    total = sum(comp.values())
    # resource utilisation
    rmap = problem.resource_map()
    tmap = problem.task_map()
    utilisation = {}
    for r in problem.resources:
        used = sum(
            tmap[t].resource_requirements.get(r.id, 0) * tmap[t].duration
            for t in starts
        )
        utilis = used / max(1, r.capacity * ms) if ms > 0 else 0.0
        utilisation[r.id] = round(min(1.0, utilis), 4)
    return {
        "makespan": ms,
        "total_completion": total,
        "mean_completion": round(total / n, 3) if n else 0.0,
        "weighted_completion": round(weighted_completion(problem, starts), 3),
        "total_tardiness": round(total_tardiness(problem, starts), 3),
        "resource_cost": round(resource_cost(problem, starts), 3),
        "soft_penalty": round(soft_penalty(problem, starts), 3),
        "utilisation": utilisation,
        "n_violations": len(hard_violations(problem, starts)),
    }
