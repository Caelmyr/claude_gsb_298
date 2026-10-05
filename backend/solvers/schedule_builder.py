"""
Serial Schedule Generation Scheme (SGS) decoder.

This is the workhorse that turns a *priority* representation into a concrete
schedule.  Meta-heuristics (genetic algorithm, simulated annealing) explore the
space of priority vectors / task orderings; the decoder deterministically maps
each one to a feasible schedule by scheduling each task as early as possible
while respecting:

* precedence constraints (explicit + hard ``precedence`` constraints),
* per-resource capacity over the task's whole execution window,
* resource availability calendars,
* task release times and hard ``time_window`` / ``fixed_start`` constraints.

The decoder also implements a *greedy* priority rule (highest priority weight,
then earliest release, then most successors) so that a deterministic baseline
solver is available for free.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .. import models


def topological_order(problem: models.Problem,
                      priorities: Optional[Dict[str, float]] = None) -> List[str]:
    """Return a precedence-feasible task order.  When ``priorities`` is given the
    available task with the *highest* priority is chosen first (random-key style
    decoding); otherwise tasks are ordered by the problem's own priority field,
    then by a stable heuristic (most successors / shortest duration)."""
    tasks = problem.task_map()
    succ = problem.successors()
    edges = problem.precedence_edges()
    indeg = {t: 0 for t in tasks}
    for b, a in edges:
        if b in indeg and a in indeg:
            indeg[a] += 1

    order: List[str] = []
    while len(order) < len(tasks):
        available = [t for t in tasks if indeg[t] == 0 and t not in order]
        if not available:
            # cycle: fall back to any unscheduled task
            available = [t for t in tasks if t not in order]

        def key(t: str) -> Tuple[float, float, float, int, str]:
            if priorities is not None:
                p = priorities.get(t, 0.0)
            else:
                p = tasks[t].priority
            # primary: priority (higher first -> negate)
            return (
                -p,
                -len(succ.get(t, [])),
                tasks[t].release_time,
                tasks[t].duration,
                t,
            )

        chosen = min(available, key=key)
        order.append(chosen)
        for s in succ.get(chosen, []):
            indeg[s] -= 1
    return order


def _resource_usage_at(task: models.Task, resource_id: str) -> float:
    return task.resource_requirements.get(resource_id, 0.0)


def _feasible_start(problem: models.Problem, task_id: str,
                    starts: Dict[str, int]) -> Optional[int]:
    """Earliest feasible start time for ``task_id`` given already-placed tasks.

    Returns None if no start time within the horizon exists (e.g. an
    availability window makes the task unschedulable)."""
    task = problem.task_map()[task_id]
    d = task.duration
    rmap = problem.resource_map()

    # earliest bound: release time + precedence completion
    earliest = task.release_time
    for dep in task.dependencies:
        if dep in starts:
            earliest = max(earliest, starts[dep] + problem.task_map()[dep].duration)
    # hard precedence constraints referencing this task as "after"
    for c in problem.hard_constraints:
        if c.type == "precedence" and c.params.get("after") == task_id:
            before = c.params.get("before")
            if before in starts:
                earliest = max(earliest, starts[before] + problem.task_map()[before].duration)

    # time_window / fixed_start hard constraints
    for c in problem.hard_constraints:
        p = c.params
        if p.get("task") != task_id:
            continue
        if c.type == "time_window":
            if p.get("release") is not None:
                earliest = max(earliest, p["release"])
        elif c.type == "fixed_start":
            s = p.get("start")
            if s is not None:
                return s if s >= earliest and s + d <= problem.horizon else None

    horizon = problem.horizon
    for t in range(earliest, horizon - d + 1):
        if _window_ok(problem, task, t, starts, rmap):
            # fixed window deadline check
            if not _window_deadline_ok(problem, task_id, t):
                continue
            return t
    return None


def _effective_capacity(problem: models.Problem, resource_id: str,
                        default: float) -> float:
    """Hard ``resource_capacity`` constraints override a resource's capacity."""
    for c in problem.hard_constraints:
        if c.type == "resource_capacity" and c.params.get("resource") == resource_id:
            return float(c.params.get("capacity", default))
    return default


def _max_concurrent_ok(problem: models.Problem, task_id: str, t: int,
                       starts: Dict[str, int]) -> bool:
    """Enforce global ``max_concurrent`` hard constraints for a candidate start."""
    d = problem.task_map()[task_id].duration
    for c in problem.hard_constraints:
        if c.type != "max_concurrent":
            continue
        limit = int(c.params.get("limit", 1))
        for slot in range(t, t + d):
            active = 1  # the task being placed
            for other_id, s in starts.items():
                if s <= slot < s + problem.task_map()[other_id].duration:
                    active += 1
            if active > limit:
                return False
    return True


def _non_overlap_ok(problem: models.Problem, task_id: str, t: int,
                    starts: Dict[str, int]) -> bool:
    """Enforce ``non_overlap`` hard constraints for a candidate start."""
    d = problem.task_map()[task_id].duration
    for c in problem.hard_constraints:
        if c.type != "non_overlap":
            continue
        group = c.params.get("tasks", [])
        if task_id not in group:
            continue
        for other_id in group:
            if other_id == task_id or other_id not in starts:
                continue
            s = starts[other_id]
            od = problem.task_map()[other_id].duration
            if t < s + od and s < t + d:
                return False
    return True


def _window_ok(problem: models.Problem, task: models.Task, t: int,
               starts: Dict[str, int], rmap: Dict[str, models.Resource]) -> bool:
    """Check resource capacity + availability over [t, t+duration)."""
    d = task.duration
    # resource availability for the task's own resource assignments
    for res_id in task.resource_requirements:
        if res_id not in rmap:
            continue
        res = rmap[res_id]
        for slot in range(t, t + d):
            if not res.available_at(slot):
                return False

    # capacity: sum requirements of overlapping placed tasks + this task
    active_res = set(task.resource_requirements)
    for res_id in active_res:
        res = rmap[res_id]
        cap = _effective_capacity(problem, res_id, res.capacity)
        for slot in range(t, t + d):
            used = task.resource_requirements.get(res_id, 0.0)
            for other_id, s in starts.items():
                other = problem.task_map()[other_id]
                if s <= slot < s + other.duration:
                    used += other.resource_requirements.get(res_id, 0.0)
            if used > cap + 1e-9:
                return False

    # global hard constraints (max concurrent, non-overlap)
    if not _max_concurrent_ok(problem, task.id, t, starts):
        return False
    if not _non_overlap_ok(problem, task.id, t, starts):
        return False
    return True


def _window_deadline_ok(problem: models.Problem, task_id: str, t: int) -> bool:
    task = problem.task_map()[task_id]
    for c in problem.hard_constraints:
        if c.type == "time_window" and c.params.get("task") == task_id:
            deadline = c.params.get("deadline")
            if deadline is not None and t + task.duration > deadline:
                return False
    return True


def decode(problem: models.Problem,
           priorities: Optional[Dict[str, float]] = None,
           seed: Optional[int] = None) -> Dict[str, int]:
    """Decode priorities into a schedule (task id -> start time).

    Tasks that cannot be placed within the horizon are omitted; the caller can
    detect infeasibility from the returned dict's length."""
    order = topological_order(problem, priorities)
    starts: Dict[str, int] = {}
    for task_id in order:
        s = _feasible_start(problem, task_id, starts)
        if s is None:
            continue
        starts[task_id] = s
    return starts


def random_keys(problem: models.Problem, rng: random.Random) -> Dict[str, float]:
    """Random priority vector (each task gets a uniform random key)."""
    return {t.id: rng.random() for t in problem.tasks}


def greedy_order(problem: models.Problem) -> Dict[str, float]:
    """Deterministic priority rule: highest problem.priority first."""
    return {t.id: float(t.priority) for t in problem.tasks}


def schedule_to_assignments(problem: models.Problem,
                            starts: Dict[str, int]) -> List[models.Assignment]:
    tmap = problem.task_map()
    out = []
    for tid in sorted(starts, key=lambda t: (starts[t], t)):
        task = tmap[tid]
        out.append(models.Assignment(
            task=tid,
            start=starts[tid],
            end=starts[tid] + task.duration,
            resources=sorted(task.resource_requirements),
        ))
    return out
