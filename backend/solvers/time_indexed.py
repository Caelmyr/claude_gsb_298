"""
Time-indexed LP/IP formulation of the resource-constrained scheduling problem.

Variables ``x_{j,t}`` = 1 iff task ``j`` starts at time ``t``.  The formulation
is the classic time-indexed MIP and is *exact* (its integer optimum is the true
optimum of the scheduling problem):

    sum_t x_{j,t} = 1                                  for every task j      (1)
    sum_{t<=s} x_{k,t} <= sum_{t<=s-d_j} x_{j,t}       precedence j -> k     (2)
    sum_{j} req_{j,r} * sum_{s<=t<s+d_j} x_{j,s} <= c_r   resource capacity  (3)

Makespan is modelled with a continuous variable ``Cmax`` and the linear
inequalities ``Cmax >= start_j + d_j``.  Weighted completion and tardiness are
linear in the ``x`` variables too, so a single builder serves the LP relaxation
(the lower-bound oracle) and the branch-and-bound integer solver.

A size guard prevents building an intractable LP for large instances: the
meta-heuristics (GA / SA) are the intended path for those.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .. import models

# Upper bound on total columns for the time-indexed formulation.  Beyond this
# the LP/IP route is refused and the caller is pointed at the meta-heuristics.
MAX_TIME_INDEXED_VARS = 30000


@dataclass
class Var:
    kind: str                 # 'start' | 'makespan' | 'tardiness'
    task: Optional[str] = None
    t: Optional[int] = None


@dataclass
class LPModel:
    c: List[float]
    A_ub: List[List[float]]
    b_ub: List[float]
    A_eq: List[List[float]]
    b_eq: List[float]
    vars: List[Var]
    index: Dict[Tuple[str, str, int], int]
    objective_note: str = ""

    def n_vars(self) -> int:
        return len(self.vars)


def feasible_start_range(problem: models.Problem, task: models.Task) -> range:
    """Allowed start times for a task given release/window constraints."""
    lo = task.release_time
    hi = problem.horizon - task.duration
    for c in problem.hard_constraints:
        p = c.params
        if p.get("task") != task.id:
            continue
        if c.type == "time_window":
            if p.get("release") is not None:
                lo = max(lo, p["release"])
            if p.get("deadline") is not None:
                hi = min(hi, p["deadline"] - task.duration)
        elif c.type == "fixed_start":
            s = p.get("start")
            if s is not None:
                lo = hi = s
    lo = max(0, lo)
    hi = max(lo - 1, hi)
    return range(lo, hi + 1)


def _linear_objective(problem: models.Problem) -> Dict[str, Tuple[str, float]]:
    """Resolve the problem objective into a dict of primitive -> weight for the
    primitives that are linear in the time-indexed variables."""
    o = problem.objective
    if o.type == "custom":
        return dict(o.weights)
    return {o.type: 1.0}


def build(problem: models.Problem) -> LPModel:
    """Construct the time-indexed LP/IP for ``problem``.

    Raises ``ValueError`` when the instance is too large for exact methods.
    """
    tasks = problem.tasks
    resources = problem.resources
    horizon = problem.horizon
    tmap = problem.task_map()
    rmap = problem.resource_map()
    succ = problem.successors()

    # ---- variables ------------------------------------------------------- #
    vars_: List[Var] = []
    index: Dict[Tuple[str, str, int], int] = {}

    def add(kind: str, task: Optional[str] = None, t: Optional[int] = None) -> int:
        key = (kind, task or "", t if t is not None else -1)
        if key in index:
            return index[key]
        idx = len(vars_)
        index[key] = idx
        vars_.append(Var(kind, task, t))
        return idx

    starts: Dict[str, Dict[int, int]] = {}  # task -> {t -> var index}
    for task in tasks:
        starts[task.id] = {}
        for t in feasible_start_range(problem, task):
            starts[task.id][t] = add("start", task.id, t)

    # ---- objective ------------------------------------------------------- #
    primitives = _linear_objective(problem)
    supported = {"makespan", "total_completion", "weighted_completion", "tardiness"}
    active = {k: w for k, w in primitives.items() if k in supported and w != 0}

    need_makespan = "makespan" in active
    need_tardiness = "tardiness" in active

    if not active:
        # cost-only (or empty) objective is constant w.r.t. starts; use makespan
        # as a sensible surrogate so the exact solvers still do something useful.
        active = {"makespan": 1.0}
        need_makespan = True

    # Auxiliary continuous variables must be created before the cost vector is
    # sized, so their columns exist when the constraint rows are built.
    cmax_idx = add("makespan") if need_makespan else None
    tard_idx: Dict[str, int] = {}
    if need_tardiness:
        for task in tasks:
            if task.due_date is not None:
                tard_idx[task.id] = add("tardiness", task.id)

    c = [0.0] * len(vars_)
    if cmax_idx is not None:
        c[cmax_idx] = active.get("makespan", 0.0)

    for task in tasks:
        dur = task.duration
        for t, idx in starts[task.id].items():
            for kind in ("total_completion", "weighted_completion"):
                if kind in active:
                    w = active[kind] * (task.weight if kind == "weighted_completion" else 1.0)
                    c[idx] += w * (t + dur)
            if "tardiness" in active and task.due_date is not None:
                c[idx] += active["tardiness"] * task.weight * (t + dur)
    for task in tasks:
        if task.id in tard_idx:
            c[tard_idx[task.id]] = active["tardiness"] * task.weight

    # ---- constraint matrices --------------------------------------------- #
    A_ub: List[List[float]] = []
    b_ub: List[float] = []
    A_eq: List[List[float]] = []
    b_eq: List[float] = []
    n = len(vars_)

    def row_ub(coeffs: Dict[int, float], rhs: float) -> None:
        r = [0.0] * n
        for idx, val in coeffs.items():
            r[idx] = val
        A_ub.append(r)
        b_ub.append(rhs)

    def row_eq(coeffs: Dict[int, float], rhs: float) -> None:
        r = [0.0] * n
        for idx, val in coeffs.items():
            r[idx] = val
        A_eq.append(r)
        b_eq.append(rhs)

    # (1) each task starts exactly once
    for task in tasks:
        coeffs = {idx: 1.0 for idx in starts[task.id].values()}
        if not coeffs:
            raise ValueError(f"task {task.id}: no feasible start time (release "
                             f"+ duration exceeds horizon)")
        row_eq(coeffs, 1.0)

    # (2) precedence  sum_{t<=s} x_{k,t} <= sum_{t<=s-d_j} x_{j,t}
    for before, after in problem.precedence_edges():
        if before not in starts or after not in starts:
            continue
        d_before = tmap[before].duration
        for s in range(horizon):
            lhs: Dict[int, float] = {}
            rhs: Dict[int, float] = {}
            for t, idx in starts[after].items():
                if t <= s:
                    lhs[idx] = lhs.get(idx, 0.0) + 1.0
            for t, idx in starts[before].items():
                if t <= s - d_before:
                    rhs[idx] = rhs.get(idx, 0.0) + 1.0
            if lhs or rhs:
                combined = dict(lhs)
                for idx, v in rhs.items():
                    combined[idx] = combined.get(idx, 0.0) - v
                row_ub(combined, 0.0)

    # (3) resource capacity per time slot (hard resource_capacity overrides)
    cap_override = {c.params.get("resource"): float(c.params.get("capacity", 0))
                    for c in problem.hard_constraints
                    if c.type == "resource_capacity" and c.params.get("resource")}
    for res in resources:
        cap = cap_override.get(res.id, res.capacity)
        for slot in range(horizon):
            coeffs: Dict[int, float] = {}
            for task in tasks:
                req = task.resource_requirements.get(res.id, 0.0)
                if req <= 0:
                    continue
                for t, idx in starts[task.id].items():
                    if t <= slot < t + task.duration:
                        coeffs[idx] = coeffs.get(idx, 0.0) + req
            if coeffs:
                row_ub(coeffs, cap)

    # (4) explicit hard constraints that translate linearly
    for hc in problem.hard_constraints:
        p = hc.params
        if hc.type == "max_concurrent":
            limit = p.get("limit", 1)
            for slot in range(horizon):
                coeffs: Dict[int, float] = {}
                for task in tasks:
                    for t, idx in starts[task.id].items():
                        if t <= slot < t + task.duration:
                            coeffs[idx] = coeffs.get(idx, 0.0) + 1.0
                if coeffs:
                    row_ub(coeffs, float(limit))
        elif hc.type == "fixed_start":
            task_id = p.get("task")
            s = p.get("start")
            if task_id in starts and s is not None and s in starts[task_id]:
                row_eq({starts[task_id][s]: 1.0}, 1.0)
        elif hc.type == "non_overlap":
            group = p.get("tasks", [])
            for slot in range(horizon):
                coeffs: Dict[int, float] = {}
                for task_id in group:
                    for t, idx in starts.get(task_id, {}).items():
                        if t <= slot < t + tmap[task_id].duration:
                            coeffs[idx] = coeffs.get(idx, 0.0) + 1.0
                if coeffs:
                    row_ub(coeffs, 1.0)

    # (5) makespan definition  sum_t (t+d_j) x_{j,t} - Cmax <= 0
    if need_makespan:
        cmax_idx = index[("makespan", "", -1)]
        for task in tasks:
            coeffs: Dict[int, float] = {cmax_idx: -1.0}
            for t, idx in starts[task.id].items():
                coeffs[idx] = (t + task.duration)
            row_ub(coeffs, 0.0)

    # (6) tardiness definition  sum_t (t+d_j) x_{j,t} - T_j <= due_j
    if need_tardiness:
        for task in tasks:
            if task.id not in tard_idx:
                continue
            tj_idx = tard_idx[task.id]
            coeffs: Dict[int, float] = {tj_idx: -1.0}
            for t, idx in starts[task.id].items():
                coeffs[idx] = (t + task.duration)
            row_ub(coeffs, float(task.due_date))

    # size guard
    if n > MAX_TIME_INDEXED_VARS:
        raise ValueError(
            f"time-indexed formulation has {n} variables (limit "
            f"{MAX_TIME_INDEXED_VARS}); use genetic / simulated_annealing")

    note = ""
    if primitives.get("cost"):
        note = "cost term is constant w.r.t. start times and is ignored in the LP/IP objective"

    return LPModel(c, A_ub, b_ub, A_eq, b_eq, vars_, index, note)


def solution_from_x(problem: models.Problem, model: LPModel,
                    x: List[float]) -> Dict[str, float]:
    """Extract expected start times from a (possibly fractional) solution."""
    expected = {}
    for task in problem.tasks:
        num = 0.0
        den = 0.0
        for var_idx, var in enumerate(model.vars):
            if var.kind == "start" and var.task == task.id:
                num += var.t * x[var_idx]
                den += x[var_idx]
        expected[task.id] = num / den if den > 1e-9 else float(task.release_time)
    return expected


def expected_starts_to_priorities(expected: Dict[str, float]) -> Dict[str, float]:
    """Turn expected start times into priorities (earlier start -> higher)."""
    if not expected:
        return {}
    lo = min(expected.values())
    hi = max(expected.values())
    span = (hi - lo) or 1.0
    return {tid: 1.0 - (v - lo) / span for tid, v in expected.items()}
