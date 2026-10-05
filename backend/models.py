"""
Domain model for the OR / scheduling solver system.

The problem is a resource-constrained project scheduling problem (RCPSP)
generalised to three resource classes (personnel, equipment, time) and to
arbitrary hard/soft constraints.  Time is discretised into integer slots so
that a time-indexed LP/IP formulation and discrete meta-heuristics share one
representation.

Every entity is a plain dataclass that round-trips through JSON.  This module
deliberately has no I/O and no solver logic: it is only the shared vocabulary
used by the storage layer, the solvers and the web API.
"""

from __future__ import annotations

import itertools
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Enumerated vocabulary
# --------------------------------------------------------------------------- #

RESOURCE_TYPES = ("personnel", "equipment", "time")

OBJECTIVE_TYPES = (
    "makespan",                 # min max completion time
    "total_completion",         # min sum of completion times
    "weighted_completion",      # min sum weight_i * C_i
    "tardiness",                # min sum weight_i * max(0, C_i - due_i)
    "cost",                     # min resource usage cost
    "custom",                   # weighted combination of the primitives
)

# Hard constraints: a violation makes a schedule *infeasible*.
HARD_CONSTRAINT_TYPES = (
    "precedence",               # task A before task B
    "time_window",              # start within [release, deadline]
    "fixed_start",              # task must start at exactly t
    "resource_capacity",        # implied by resource.capacity, still expressible
    "non_overlap",              # two tasks may not overlap on a resource
    "max_concurrent",           # at most k tasks running simultaneously
    "resource_assignment",      # task requires specific resource set
)

# Soft constraints: a violation contributes a *penalty* to the objective.
SOFT_CONSTRAINT_TYPES = (
    "due_date",                 # penalise late completion (weighted tardiness)
    "preferred_window",         # penalise starts outside [a, b]
    "min_gap",                  # penalise insufficient gap between two tasks
    "resource_balance",         # penalise uneven resource load
    "setup_time",               # penalise missing setup between consecutive jobs
    "max_makespan",             # penalise exceeding a target makespan
)

SOLVER_NAMES = ("lp", "ip", "genetic", "simulated_annealing", "greedy")

SOLUTION_STATUS = ("optimal", "feasible", "infeasible", "timeout", "error")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# --------------------------------------------------------------------------- #
# Resource
# --------------------------------------------------------------------------- #

@dataclass
class Resource:
    """A schedulable resource.

    ``availability`` is a list of half-open ``[start, end)`` intervals during
    which the resource is usable; ``None`` means always available.  ``skills``
    is used by the ``resource_assignment`` constraint so a task can require a
    person or machine that possesses a particular capability rather than one
    specific instance.
    """
    id: str
    name: str = ""
    type: str = "personnel"                 # personnel | equipment | time
    capacity: float = 1.0                   # units available simultaneously
    skills: List[str] = field(default_factory=list)
    cost_per_unit: float = 0.0              # cost per unit-time of use
    availability: Optional[List[List[int]]] = None   # [[start, end), ...]

    def __post_init__(self) -> None:
        if self.type not in RESOURCE_TYPES:
            raise ValueError(f"unknown resource type: {self.type}")
        if self.capacity < 0:
            raise ValueError(f"resource {self.id}: negative capacity")

    def available_at(self, t: int) -> bool:
        if self.availability is None:
            return True
        return any(a <= t < b for (a, b) in self.availability)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Resource":
        return cls(**d)


# --------------------------------------------------------------------------- #
# Task
# --------------------------------------------------------------------------- #

@dataclass
class Task:
    """A unit of work.  ``duration`` is an integer number of time slots and
    ``resource_requirements`` maps a resource id to the number of capacity
    units the task occupies for every slot of its execution."""
    id: str
    name: str = ""
    duration: int = 1
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)   # task ids
    release_time: int = 0
    due_date: Optional[int] = None
    weight: float = 1.0
    priority: float = 1.0                     # used by greedy / tie-breaks

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError(f"task {self.id}: duration must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Task":
        return cls(**d)


# --------------------------------------------------------------------------- #
# Constraints
# --------------------------------------------------------------------------- #

@dataclass
class HardConstraint:
    """A hard constraint.  ``params`` is a type-specific payload.

    * precedence            params: {"before": task_id, "after": task_id}
    * time_window           params: {"task": task_id, "release": t, "deadline": t}
    * fixed_start           params: {"task": task_id, "start": t}
    * non_overlap           params: {"tasks": [task_id, ...]}
    * max_concurrent        params: {"limit": int}
    * resource_assignment   params: {"task": task_id, "resources": [res_id, ...]}
    * resource_capacity     params: {"resource": res_id, "capacity": float}
    """
    id: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in HARD_CONSTRAINT_TYPES:
            raise ValueError(f"unknown hard constraint type: {self.type}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HardConstraint":
        return cls(**d)


@dataclass
class SoftConstraint:
    """A soft constraint with a linear penalty ``penalty`` and optional weight
    ``factor`` used to scale the contribution inside the objective."""
    id: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    penalty: float = 1.0
    factor: float = 1.0

    def __post_init__(self) -> None:
        if self.type not in SOFT_CONSTRAINT_TYPES:
            raise ValueError(f"unknown soft constraint type: {self.type}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SoftConstraint":
        return cls(**d)


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #

@dataclass
class Objective:
    """The optimisation goal.  ``type`` selects a primitive; ``custom`` can
    combine primitives through ``weights`` (a map of primitive name -> weight).
    ``minimize`` is True for a minimisation problem."""
    type: str = "makespan"
    weights: Dict[str, float] = field(default_factory=dict)
    minimize: bool = True
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in OBJECTIVE_TYPES:
            raise ValueError(f"unknown objective type: {self.type}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Objective":
        return cls(**d)


# --------------------------------------------------------------------------- #
# Problem
# --------------------------------------------------------------------------- #

@dataclass
class Problem:
    id: str
    name: str = ""
    description: str = ""
    horizon: int = 100
    time_unit: str = "hour"
    resources: List[Resource] = field(default_factory=list)
    tasks: List[Task] = field(default_factory=list)
    hard_constraints: List[HardConstraint] = field(default_factory=list)
    soft_constraints: List[SoftConstraint] = field(default_factory=list)
    objective: Objective = field(default_factory=Objective)
    version: int = 1
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    # -- convenience index helpers ---------------------------------------- #
    def resource_map(self) -> Dict[str, Resource]:
        return {r.id: r for r in self.resources}

    def task_map(self) -> Dict[str, Task]:
        return {t.id: t for t in self.tasks}

    def precedence_edges(self) -> List[Tuple[str, str]]:
        """Return explicit (before, after) edges from task.dependencies plus
        explicit ``precedence`` hard constraints, de-duplicated."""
        edges: List[Tuple[str, str]] = []
        seen = set()
        for t in self.tasks:
            for dep in t.dependencies:
                if (dep, t.id) not in seen:
                    seen.add((dep, t.id))
                    edges.append((dep, t.id))
        for c in self.hard_constraints:
            if c.type == "precedence":
                b, a = c.params.get("before"), c.params.get("after")
                if b and a and (b, a) not in seen:
                    seen.add((b, a))
                    edges.append((b, a))
        return edges

    def successors(self) -> Dict[str, List[str]]:
        succ: Dict[str, List[str]] = {t.id: [] for t in self.tasks}
        for b, a in self.precedence_edges():
            succ.setdefault(b, []).append(a)
        return succ

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Problem":
        d = dict(d)
        d["resources"] = [Resource.from_dict(r) for r in d.get("resources", [])]
        d["tasks"] = [Task.from_dict(t) for t in d.get("tasks", [])]
        d["hard_constraints"] = [
            HardConstraint.from_dict(c) for c in d.get("hard_constraints", [])
        ]
        d["soft_constraints"] = [
            SoftConstraint.from_dict(c) for c in d.get("soft_constraints", [])
        ]
        d["objective"] = Objective.from_dict(d.get("objective", {}))
        return cls(**d)


# --------------------------------------------------------------------------- #
# Solution
# --------------------------------------------------------------------------- #

@dataclass
class Assignment:
    task: str
    start: int
    end: int
    resources: List[str] = field(default_factory=list)

    @property
    def duration(self) -> int:
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Assignment":
        return cls(**d)


@dataclass
class Solution:
    id: str
    problem_id: str
    solver: str
    status: str = "feasible"            # optimal|feasible|infeasible|timeout|error
    objective_value: Optional[float] = None
    makespan: Optional[int] = None
    assignments: List[Assignment] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    solve_time: float = 0.0
    message: str = ""
    lower_bound: Optional[float] = None
    created_at: str = field(default_factory=now_iso)
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Solution":
        d = dict(d)
        d["assignments"] = [Assignment.from_dict(a) for a in d.get("assignments", [])]
        return cls(**d)


# --------------------------------------------------------------------------- #
# Solver configuration
# --------------------------------------------------------------------------- #

@dataclass
class SolverConfig:
    id: str
    problem_id: str
    solver: str
    params: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SolverConfig":
        return cls(**d)


# --------------------------------------------------------------------------- #
# Sensitivity & report wrappers
# --------------------------------------------------------------------------- #

@dataclass
class SensitivityResult:
    id: str
    problem_id: str
    solver: str
    base_objective: float
    parameter: str
    variations: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SensitivityResult":
        return cls(**d)


@dataclass
class Report:
    id: str
    problem_id: str
    title: str = ""
    format: str = "markdown"
    content: str = ""
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Report":
        return cls(**d)


# --------------------------------------------------------------------------- #
# Cross-cutting validation used by storage and API
# --------------------------------------------------------------------------- #

def validate_problem(problem: Problem) -> List[str]:
    """Return a list of human-readable validation errors (empty == valid)."""
    errors: List[str] = []

    ids = [t.id for t in problem.tasks]
    if len(ids) != len(set(ids)):
        errors.append("task ids must be unique")

    rids = [r.id for r in problem.resources]
    if len(rids) != len(set(rids)):
        errors.append("resource ids must be unique")

    tasks = problem.task_map()
    res = problem.resource_map()

    for t in problem.tasks:
        if t.duration <= 0:
            errors.append(f"task {t.id}: duration must be > 0")
        if t.release_time < 0:
            errors.append(f"task {t.id}: release_time must be >= 0")
        if t.release_time + t.duration > problem.horizon:
            errors.append(f"task {t.id}: release + duration exceeds horizon")
        for dep in t.dependencies:
            if dep not in tasks:
                errors.append(f"task {t.id}: unknown dependency '{dep}'")
        for r, amount in t.resource_requirements.items():
            if r not in res:
                errors.append(f"task {t.id}: unknown resource '{r}'")
            if amount < 0:
                errors.append(f"task {t.id}: negative requirement for '{r}'")

    # dependency cycle detection (Kahn's algorithm)
    succ = problem.successors()
    indeg = {i: 0 for i in ids}
    for b, a in problem.precedence_edges():
        if b not in indeg or a not in indeg:
            continue
        indeg[a] += 1
    queue = [i for i in ids if indeg.get(i, 0) == 0]
    topo_count = 0
    while queue:
        n = queue.pop()
        topo_count += 1
        for s in succ.get(n, []):
            indeg[s] -= 1
            if indeg[s] == 0:
                queue.append(s)
    if topo_count != len(ids):
        errors.append("dependency graph contains a cycle")

    for c in problem.hard_constraints:
        p = c.params
        if c.type == "precedence":
            if p.get("before") not in tasks or p.get("after") not in tasks:
                errors.append(f"precedence '{c.id}': unknown task reference")
        elif c.type in ("time_window", "fixed_start", "resource_assignment"):
            if p.get("task") not in tasks:
                errors.append(f"{c.type} '{c.id}': unknown task '{p.get('task')}'")
        elif c.type == "resource_capacity":
            if p.get("resource") not in res:
                errors.append(f"resource_capacity '{c.id}': unknown resource")

    if problem.horizon <= 0:
        errors.append("horizon must be positive")

    return errors


def objective_summary(problem: Problem) -> str:
    """Short human-readable objective label for UI titles."""
    o = problem.objective
    if o.type == "custom":
        parts = ", ".join(f"{k}={v}" for k, v in sorted(o.weights.items()))
        return f"custom({parts})"
    return o.type
