"""
Automatic infeasibility diagnosis for scheduling problems.

When a solve comes back "infeasible" the user is left guessing which of their
hard constraints fight each other.  This module answers three questions:

1. *Which constraints (and tasks) are jointly responsible?*  Every hard
   restriction is expressed as an independently removable **participant** --
   a time-window bound, a precedence edge, one resource's whole capacity
   family, a non-overlap group, ... -- and the diagnosis returns one or more
   **conflict sets** of participants that are *jointly* infeasible.
2. *Is the set as small as possible?*  For small/medium instances a
   deletion filter shrinks every conflict set to an irreducible infeasible
   subsystem (IIS): removing any single participant restores feasibility.
3. *What should I change?*  Two flavours of suggestion are produced and then
   re-checked by actually rebuilding and re-solving the modified problem:
   "relax" (widen a window, raise a capacity, move a fixed start) with a
   quantified minimum amount, and "remove" (delete one constraint or one
   task).  Suggestions that survive the re-solve are marked ``verified``.

Layered pipeline (each layer has a hard size/time budget, so diagnosis time
grows gently even as problems get large):

    layer 1  temporal bound propagation  -- O(V+E), exact for everything
             expressible on release/deadline + precedence alone;
    layer 2  time-indexed feasibility LP -- elastic LP localises the fight,
             deletion filter certifies minimality, IP verifies suggestions;
    layer 3  multi-order SGS sweep       -- heuristic fallback for instances
             too big for the LP; reports are explicitly labelled heuristic.

Freshness is the caller's concern but the fingerprints used to judge it live
here (:func:`problem_fingerprint`): a report embeds the problem version and a
hash of every field that can affect feasibility, so any later edit makes the
old report visibly stale instead of silently misleading.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from . import models
from .solvers import simplex

# --------------------------------------------------------------------------- #
# Size / time budgets
# --------------------------------------------------------------------------- #

# Above this many time-indexed columns the exact LP route is refused.  Kept
# well below the solver builder's own guard because diagnosis solves the LP
# many times (deletion filter) instead of once.
DIAG_MAX_LP_VARS = 8000
# ... and this many rows (dense tableau, so rows quadratic-ish in simplex
# iterations and quadratic in memory via repeated pivots).
DIAG_MAX_LP_ROWS = 12000
# Default wall-clock budget for a full diagnosis (seconds).
DEFAULT_TIME_BUDGET = 5.0
# How many distinct conflict sets to report.
MAX_CONFLICTS = 3
# How many suggestions of each kind to verify per conflict.
MAX_SUGGESTIONS = 6


# --------------------------------------------------------------------------- #
# Freshness
# --------------------------------------------------------------------------- #

def problem_fingerprint(problem: models.Problem) -> str:
    """SHA-256 over every problem field that can change feasibility.

    Soft constraints and the objective are deliberately excluded: they never
    make a schedule infeasible.
    """
    payload = {
        "horizon": problem.horizon,
        "resources": [r.to_dict() for r in problem.resources],
        "tasks": [
            {
                "id": t.id, "duration": t.duration,
                "resource_requirements": t.resource_requirements,
                "dependencies": t.dependencies,
                "release_time": t.release_time,
            }
            for t in problem.tasks
        ],
        "hard_constraints": [c.to_dict() for c in problem.hard_constraints],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Result dataclasses (plain dicts once serialised)
# --------------------------------------------------------------------------- #

@dataclass
class Participant:
    """One independently removable hard restriction."""
    id: str                       # stable id within one diagnosis
    kind: str                     # window_release | window_deadline | fixed_start |
                                  # precedence | resource_capacity | max_concurrent |
                                  # non_overlap | task_release | task_horizon |
                                  # resource_assignment
    label: str                    # human-readable, Chinese
    constraint_id: Optional[str] = None   # None for task/resource-implied participants
    task_ids: List[str] = field(default_factory=list)
    resource_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "constraint_id": self.constraint_id,
            "task_ids": self.task_ids,
            "resource_id": self.resource_id,
        }


@dataclass
class ConflictSet:
    """A jointly-infeasible group of participants, ideally an IIS."""
    participants: List[Participant]
    tasks: List[str]
    explanation: str
    minimal: bool = False              # deletion-filter certified?
    method: str = "lp"                 # temporal | lp | heuristic
    witness: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "participants": [p.to_dict() for p in self.participants],
            "tasks": self.tasks,
            "explanation": self.explanation,
            "minimal": self.minimal,
            "method": self.method,
            "witness": self.witness,
        }


@dataclass
class Suggestion:
    """One concrete edit that makes the conflict satisfiable."""
    action: str                       # relax | remove_constraint | remove_task
    kind: str                         # what to change (same vocab as Participant)
    label: str                        # human-readable
    constraint_id: Optional[str] = None
    task_id: Optional[str] = None
    resource_id: Optional[str] = None
    # For relax: current value and suggested new value / minimum delta.
    current_value: Optional[float] = None
    suggested_value: Optional[float] = None
    delta: Optional[float] = None
    unit: str = ""
    resolves_conflict: int = 0        # 1-based index of the conflict addressed
    sufficient: bool = False          # alone makes the WHOLE problem feasible?
    verified: bool = False            # re-solved the modified problem?
    verification: str = ""            # status / note
    expected_effect: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "kind": self.kind,
            "label": self.label,
            "constraint_id": self.constraint_id,
            "task_id": self.task_id,
            "resource_id": self.resource_id,
            "current_value": self.current_value,
            "suggested_value": self.suggested_value,
            "delta": self.delta,
            "unit": self.unit,
            "resolves_conflict": self.resolves_conflict,
            "sufficient": self.sufficient,
            "verified": self.verified,
            "verification": self.verification,
            "expected_effect": self.expected_effect,
        }


@dataclass
class Diagnosis:
    feasible: bool
    fingerprint: str
    problem_version: int
    method: str                       # temporal | lp | heuristic | mixed
    conflicts: List[ConflictSet] = field(default_factory=list)
    suggestions: List[Suggestion] = field(default_factory=list)
    combined_fixes: List[Dict[str, Any]] = field(default_factory=list)
    unschedulable_tasks: List[str] = field(default_factory=list)
    timed_out: bool = False
    diagnose_time: float = 0.0
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feasible": self.feasible,
            "fingerprint": self.fingerprint,
            "problem_version": self.problem_version,
            "method": self.method,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "suggestions": [s.to_dict() for s in self.suggestions],
            "combined_fixes": self.combined_fixes,
            "unschedulable_tasks": self.unschedulable_tasks,
            "timed_out": self.timed_out,
            "diagnose_time": round(self.diagnose_time, 4),
            "summary": self.summary,
        }


# --------------------------------------------------------------------------- #
# Participant construction
# --------------------------------------------------------------------------- #

def _task_name(problem: models.Problem, tid: str) -> str:
    t = problem.task_map().get(tid)
    return f"{t.name}（{tid}）" if t and t.name else tid


def _resource_name(problem: models.Problem, rid: str) -> str:
    r = problem.resource_map().get(rid)
    return f"{r.name}（{rid}）" if r and r.name else rid


def build_participants(problem: models.Problem) -> List[Participant]:
    """Enumerate every removable hard restriction in the problem."""
    ps: List[Participant] = []

    # per-task: release time and the horizon end bound
    for t in problem.tasks:
        ps.append(Participant(
            id=f"release:{t.id}", kind="task_release",
            label=f"任务 {_task_name(problem, t.id)} 的最早释放时间 t≥{t.release_time}",
            task_ids=[t.id]))
        ps.append(Participant(
            id=f"horizon:{t.id}", kind="task_horizon",
            label=f"任务 {_task_name(problem, t.id)} 必须在规划期 {problem.horizon} 内完成"
                  f"（工期 {t.duration}）",
            task_ids=[t.id]))

    # dependency-implied precedence edges (structural, but removable together
    # with the dependency)
    explicit_edges = {(c.params.get("before"), c.params.get("after"))
                      for c in problem.hard_constraints if c.type == "precedence"}
    for t in problem.tasks:
        for dep in t.dependencies:
            if (dep, t.id) in explicit_edges:
                continue
            ps.append(Participant(
                id=f"edge:{dep}:{t.id}", kind="precedence",
                task_ids=[dep, t.id],
                label=f"先后顺序（依赖）：{_task_name(problem, dep)} 完成后才能开始 "
                      f"{_task_name(problem, t.id)}"))

    # explicit hard constraints
    for c in problem.hard_constraints:
        p = c.params
        if c.type == "precedence":
            b, a = p.get("before"), p.get("after")
            ps.append(Participant(
                id=f"hc:{c.id}", kind="precedence",
                constraint_id=c.id,
                label=f"先后顺序：{_task_name(problem, b)} 完成后才能开始 {_task_name(problem, a)}",
                task_ids=[x for x in (b, a) if x]))
        elif c.type == "time_window":
            tid = p.get("task")
            if p.get("release") is not None:
                ps.append(Participant(
                    id=f"hc:{c.id}:release", kind="window_release",
                    constraint_id=c.id, task_ids=[tid],
                    label=f"时间窗 {c.id}：{_task_name(problem, tid)} 不早于 t={p['release']} 开始"))
            if p.get("deadline") is not None:
                ps.append(Participant(
                    id=f"hc:{c.id}:deadline", kind="window_deadline",
                    constraint_id=c.id, task_ids=[tid],
                    label=f"时间窗 {c.id}：{_task_name(problem, tid)} 不晚于 t={p['deadline']} 完成"))
        elif c.type == "fixed_start":
            tid = p.get("task")
            ps.append(Participant(
                id=f"hc:{c.id}", kind="fixed_start",
                constraint_id=c.id, task_ids=[tid],
                label=f"固定开工 {c.id}：{_task_name(problem, tid)} 必须在 t={p.get('start')} 开始"))
        elif c.type == "resource_capacity":
            rid = p.get("resource")
            ps.append(Participant(
                id=f"hc:{c.id}", kind="resource_capacity",
                constraint_id=c.id, resource_id=rid,
                task_ids=[t.id for t in problem.tasks
                          if t.resource_requirements.get(rid, 0) > 0],
                label=f"容量约束 {c.id}：{_resource_name(problem, rid)} 容量上限 "
                      f"{p.get('capacity', 1)}"))
        elif c.type == "non_overlap":
            group = list(p.get("tasks", []))
            ps.append(Participant(
                id=f"hc:{c.id}", kind="non_overlap",
                constraint_id=c.id, task_ids=group,
                label=f"互斥约束 {c.id}：{ '、'.join(_task_name(problem, x) for x in group) } 不得重叠"))
        elif c.type == "max_concurrent":
            ps.append(Participant(
                id=f"hc:{c.id}", kind="max_concurrent",
                constraint_id=c.id, task_ids=[t.id for t in problem.tasks],
                label=f"全局并发上限 {c.id}：同一时刻最多 {p.get('limit', 1)} 个任务"))
        elif c.type == "resource_assignment":
            tid = p.get("task")
            ps.append(Participant(
                id=f"hc:{c.id}", kind="resource_assignment",
                constraint_id=c.id, task_ids=[tid],
                resource_id=(p.get("resources") or [None])[0],
                label=f"资源指派 {c.id}：{_task_name(problem, tid)} 必须使用 "
                      f"{'、'.join(p.get('resources', []))}"))

    # per-resource implied capacity + availability calendar
    for r in problem.resources:
        users = [t.id for t in problem.tasks
                 if t.resource_requirements.get(r.id, 0) > 0]
        ps.append(Participant(
            id=f"cap:{r.id}", kind="resource_capacity",
            resource_id=r.id, task_ids=users,
            label=f"资源 {_resource_name(problem, r.id)} 的容量上限 {r.capacity}"))
        if r.availability:
            ps.append(Participant(
                id=f"avail:{r.id}", kind="resource_availability",
                resource_id=r.id, task_ids=users,
                label=f"资源 {_resource_name(problem, r.id)} 的可用时段日历"
                      f"（{'，'.join(f'[{a},{b})' for a, b in r.availability)}）"))

    return ps


# --------------------------------------------------------------------------- #
# Feasibility LP with tagged, independently removable participants
# --------------------------------------------------------------------------- #

@dataclass
class _FeasLP:
    c: List[float]
    A_ub: List[List[float]]
    b_ub: List[float]
    A_eq: List[List[float]]
    b_eq: List[float]
    # row -> participant id ("" for structural rows that cannot be removed)
    row_participant: List[str]
    start_cols: Dict[str, Dict[int, int]]   # task -> t -> column
    n_base: int                             # number of x columns (no elastic)


def _bounds_for_task(problem: models.Problem,
                     task: models.Task) -> Tuple[int, int]:
    """Static (propagation-free) allowed start range; returns (lo, hi) with
    hi < lo meaning the task has no start time at all."""
    lo = max(0, task.release_time)
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
        elif c.type == "fixed_start" and p.get("start") is not None:
            lo = hi = p["start"]
    return lo, hi


def build_feasibility_lp(problem: models.Problem,
                         active: Optional[Set[str]] = None,
                         ) -> Optional[_FeasLP]:
    """Build the pure-feasibility time-indexed LP (zero objective).

    Every generated row is tagged with the participant id responsible for it.
    ``active`` is a set of participant ids that remain enforced; participants
    absent from the set are dropped entirely.  Structural rows (each task
    starts exactly once) are always present.

    Returns None if a task has zero candidate start columns even before any
    other constraint is considered -- that itself is reported separately.
    """
    tmap = problem.task_map()
    rmap = problem.resource_map()
    horizon = problem.horizon
    n_slots = horizon

    def on(pid: str) -> bool:
        return active is None or pid in active

    # columns ----------------------------------------------------------------
    cols: List[str] = []
    col_index: Dict[str, int] = {}
    start_cols: Dict[str, Dict[int, int]] = {}

    def add_col(name: str) -> int:
        if name in col_index:
            return col_index[name]
        idx = len(cols)
        col_index[name] = idx
        cols.append(name)
        return idx

    ranges: Dict[str, Tuple[int, int]] = {}
    for t in problem.tasks:
        lo, hi = _bounds_for_task(problem, t)
        if hi < lo:
            return None
        start_cols[t.id] = {}
        for s in range(lo, hi + 1):
            start_cols[t.id][s] = add_col(f"x:{t.id}:{s}")
        ranges[t.id] = (lo, hi)

    n = len(cols)

    A_ub: List[List[float]] = []
    b_ub: List[float] = []
    A_eq: List[List[float]] = []
    b_eq: List[float] = []
    row_pid: List[str] = []

    def add_ub(pid: str, coeffs: Dict[int, float], rhs: float) -> None:
        row = [0.0] * n
        for idx, v in coeffs.items():
            row[idx] = v
        A_ub.append(row)
        b_ub.append(rhs)
        row_pid.append(pid)

    def add_eq(pid: str, coeffs: Dict[int, float], rhs: float) -> None:
        row = [0.0] * n
        for idx, v in coeffs.items():
            row[idx] = v
        A_eq.append(row)
        b_eq.append(rhs)

    # structural: every task starts exactly once ----------------------------
    for tid, scols in start_cols.items():
        add_eq("", {idx: 1.0 for idx in scols.values()}, 1.0)

    # precedence (edges implied by dependencies carry the *successor task's*
    # release participant only loosely -- each explicit edge is its own
    # participant; dependency edges are structural-ish, tagged on the tasks) -
    for b, a in problem.precedence_edges():
        if b not in start_cols or a not in start_cols:
            continue
        # explicit precedence constraint?
        pid = f"edge:{b}:{a}"
        explicit = next((c.id for c in problem.hard_constraints
                         if c.type == "precedence"
                         and c.params.get("before") == b
                         and c.params.get("after") == a), None)
        if explicit:
            pid = f"hc:{explicit}"
        if not on(pid):
            continue
        d_b = tmap[b].duration
        for s in range(horizon):
            lhs = {idx: 1.0 for tt, idx in start_cols[a].items() if tt <= s}
            coeffs: Dict[int, float] = dict(lhs)
            for tt, idx in start_cols[b].items():
                if tt <= s - d_b:
                    coeffs[idx] = coeffs.get(idx, 0.0) - 1.0
            if coeffs:
                add_ub(pid, coeffs, 0.0)

    # capacity per resource per slot + availability calendars ---------------
    cap_override = {c.params.get("resource"): float(c.params.get("capacity", 0))
                    for c in problem.hard_constraints
                    if c.type == "resource_capacity" and c.params.get("resource")}
    override_pid = {c.params.get("resource"): f"hc:{c.id}"
                    for c in problem.hard_constraints
                    if c.type == "resource_capacity" and c.params.get("resource")}

    for res in problem.resources:
        rid = res.id
        cap_pid = override_pid.get(rid, f"cap:{rid}")
        cap = cap_override.get(rid, res.capacity)
        if not on(cap_pid):
            cap = float("inf")
        avail_pid = f"avail:{rid}"
        enforce_avail = res.availability is not None and on(avail_pid)
        for slot in range(horizon):
            coeffs: Dict[int, float] = {}
            for t in problem.tasks:
                req = t.resource_requirements.get(rid, 0.0)
                if req <= 0:
                    continue
                for s, idx in start_cols[t.id].items():
                    if s <= slot < s + t.duration:
                        coeffs[idx] = coeffs.get(idx, 0.0) + req
            if coeffs and cap != float("inf"):
                add_ub(cap_pid, coeffs, float(cap))
            # availability: no task may run on the resource in a closed slot
            if enforce_avail and not res.available_at(slot) and coeffs:
                add_ub(avail_pid, coeffs, 0.0)

    # explicit non-capacity hard constraints --------------------------------
    for hc in problem.hard_constraints:
        p = hc.params
        if hc.type == "max_concurrent":
            pid = f"hc:{hc.id}"
            if not on(pid):
                continue
            limit = int(p.get("limit", 1))
            for slot in range(horizon):
                coeffs: Dict[int, float] = {}
                for t in problem.tasks:
                    for s, idx in start_cols[t.id].items():
                        if s <= slot < s + t.duration:
                            coeffs[idx] = coeffs.get(idx, 0.0) + 1.0
                if coeffs:
                    add_ub(pid, coeffs, float(limit))
        elif hc.type == "non_overlap":
            pid = f"hc:{hc.id}"
            if not on(pid):
                continue
            group = list(p.get("tasks", []))
            for slot in range(horizon):
                coeffs: Dict[int, float] = {}
                for tid in group:
                    if tid not in tmap:
                        continue
                    for s, idx in start_cols.get(tid, {}).items():
                        if s <= slot < s + tmap[tid].duration:
                            coeffs[idx] = coeffs.get(idx, 0.0) + 1.0
                if coeffs:
                    add_ub(pid, coeffs, 1.0)
        # window/fixed start rows are already embedded in the column ranges;
        # when their participant is inactive we must widen the ranges instead
        # -- handled below via _bounds expansion, so skip here.

    return _FeasLP(
        c=[0.0] * n, A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        row_participant=row_pid,
        start_cols=start_cols, n_base=n)


def _lp_is_feasible(flp: _FeasLP) -> bool:
    res = simplex.linprog(flp.c, flp.A_ub, flp.b_ub, flp.A_eq, flp.b_eq)
    return res.status == "optimal"


# Column-range widening when window/fixed/release participants are removed:
# the simplest robust approach is to rebuild with a problem copy on which the
# corresponding bounds are relaxed.  We therefore solve sub-problems on clones.

def _clone_without(problem: models.Problem,
                   removed: Set[str]) -> models.Problem:
    """Return a problem copy with the given participants neutralised."""
    p = copy.deepcopy(problem)

    def keep(hc: models.HardConstraint) -> bool:
        if f"hc:{hc.id}" in removed:
            return False
        if hc.type == "time_window":
            if f"hc:{hc.id}:release" in removed and "release" in hc.params:
                hc.params = {k: v for k, v in hc.params.items() if k != "release"}
            if f"hc:{hc.id}:deadline" in removed and "deadline" in hc.params:
                hc.params = {k: v for k, v in hc.params.items() if k != "deadline"}
            if not hc.params:
                return False
        return True

    p.hard_constraints = [c for c in p.hard_constraints if keep(c)]

    # capacity overrides removed? resource_capacity dropped entirely -> resource
    # native capacity applies (already the fallback).
    for t in p.tasks:
        if f"release:{t.id}" in removed:
            t.release_time = 0
        if f"horizon:{t.id}" in removed:
            # widen horizon for this one task by stretching the problem horizon
            p.horizon = max(p.horizon, t.release_time + t.duration)
    # resource capacity participant removal means ignoring *native* capacity:
    # emulate by raising it (the LP builder treats removed override via active
    # sets; here we rebuild without tagging, so bump numeric capacities).
    for r in p.resources:
        if f"cap:{r.id}" in removed:
            r.capacity = 1e9
        if f"avail:{r.id}" in removed:
            r.availability = None
    # dependency-implied precedence edges cannot be deleted without removing
    # the dependency itself.
    for t in p.tasks:
        t.dependencies = [d for d in t.dependencies if f"edge:{d}:{t.id}" not in removed]
    return p


def feasible_with(problem: models.Problem, removed: Set[str],
                  active: Optional[Set[str]] = None) -> bool:
    """Feasibility oracle with a set of participants removed.

    Uses the tagged LP when removals only concern LP-tagged rows; falls back to
    cloning the problem for participants encoded inside column ranges
    (windows / fixed starts / releases / horizon / availability)."""
    needs_clone = any(
        pid.startswith(("release:", "horizon:", "avail:")) or
        pid.startswith("hc:") and _is_range_participant(problem, pid)
        for pid in removed)
    if needs_clone:
        p2 = _clone_without(problem, removed)
        flp = build_feasibility_lp(p2)
        if flp is None:
            return False
        # also drop LP-row participants (capacity/concurrency/...) from clone
        row_removals = {pid for pid in removed
                        if not (pid.startswith(("release:", "horizon:", "avail:")))}
        if row_removals:
            keep = {pid for pid in _all_row_pids(problem) if pid not in row_removals}
            flp = build_feasibility_lp(p2, active=keep)
            if flp is None:
                return False
        return _lp_is_feasible(flp)

    keep = {pid for pid in _all_row_pids(problem) if pid not in removed}
    flp = build_feasibility_lp(problem, active=keep)
    if flp is None:
        return False
    return _lp_is_feasible(flp)


def _is_range_participant(problem: models.Problem, pid: str) -> bool:
    if not pid.startswith("hc:"):
        return False
    cid = pid.split(":")[1].split(":")[0]
    c = next((c for c in problem.hard_constraints if c.id == cid), None)
    return c is not None and c.type in ("time_window", "fixed_start")


def _all_row_pids(problem: models.Problem) -> Set[str]:
    pids: Set[str] = set()
    for b, a in problem.precedence_edges():
        pids.add(f"edge:{b}:{a}")
    for c in problem.hard_constraints:
        pids.add(f"hc:{c.id}")
    for r in problem.resources:
        pids.add(f"cap:{r.id}")
        if r.availability:
            pids.add(f"avail:{r.id}")
    return pids


# --------------------------------------------------------------------------- #
# Layer 1: temporal bound propagation
# --------------------------------------------------------------------------- #

@dataclass
class _Bound:
    value: int
    why_pid: Optional[str]
    why_task: Optional[str]

    def desc(self, problem: models.Problem) -> str:
        if self.why_pid:
            return self.why_pid
        return f"release:{self.why_task}" if self.why_task else "?"


def temporal_contradictions(problem: models.Problem,
                            ) -> List[ConflictSet]:
    """Propagate earliest-start / latest-finish over the precedence DAG.

    A contradiction (latest finish < earliest start) is an *exact* proof of
    infeasibility of the temporal sub-problem.  Provenance is tracked for every
    bound so the witness is already a chain of real constraints; a deletion
    filter against the temporal oracle then certifies irreducibility.
    """
    tmap = problem.task_map()
    horizon = problem.horizon

    # static per-task bounds and their participant
    es: Dict[str, _Bound] = {}
    lf: Dict[str, _Bound] = {}
    for t in problem.tasks:
        es[t.id] = _Bound(t.release_time, None, t.id)
        lf[t.id] = _Bound(horizon, f"horizon:{t.id}", t.id)
        for c in problem.hard_constraints:
            p = c.params
            if p.get("task") != t.id:
                continue
            if c.type == "time_window":
                if p.get("release") is not None:
                    v = int(p["release"])
                    if v > es[t.id].value:
                        es[t.id] = _Bound(v, f"hc:{c.id}:release", t.id)
                if p.get("deadline") is not None:
                    lf[t.id] = _Bound(int(p["deadline"]), f"hc:{c.id}:deadline", t.id)
            elif c.type == "fixed_start" and p.get("start") is not None:
                s = int(p["start"])
                es[t.id] = _Bound(s, f"hc:{c.id}", t.id)
                lf[t.id] = _Bound(s + t.duration, f"hc:{c.id}", t.id)

    succ = problem.successors()
    pred: Dict[str, List[str]] = {t.id: [] for t in problem.tasks}
    for b, a in problem.precedence_edges():
        pred.setdefault(a, []).append(b)

    def edge_pid(b: str, a: str) -> str:
        c = next((c for c in problem.hard_constraints
                  if c.type == "precedence" and c.params.get("before") == b
                  and c.params.get("after") == a), None)
        return f"hc:{c.id}" if c else f"edge:{b}:{a}"

    # provenance maps: bound at task -> set of participant ids supporting it
    es_why: Dict[str, Set[str]] = {t.id: set() for t in problem.tasks}
    lf_why: Dict[str, Set[str]] = {t.id: set() for t in problem.tasks}
    for t in problem.tasks:
        if es[t.id].why_pid:
            es_why[t.id].add(es[t.id].why_pid)
        else:
            es_why[t.id].add(f"release:{t.id}")
        lf_why[t.id].add(lf[t.id].why_pid or f"horizon:{t.id}")

    # forward propagation ES_a >= ES_b + d_b
    order = _topo(problem)
    changed = True
    while changed:
        changed = False
        for a in order:
            for b in pred.get(a, []):
                cand = es[b].value + tmap[b].duration
                if cand > es[a].value:
                    es[a] = _Bound(cand, edge_pid(b, a), a)
                    es_why[a] = es_why[b] | {edge_pid(b, a)}
                    changed = True

    # backward propagation LF_b <= LF_a - d_b
    changed = True
    while changed:
        changed = False
        for b in reversed(order):
            for a in succ.get(b, []):
                cand = lf[a].value - tmap[b].duration
                if cand < lf[b].value:
                    lf[b] = _Bound(cand, edge_pid(b, a), b)
                    lf_why[b] = lf_why[a] | {edge_pid(b, a)}
                    changed = True

    conflicts: List[Tuple[str, Set[str], str]] = []
    seen: Set[frozenset] = set()
    for t in problem.tasks:
        if es[t.id].value + t.duration > lf[t.id].value + 1e-9:
            why = es_why[t.id] | lf_why[t.id]
            key = frozenset(why)
            if key in seen:
                continue
            seen.add(key)
            detail = (f"任务 {_task_name(problem, t.id)} 最早可开始于 t={es[t.id].value}"
                      f"（最早完工 t={es[t.id].value + t.duration}），"
                      f"但要求最晚 t={lf[t.id].value} 完工，时间上不可能同时满足。")
            conflicts.append((t.id, why, detail))

    pmap = {p.id: p for p in build_participants(problem)}
    out: List[ConflictSet] = []
    for tid, why, detail in conflicts[:MAX_CONFLICTS]:
        # deletion filter against the temporal oracle -> irreducible witness
        minimal, kept = _temporal_deletion_filter(problem, set(why))
        parts = [pmap[pid] for pid in sorted(kept) if pid in pmap]
        tasks = sorted({x for p in parts for x in p.task_ids})
        out.append(ConflictSet(
            participants=parts, tasks=tasks,
            explanation=detail, minimal=minimal, method="temporal",
            witness={"tight_task": tid,
                     "earliest_start": es[tid].value,
                     "latest_finish": lf[tid].value}))
    return out


def _topo(problem: models.Problem) -> List[str]:
    succ = problem.successors()
    indeg = {t.id: 0 for t in problem.tasks}
    for b, a in problem.precedence_edges():
        if b in indeg and a in indeg:
            indeg[a] += 1
    queue = [i for i, d in indeg.items() if d == 0]
    order: List[str] = []
    while queue:
        n = queue.pop()
        order.append(n)
        for s in succ.get(n, []):
            indeg[s] -= 1
            if indeg[s] == 0:
                queue.append(s)
    # cycles are caught at validation; append leftovers defensively
    for t in problem.tasks:
        if t.id not in order:
            order.append(t.id)
    return order


def earliest_starts(problem: models.Problem) -> Dict[str, int]:
    """Forward longest-path propagation: earliest feasible start per task."""
    tmap = problem.task_map()
    es = {t.id: t.release_time for t in problem.tasks}
    for c in problem.hard_constraints:
        p = c.params
        tid = p.get("task")
        if tid not in es:
            continue
        if c.type == "time_window" and p.get("release") is not None:
            es[tid] = max(es[tid], int(p["release"]))
        elif c.type == "fixed_start" and p.get("start") is not None:
            es[tid] = max(es[tid], int(p["start"]))
    for _ in range(len(problem.tasks) + 1):
        changed = False
        for b, a in problem.precedence_edges():
            cand = es[b] + tmap[b].duration
            if cand > es[a]:
                es[a] = cand
                changed = True
        if not changed:
            break
    return es


def _temporal_feasible(problem: models.Problem) -> bool:
    es = earliest_starts(problem)
    tmap = problem.task_map()
    lf = {t.id: problem.horizon for t in problem.tasks}
    for c in problem.hard_constraints:
        p = c.params
        tid = p.get("task")
        if tid not in lf:
            continue
        if c.type == "time_window" and p.get("deadline") is not None:
            lf[tid] = min(lf[tid], int(p["deadline"]))
        elif c.type == "fixed_start" and p.get("start") is not None:
            lf[tid] = min(lf[tid], int(p["start"]) + tmap[tid].duration)
    pred: Dict[str, List[str]] = {t.id: [] for t in problem.tasks}
    for b, a in problem.precedence_edges():
        pred.setdefault(a, []).append(b)
    for _ in range(len(problem.tasks) + 1):
        changed = False
        for a in lf:
            for b in pred.get(a, []):
                cand = lf[a] - tmap[b].duration
                if cand < lf[b]:
                    lf[b] = cand
                    changed = True
        if not changed:
            break
    return all(es[t.id] + t.duration <= lf[t.id] for t in problem.tasks)


def _temporal_deletion_filter(problem: models.Problem,
                              witness: Set[str]) -> Tuple[bool, Set[str]]:
    # The classic deletion filter is order-independent only when necessity is
    # tested against the WHOLE witness: a member is essential if, with every
    # OTHER witness member still enforced, deleting it alone restores
    # feasibility.  Testing against the shrinking set makes members look
    # interchangeable (drop edge first and release then seems redundant).
    essential: Set[str] = set()
    for pid in sorted(witness):
        p2 = _clone_without(problem, {pid})
        if _temporal_feasible(p2):
            essential.add(pid)
    # guard: if nothing single-handedly matters (over-coupled witness), keep
    # the original witness rather than returning an empty set.
    kept = essential if essential else set(witness)
    minimal = all(_temporal_feasible(_clone_without(problem, {pid}))
                  for pid in kept)
    return minimal, kept


# --------------------------------------------------------------------------- #
# Layer 2: elastic LP localisation + deletion filter
# --------------------------------------------------------------------------- #

def _lp_size(problem: models.Problem) -> Tuple[int, int]:
    """Estimate (columns, rows) of the feasibility LP without building it."""
    ncol = 0
    for t in problem.tasks:
        lo, hi = _bounds_for_task(problem, t)
        if hi < lo:
            return (-1, -1)
        ncol += hi - lo + 1
    h = problem.horizon
    nrow = len(problem.tasks)
    nrow += h * len(problem.precedence_edges())
    nrow += h * len(problem.resources)
    for c in problem.hard_constraints:
        if c.type in ("max_concurrent", "non_overlap"):
            nrow += h
    return ncol, nrow



def deletion_filter_lp(problem: models.Problem,
                       candidates: Set[str],
                       deadline: float) -> Tuple[Set[str], bool]:
    """Deletion filter producing an IIS, order-independent.

    Necessity is tested against the WHOLE candidate set: a participant is
    essential if deleting it ALONE (everything else still enforced) restores
    feasibility.  Testing against a shrinking set is order-dependent -- once
    another member has been dropped, redundant-looking members slip through.
    Every oracle call therefore removes exactly one participant."""
    essential: Set[str] = set()
    certified = True
    for pid in sorted(candidates):
        if time.time() > deadline:
            certified = False
            break
        if feasible_with(problem, {pid}):
            essential.add(pid)
    if not essential:
        # over-coupled witness where nothing single-handedly matters
        return set(candidates), False
    return essential, certified


def elastic_localize(problem: models.Problem,
                     deadline: float,
                     blocked: Optional[Set[str]] = None
                     ) -> Optional[Tuple[Set[str], Dict[str, float]]]:
    """Add one non-negative elastic violation variable per participant family;
    minimising their sum picks out the constraints that must bend.

    ``blocked`` participants are excluded from the model entirely, so a
    re-solve can surface a *different* conflict rather than the same elastic
    optimum again.

    Returns (participant ids with positive elasticity, elastic values) or
    None if the LP cannot be solved (e.g. simplex iteration limit)."""
    blocked = blocked or set()
    flp = build_feasibility_lp(problem,
                               active=_all_row_pids(problem) - blocked
                               if blocked else None)
    if flp is None:
        return None

    # group rows by participant, ignoring structural rows
    groups: Dict[str, List[int]] = {}
    for i, pid in enumerate(flp.row_participant):
        if pid:
            groups.setdefault(pid, []).append(i)
    # participants encoded via column ranges (windows/fixed/releases/horizon)
    # get their own rows by widening ranges: we add them as explicit bounds on
    # the start-selection columns instead.  Rebuild those as simple rows.
    extra_rows: List[Tuple[str, Dict[int, float], float]] = []
    for t in problem.tasks:
        scols = flp.start_cols[t.id]
        if not scols:
            continue
        # release: sum s*x >= release  -> -sum s x <= -release
        rel_pid = f"release:{t.id}"
        coeffs = {idx: -float(s) for s, idx in scols.items()}
        extra_rows.append((rel_pid, coeffs, -float(t.release_time)))
        # latest completion: sum (s+d)x <= horizon
        hor_pid = f"horizon:{t.id}"
        coeffs = {idx: float(s + t.duration) for s, idx in scols.items()}
        extra_rows.append((hor_pid, coeffs, float(problem.horizon)))
        for c in problem.hard_constraints:
            p = c.params
            if p.get("task") != t.id:
                continue
            if c.type == "time_window":
                if p.get("release") is not None:
                    pid = f"hc:{c.id}:release"
                    coeffs = {idx: -float(s) for s, idx in scols.items()}
                    extra_rows.append((pid, coeffs, -float(p["release"])))
                if p.get("deadline") is not None:
                    pid = f"hc:{c.id}:deadline"
                    coeffs = {idx: float(s + t.duration) for s, idx in scols.items()}
                    extra_rows.append((pid, coeffs, float(p["deadline"])))
            elif c.type == "fixed_start" and p.get("start") is not None:
                pid = f"hc:{c.id}"
                coeffs = {idx: 1.0 for s, idx in scols.items() if s != p["start"]}
                if coeffs:
                    extra_rows.append((pid, coeffs, 0.0))

    # participant -> elastic column
    pids = sorted((groups.keys() | {pid for pid, _, _ in extra_rows}) - blocked)
    if not pids:
        return set(), {}
    n = flp.n_base
    z_index = {pid: n + i for i, pid in enumerate(pids)}

    ncols = n + len(pids)
    A: List[List[float]] = []
    b: List[float] = []

    def densify(coeffs: Dict[int, float]) -> List[float]:
        row = [0.0] * ncols
        for k, v in coeffs.items():
            row[k] = v
        return row

    # existing ub rows: a x <= b  -> a x - z <= b
    for row, rhs, pid in zip(flp.A_ub, flp.b_ub, flp.row_participant):
        r = list(row) + [0.0] * len(pids)
        if pid:
            r[z_index[pid]] = -1.0
        A.append(r)
        b.append(rhs)
    # extra bound rows likewise (range participants live in the column ranges,
    # so blocking them means the elastic row is simply omitted)
    for pid, coeffs, rhs in extra_rows:
        if pid in blocked:
            continue
        r = densify(coeffs)
        r[z_index[pid]] = -1.0
        A.append(r)
        b.append(rhs)
    # equalities stay exact
    A_eq = [list(row) + [0.0] * len(pids) for row in flp.A_eq]
    b_eq = list(flp.b_eq)

    # weights: task assignment-ish violations are costly; capacity cheap.
    weights: Dict[str, float] = {}
    for pid in pids:
        if pid.startswith(("window_deadline", "hc:")):
            w = 10.0
        elif pid.startswith(("cap:", "avail:")):
            w = 1.0
        else:
            w = 5.0
        weights[pid] = w
    c = [0.0] * n + [weights[pid] for pid in pids]

    res = simplex.linprog(c, A, b, A_eq, b_eq)
    if res.status != "optimal":
        return None

    z = {pid: max(0.0, res.x[z_index[pid]]) for pid in pids}
    active = {pid for pid, v in z.items() if v > 1e-6}
    return active, z


def _rigid_partners(problem: models.Problem,
                    elastic_group: Set[str]) -> Set[str]:
    """Rigid participants that co-drive an elastic violation.

    For every capacity / availability / concurrency participant in the elastic
    group, collect the fixed-start, window-bound, release and precedence
    participants attached to the tasks that actually use that resource -- the
    deletion filter then decides which of them truly belong to the IIS.
    """
    tmap = problem.task_map()
    out: Set[str] = set()
    resources_hit = {pid.split(":", 1)[1]
                     for pid in elastic_group
                     if pid.startswith(("cap:", "avail:"))}
    tasks_hit: Set[str] = set()
    for t in problem.tasks:
        if any(r in resources_hit for r in t.resource_requirements):
            tasks_hit.add(t.id)
    # non_overlap / max_concurrent groups pull in every member task
    for pid in elastic_group:
        if pid.startswith("hc:"):
            c = next((c for c in problem.hard_constraints
                      if c.id == pid.split(":", 1)[1]), None)
            if c is not None and c.type == "non_overlap":
                tasks_hit.update(c.params.get("tasks", []))
    for tid in tasks_hit:
        out.add(f"release:{tid}")
        for c in problem.hard_constraints:
            if c.params.get("task") != tid:
                continue
            if c.type == "fixed_start":
                out.add(f"hc:{c.id}")
            elif c.type == "time_window":
                if c.params.get("release") is not None:
                    out.add(f"hc:{c.id}:release")
                if c.params.get("deadline") is not None:
                    out.add(f"hc:{c.id}:deadline")
    # precedence edges touching the hit tasks
    for b, a in problem.precedence_edges():
        if b in tasks_hit or a in tasks_hit:
            cid = next((c.id for c in problem.hard_constraints
                        if c.type == "precedence"
                        and c.params.get("before") == b
                        and c.params.get("after") == a), None)
            out.add(f"hc:{cid}" if cid else f"edge:{b}:{a}")
    return out


def lp_conflicts(problem: models.Problem,
                 pmap: Dict[str, Participant],
                 deadline: float
                 ) -> Tuple[List[ConflictSet], Dict[str, float], bool]:
    """Find multiple independent infeasible subsystems.

    A plain deletion filter over *all* elastic participants is defeated by
    multiple independent conflicts: deleting a member of conflict A leaves
    the instance infeasible due to conflict B, so the filter wrongly discards
    it and degenerates to a single IIS.  We therefore first **partition**
    participants into groups (members that consistently co-occur in elastic
    optima, via hit counts over a few blocked re-solves), then run the
    deletion filter independently within each group."""
    found: List[ConflictSet] = []
    amounts: Dict[str, float] = {}
    timed_out = False

    groups, hit_counts, amounts = _partition_conflicts(problem, deadline)
    if groups is None:
        return found, amounts, True

    for group in groups[:MAX_CONFLICTS]:
        if time.time() > deadline:
            timed_out = True
            break
        # The elastic optimum only contains participants that can *bend*;
        # rigid rows driving the violation (fixed starts, tight releases,
        # precedence edges) carry no elastic variable yet are essential
        # members of the true IIS.  Seed the deletion filter with the elastic
        # group plus the rigid participants touching the same tasks/slots.
        candidates = set(group)
        candidates |= _rigid_partners(problem, group)
        kept, certified = deletion_filter_lp(problem, candidates, deadline)
        if not kept:
            continue
        parts = [pmap[pid] for pid in sorted(kept) if pid in pmap]
        tasks = sorted({x for p in parts for x in p.task_ids})
        found.append(ConflictSet(
            participants=parts, tasks=tasks,
            explanation=_lp_explanation(problem, parts),
            minimal=certified, method="lp",
            witness={"participant_kinds": sorted({p.kind for p in parts}),
                     "elastic_hits": {pid: hit_counts.get(pid, 0)
                                      for pid in kept}}))
    return found, amounts, timed_out


def _partition_conflicts(problem: models.Problem,
                         deadline: float
                         ) -> Tuple[Optional[List[Set[str]]], Dict[str, int],
                                    Dict[str, float]]:
    """Group participants into independent infeasible subsystems.

    Repeatedly solves the elastic LP; after each solve the active participants
    form one candidate group, and one of them is blocked (excluded from later
    solves) to surface the *next* independent fight.  Hit counts over the
    re-solves let members of the same underlying IIS cluster together."""
    hit_counts: Dict[str, int] = {}
    amounts: Dict[str, float] = {}
    groups: List[Set[str]] = []
    blocked: Set[str] = set()

    for _ in range(MAX_CONFLICTS + 1):
        if time.time() > deadline:
            break
        loc = elastic_localize(problem, deadline, blocked=blocked)
        if loc is None:
            break
        active, z = loc
        if not active:
            break
        for pid in active:
            hit_counts[pid] = hit_counts.get(pid, 0) + 1
        for pid, v in z.items():
            if v > 1e-6:
                amounts[pid] = v
        groups.append(set(active))
        # block a representative so the next solve finds a *different* fight;
        # prefer explicit constraints / capacities over structural rows.
        rep = next((pid for pid in sorted(active)
                    if pid.startswith("hc:") or pid.startswith("cap:")
                    or pid.startswith("edge:")),
                   sorted(active)[0])
        blocked.add(rep)

    # Merge heavily-overlapping groups (different elastic optima of the same
    # IIS often differ by one or two "bystander" participants).
    merged: List[Set[str]] = []
    for g in groups:
        for m in merged:
            if g & m and len(g & m) >= min(len(g), len(m)) - 1:
                m |= g
                break
        else:
            merged.append(set(g))
    return merged, hit_counts, amounts


def _lp_explanation(problem: models.Problem, parts: List[Participant]) -> str:
    tasks = sorted({x for p in parts for x in p.task_ids})
    task_txt = "、".join(_task_name(problem, t) for t in tasks[:6])
    if len(tasks) > 6:
        task_txt += f" 等 {len(tasks)} 个任务"
    kinds = {p.kind for p in parts}
    if "resource_capacity" in kinds and (
            "window_deadline" in kinds or "fixed_start" in kinds
            or "precedence" in kinds):
        return (f"时间限制（窗口/先后/固定开工）把 {task_txt} 挤进了资源容量放不下的时段，"
                f"容量与时间限制互相冲突。")
    if kinds <= {"resource_capacity"}:
        res_parts = [p for p in parts if p.resource_id]
        res_txt = "、".join(_resource_name(problem, p.resource_id)
                           for p in res_parts if p.resource_id)
        users = sorted({t for p in res_parts for t in p.task_ids})
        u_txt = "、".join(_task_name(problem, t) for t in users[:8])
        return (f"资源 {res_txt or '容量'} 在某个时段被占满：使用它的任务（{u_txt}）"
                f"无论如何错开都无法同时满足容量上限——要么提高容量，要么移走其中一个任务。")
    if kinds <= {"resource_capacity", "resource_availability", "max_concurrent",
                 "non_overlap"}:
        return f"同一时段内 {task_txt} 对资源/并发的占用超过了允许上限。"
    if kinds <= {"window_release", "window_deadline", "fixed_start", "precedence",
                 "task_release", "task_horizon"}:
        return f"作用在 {task_txt} 上的时间限制与先后顺序首尾不能相顾。"
    return f"以下 {len(parts)} 条限制同时成立时排不出任何方案（涉及 {task_txt}）。"


# --------------------------------------------------------------------------- #
# Suggestions
# --------------------------------------------------------------------------- #

def _suggestions_for_conflict(problem: models.Problem,
                              conflict: ConflictSet,
                              amounts: Dict[str, float],
                              verify: Callable[[models.Problem], str],
                              deadline: float,
                              conflict_index: int = 1,
                              ) -> List[Suggestion]:
    """Build candidate edits, then verify each by cloning + re-solving.

    ``verified`` means the edit produces a feasible schedule for the WHOLE
    problem (not just this conflict); ``sufficient`` mirrors that.  When
    several independent conflicts coexist a single-conflict edit is still
    listed (it resolves *this* fight) but is marked insufficient, and
    :func:`_combined_fix` assembles one edit per conflict into a verified
    end-to-end plan."""
    sugg: List[Suggestion] = []

    def finalize(s: Suggestion, p2: models.Problem) -> None:
        status = verify(p2)
        s.verified = status in ("feasible", "optimal")
        s.sufficient = s.verified
        s.verification = (f"应用后重新求解：{status}" +
                          ("" if s.verified else "（仍有其他冲突需要处理）"))
        s.resolves_conflict = conflict_index
        sugg.append(s)

    # 1) relax each quantitative participant
    for part in conflict.participants:
        if time.time() > deadline:
            break
        pid = part.id
        amount = amounts.get(pid)

        if part.kind == "resource_capacity" and part.resource_id:
            r = problem.resource_map().get(part.resource_id)
            override = next((c for c in problem.hard_constraints
                             if c.id == part.constraint_id), None)
            cur = float(override.params["capacity"]) if override else (
                r.capacity if r else 0.0)
            delta = _capacity_delta(problem, part.resource_id)
            newv = cur + delta
            s = Suggestion(
                action="relax", kind="resource_capacity",
                label=f"把 {part.label.split('：',1)[-1]} 从 {cur:g} 提高到至少 {newv:g}",
                constraint_id=part.constraint_id, resource_id=part.resource_id,
                current_value=cur, suggested_value=newv, delta=delta,
                unit="容量单位",
                expected_effect="缓解最紧时段的资源争抢")
            p2 = _apply_relax(problem, part, new_value=newv)
            finalize(s, p2)

        elif part.kind == "window_deadline" and part.constraint_id:
            c = next(c for c in problem.hard_constraints if c.id == part.constraint_id)
            # time slots are integers; keep int types so the time-indexed
            # builder (which iterates integer ranges) is not handed a float.
            cur = int(c.params["deadline"])
            # Exact minimum: drop this one deadline, propagate earliest starts,
            # and read off when this task can actually finish.
            p2 = _clone_without(problem, {f"hc:{c.id}:deadline"})
            es = earliest_starts(p2)
            need = es[c.params["task"]] + p2.task_map()[c.params["task"]].duration
            newv = int(max(cur + 1, need))
            s = Suggestion(
                action="relax", kind="window_deadline",
                label=f"把 {part.label.split('：',1)[0]} 的最晚完工时间从 {cur:g} 后移到 ≥ {newv:g}",
                constraint_id=part.constraint_id,
                task_id=c.params.get("task"),
                current_value=cur, suggested_value=newv, delta=newv - cur,
                unit="时间单位",
                expected_effect="给关键链留出足够的执行时间")
            finalize(s, _apply_relax(problem, part, new_value=newv))

        elif part.kind == "window_release" and part.constraint_id:
            c = next(c for c in problem.hard_constraints if c.id == part.constraint_id)
            cur = float(c.params["release"])
            s = Suggestion(
                action="relax", kind="window_release",
                label=f"把 {part.label.split('：',1)[0]} 的最早开始限制从 {cur:g} 提前（或删除该下界）",
                constraint_id=part.constraint_id,
                task_id=c.params.get("task"),
                current_value=cur, suggested_value=0.0,
                unit="时间单位",
                expected_effect="允许任务更早开始以避开资源高峰")
            finalize(s, _clone_without(problem, {part.id}))

        elif part.kind == "fixed_start" and part.constraint_id:
            s = Suggestion(
                action="remove_constraint", kind="fixed_start",
                label=f"取消固定开工（{part.label.split('：',1)[-1]}），改由求解器在时间窗内自选开工点",
                constraint_id=part.constraint_id,
                task_id=part.task_ids[0] if part.task_ids else None,
                expected_effect="释放一个把资源钉死在某一时段的约束")
            finalize(s, _clone_without(problem, {pid}))

        elif part.kind == "task_release":
            tid = part.task_ids[0]
            cur_rel = problem.task_map()[tid].release_time
            # suggesting "move 0 to 0" is noise -- only emit when there is
            # actually a positive release bound to relax.
            if cur_rel > 0:
                s = Suggestion(
                    action="relax", kind="task_release",
                    label=f"把任务 {_task_name(problem, tid)} 的最早释放时间 {cur_rel} 提前到 0",
                    task_id=tid,
                    current_value=float(cur_rel),
                    suggested_value=0.0, delta=float(cur_rel), unit="时间单位")
                finalize(s, _clone_without(problem, {pid}))

        elif part.kind == "task_horizon":
            tid = part.task_ids[0]
            t = problem.task_map()[tid]
            # earliest possible completion given the propagated start
            es = earliest_starts(problem)
            need = es.get(tid, t.release_time) + t.duration
            s = Suggestion(
                action="extend_horizon", kind="task_horizon",
                label=(f"把规划期上限从 {problem.horizon} 延长到 ≥ {need}"
                       f"（或缩短任务 {_task_name(problem, tid)} 的工期 {t.duration}）"),
                task_id=tid,
                current_value=float(problem.horizon),
                suggested_value=float(need),
                delta=float(need - problem.horizon), unit="时间单位",
                expected_effect="给放不下的任务留出合法的执行区间")
            p2 = copy.deepcopy(problem)
            p2.horizon = max(problem.horizon, int(need))
            finalize(s, p2)

        elif part.kind in ("precedence",):
            s = Suggestion(
                action="remove_constraint", kind="precedence",
                label=f"删除先后顺序：{part.label.split('：',1)[-1]}",
                constraint_id=part.constraint_id,
                task_id=part.task_ids[-1] if part.task_ids else None,
                expected_effect="断开关键矛盾链中的一环")
            finalize(s, _clone_without(problem, {pid}))

        elif part.kind in ("max_concurrent", "non_overlap", "resource_availability"):
            s = Suggestion(
                action="remove_constraint", kind=part.kind,
                label=f"暂时去掉 {part.label}",
                constraint_id=part.constraint_id,
                resource_id=part.resource_id,
                expected_effect="移除一个并发/互斥/容量/日历上限")
            finalize(s, _clone_without(problem, {pid}))

    # 2) remove a whole task (useful when the task is a structural bottleneck)
    for tid in conflict.tasks[:3]:
        if time.time() > deadline:
            break
        s = Suggestion(
            action="remove_task", kind="task",
            label=f"从本次排程中移除任务 {_task_name(problem, tid)}",
            task_id=tid,
            expected_effect="同时去掉该任务带来的容量占用与先后关系")
        finalize(s, _problem_without_task(problem, tid))

    # sufficient (whole-problem) suggestions first; cap total
    sugg.sort(key=lambda s: (not s.sufficient, s.action != "relax"))
    return sugg[:MAX_SUGGESTIONS]


def _capacity_delta(problem: models.Problem, rid: str) -> float:
    """Minimum extra capacity that ever helps = smallest positive requirement
    among users; round up to a clean unit for integer capacities."""
    reqs = [t.resource_requirements.get(rid, 0.0)
            for t in problem.tasks
            if t.resource_requirements.get(rid, 0.0) > 0]
    if not reqs:
        return 1.0
    step = min(reqs)
    return float(max(1, round(step)))


def _apply_relax(problem: models.Problem, part: Participant,
                 new_value: Optional[float] = None) -> models.Problem:
    """Return a problem clone with one quantitative participant relaxed."""
    p2 = copy.deepcopy(problem)
    if part.constraint_id:
        c = next(c for c in p2.hard_constraints if c.id == part.constraint_id)
        if part.kind == "resource_capacity":
            c.params["capacity"] = new_value
        elif part.kind == "window_deadline":
            c.params["deadline"] = new_value
        elif part.kind == "window_release":
            c.params["release"] = new_value
        elif part.kind == "fixed_start":
            c.params["start"] = new_value
    elif part.kind == "resource_capacity" and part.resource_id:
        r = next(r for r in p2.resources if r.id == part.resource_id)
        r.capacity = new_value
    return p2


def _problem_without_task(problem: models.Problem, tid: str) -> models.Problem:
    p2 = copy.deepcopy(problem)
    p2.tasks = [t for t in p2.tasks if t.id != tid]
    for t in p2.tasks:
        t.dependencies = [d for d in t.dependencies if d != tid]
    p2.hard_constraints = [
        c for c in p2.hard_constraints
        if tid not in (c.params.get("task"),)
        and tid not in c.params.get("tasks", [])
        and c.params.get("before") != tid and c.params.get("after") != tid]
    return p2


# --------------------------------------------------------------------------- #
# Layer 3: heuristic multi-order SGS fallback
# --------------------------------------------------------------------------- #

def heuristic_diagnosis(problem: models.Problem,
                        deadline: float
                        ) -> Tuple[List[ConflictSet], List[str], bool]:
    """Try the serial SGS decoder under many priority rules; collect tasks that
    *no* ordering manages to place, and the participants blocking them."""
    from .solvers import schedule_builder

    tmap = problem.task_map()
    never: Dict[str, int] = {t.id: 0 for t in problem.tasks}
    blockers: Dict[str, Dict[str, int]] = {t.id: {} for t in problem.tasks}
    runs = 0
    import random
    rng = random.Random(42)

    timed_out = False
    # always run the deterministic greedy order first (it is the cheapest
    # useful signal); random restarts only while budget remains.
    rules: List[Dict[str, float]] = [schedule_builder.greedy_order(problem)]
    rules.append({t.id: -float(t.release_time) for t in problem.tasks})
    rules.append({t.id: float(t.duration) for t in problem.tasks})
    rules.append({t.id: -float(sum(t.resource_requirements.values()))
                  for t in problem.tasks})
    import random
    rng = random.Random(42)

    for priorities in rules:
        if runs > 0 and time.time() > deadline:
            timed_out = True
            break
        # a single decode must itself respect the remaining budget; if there
        # is no room for even one, stop rather than record a bogus 0-run sweep
        if runs > 0 and deadline - time.time() <= 0:
            timed_out = True
            break
        runs += 1
        starts = schedule_builder.decode(problem, priorities)
        for tid in never:
            if tid not in starts:
                never[tid] += 1
                for pid in _blocking_participants(problem, tid, starts):
                    blockers[tid][pid] = blockers[tid].get(pid, 0) + 1
        # every task placed under greedy is already strong evidence of
        # feasibility -- do not burn the whole budget on random restarts
        if len(starts) == len(problem.tasks):
            break
        for _ in range(6):
            if time.time() > deadline:
                timed_out = True
                break
            runs += 1
            pri = {t.id: rng.random() for t in problem.tasks}
            st = schedule_builder.decode(problem, pri)
            for tid in never:
                if tid not in st:
                    never[tid] += 1
                    for pid in _blocking_participants(problem, tid, st):
                        blockers[tid][pid] = blockers[tid].get(pid, 0) + 1
            if len(st) == len(problem.tasks):
                break

    # only trust "never placed" when at least one ordering was actually tried;
    # with zero runs every counter is 0 and n==runs would falsely accuse all.
    unsched = [tid for tid, n in never.items() if runs > 0 and n == runs]
    pmap = {p.id: p for p in build_participants(problem)}
    conflicts: List[ConflictSet] = []
    for tid in unsched[:MAX_CONFLICTS]:
        top = sorted(blockers[tid].items(), key=lambda kv: -kv[1])[:8]
        pids = [pid for pid, _ in top]
        # always include the task's own temporal participants
        pids.append(f"horizon:{tid}")
        parts = [pmap[pid] for pid in dict.fromkeys(pids) if pid in pmap]
        tasks = sorted({x for p in parts for x in p.task_ids} | {tid})
        conflicts.append(ConflictSet(
            participants=parts, tasks=tasks,
            explanation=(f"任务 {_task_name(problem, tid)} 在 {runs} 种不同优先级"
                         f"排序下都无法放进规划期；以下限制最频繁地挡住它"
                         f"（启发式结果，未证明为极小冲突集）。"),
            minimal=False, method="heuristic",
            witness={"sgs_runs": runs, "failed_runs": never[tid]}))
    return conflicts, unsched, timed_out


def _blocking_participants(problem: models.Problem, tid: str,
                           starts: Dict[str, int]) -> List[str]:
    """Cheap attribution for why a task could not be placed: window/deadline
    participants plus the resource/concurrency participants that constrain its
    requirement-heavy slots."""
    task = problem.task_map().get(tid)
    if task is None:
        return []
    out: List[str] = []
    if task.release_time + task.duration > problem.horizon:
        out.append(f"horizon:{tid}")
    for c in problem.hard_constraints:
        p = c.params
        if p.get("task") != tid:
            continue
        out.append(f"hc:{c.id}")
        if c.type == "time_window":
            if p.get("release") is not None:
                out.append(f"hc:{c.id}:release")
            if p.get("deadline") is not None:
                out.append(f"hc:{c.id}:deadline")
    for rid in task.resource_requirements:
        out.append(f"cap:{rid}")
        r = problem.resource_map().get(rid)
        if r is not None and r.availability:
            out.append(f"avail:{rid}")
    for c in problem.hard_constraints:
        if c.type in ("max_concurrent", "non_overlap"):
            if c.type != "non_overlap" or tid in c.params.get("tasks", []):
                out.append(f"hc:{c.id}")
    return out


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

def diagnose(problem: models.Problem,
             *,
             time_budget: float = DEFAULT_TIME_BUDGET,
             verifier: Optional[Callable[[models.Problem], str]] = None,
             build_fixes: bool = True,
             ) -> Diagnosis:
    """Full diagnosis pipeline.

    ``verifier`` maps a modified problem to a solve status; it is used to
    confirm suggestions.  It defaults to the LP/IP route on small instances and
    a multi-order SGS check on large ones.  ``build_fixes`` is turned off by
    the internal repair walk so its recursive diagnoses do not themselves
    assemble (recursive) combined fixes.
    """
    t0 = time.time()
    deadline = t0 + time_budget
    diag = Diagnosis(
        feasible=True,
        fingerprint=problem_fingerprint(problem),
        problem_version=problem.version,
        method="temporal")

    if verifier is None:
        verifier = _default_verifier

    participants = build_participants(problem)
    pmap = {p.id: p for p in participants}

    # quick structural check: a task with zero start columns on its own
    empty_tasks = [t.id for t in problem.tasks
                   if (lambda r: r[1] < r[0])(_bounds_for_task(problem, t))]
    if empty_tasks:
        diag.feasible = False
        diag.method = "temporal"
        diag.unschedulable_tasks = empty_tasks
        for tid in empty_tasks[:MAX_CONFLICTS]:
            t = problem.task_map()[tid]
            why = [f"release:{tid}", f"horizon:{tid}"]
            for c in problem.hard_constraints:
                if c.params.get("task") == tid and c.type in (
                        "time_window", "fixed_start"):
                    why.append(f"hc:{c.id}")
            parts = [pmap[p] for p in why if p in pmap]
            diag.conflicts.append(ConflictSet(
                participants=parts, tasks=[tid], minimal=True, method="temporal",
                explanation=(f"任务 {_task_name(problem, tid)}（工期 {t.duration}）"
                             f"在其释放时间/时间窗/固定开工与规划期之间没有任何"
                             f"合法开始时刻。")))
        diag.suggestions = _all_suggestions(problem, diag.conflicts, {},
                                            verifier, deadline)
        if build_fixes:
            diag.combined_fixes = build_combined_fixes(problem, diag, verifier, deadline)
        diag.diagnose_time = time.time() - t0
        diag.summary = _summarize(diag)
        return diag

    # Iterative "peeling" pipeline.  Both temporal propagation and the elastic
    # LP can be fooled by co-existing conflicts (a chain and a capacity fight
    # couple inside one elastic optimum), so after every conflict set is
    # certified we remove *all of its participants* from the working problem
    # and diagnose the remainder again.  Each reported set is therefore a
    # genuinely independent fight, and the suggestion for any one of them --
    # "delete this set's constraint" -- is verified against the ORIGINAL
    # problem separately by the verifier below.
    ncol, nrow = _lp_size(problem)
    lp_ok = ncol > 0 and ncol <= DIAG_MAX_LP_VARS and nrow <= DIAG_MAX_LP_ROWS

    conflicts: List[ConflictSet] = []
    amounts: Dict[str, float] = {}
    methods: Set[str] = set()
    timed_out = False
    # cached heuristic result for the original problem, reused by the fallback
    # so a big instance is swept only once even when the deadline bites
    hresult: Optional[Tuple[List[ConflictSet], List[str], bool]] = None

    work = copy.deepcopy(problem)
    # remember original participant objects for labels (ids are stable)
    original_pmap = dict(pmap)

    while len(conflicts) < MAX_CONFLICTS and time.time() <= deadline:
        wp = {p.id: p for p in build_participants(work)}
        wp.update(original_pmap)

        # 1) temporal layer on the remaining problem
        tcs = temporal_contradictions(work)
        if tcs:
            cf = tcs[0]
            conflicts.append(cf)
            methods.add("temporal")
            work = _clone_without(
                work, {p.id for p in cf.participants
                       if not p.id.startswith("horizon:")})
            continue

        # 2) exact LP layer
        if lp_ok:
            flp = build_feasibility_lp(work)
            if flp is not None and not _lp_is_feasible(flp):
                lconf, lam, tmo = lp_conflicts(work, wp, deadline)
                timed_out = tmo
                amounts.update(lam)
                if not lconf:
                    break
                cf = lconf[0]
                # relabel participants with original labels
                cf.participants = [original_pmap.get(p.id, p)
                                   for p in cf.participants]
                conflicts.append(cf)
                methods.add("lp")
                work = _clone_without(
                    work, {p.id for p in cf.participants
                           if not p.id.startswith("horizon:")})
                continue
            break

        # 3) heuristic layer for big instances (swept once; result is cached
        # for the fallback below to avoid a second, zero-run sweep)
        if hresult is None:
            hresult = heuristic_diagnosis(work, deadline)
        hconf, unsched, tmo = hresult
        timed_out = tmo
        if hconf:
            cf = hconf[0]
            cf.participants = [original_pmap.get(p.id, p) for p in cf.participants]
            conflicts.append(cf)
            methods.add("heuristic")
            diag.unschedulable_tasks.extend(
                t for t in unsched if t not in diag.unschedulable_tasks)
            hresult = None   # force a fresh sweep on the peeled remainder
            work = _clone_without(
                work, {p.id for p in cf.participants
                       if not p.id.startswith("horizon:")})
            continue
        break

    if conflicts:
        diag.feasible = False
        diag.method = ("mixed" if len(methods) > 1
                       else next(iter(methods)))
        diag.conflicts = conflicts[:MAX_CONFLICTS]
        diag.timed_out = timed_out
        diag.suggestions = _all_suggestions(
            problem, diag.conflicts, amounts, verifier, deadline)
        if build_fixes:
            diag.combined_fixes = build_combined_fixes(problem, diag, verifier, deadline)
        diag.diagnose_time = time.time() - t0
        diag.summary = _summarize(diag)
        return diag

    if not lp_ok:
        # large instance: reuse the heuristic sweep already performed in the
        # loop (never re-sweep -- a second call with no budget left would run
        # zero orderings and falsely accuse every task).
        if hresult is not None:
            cf, unsched, tmo = hresult
            diag.timed_out = tmo
            if unsched:
                diag.feasible = False
                diag.method = "heuristic"
                diag.conflicts = cf
                diag.unschedulable_tasks = unsched
                diag.suggestions = _all_suggestions(problem, cf, {},
                                                    verifier, deadline)
                if build_fixes:
                    diag.combined_fixes = build_combined_fixes(problem, diag, verifier, deadline)
            else:
                diag.feasible = True
                diag.method = "heuristic"
                if tmo:
                    diag.timed_out = True
                    diag.summary = ("实例过大且诊断在时间预算内未完成；已尝试的优先级"
                                    "排序都能放下全部任务，未发现明确冲突（非证明性）。")
                else:
                    diag.summary = ("实例过大未使用精确 LP；多排序启发式检查均能放下全部任务，"
                                    "未发现明确冲突（非证明性结论）。")
        diag.diagnose_time = time.time() - t0
        if not diag.summary:
            diag.summary = _summarize(diag)
        return diag

    diag.feasible = True
    diag.method = "lp"
    diag.diagnose_time = time.time() - t0
    diag.summary = "可行性检查通过：时间索引 LP 松弛存在可行解。"
    return diag


def _all_suggestions(problem: models.Problem,
                     conflicts: List[ConflictSet],
                     amounts: Dict[str, float],
                     verifier: Callable[[models.Problem], str],
                     deadline: float) -> List[Suggestion]:
    seen: Set[Tuple[str, str]] = set()
    out: List[Suggestion] = []
    for idx, cf in enumerate(conflicts, 1):
        for s in _suggestions_for_conflict(problem, cf, amounts,
                                           verifier, deadline,
                                           conflict_index=idx):
            key = (s.action, s.constraint_id or s.task_id or "")
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
    return out


def build_combined_fixes(problem: models.Problem,
                         diag: Diagnosis,
                         verifier: Callable[[models.Problem], str],
                         deadline: float) -> List[Dict[str, Any]]:
    """Assemble an end-to-end fix and verify it.

    With several independent conflicts this picks one suggestion per conflict
    and applies them cumulatively.  But a *single* reported conflict can also
    require editing more than one participant (a tight chain coupling a
    deadline and an overloaded machine), so a second strategy performs a
    greedy *repair walk*: apply the best single edit, re-diagnose the result,
    and append the next edit until the modified problem solves."""
    fixes: List[Dict[str, Any]] = []
    seen_plans: Set[Tuple[str, ...]] = set()

    def record(name: str, steps: List[Dict[str, Any]], status: str) -> None:
        key = tuple(st["label"] for st in steps)
        if not key or key in seen_plans:
            return
        seen_plans.add(key)
        fixes.append({
            "name": name,
            "steps": steps,
            "verified": status in ("feasible", "optimal"),
            "verification": f"按 {len(steps)} 步全部修改后重新求解：{status}",
        })

    # strategy 1: one edit per already-reported conflict (prefer relax)
    if len(diag.conflicts) >= 1:
        chosen: List[Suggestion] = []
        for idx, cf in enumerate(diag.conflicts, 1):
            cands = [s for s in diag.suggestions if s.resolves_conflict == idx]
            pick = next((s for s in cands if s.action == "relax"),
                        cands[0] if cands else None)
            if pick is not None:
                chosen.append(pick)
        if chosen:
            p2 = copy.deepcopy(problem)
            steps: List[Dict[str, Any]] = []
            ok = True
            for s in chosen:
                p2, applied = _apply_suggestion(p2, s)
                if not applied:
                    ok = False
                    break
                steps.append(_step_dict(s))
            if ok:
                record("放宽优先组合", steps, verifier(p2))

    # strategy 2: greedy repair walk (handles coupled single conflicts and
    # picks up anything strategy 1 missed).  Budgeted by both wall-clock and
    # a step cap so it can never chase edits for long.
    walk_steps, walk_status = _repair_walk(problem, verifier, deadline)
    if walk_steps:
        record("链式修复（逐步放宽/删除）", walk_steps, walk_status)

    verified = [f for f in fixes if f["verified"]]
    return verified or fixes[:1]


def _step_dict(s: Suggestion) -> Dict[str, Any]:
    return {"conflict": s.resolves_conflict,
            "action": s.action,
            "label": s.label,
            "constraint_id": s.constraint_id,
            "task_id": s.task_id,
            "resource_id": s.resource_id,
            "suggested_value": s.suggested_value,
            "unit": s.unit}


def _repair_walk(problem: models.Problem,
                 verifier: Callable[[models.Problem], str],
                 deadline: float,
                 max_steps: int = 4) -> Tuple[List[Dict[str, Any]], str]:
    """Repeatedly apply the leading relaxation/removal, then re-diagnose,
    until feasible or the budget is exhausted.  Prefers relaxations so the
    suggested plan changes the least restrictive data."""
    cur = copy.deepcopy(problem)
    steps: List[Dict[str, Any]] = []
    tried: Set[Tuple[str, str]] = set()

    for _ in range(max_steps):
        if time.time() > deadline:
            return steps, "timeout"
        status = verifier(cur)
        if status in ("feasible", "optimal"):
            return steps, status
        sub = diagnose(cur, time_budget=max(0.5, deadline - time.time()),
                       verifier=verifier, build_fixes=False)
        if not sub.suggestions:
            return steps, "infeasible"
        # choose: a not-yet-tried relaxation first, then any not-yet-tried
        pick = next((s for s in sub.suggestions
                     if s.action == "relax"
                     and (s.action, s.constraint_id or s.task_id or "") not in tried),
                    next((s for s in sub.suggestions
                          if (s.action, s.constraint_id or s.task_id or "") not in tried),
                         None))
        if pick is None:
            return steps, "infeasible"
        tried.add((pick.action, pick.constraint_id or pick.task_id or ""))
        cur, applied = _apply_suggestion(cur, pick)
        if not applied:
            return steps, "error"
        steps.append(_step_dict(pick))
    return steps, verifier(cur)


def _apply_suggestion(problem: models.Problem,
                      s: Suggestion) -> Tuple[models.Problem, bool]:
    """Apply a Suggestion edit to a problem clone; returns (clone, applied)."""
    p2 = copy.deepcopy(problem)
    if s.action == "remove_task" and s.task_id:
        return _problem_without_task(p2, s.task_id), True
    if s.action == "remove_constraint":
        if s.constraint_id:
            p2.hard_constraints = [c for c in p2.hard_constraints
                                   if c.id != s.constraint_id]
            return p2, True
        if s.kind == "resource_availability" and s.resource_id:
            r = next((r for r in p2.resources if r.id == s.resource_id), None)
            if r is not None:
                r.availability = None
            return p2, True
    if s.action == "relax":
        if s.constraint_id:
            c = next((c for c in p2.hard_constraints
                      if c.id == s.constraint_id), None)
            if c is None:
                return p2, False
            if s.kind == "resource_capacity":
                c.params["capacity"] = s.suggested_value
            elif s.kind == "window_deadline":
                c.params["deadline"] = s.suggested_value
            elif s.kind == "window_release":
                if s.suggested_value is not None:
                    c.params["release"] = s.suggested_value
                else:
                    c.params.pop("release", None)
            elif s.kind == "fixed_start":
                c.params["start"] = s.suggested_value
            return p2, True
        if s.kind == "resource_capacity" and s.resource_id:
            r = next((r for r in p2.resources if r.id == s.resource_id), None)
            if r is not None:
                r.capacity = s.suggested_value
            return p2, True
        if s.kind == "task_release" and s.task_id:
            t = next((t for t in p2.tasks if t.id == s.task_id), None)
            if t is not None:
                t.release_time = int(s.suggested_value or 0)
            return p2, True
        if s.kind == "task_horizon" and s.suggested_value is not None:
            p2.horizon = int(s.suggested_value)
            return p2, True
    return p2, False


def _default_verifier(problem: models.Problem) -> str:
    """Re-solve a modified problem to confirm a suggestion.

    Ordering, cheapest-and-soundest first:

    1. tagged feasibility LP -- if the *relaxation* is infeasible the integer
       problem is certainly infeasible (a one-way proof);
    2. exact B&B IP on small instances -- a feasible/optimal status proves the
       edit really schedules every task;
    3. multi-order SGS sweep -- fast heuristic for larger instances; it can
       miss a feasible arrangement, so callers treat this as approximate.
    """
    from .solvers import schedule_builder

    ncol, _ = _lp_size(problem)
    if 0 < ncol <= DIAG_MAX_LP_VARS:
        try:
            flp = build_feasibility_lp(problem)
            if flp is not None and not _lp_is_feasible(flp):
                return "infeasible"                  # certified by LP
        except Exception:
            pass
        try:
            from .solvers.ip import IPSolver
            sol = IPSolver().solve(problem, {"time_limit": 2.0,
                                             "node_limit": 2000})
            if sol.status in ("optimal", "feasible"):
                return sol.status
            if sol.status == "infeasible":
                return "infeasible"
            # timeout / error: fall through to the heuristic check
        except Exception:
            pass

    best: Dict[str, int] = {}
    import random
    rng = random.Random(7)
    for i in range(12):
        pri = schedule_builder.greedy_order(problem) if i == 0 else {
            t.id: rng.random() for t in problem.tasks}
        starts = schedule_builder.decode(problem, pri)
        if len(starts) == len(problem.tasks):
            return "feasible"
        if len(starts) > len(best):
            best = starts
    return "infeasible"


def _summarize(diag: Diagnosis) -> str:
    if diag.feasible:
        return diag.summary or "未发现不可行。"
    method_txt = {"temporal": "时间关系精确诊断",
                  "lp": "时间索引 LP 精确诊断",
                  "heuristic": "启发式诊断（未证明极小）"}.get(diag.method, diag.method)
    n = len(diag.conflicts)
    verified = sum(1 for s in diag.suggestions if s.verified)
    bits = [f"判定为不可行（{method_txt}），找到 {n} 组冲突。"]
    if diag.unschedulable_tasks:
        bits.append("无论如何放宽都放不下的任务："
                    + "、".join(diag.unschedulable_tasks) + "。")
    if verified:
        bits.append(f"其中 {verified} 条建议已经过修改后重新求解验证。")
    if diag.timed_out:
        bits.append("诊断在时间预算内只完成了部分检查，结论可能不完整。")
    return "".join(bits)


def is_stale(report_fp: str, report_version: int,
             problem: models.Problem) -> bool:
    """True when a stored diagnosis no longer matches the current problem."""
    return (report_version != problem.version
            or report_fp != problem_fingerprint(problem))
