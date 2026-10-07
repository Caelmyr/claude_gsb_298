"""
Infeasibility diagnosis: explain *why* a problem cannot be scheduled.

When the solvers report ``infeasible`` the only information the user gets is
the word itself.  This module turns that into an actionable diagnosis:

* a **conflict set** of user-level objects (tasks, hard constraints, resources,
  the planning horizon) that are jointly infeasible, shrunk to an irreducible
  *infeasible core* (an IIS): removing *any single* member makes the full
  remaining problem feasible;
* **relaxation suggestions** ("delete constraint X", "push the deadline of
  task T to >= k", "raise resource R capacity to >= c", "drop task T",
  "extend the horizon to >= h"), each re-tested against the feasibility oracle
  so a reported suggestion is *verified* to break the deadlock, with the
  minimum numeric relaxation found by binary search.

Two layers are used:

1. Structural checks -- pure graph/interval arithmetic, no solver.  They catch
   the common deadlocks instantly and with perfect precision (empty start
   domains, precedence chains vs. time windows, forced overloads, bad resource
   assignments / calendars).
2. An *elastic continuous-time feasibility LP*.  Every user-removable object
   is a labelled constraint *group* with its own artificial violation
   variable; minimising total violation produces a small hitting set, which is
   then reduced to a minimal infeasible core by a deletion filter (at most
   ``n_seed`` feasibility queries because every constraint is tried once).

The LP is a *relaxation* of the discrete scheduling problem (continuous start
times).  LP-infeasibility therefore *certifies* true infeasibility -- every
integral schedule is contained in the relaxation; LP-feasibility cannot by
itself certify feasibility, so when the relaxation is feasible while no
schedule exists the honest status ``relaxation_feasible`` is returned with a
hint instead of a fabricated conflict set.

Diagnoses carry a content :func:`fingerprint`; a diagnosis whose fingerprint no
longer matches the current problem is marked ``stale`` by the storage layer.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

from . import models
from .solvers import simplex

# --------------------------------------------------------------------------- #
# Size / effort guards -- the LP route is refused past these so a big instance
# never produces a huge dense tableau.  Structural diagnosis always runs.
# --------------------------------------------------------------------------- #
MAX_DIAG_TASKS = 120
MAX_DIAG_ROWS = 20000
DEFAULT_TIME_BUDGET = 10.0
DEFAULT_SIMPLEX_ITER = 20000
MAX_CORES = 3                       # disjoint cores offered in one diagnosis


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass
class ConflictMember:
    """One user-level object participating in a conflict core."""
    kind: str               # task | hard_constraint | resource | horizon
    ref_id: str
    type: str = ""
    label: str = ""
    detail: str = ""

    def key(self) -> Tuple[str, str]:
        return (self.kind, self.ref_id)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConflictMember":
        return cls(**d)


@dataclass
class RelaxationSuggestion:
    """A proposed edit that resolves a core.

    ``action`` is one of: delete_constraint | relax_time_window |
    relax_capacity | delete_task | extend_horizon.  ``status == "verified"``
    only after the feasibility oracle confirmed the edited problem.
    """
    action: str
    target_kind: str = ""
    target_id: str = ""
    label: str = ""
    description: str = ""
    patch: Dict[str, Any] = field(default_factory=dict)
    verified: bool = False
    status: str = "unverified"         # verified | predicted | failed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RelaxationSuggestion":
        return cls(**d)


@dataclass
class ConflictCore:
    members: List[ConflictMember] = field(default_factory=list)
    explanation: str = ""
    source: str = "elastic_lp"         # structural | elastic_lp
    suggestions: List[RelaxationSuggestion] = field(default_factory=list)

    def member_keys(self) -> Set[Tuple[str, str]]:
        return {m.key() for m in self.members}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "members": [m.to_dict() for m in self.members],
            "explanation": self.explanation,
            "source": self.source,
            "suggestions": [s.to_dict() for s in self.suggestions],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConflictCore":
        return cls(
            members=[ConflictMember.from_dict(m) for m in d.get("members", [])],
            explanation=d.get("explanation", ""),
            source=d.get("source", "elastic_lp"),
            suggestions=[RelaxationSuggestion.from_dict(s)
                         for s in d.get("suggestions", [])],
        )


@dataclass
class Diagnosis:
    id: str
    problem_id: str
    feasible: bool
    status: str               # infeasible | feasible | relaxation_feasible
    cores: List[ConflictCore] = field(default_factory=list)
    summary: str = ""
    fingerprint: str = ""
    problem_version: int = 1
    stale: bool = False
    method: str = ""          # structural | elastic_lp | mixed
    elapsed: float = 0.0
    warnings: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=models.now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "problem_id": self.problem_id,
            "feasible": self.feasible,
            "status": self.status,
            "cores": [c.to_dict() for c in self.cores],
            "summary": self.summary,
            "fingerprint": self.fingerprint,
            "problem_version": self.problem_version,
            "stale": self.stale,
            "method": self.method,
            "elapsed": round(self.elapsed, 4),
            "warnings": self.warnings,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Diagnosis":
        return cls(
            id=d["id"],
            problem_id=d["problem_id"],
            feasible=d.get("feasible", True),
            status=d.get("status", "feasible"),
            cores=[ConflictCore.from_dict(c) for c in d.get("cores", [])],
            summary=d.get("summary", ""),
            fingerprint=d.get("fingerprint", ""),
            problem_version=d.get("problem_version", 1),
            stale=d.get("stale", False),
            method=d.get("method", ""),
            elapsed=d.get("elapsed", 0.0),
            warnings=d.get("warnings", []),
            created_at=d.get("created_at", models.now_iso()),
        )


# --------------------------------------------------------------------------- #
# Fingerprinting
# --------------------------------------------------------------------------- #

def fingerprint(problem: models.Problem) -> str:
    """Hash every input feasibility depends on.  Objective/soft constraints
    are excluded, so editing only the objective keeps a diagnosis valid."""
    payload = {
        "horizon": problem.horizon,
        "tasks": [(t.id, t.duration, t.release_time, t.due_date,
                   sorted(t.dependencies),
                   sorted(t.resource_requirements.items()))
                  for t in sorted(problem.tasks, key=lambda t: t.id)],
        "resources": [(r.id, r.type, r.capacity, r.availability)
                      for r in sorted(problem.resources, key=lambda r: r.id)],
        "hard": [(c.id, c.type, sorted(c.params.items(), key=str))
                 for c in sorted(problem.hard_constraints, key=lambda c: c.id)],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Groups & labels
# --------------------------------------------------------------------------- #

HORIZON_GROUP = ("horizon", "")
SYSTEM_GROUP = ("system", "")      # fixed background rows, never removable


@dataclass
class _Row:
    coeffs: Dict[int, float]      # var index -> coefficient (start vars only)
    rhs: float                    # coeffs . s <= rhs
    group: Tuple[str, str]
    fixed: bool = False           # system row: cannot be removed / violated


@dataclass
class _FeasModel:
    n: int
    rows: List[_Row]
    task_var: Dict[str, int]
    labels: Dict[Tuple[str, str], ConflictMember]


def _capacity_for(problem: models.Problem, r: models.Resource) -> float:
    for c in problem.hard_constraints:
        if c.type == "resource_capacity" and c.params.get("resource") == r.id:
            return float(c.params.get("capacity", r.capacity))
    return r.capacity


def _constraint_detail(problem: models.Problem, c: models.HardConstraint) -> str:
    p = c.params
    tmap = problem.task_map()

    def tname(tid: Optional[str]) -> str:
        if not tid:
            return "?"
        t = tmap.get(tid)
        return (t.name or tid) if t else tid

    if c.type == "precedence":
        return f"先后顺序：{tname(p.get('before'))} → {tname(p.get('after'))}"
    if c.type == "time_window":
        return (f"时间窗口：{tname(p.get('task'))} 需在 "
                f"[{p.get('release')}, {p.get('deadline')}) 内执行")
    if c.type == "fixed_start":
        return f"固定开始：{tname(p.get('task'))} 必须在 t={p.get('start')} 开始"
    if c.type == "non_overlap":
        return ("互斥：" + "、".join(tname(x) for x in p.get("tasks", []))
                + " 不得重叠")
    if c.type == "max_concurrent":
        return f"全局最大并发数 ≤ {p.get('limit', 1)}"
    if c.type == "resource_capacity":
        return f"资源 {p.get('resource')} 容量 ≤ {p.get('capacity')}"
    if c.type == "resource_assignment":
        return (f"资源指派：{tname(p.get('task'))} 必须使用 "
                f"{'/'.join(p.get('resources', []))}")
    return c.id


def _member_labels(problem: models.Problem) -> Dict[Tuple[str, str], ConflictMember]:
    labels: Dict[Tuple[str, str], ConflictMember] = {}
    for t in problem.tasks:
        labels[("task", t.id)] = ConflictMember(
            "task", t.id, "task", t.name or t.id,
            f"任务 {t.name or t.id}（工期 {t.duration}）")
    for c in problem.hard_constraints:
        labels[("hard_constraint", c.id)] = ConflictMember(
            "hard_constraint", c.id, c.type, c.id,
            _constraint_detail(problem, c))
    for r in problem.resources:
        labels[("resource", r.id)] = ConflictMember(
            "resource", r.id, "resource", r.name or r.id,
            f"资源 {r.name or r.id}（容量 {_capacity_for(problem, r):g}）")
    labels[HORIZON_GROUP] = ConflictMember(
        "horizon", "", "horizon", "horizon", f"计划周期 H={problem.horizon}")
    return labels


# --------------------------------------------------------------------------- #
# Earliest / latest start propagation (shared by structural checks and the LP)
# --------------------------------------------------------------------------- #

def _bounds(problem: models.Problem
           ) -> Tuple[Dict[str, float], Dict[str, float],
                      Dict[str, List[str]], List[str]]:
    """Propagate earliest starts forward and latest starts backward over the
    precedence graph, taking releases / windows / fixed starts / horizon into
    account."""
    tmap = problem.task_map()
    H = problem.horizon
    win_rel: Dict[str, float] = {}
    win_dl: Dict[str, float] = {}
    fixed: Dict[str, float] = {}
    for c in problem.hard_constraints:
        tid = c.params.get("task")
        if tid not in tmap:
            continue
        if c.type == "time_window":
            if c.params.get("release") is not None:
                win_rel[tid] = float(c.params["release"])
            if c.params.get("deadline") is not None:
                win_dl[tid] = float(c.params["deadline"])
        elif c.type == "fixed_start" and c.params.get("start") is not None:
            fixed[tid] = float(c.params["start"])

    preds: Dict[str, List[str]] = {t.id: [] for t in problem.tasks}
    for b, a in problem.precedence_edges():
        if b in tmap and a in tmap:
            preds.setdefault(a, []).append(b)

    order = _topo_order(problem, preds)

    earliest: Dict[str, float] = {}
    for t in problem.tasks:
        e = float(t.release_time)
        if t.id in win_rel:
            e = max(e, win_rel[t.id])
        if t.id in fixed:
            e = fixed[t.id]
        earliest[t.id] = e
    for j in order:
        for p in preds.get(j, []):
            earliest[j] = max(earliest[j], earliest[p] + tmap[p].duration)

    latest: Dict[str, float] = {t.id: float(H - t.duration)
                                for t in problem.tasks}
    for tid, dl in win_dl.items():
        latest[tid] = min(latest[tid], dl - tmap[tid].duration)
    for tid, s_ in fixed.items():
        latest[tid] = s_
    for j in reversed(order):
        for p in preds.get(j, []):
            latest[p] = min(latest[p], latest[j] - tmap[p].duration)
    return earliest, latest, preds, order


def _topo_order(problem: models.Problem,
                preds: Optional[Dict[str, List[str]]] = None) -> List[str]:
    tmap = problem.task_map()
    if preds is None:
        preds = {t.id: [] for t in problem.tasks}
        for b, a in problem.precedence_edges():
            if b in tmap and a in tmap:
                preds.setdefault(a, []).append(b)
    succ: Dict[str, List[str]] = {t.id: [] for t in problem.tasks}
    indeg = {t.id: 0 for t in problem.tasks}
    for a, ps in preds.items():
        indeg[a] = len(ps)
        for b in ps:
            succ.setdefault(b, []).append(a)
    queue = [j for j, d in indeg.items() if d == 0]
    order: List[str] = []
    while queue:
        n = queue.pop()
        order.append(n)
        for s in succ.get(n, []):
            indeg[s] -= 1
            if indeg[s] == 0:
                queue.append(s)
    if len(order) != len(indeg):  # defensive: validation normally rejects cycles
        order += [j for j in indeg if j not in order]
    return order


# --------------------------------------------------------------------------- #
# Elastic feasibility model (continuous start times, labelled groups)
# --------------------------------------------------------------------------- #

def build_feasibility_model(
        problem: models.Problem,
        disabled: Optional[Set[Tuple[str, str]]] = None
) -> Optional[_FeasModel]:
    """Build the labelled continuous-time feasibility LP.

    One unrestricted variable ``s_j`` per active task.  Rows come in two
    flavours:

    * **fixed** rows (``fixed=True``) -- release times and dependency edges
      declared directly on tasks: immutable background facts that never take
      an elastic variable and never appear as a removable group.
    * **removable** rows -- owned by a user-visible group: explicit hard
      constraints, resource capacities and the planning horizon.  Only these
      rows receive elastic artificial variables, so an infeasible core names
      objects the user can actually delete/relax; tasks implicated by a core
      are derived empirically by the caller (delete-task re-test).

    Pairwise ordering rows for tight resources / non-overlap are asserted only
    in the direction the tasks' windows make unavoidable; when windows force a
    pair to overlap, both directions appear and directly contradict.

    Returns ``None`` past the size guards.
    """
    disabled = disabled or set()
    tasks = [t for t in problem.tasks if ("task", t.id) not in disabled]
    horizon_enabled = HORIZON_GROUP not in disabled
    hcs = [c for c in problem.hard_constraints
           if ("hard_constraint", c.id) not in disabled]
    active = {t.id for t in tasks}
    tmap = {t.id: t for t in tasks}
    suppressed_res = {r.id for r in problem.resources
                      if ("resource", r.id) in disabled}

    n = len(tasks)
    task_var = {t.id: i for i, t in enumerate(tasks)}
    rows: List[_Row] = []

    win_by_task: Dict[str, models.HardConstraint] = {}
    fixed_by_task: Dict[str, models.HardConstraint] = {}
    for c in hcs:
        tid = c.params.get("task")
        if c.type == "time_window":
            win_by_task[tid] = c
        elif c.type == "fixed_start":
            fixed_by_task[tid] = c

    # -- domains: fixed release/horizon + removable window/fixed rows ----- #
    for t in tasks:
        j, d = t.id, t.duration
        sv = task_var[j]
        wc = win_by_task.get(j)
        fc = fixed_by_task.get(j)

        rows.append(_Row({sv: -1.0}, -float(t.release_time),
                         SYSTEM_GROUP, fixed=True))
        if horizon_enabled:
            rows.append(_Row({sv: 1.0}, float(problem.horizon - d),
                             HORIZON_GROUP))
        if wc and wc.params.get("release") is not None:
            rows.append(_Row({sv: -1.0}, -float(wc.params["release"]),
                             ("hard_constraint", wc.id)))
        if wc and wc.params.get("deadline") is not None:
            rows.append(_Row({sv: 1.0},
                             float(wc.params["deadline"]) - d,
                             ("hard_constraint", wc.id)))
        if fc and fc.params.get("start") is not None:
            s_ = float(fc.params["start"])
            g = ("hard_constraint", fc.id)
            rows.append(_Row({sv: -1.0}, -s_, g))
            rows.append(_Row({sv: 1.0}, s_, g))

    # -- precedence: explicit HC rows are removable, task dependencies fixed #
    explicit_edges: Set[Tuple[str, str]] = set()
    for c in hcs:
        if c.type != "precedence":
            continue
        b, a = c.params.get("before"), c.params.get("after")
        explicit_edges.add((b, a))
        if b in active and a in active:
            rows.append(_Row(
                {task_var[a]: -1.0, task_var[b]: 1.0},
                -float(tmap[b].duration),
                ("hard_constraint", c.id)))
    for b, a in problem.precedence_edges():
        if (b, a) in explicit_edges or b not in active or a not in active:
            continue
        rows.append(_Row(
            {task_var[a]: -1.0, task_var[b]: 1.0},
            -float(tmap[b].duration), SYSTEM_GROUP, fixed=True))

    # -- resources: capacity rows owned by HC or by the resource itself --- #
    disabled_cap = {c.params.get("resource") for c in problem.hard_constraints
                    if c.type == "resource_capacity"
                    and ("hard_constraint", c.id) in disabled}
    for r in problem.resources:
        if r.id in suppressed_res or r.id in disabled_cap:
            continue
        cap = _capacity_for(problem, r)
        owner = next((c for c in hcs if c.type == "resource_capacity"
                      and c.params.get("resource") == r.id), None)
        grp = (("hard_constraint", owner.id) if owner
               else ("resource", r.id))
        users = [t for t in tasks if t.resource_requirements.get(r.id, 0) > 0]
        _pair_order_rows(users, tmap, rows, task_var, grp, cap, resource=r.id)

    # -- non_overlap groups ------------------------------------------------ #
    for c in hcs:
        if c.type != "non_overlap":
            continue
        grp = ("hard_constraint", c.id)
        users = [tmap[j] for j in c.params.get("tasks", []) if j in active]
        _pair_order_rows(users, tmap, rows, task_var, grp, 1.0)

    # -- max_concurrent: cliques forced simultaneously -------------------- #
    for c in hcs:
        if c.type != "max_concurrent":
            continue
        limit = int(c.params.get("limit", 1))
        grp = ("hard_constraint", c.id)
        # windows alone (ignoring precedence) give per-task [E, L]; use the
        # propagated bounds for stronger clique detection.
        earliest, latest, _p, _o = _bounds(problem)
        for clique in _forced_cliques(
                [t for t in tasks if t.id in earliest],
                earliest, latest, limit + 1):
            cm = [tmap[j] for j in clique]
            _pair_order_rows(cm, tmap, rows, task_var, grp, 1.0,
                             earliest=earliest, latest=latest)

    if len(rows) > MAX_DIAG_ROWS:
        return None
    return _FeasModel(n, rows, task_var, _member_labels(problem))


def _pair_order_rows(users, tmap, rows, task_var, grp, cap,
                     resource=None, earliest=None, latest=None):
    """Add pairwise ordering rows for tasks that cannot overlap (demand sum
    over capacity).  A direction is asserted only when the tasks' own windows
    make the opposite direction impossible, which keeps every row valid."""
    if earliest is None:
        earliest = {}
        latest = {}
        for t in users:
            e = float(t.release_time)
            latest[t.id] = None
            earliest[t.id] = e

    def E(t):
        return earliest.get(t.id, float(t.release_time))

    def L(t):
        return latest[t.id] if t.id in latest else None

    for x in range(len(users)):
        for y in range(x + 1, len(users)):
            tj, tk = users[x], users[y]
            if resource is not None:
                req = (tj.resource_requirements[resource]
                       + tk.resource_requirements[resource])
                if req <= cap + 1e-9:
                    continue
            j, k = tj.id, tk.id
            sj, sk = task_var[j], task_var[k]
            lj, lk = L(tj), L(tk)
            # j-before-k impossible when j's earliest completion exceeds k's
            # latest start -> k must precede j:  s_j - s_k >= d_k.
            if lk is not None and E(tj) + tj.duration > lk:
                rows.append(_Row({sj: -1.0, sk: 1.0}, -float(tk.duration), grp))
            # k-before-j impossible -> j must precede k:  s_k - s_j >= d_j.
            if lj is not None and E(tk) + tk.duration > lj:
                rows.append(_Row({sk: -1.0, sj: 1.0}, -float(tj.duration), grp))


def _forced_cliques(tasks, earliest, latest, size):
    """Greedy cliques of >= ``size`` tasks whose forced-execution intervals
    [latest_start, earliest_start+duration) all pairwise intersect."""
    tmap = {t.id: t for t in tasks}
    ids = [t.id for t in tasks
           if earliest[t.id] + t.duration > latest[t.id]]
    adj: Dict[str, Set[str]] = {j: set() for j in ids}
    for x in range(len(ids)):
        j = ids[x]
        for y in range(x + 1, len(ids)):
            k = ids[y]
            # forced intervals [L, E+d) overlap pairwise
            if (latest[j] < earliest[k] + tmap[k].duration
                    and latest[k] < earliest[j] + tmap[j].duration):
                adj[j].add(k)
                adj[k].add(j)
    cliques: List[List[str]] = []
    remaining = set(ids)
    while remaining:
        seed = max(remaining, key=lambda j: len(adj[j] & remaining))
        clique = [seed]
        candidates = adj[seed] & remaining
        while candidates:
            nxt = max(candidates, key=lambda j: len(adj[j] & candidates))
            clique.append(nxt)
            candidates &= adj[nxt]
        remaining -= set(clique)
        if len(clique) >= size:
            cliques.append(sorted(clique))
    return cliques


# --------------------------------------------------------------------------- #
# Elastic LP oracle
# --------------------------------------------------------------------------- #

def _solve_elastic(model: Optional[_FeasModel],
                   disabled: Set[Tuple[str, str]],
                   max_iter: int = DEFAULT_SIMPLEX_ITER
                   ) -> Tuple[str, List[Tuple[str, Tuple[str, str], float]], float]:
    """Minimise total violation of the *removable* rows.

    Fixed system rows are added as hard inequalities; every active removable
    row gets one non-negative artificial ``v`` (``a·s - v ≤ b``) with cost 1.
    A positive artificial identifies the user-visible group whose constraint
    must bend.
    """
    if model is None:
        return "model_too_large", [], 0.0

    fixed_rows: List[_Row] = []
    active: List[_Row] = []
    for row in model.rows:
        if row.fixed:
            fixed_rows.append(row)
        elif row.group not in disabled:
            active.append(row)

    m = len(active)
    n = model.n + m
    c = [0.0] * n
    A_ub: List[List[float]] = []
    b_ub: List[float] = []

    # Fixed rows stay plain inequalities; each removable row gets its own
    # non-negative artificial column.
    art_group: List[Tuple[str, str]] = []
    for row in fixed_rows:
        r = [0.0] * n
        for idx, v in row.coeffs.items():
            r[idx] = v
        A_ub.append(r)
        b_ub.append(row.rhs)
    for row in active:
        col = model.n + len(art_group)
        r = [0.0] * n
        for idx, v in row.coeffs.items():
            r[idx] = v
        r[col] = -1.0
        c[col] = 1.0
        A_ub.append(r)
        b_ub.append(row.rhs)
        art_group.append(row.group)

    res = simplex.linprog(c, A_ub=A_ub, b_ub=b_ub, max_iter=max_iter)
    if res.status != "optimal":
        return res.status, [], float(res.objective or 0.0)
    obj = res.objective if res.objective is not None else 0.0
    violated: List[Tuple[str, Tuple[str, str], float]] = []
    seen: Set[Tuple[str, str]] = set()
    for i, g in enumerate(art_group):
        if res.x[model.n + i] > 1e-7 and g not in seen:
            seen.add(g)
            violated.append(("group", g, res.x[model.n + i]))
    return "optimal", violated, obj


def is_feasible(model: Optional[_FeasModel],
                disabled: Optional[Set[Tuple[str, str]]] = None,
                max_iter: int = DEFAULT_SIMPLEX_ITER) -> bool:
    status, _v, obj = _solve_elastic(model, disabled or set(), max_iter)
    return status == "optimal" and obj <= 1e-7


def _implicated_tasks(model: _FeasModel,
                      core: Set[Tuple[str, str]],
                      problem: models.Problem,
                      deadline: float,
                      cap: int = 3) -> List[str]:
    """Tasks whose deletion resolves the contradiction named by ``core``.

    This is tested empirically (delete one task, keep the core constraints)
    rather than guessed, so every named task gets a *verified* delete-task
    suggestion later.  Stop after ``cap`` to keep the core small.
    """
    out: List[str] = []
    removable = {row.group for row in model.rows if not row.fixed}
    base_removed = (removable - set(core))   # keep exactly the core active
    for tid in model.task_var:
        if time.time() > deadline or len(out) >= cap:
            break
        if is_feasible(model, base_removed | {("task", tid)}):
            out.append(tid)
    return out


def _minimal_core(model: _FeasModel,
                  seed: Set[Tuple[str, str]],
                  deadline: float
                  ) -> Optional[Set[Tuple[str, str]]]:
    """Deletion filter: shrink a feasibility-breaking ``seed`` (removable
    groups whose artificials were positive) to an inclusion-minimal
    infeasible core, tested against the *full* problem.

    Every other removable group is feasible jointly with ``seed`` (the elastic
    optimum violated exactly the seed groups), so dropping them all keeps an
    infeasible subsystem.  Each seed member is then tried once; it survives
    only if removing it makes the system feasible.  Fixed system rows always
    remain.  The result is an IIS: with any one surviving member removed the
    full problem is feasible.
    """
    removable = {row.group for row in model.rows if not row.fixed}
    removed = (removable - set(seed))   # baseline: only seed groups active
    core: Set[Tuple[str, str]] = set(seed)
    timed_out = False
    for g in seed:
        if time.time() > deadline:
            timed_out = True
            break
        if is_feasible(model, removed | {g}):
            continue                 # g is essential to the contradiction
        removed.add(g)
        core.discard(g)
    if timed_out and not core:
        return None
    return core


# --------------------------------------------------------------------------- #
# Structural diagnosis (solver-free, exact)
# --------------------------------------------------------------------------- #

def structural_diagnosis(problem: models.Problem) -> List[ConflictCore]:
    """Conflict cores provable without any solver call."""
    cores: List[ConflictCore] = []
    labels = _member_labels(problem)
    tmap = problem.task_map()
    H = problem.horizon
    earliest, latest, preds, order = _bounds(problem)

    win_by_task: Dict[str, models.HardConstraint] = {}
    fixed_by_task: Dict[str, models.HardConstraint] = {}
    for c in problem.hard_constraints:
        tid = c.params.get("task")
        if c.type == "time_window":
            win_by_task[tid] = c
        elif c.type == "fixed_start":
            fixed_by_task[tid] = c

    edge_owner: Dict[Tuple[str, str], str] = {}
    for c in problem.hard_constraints:
        if c.type == "precedence":
            edge_owner[(c.params.get("before"), c.params.get("after"))] = c.id

    # 1) unary contradictions --------------------------------------------- #
    for t in problem.tasks:
        j = t.id
        wc = win_by_task.get(j)
        fc = fixed_by_task.get(j)
        e0 = float(t.release_time)
        if wc and wc.params.get("release") is not None:
            e0 = max(e0, float(wc.params["release"]))
        f0 = float(fc.params["start"]) if fc and fc.params.get("start") is not None else None

        # fixed start earlier than release / window release
        if f0 is not None:
            bound = float(t.release_time)
            owner = None
            if wc and wc.params.get("release") is not None and \
                    float(wc.params["release"]) > bound:
                bound = float(wc.params["release"])
                owner = wc
            if f0 < bound:
                members = [labels[("task", j)],
                           labels[("hard_constraint", fc.id)]]
                if owner is not None:
                    members.append(labels[("hard_constraint", owner.id)])
                cores.append(ConflictCore(
                    members=members,
                    explanation=(f"{t.name or j} 被固定在 t={f0:g}，"
                                 f"早于最早可开工时刻 {bound:g}"),
                    source="structural"))
            if f0 + t.duration > H:
                cores.append(ConflictCore(
                    members=[labels[("task", j)],
                             labels[("hard_constraint", fc.id)],
                             labels[HORIZON_GROUP]],
                    explanation=(f"固定开始 t={f0:g} 时，{t.name or j} "
                                 f"完工 {f0 + t.duration:g} 超过周期 {H}"),
                    source="structural"))
            if wc and wc.params.get("deadline") is not None and \
                    f0 + t.duration > float(wc.params["deadline"]):
                cores.append(ConflictCore(
                    members=[labels[("task", j)],
                             labels[("hard_constraint", fc.id)],
                             labels[("hard_constraint", wc.id)]],
                    explanation=(f"{t.name or j} 固定在 t={f0:g}，"
                                 f"完工晚于窗口截止 {wc.params['deadline']}"),
                    source="structural"))
            if f0 < 0:
                cores.append(ConflictCore(
                    members=[labels[("task", j)],
                             labels[("hard_constraint", fc.id)]],
                    explanation=f"{t.name or j} 被固定在负时间 t={f0:g}",
                    source="structural"))

        # earliest completion beyond horizon
        check_lo = f0 if f0 is not None else e0
        if check_lo + t.duration > H:
            members = [labels[("task", j)], labels[HORIZON_GROUP]]
            if f0 is not None:
                members.append(labels[("hard_constraint", fc.id)])
            elif wc and wc.params.get("release") is not None:
                members.append(labels[("hard_constraint", wc.id)] if False
                               else labels[("hard_constraint", wc.id)]
                               if check_lo == float(wc.params["release"])
                               else labels[HORIZON_GROUP])
                members = _uniq(members)
            cores.append(ConflictCore(
                members=members,
                explanation=(f"{t.name or j} 最早 t={check_lo:g} 开工，"
                             f"工期 {t.duration}，最早完工 "
                             f"{check_lo + t.duration:g} > 周期 {H}"),
                source="structural"))

        # earliest completion beyond window deadline
        if wc and wc.params.get("deadline") is not None and \
                check_lo + t.duration > float(wc.params["deadline"]):
            cores.append(ConflictCore(
                members=[labels[("task", j)],
                         labels[("hard_constraint", wc.id)]],
                explanation=(f"{t.name or j} 最早完工 {check_lo + t.duration:g} "
                             f"晚于时间窗截止 {wc.params['deadline']}"),
                source="structural"))

    # 2) propagated earliest vs. latest along precedence ------------------- #
    # Only one core per contradictory chain: process tight tasks and skip a
    # task once one of its tight incoming edges has already been reported, so
    # a long infeasible chain yields one minimal explanation (the leaf task,
    # the chain and the violated bound) instead of one report per node.
    reported: Set[Tuple[str, ...]] = set()
    reported_edges: Set[Tuple[str, str]] = set()
    reported_tasks: Set[str] = set()
    # Leaves first (reverse topological order) so one contradiction chain is
    # explained by its longest path rather than re-reported at every node.
    for t in (tmap[j] for j in reversed(order)):
        j = t.id
        if earliest[j] <= latest[j] + 1e-9:
            continue
        tight_in = {(p, j) for p in preds.get(j, [])
                    if abs(earliest[j] - (earliest[p] + tmap[p].duration)) < 1e-9}
        if (tight_in & reported_edges) or j in reported_tasks:
            reported_edges |= tight_in
            continue
        reported_edges |= tight_in
        chain = _tight_chain(j, earliest, preds, tmap)
        reported_tasks.update(chain)
        members: List[ConflictMember] = []
        seen: Set[Tuple[str, str]] = set()

        def add(key):
            if key not in seen and key in labels:
                seen.add(key)
                members.append(labels[key])

        for cid in chain:
            add(("task", cid))
        for x in range(len(chain) - 1):
            e = (chain[x], chain[x + 1])
            if e in edge_owner:
                add(("hard_constraint", edge_owner[e]))
        fc = fixed_by_task.get(j)
        wc = win_by_task.get(j)
        if fc is not None:
            add(("hard_constraint", fc.id))
        elif wc is not None and wc.params.get("deadline") is not None and \
                earliest[j] + t.duration > float(wc.params["deadline"]):
            add(("hard_constraint", wc.id))
        else:
            add(HORIZON_GROUP)
        # window releases that push the chain
        for cid in chain:
            w = win_by_task.get(cid)
            if w is not None and w.params.get("release") is not None and \
                    abs(earliest[cid] - float(w.params["release"])) < 1e-9:
                add(("hard_constraint", w.id))
        key = tuple(sorted(f"{m.kind}:{m.ref_id}" for m in members))
        if key not in reported:
            reported.add(key)
            cores.append(ConflictCore(
                members=members,
                explanation=(f"先后关系链把 {t.name or j} 的最早开工推到 "
                             f"{earliest[j]:g}，但它最晚必须在 "
                             f"{latest[j]:g} 开工"),
                source="structural"))

    # 3) resource assignment vs. task requirements ------------------------ #
    for c in problem.hard_constraints:
        if c.type != "resource_assignment":
            continue
        tid = c.params.get("task")
        required = set(c.params.get("resources", []))
        if tid in tmap and required and required.isdisjoint(
                tmap[tid].resource_requirements):
            cores.append(ConflictCore(
                members=[labels[("task", tid)],
                         labels[("hard_constraint", c.id)]],
                explanation=(f"资源指派要求 {tmap[tid].name or tid} 使用 "
                             f"{'/'.join(sorted(required))}，"
                             f"但该任务的资源需求里没有其中任何资源"),
                source="structural"))

    # 4) resource calendars ------------------------------------------------ #
    for t in problem.tasks:
        if earliest[t.id] > latest[t.id] + 1e-9:
            continue  # already reported
        for rid in t.resource_requirements:
            r = problem.resource_map().get(rid)
            if r is None or r.availability is None:
                continue
            open_slots = set()
            for a, b in r.availability:
                open_slots.update(range(max(0, int(a)), min(H, int(b))))
            ok = any(all(s + k in open_slots for k in range(t.duration))
                     for s in range(int(earliest[t.id]),
                                    int(latest[t.id]) + 1))
            if not ok:
                cores.append(ConflictCore(
                    members=[labels[("task", t.id)],
                             labels[("resource", rid)]],
                    explanation=(f"资源 {r.name or rid} 的可用时段无法覆盖 "
                                 f"{t.name or t.id} 在 [{earliest[t.id]:g}, "
                                 f"{latest[t.id] + t.duration:g}] 内任何长度 "
                                 f"{t.duration} 的连续执行区间"),
                    source="structural"))

    # 5) forced overloads --------------------------------------------------- #
    pin = _pin_groups(problem)
    for r in problem.resources:
        cap = _capacity_for(problem, r)
        owner = next((c for c in problem.hard_constraints
                      if c.type == "resource_capacity"
                      and c.params.get("resource") == r.id), None)
        owner_key = (("hard_constraint", owner.id) if owner
                     else ("resource", r.id))
        hit = _forced_overload(
            [t for t in problem.tasks
             if t.resource_requirements.get(r.id, 0) > 0],
            earliest, latest,
            lambda t: t.resource_requirements[r.id], cap)
        if hit:
            members, seen = [], set()

            def add2(key):
                if key not in seen and key in labels:
                    seen.add(key)
                    members.append(labels[key])

            add2(owner_key)
            for tid in hit:
                add2(("task", tid))
                for g in pin.get(tid, ()):
                    add2(g)
            cores.append(ConflictCore(
                members=members,
                explanation=(f"时间窗/固定时间强制这些任务在同一时段占用资源 "
                             f"{r.name or r.id}，合计需求超过容量 {cap:g}"),
                source="structural"))

        # 5b) cumulative Hall squeeze: no single slot is forced overloaded,
        # but a set of tasks whose whole feasible span lies inside some
        # interval [a,b) cannot fit (total required work > cap*(b-a)).
        hit5b = _cumulative_overload(
            [t for t in problem.tasks
             if t.resource_requirements.get(r.id, 0) > 0],
            earliest, latest,
            lambda t: t.resource_requirements[r.id], cap, H)
        if hit5b:
            members, seen = [], set()

            def add5b(key):
                if key not in seen and key in labels:
                    seen.add(key)
                    members.append(labels[key])

            add5b(owner_key)
            for tid in hit5b:
                add5b(("task", tid))
                for g in pin.get(tid, ()):
                    add5b(g)
            cores.append(ConflictCore(
                members=members,
                explanation=(f"这些任务的可行执行窗口被先后关系/时间窗挤压在同一段"
                             f"时间内，所需占用资源 {r.name or r.id} 的总工作量超过"
                             f"容量 {cap:g} 在该段时间内能提供的总产能"),
                source="structural"))

    for c in problem.hard_constraints:
        if c.type == "non_overlap":
            ids = [x for x in c.params.get("tasks", []) if x in tmap]
            hit = _forced_overload([tmap[x] for x in ids],
                                   earliest, latest, lambda t: 1.0, 1.0)
            if hit:
                members, seen = [], set()

                def add3(key, c=c):
                    if key not in seen and key in labels:
                        seen.add(key)
                        members.append(labels[key])

                add3(("hard_constraint", c.id))
                for tid in hit:
                    add3(("task", tid))
                    for g in pin.get(tid, ()):
                        add3(g)
                cores.append(ConflictCore(
                    members=members,
                    explanation="时间窗/固定时间强制这些互斥任务在同一时段执行",
                    source="structural"))
            hall = _cumulative_overload([tmap[x] for x in ids],
                                        earliest, latest, lambda t: 1.0, 1.0, H)
            if hall:
                members2, seen2 = [], set()

                def add3b(key, c=c):
                    if key not in seen2 and key in labels:
                        seen2.add(key)
                        members2.append(labels[key])

                add3b(("hard_constraint", c.id))
                for tid in hall:
                    add3b(("task", tid))
                    for g in pin.get(tid, ()):
                        add3b(g)
                cores.append(ConflictCore(
                    members=members2,
                    explanation=("这些互斥任务的可行执行窗口被挤压在同一段时间内，"
                                 "总工期超过该段可用时间，无法全部排下"),
                    source="structural"))
        elif c.type == "max_concurrent":
            limit = float(c.params.get("limit", 1))
            hit = _forced_overload(problem.tasks, earliest, latest,
                                   lambda t: 1.0, limit)
            if hit:
                members, seen = [], set()

                def add4(key, c=c):
                    if key not in seen and key in labels:
                        seen.add(key)
                        members.append(labels[key])

                add4(("hard_constraint", c.id))
                for tid in hit:
                    add4(("task", tid))
                    for g in pin.get(tid, ()):
                        add4(g)
                cores.append(ConflictCore(
                    members=members,
                    explanation=(f"时间窗/固定时间强制至少 {len(hit)} 个任务"
                                 f"同时运行，超过最大并发 {int(limit)}"),
                    source="structural"))

    return _dedupe_cores(cores)


def _uniq(members: List[ConflictMember]) -> List[ConflictMember]:
    seen = set()
    out = []
    for m in members:
        k = m.key()
        if k not in seen:
            seen.add(k)
            out.append(m)
    return out


def _pin_groups(problem: models.Problem
                ) -> Dict[str, List[Tuple[str, str]]]:
    """Constraint groups pinning each task's start (fixed / tight window)."""
    out: Dict[str, List[Tuple[str, str]]] = {}
    earliest, latest, _p, _o = _bounds(problem)
    for c in problem.hard_constraints:
        tid = c.params.get("task")
        if tid is None:
            continue
        if c.type == "fixed_start":
            out.setdefault(tid, []).append(("hard_constraint", c.id))
        elif c.type == "time_window":
            tight = (c.params.get("release") is not None
                     and abs(earliest.get(tid, 0)
                             - float(c.params["release"])) < 1e-9)
            tight_deadline = (c.params.get("deadline") is not None
                              and latest.get(tid, 0)
                              <= float(c.params["deadline"])
                              - problem.task_map()[tid].duration + 1e-9)
            if tight or tight_deadline:
                out.setdefault(tid, []).append(("hard_constraint", c.id))
    return out


def _forced_overload(users, earliest, latest, demand, cap) -> Optional[List[str]]:
    """Tasks forced to run simultaneously by their own windows.

    A task is guaranteed active at slot τ for every feasible start iff
    ``latest_start ≤ τ < earliest_start + duration``; sweep these forced
    intervals and return the tasks behind the worst forced overload.
    """
    tmap = {t.id: t for t in users}
    events = []
    for t in users:
        a = latest[t.id]
        b = earliest[t.id] + t.duration
        if b <= a + 1e-9:
            continue            # no forced slot
        events.append((a, 1, t))
        events.append((b, -1, t))
    if not events:
        return None
    active: Dict[str, float] = {}
    worst_total = cap
    worst: List[str] = []
    for _t0, kind, t in sorted(events, key=lambda e: (e[0], -e[1])):
        if kind == 1:
            active[t.id] = demand(t)
        total = sum(active.values())
        if total > cap + 1e-9 and (not worst or total > worst_total + 1e-9
                                   or len(active) > len(worst)):
            worst_total, worst = total, sorted(active)
        if kind == -1:
            active.pop(t.id, None)
    return worst or None


def _cumulative_overload(users, earliest, latest, demand, cap, H
                         ) -> Optional[List[str]]:
    """Hall-type cumulative squeeze.

    If tasks exist whose *entire* feasible execution span lies inside some
    interval [a, b), their total required work cannot exceed
    ``cap * (b - a)``; otherwise no schedule exists.  Candidate bounds are the
    propagated earliest starts and latest completions.  The offending task set
    is reduced greedily so the reported core stays small.
    """
    if not users:
        return None
    tmap = {t.id: t for t in users}
    # span of task i: [E_i, L_i + d_i)
    span = {}
    for t in users:
        a = earliest[t.id]
        b = latest[t.id] + t.duration
        if b > a + 1e-9 and b <= H + 1e-9:
            span[t.id] = (a, b, demand(t) * t.duration)
    if not span:
        return None
    a_ends = sorted({v[0] for v in span.values()})
    b_ends = sorted({v[1] for v in span.values()})
    worst: List[str] = []
    worst_excess = 0.0
    for a in a_ends:
        for b in b_ends:
            if b <= a:
                continue
            inside = [tid for tid, (ea, eb, _w) in span.items()
                      if ea >= a - 1e-9 and eb <= b + 1e-9]
            work = sum(span[tid][2] for tid in inside)
            if work > cap * (b - a) + 1e-9:
                excess = work - cap * (b - a)
                if excess > worst_excess + 1e-9:
                    worst_excess, worst = excess, inside
    if not worst:
        return None
    # shrink: drop any task whose removal keeps the certificate valid
    work_of = {tid: span[tid][2] for tid in worst}
    reduced = set(worst)
    for tid in worst:
        rest = reduced - {tid}
        # certificate valid if the same Hall interval still overflows; the
        # tight interval for the subset is [min E, max L+d] of the subset.
        if not rest:
            continue
        aa = min(span[x][0] for x in rest)
        bb = max(span[x][1] for x in rest)
        ww = sum(work_of[x] for x in rest)
        if ww > cap * (bb - aa) + 1e-9:
            reduced.discard(tid)
    return sorted(reduced)


def _tight_chain(tid, earliest, preds, tmap) -> List[str]:
    """Predecessor chain achieving the propagated earliest start."""
    chain = [tid]
    cur, guard = tid, 0
    while guard < 10000:
        guard += 1
        nxt = None
        for p in preds.get(cur, []):
            if abs(earliest[cur] - (earliest[p] + tmap[p].duration)) < 1e-9:
                nxt = p
                break
        if nxt is None or nxt in chain:
            break
        chain.append(nxt)
        cur = nxt
    chain.reverse()
    return chain


def _dedupe_cores(cores: List[ConflictCore]) -> List[ConflictCore]:
    seen: Set[Tuple[Tuple[str, str], ...]] = set()
    out: List[ConflictCore] = []
    for core in cores:
        key = tuple(sorted(m.key() for m in core.members))
        if key in seen:
            continue
        seen.add(key)
        core.members.sort(key=lambda m: (m.kind, m.ref_id))
        out.append(core)
    return out


# --------------------------------------------------------------------------- #
# Suggestions
# --------------------------------------------------------------------------- #

def _clone(problem: models.Problem) -> models.Problem:
    return models.Problem.from_dict(problem.to_dict())


def _candidate_suggestions(problem: models.Problem,
                           core: ConflictCore) -> List[RelaxationSuggestion]:
    suggestions: List[RelaxationSuggestion] = []
    tmap = problem.task_map()

    for m in core.members:
        if m.kind == "hard_constraint":
            c = next((c for c in problem.hard_constraints if c.id == m.ref_id),
                     None)
            if c is None:
                continue
            if c.type == "time_window" and c.params.get("deadline") is not None:
                tid = c.params.get("task")
                t = tmap.get(tid)
                if t is not None:
                    lo = max(t.release_time, c.params.get("release", 0) or 0)
                    min_dl = int(lo) + t.duration
                    suggestions.append(RelaxationSuggestion(
                        action="relax_time_window",
                        target_kind="hard_constraint", target_id=c.id,
                        label=f"放宽 {t.name or tid} 的截止时间",
                        description=(f"把截止时间从 {c.params.get('deadline')} "
                                     f"放宽到 ≥ {min_dl}"),
                        patch={"hard_constraint": c.id,
                               "params": {"deadline": min_dl}}))
            elif c.type == "resource_capacity":
                rid = c.params.get("resource")
                need = _sufficient_capacity(problem, rid)
                suggestions.append(RelaxationSuggestion(
                    action="relax_capacity",
                    target_kind="hard_constraint", target_id=c.id,
                    label=f"提高资源 {rid} 的容量",
                    description=f"把容量从 {c.params.get('capacity'):g} 提高",
                    patch={"hard_constraint": c.id,
                           "params": {"capacity": need}}))
            else:
                descriptions = {
                    "precedence": "删除后两个任务不再有先后要求，可解除此冲突。",
                    "non_overlap": "删除后这些任务允许在同一时段执行。",
                    "max_concurrent": "删除后不再限制同时运行的任务数量。",
                    "resource_assignment": "删除后任务可使用其需求中的任意资源。",
                    "fixed_start": "删除后任务可在允许窗口内任意时刻开始。",
                    "time_window": "删除该时间窗口约束。",
                }
                suggestions.append(RelaxationSuggestion(
                    action="delete_constraint",
                    target_kind="hard_constraint", target_id=c.id,
                    label=f"删除约束「{m.detail}」",
                    description=descriptions.get(c.type, "删除该约束。"),
                    patch={"remove_hard_constraint": c.id}))
        elif m.kind == "resource":
            r = next((r for r in problem.resources if r.id == m.ref_id), None)
            if r is None:
                continue
            if any(c.type == "resource_capacity"
                   and c.params.get("resource") == r.id
                   for c in problem.hard_constraints):
                continue
            need = _sufficient_capacity(problem, r.id)
            suggestions.append(RelaxationSuggestion(
                action="relax_capacity",
                target_kind="resource", target_id=r.id,
                label=f"提高资源 {r.name or r.id} 的容量",
                description=(f"把容量从 {r.capacity:g} 提高（验证后给出最小值），"
                             f"或错峰执行相关任务。"),
                patch={"resource": r.id, "capacity": need}))
        elif m.kind == "task":
            t = tmap.get(m.ref_id)
            if t is None:
                continue
            suggestions.append(RelaxationSuggestion(
                action="delete_task",
                target_kind="task", target_id=t.id,
                label=f"去掉任务「{t.name or t.id}」",
                description=("该任务是冲突链上的必要一环；去掉它（同时清理"
                             "引用它的约束）后，其余任务可以排出。"),
                patch={"remove_task": t.id}))
        elif m.kind == "horizon":
            h_need = _sufficient_horizon(problem)
            suggestions.append(RelaxationSuggestion(
                action="extend_horizon",
                target_kind="horizon", target_id="",
                label="延长计划周期",
                description=f"把计划周期从 {problem.horizon} 延长到 ≥ {h_need}。",
                patch={"horizon": h_need}))

    out, seen = [], set()
    for s in suggestions:
        k = (s.action, s.target_kind, s.target_id)
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def _sufficient_capacity(problem: models.Problem, rid: str) -> float:
    """Capacity certainly sufficient for this resource: sum of every user's
    demand (all tasks at once).  Binary search tightens to the minimum that is
    actually feasible."""
    total = sum(t.resource_requirements.get(rid, 0) for t in problem.tasks)
    if abs(total - round(total)) < 1e-9:
        total = float(int(round(total)))
    return round(total, 3)


def _sufficient_horizon(problem: models.Problem) -> int:
    tmap = problem.task_map()
    preds = {t.id: [] for t in problem.tasks}
    for b, a in problem.precedence_edges():
        preds.setdefault(a, []).append(b)
    earliest = {t.id: float(t.release_time) for t in problem.tasks}
    for j in _topo_order(problem, preds):
        for p in preds.get(j, []):
            earliest[j] = max(earliest[j], earliest[p] + tmap[p].duration)
    end = int(max((earliest[t.id] + t.duration for t in problem.tasks),
                  default=problem.horizon))
    return max(end, problem.horizon)


def _referenced_tasks(c: models.HardConstraint) -> List[str]:
    p = c.params
    out = []
    for key in ("task", "before", "after"):
        if p.get(key):
            out.append(p[key])
    out.extend(p.get("tasks", []))
    return out


def apply_patch(problem: models.Problem,
                patch: Dict[str, Any]) -> models.Problem:
    """Clone ``problem`` and apply one suggestion patch."""
    p = _clone(problem)
    if "remove_hard_constraint" in patch:
        p.hard_constraints = [c for c in p.hard_constraints
                              if c.id != patch["remove_hard_constraint"]]
    if "remove_task" in patch:
        tid = patch["remove_task"]
        p.tasks = [t for t in p.tasks if t.id != tid]
        for t in p.tasks:
            t.dependencies = [d for d in t.dependencies if d != tid]
        p.hard_constraints = [
            c for c in p.hard_constraints
            if not ({tid} & set(_referenced_tasks(c)))]
    if "horizon" in patch:
        p.horizon = int(patch["horizon"])
    if "resource" in patch and "capacity" in patch:
        for r in p.resources:
            if r.id == patch["resource"]:
                r.capacity = float(patch["capacity"])
    if "hard_constraint" in patch:
        for c in p.hard_constraints:
            if c.id == patch["hard_constraint"]:
                c.params.update(patch.get("params", {}))
    return p


# --------------------------------------------------------------------------- #
# Verification & tightening
# --------------------------------------------------------------------------- #

def _schedule_witness(problem: models.Problem, thorough: bool = True) -> bool:
    """True iff an actual integral schedule placing every task exists.

    Tries serial-SGS priority rules ordered cheapest-first: earliest-deadline
    and greedy first (they resolve the common "tight window must run first"
    case), then more orders and an LP-guided order when ``thorough``.  During
    numeric binary searches callers pass ``thorough=False`` to keep each
    feasibility probe O(n²) instead of O(n³).
    """
    from .solvers import schedule_builder
    from . import objectives

    def good(starts) -> bool:
        return (len(starts) == len(problem.tasks)
                and not objectives.hard_violations(problem, starts))

    # latest-start / EDD: tightest latest start scheduled first (decode picks
    # the largest priority), so negate to give small latest starts high prio.
    _e, latest, _p, _o = _bounds(problem)
    edd = {t.id: -float(latest[t.id]) for t in problem.tasks}
    orders = [
        edd,
        schedule_builder.greedy_order(problem),
        {t.id: -float(t.release_time) for t in problem.tasks},
    ]
    if thorough:
        succ = problem.successors()
        orders += [
            {t.id: float(t.release_time) for t in problem.tasks},
            {t.id: -float(t.duration) for t in problem.tasks},
            {t.id: float(t.duration) for t in problem.tasks},
            {t.id: -float(len(succ.get(t.id, []))) for t in problem.tasks},
        ]
    for prio in orders:
        if good(schedule_builder.decode(problem, prio)):
            return True

    if thorough:
        try:
            from .solvers import time_indexed, simplex
            model = time_indexed.build(problem)
            res = simplex.linprog(model.c, model.A_ub, model.b_ub,
                                  model.A_eq, model.b_eq, max_iter=30000)
            if res.status == "optimal":
                expected = time_indexed.solution_from_x(problem, model, res.x)
                prio = time_indexed.expected_starts_to_priorities(expected)
                if good(schedule_builder.decode(problem, prio)):
                    return True
        except Exception:
            pass
    return False


def _verify(edited: models.Problem, thorough: bool = True) -> Tuple[bool, str]:
    """Feasibility status of an edited problem.

    Returns ``(True, "feasible")`` only with an actual schedule witness;
    ``(False, "infeasible")`` when the relaxation certifies infeasibility;
    ``(False, "inconclusive")`` when the relaxation is feasible but no
    integral witness was found.  ``thorough=False`` uses the cheap witness
    orders only (for inner binary-search probes).
    """
    if structural_diagnosis(edited):
        return False, "infeasible"
    if len(edited.tasks) > MAX_DIAG_TASKS:
        return False, "inconclusive"
    try:
        model = build_feasibility_model(edited)
    except Exception as exc:
        return False, f"build_failed:{exc}"
    if model is None:
        return False, "inconclusive"
    status, _v, obj = _solve_elastic(model, set())
    if status != "optimal":
        return False, "inconclusive"
    if obj > 1e-7:
        return False, "infeasible"
    # relaxation feasible: demand a concrete integral witness
    witness = _schedule_witness(edited, thorough=thorough)
    return witness, ("feasible" if witness else "inconclusive")


def _core_resolved(original: models.Problem,
                   edited: models.Problem,
                   core_keys: Set[Tuple[str, str]]) -> bool:
    """True iff the contradiction named by ``core_keys`` no longer exists in
    ``edited`` (other, independent cores may remain).

    Semantics per member kind: a deleted task/constraint/resource releases it;
    a relaxed constraint / extended horizon must no longer reproduce the
    contradiction.  Test: with every removable group that is *not* a surviving
    core member disabled, the remaining system is feasible.
    """
    surviving: Set[Tuple[str, str]] = set()
    task_ids = {t.id for t in edited.tasks}
    hc_ids = {c.id for c in edited.hard_constraints}
    res_ids = {r.id for r in edited.resources}
    for kind, ref in core_keys:
        if kind == "task" and ref in task_ids:
            surviving.add((kind, ref))
        elif kind == "hard_constraint" and ref in hc_ids:
            surviving.add((kind, ref))
        elif kind == "resource" and ref in res_ids:
            surviving.add((kind, ref))
        elif kind == "horizon":
            surviving.add(HORIZON_GROUP)
    if not surviving:
        return True
    # A changed-but-present constraint (e.g. relaxed deadline) counts as
    # resolved when structural analysis no longer flags the same member set.
    struct = structural_diagnosis(edited)
    for rc in struct:
        rk = rc.member_keys()
        if rk and rk <= core_keys:
            return False
    try:
        model = build_feasibility_model(edited)
    except Exception:
        return True
    if model is None:
        return True
    removable = {row.group for row in model.rows if not row.fixed}
    disable = removable - surviving
    return is_feasible(model, disable)


def _evaluate_suggestion(problem: models.Problem,
                         sug: RelaxationSuggestion,
                         deadline: float,
                         core_keys: Optional[Set[Tuple[str, str]]] = None
                         ) -> RelaxationSuggestion:
    """Verify a suggestion and, for numeric relaxations, find the *minimum*
    value that works.

    Statuses:
      verified       -- the edited whole problem is fully schedulable;
      resolves_core  -- the edited problem still has *other* cores, but this
                        suggestion's own core is gone;
      predicted      -- relaxation feasible but no witness was built/timeout;
      blocked        -- the edit does not even remove its own core (dropped).
    """
    core_keys = core_keys or set()
    if time.time() > deadline:
        sug.status = "predicted"
        return sug

    def status_of(edited: models.Problem) -> str:
        ok, note = _verify(edited)
        if ok:
            return "verified"
        if note == "inconclusive":
            # can't build a witness, but the target core may still be gone
            return ("resolves_core"
                    if _core_resolved(problem, edited, core_keys)
                    else "predicted")
        # note == "infeasible": still some contradiction.  If *this* core is
        # gone, the remaining infeasibility belongs to other cores.
        return ("resolves_core"
                if _core_resolved(problem, edited, core_keys)
                else "blocked")

    if sug.action in ("delete_constraint", "delete_task"):
        st = status_of(apply_patch(problem, sug.patch))
        sug.status = st
        sug.verified = (st == "verified")
        if st == "resolves_core":
            sug.description = ("可解除本组冲突（其它冲突组仍需分别处理）。")
        return sug

    # Numeric relaxations: search for the smallest value that at least
    # resolves the target core, preferring one that makes the whole problem
    # feasible.  The search oracle accepts both outcomes.
    def probe(v_patch: Dict[str, Any]) -> str:
        return status_of(apply_patch(problem, v_patch))

    if sug.action == "extend_horizon":
        cur = problem.horizon
        hi0 = int(sug.patch["horizon"])
        best, st = _search_numeric(
            cur, hi0, deadline, integral=True,
            make=lambda v: {"horizon": int(v)}, probe=probe)
        if best is None:
            sug.status = "predicted" if time.time() > deadline else "blocked"
            return sug
        sug.patch["horizon"] = best
        sug.description = f"把计划周期从 {cur} 延长到 ≥ {best}（最小验证值）。"
        sug.status, sug.verified = st, st == "verified"
        if st == "resolves_core":
            sug.description += " 可解除本组冲突（其它冲突组仍需分别处理）。"
        return sug

    if sug.action == "relax_time_window":
        cid = sug.patch["hard_constraint"]
        c = next(c for c in problem.hard_constraints if c.id == cid)
        cur_dl = int(c.params.get("deadline"))
        hi0 = int(sug.patch["params"]["deadline"])
        best, st = _search_numeric(
            cur_dl, hi0, deadline, integral=True,
            make=lambda v: {"hard_constraint": cid,
                            "params": {"deadline": int(v)}}, probe=probe)
        if best is None:
            sug.status = "predicted" if time.time() > deadline else "blocked"
            return sug
        sug.patch["params"]["deadline"] = best
        sug.description = f"把截止时间从 {cur_dl} 放宽到 ≥ {best}（最小验证值）。"
        sug.status, sug.verified = st, st == "verified"
        if st == "resolves_core":
            sug.description += " 可解除本组冲突（其它冲突组仍需分别处理）。"
        return sug

    if sug.action == "relax_capacity":
        is_resource = "resource" in sug.patch
        if is_resource:
            rid = sug.patch["resource"]
            cur = next((r.capacity for r in problem.resources
                        if r.id == rid), 0.0)
        else:
            cid = sug.patch["hard_constraint"]
            c = next(c for c in problem.hard_constraints if c.id == cid)
            rid = c.params.get("resource")
            cur = float(c.params.get("capacity", 0.0))
        hi0 = float(sug.patch.get("capacity",
                                 sug.patch.get("params", {}).get("capacity",
                                                                  cur)))

        # Only capacity values that line up with the demand granularity can
        # make a physical difference (e.g. unit demands need an integral
        # capacity); search on that quantum grid.
        reqs = [t.resource_requirements.get(rid, 0.0)
                for t in problem.tasks
                if t.resource_requirements.get(rid, 0.0) > 0]
        quantum = _demand_quantum(reqs)
        import math as _math
        cur_k = int(_math.ceil(cur / quantum - 1e-9))
        hi_k = int(_math.ceil(hi0 / quantum - 1e-9))

        def make(v: float) -> Dict[str, Any]:
            return ({"resource": rid, "capacity": float(v)} if is_resource
                    else {"hard_constraint": cid,
                          "params": {**sug.patch.get("params", {}),
                                     "capacity": float(v)}})

        best_k, st = _search_grid(cur_k, hi_k, quantum, deadline,
                                  make=make, probe=probe)
        if best_k is None:
            sug.status = "predicted" if time.time() > deadline else "blocked"
            return sug
        best = best_k * quantum
        best_v = (int(best) if abs(best - round(best)) < 1e-6
                  else round(best, 3))
        if is_resource:
            sug.patch["capacity"] = best_v
        else:
            sug.patch["params"] = {**sug.patch.get("params", {}),
                                   "capacity": best_v}
        sug.description = (f"把资源 {rid} 的容量从 {cur:g} 提高到 ≥ "
                           f"{best_v:g}（最小验证值）。")
        sug.status, sug.verified = st, st == "verified"
        if st == "resolves_core":
            sug.description += " 可解除本组冲突（其它冲突组仍需分别处理）。"
        return sug

    st = status_of(apply_patch(problem, sug.patch))
    sug.status, sug.verified = st, st == "verified"
    return sug


def _demand_quantum(reqs: List[float]) -> float:
    """Greatest common divisor (scaled) of resource demands -- the smallest
    capacity increment that can physically change which task sets fit."""
    vals = []
    for r in reqs:
        if r > 1e-9:
            vals.append(round(r, 6))
    if not vals:
        return 1.0
    # work on integers after a 1e6 scaling
    ints = [int(round(v * 1e6)) for v in vals]
    g = ints[0]
    for x in ints[1:]:
        while x:
            g, x = x, g % x
    return max(g / 1e6, 1e-6)


def _search_numeric(cur, hi, deadline, *, integral: bool, make, probe):
    """Integral expansion + binary search for the smallest value in
    [cur+1 ..] whose patch yields verified/resolves_core.  Returns
    (value, best_status)."""
    lo0 = int(cur)
    v = max(int(hi), lo0 + 1)
    found_v, found_st = None, None
    steps = 0
    while time.time() <= deadline and steps < 40:
        steps += 1
        st = probe(make(v))
        if st in ("verified", "resolves_core"):
            found_v, found_st = v, st
            break
        v = v + max(1, v - lo0)
        if v > 10 ** 9:
            break
    if found_v is None:
        return None, None
    lo, hi2 = lo0, int(found_v)
    best, bst = found_v, found_st
    while lo < hi2 and time.time() <= deadline:
        mid = (lo + hi2) // 2
        st = probe(make(mid))
        if st in ("verified", "resolves_core"):
            best, bst, hi2 = mid, st, mid
        else:
            lo = mid + 1
    return best, bst


def _search_grid(cur_k, hi_k, quantum, deadline, *, make, probe):
    """Same expansion+binary-search as :func:`_search_numeric` but over an
    integer grid index ``k`` where the tried value is ``k * quantum``."""
    k = max(hi_k, cur_k + 1)
    found_k, found_st = None, None
    steps = 0
    while time.time() <= deadline and steps < 40:
        steps += 1
        st = probe(make(k * quantum))
        if st in ("verified", "resolves_core"):
            found_k, found_st = k, st
            break
        k = k + max(1, k - cur_k)
        if k > 10 ** 9:
            break
    if found_k is None:
        return None, None
    lo, hi2 = cur_k, found_k
    best, bst = found_k, found_st
    while lo < hi2 and time.time() <= deadline:
        mid = (lo + hi2) // 2
        st = probe(make(mid * quantum))
        if st in ("verified", "resolves_core"):
            best, bst, hi2 = mid, st, mid
        else:
            lo = mid + 1
    return best, bst


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def diagnose(problem: models.Problem,
             time_budget: float = DEFAULT_TIME_BUDGET,
             persist: bool = False) -> Diagnosis:
    """Diagnose infeasibility of ``problem`` end to end."""
    t0 = time.time()
    deadline = t0 + time_budget
    diag = Diagnosis(
        id=models.new_id("diag"),
        problem_id=problem.id,
        feasible=False,
        status="infeasible",
        fingerprint=fingerprint(problem),
        problem_version=problem.version,
    )

    if not problem.tasks:
        diag.feasible = True
        diag.status = "feasible"
        diag.summary = "问题中没有任务，自然可行。"
        if persist:
            _persist(problem, diag)
        return diag

    warnings: List[str] = []
    structural = structural_diagnosis(problem)
    diag.cores.extend(structural)

    model = None
    if len(problem.tasks) <= MAX_DIAG_TASKS:
        try:
            model = build_feasibility_model(problem)
        except Exception as exc:
            warnings.append(f"诊断模型构建失败：{exc}")
    else:
        warnings.append(
            f"任务数 {len(problem.tasks)} 超过 LP 诊断上限 "
            f"{MAX_DIAG_TASKS}，仅给出结构化诊断。")

    if model is not None:
        status, violated, obj = _solve_elastic(model, set())
        if status != "optimal":
            warnings.append(f"弹性 LP 求解器返回 {status}，该层结果不可用。")
            # still try the structural route below
        elif obj <= 1e-7 and not structural:
            # Relaxation feasible: either a real schedule exists or the gap is
            # purely integrality.  Demand a concrete witness before claiming
            # feasibility; otherwise report the honest inconclusive status.
            if _schedule_witness(problem):
                diag.feasible = True
                diag.status = "feasible"
                diag.summary = "存在可行排程（已用贪心解码构造出完整排程）。"
                diag.method = "structural"
                diag.elapsed = time.time() - t0
                if persist:
                    _persist(problem, diag)
                return diag
            diag.status = "relaxation_feasible"
            diag.summary = (
                "连续时间松弛模型可行，但没能构造出完整的整数排程：卡死很可能"
                "来自容量/互斥类约束在离散组合下的相互挤压"
                "（分数意义下排得下、整数意义下排不下）。"
                "建议优先尝试：放宽最紧资源的容量、放宽截止时间窗，"
                "或去掉一个最争用资源上的任务后重试诊断。")
            diag.method = "structural"
            diag.warnings = warnings
            diag.elapsed = time.time() - t0
            if persist:
                _persist(problem, diag)
            return diag
        elif obj > 1e-7:
            # removable groups already implicated by a structural proof
            covered: Set[Tuple[str, str]] = set()
            for c in structural:
                for m in c.members:
                    if m.kind in ("hard_constraint", "resource", "horizon"):
                        covered.add(m.key())

            blocked: Set[Tuple[str, str]] = set()
            existing_keys = {tuple(sorted(m.key() for m in c.members))
                             for c in diag.cores}
            n_lp = 0
            while n_lp < MAX_CORES and time.time() < deadline:
                st, vv, oo = (_solve_elastic(model, blocked)
                              if blocked else (status, violated, obj))
                if st != "optimal" or oo <= 1e-7:
                    break
                cs = _minimal_core(model, {g for _r, g, _a in vv}, deadline)
                if not cs:
                    break
                blocked |= cs
                # the LP core only adds value beyond the structural layer if
                # it names at least one removable group structural analysis
                # did not already prove; otherwise it is the same deadlock.
                if cs <= covered:
                    continue
                covered |= cs
                members = [model.labels[g] for g in sorted(cs)
                           if g in model.labels]
                # name involved tasks empirically: only tasks whose deletion
                # actually resolves *this* contradiction are listed, capped so
                # a big instance cannot bury the user in task names.
                implicated = _implicated_tasks(model, cs, problem, deadline)
                for tid in implicated:
                    key = ("task", tid)
                    if key in model.labels:
                        members.append(model.labels[key])
                key = tuple(sorted(m.key() for m in members))
                if key in existing_keys:
                    continue
                existing_keys.add(key)
                diag.cores.append(ConflictCore(
                    members=members,
                    explanation=_explain_lp_core(members),
                    source="elastic_lp"))
                n_lp += 1

    diag.cores = _dedupe_cores(diag.cores)
    sources = {c.source for c in diag.cores}
    diag.method = ("mixed" if len(sources) > 1
                   else next(iter(sources), "structural"))

    for c in diag.cores:
        core_keys = c.member_keys()
        for sug in _candidate_suggestions(problem, c):
            sug = _evaluate_suggestion(problem, sug, deadline, core_keys)
            # A "blocked" patch does not even resolve *this* core (e.g.
            # deleting a task that an overlapping core also needs) -- drop it
            # rather than imply it is a fix.
            if sug.status != "blocked":
                c.suggestions.append(sug)
        rank = {"verified": 0, "unverified": 1, "predicted": 1,
                "resolves_core": 1}
        c.suggestions.sort(key=lambda s: (rank.get(s.status, 1), s.action))

    task_ids = sorted({m.ref_id for c in diag.cores for m in c.members
                       if m.kind == "task"})
    n_verified = sum(1 for c in diag.cores for s in c.suggestions
                     if s.status == "verified")
    if diag.cores:
        diag.summary = (
            f"发现 {len(diag.cores)} 组最小冲突（涉及任务 "
            f"{'、'.join(task_ids[:8]) or '—'}"
            f"{'…' if len(task_ids) > 8 else ''}）。"
            f"每一组都已最小化：去掉组内任意一条，该组冲突即解除；"
            f"共生成 {n_verified} 条经实际复测验证可行的放宽建议。")
    else:
        diag.summary = "未检测到明确的冲突集合。"

    diag.warnings = warnings
    diag.elapsed = time.time() - t0
    if persist:
        _persist(problem, diag)
    return diag


def _explain_lp_core(members: List[ConflictMember]) -> str:
    tasks = [m.label for m in members if m.kind == "task"]
    cons = [m.detail for m in members if m.kind == "hard_constraint"]
    res = [m.label for m in members if m.kind == "resource"]
    parts = []
    if tasks:
        parts.append("任务 " + "、".join(tasks))
    if cons:
        parts.append("约束（" + "；".join(cons) + "）")
    if res:
        parts.append("资源 " + "、".join(res))
    if any(m.kind == "horizon" for m in members):
        parts.append("计划周期上限")
    return ("以下条件同时存在时不存在任何可行排程（已验证为最小集合）："
            + "，".join(parts) + "。")


def _persist(problem: models.Problem, diag: Diagnosis) -> None:
    from . import storage
    storage.save_diagnosis(problem.id, diag)
