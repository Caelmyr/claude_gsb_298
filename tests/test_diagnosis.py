"""Tests for automatic infeasibility diagnosis (backend.diagnosis).

Run with:  python3 -m pytest tests/ -q   (or `python3 tests/test_diagnosis.py`)

These tests pin down the four properties the feature promises:

* conflicts are *jointly* infeasible and, on the exact layers, minimal
  (deleting any one participant makes the remainder feasible);
* every "sufficient" / combined suggestion is verified by a real re-solve;
* fingerprints go stale exactly when feasibility-relevant data changes;
* diagnosis stays fast and never falsely reports a feasible instance as
  infeasible (the large-instance heuristic guard).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import models, diagnosis  # noqa: E402


def _res(rid="M", cap=1.0):
    return models.Resource(id=rid, type="equipment", capacity=cap)


def _task(tid, dur=6, res=("M",), deps=None):
    return models.Task(id=tid, duration=dur,
                       resource_requirements={r: 1.0 for r in res},
                       dependencies=list(deps or []))


# --------------------------------------------------------------------------- #
def test_temporal_contradiction_is_minimal():
    p = models.Problem(
        id="x", horizon=40, resources=[_res()],
        tasks=[_task("A", 10), _task("B", 10, deps=["A"])],
        hard_constraints=[
            models.HardConstraint(id="w", type="time_window",
                                  params={"task": "B", "deadline": 15})])
    d = diagnosis.diagnose(p, time_budget=10)
    assert not d.feasible
    assert any(c.method == "temporal" and c.minimal for c in d.conflicts)
    cf = d.conflicts[0]
    # removing ANY single participant restores temporal feasibility
    pids = {pt.id for pt in cf.participants}
    for pid in pids:
        assert diagnosis._temporal_feasible(
            diagnosis._clone_without(p, {pid})), f"{pid} removal still infeasible"


# --------------------------------------------------------------------------- #
def test_capacity_overload_lp_conflict():
    p = models.Problem(
        id="x", horizon=20, resources=[_res(cap=1)],
        tasks=[_task("X", 8), _task("Y", 8), _task("Z", 8)])
    d = diagnosis.diagnose(p, time_budget=10)
    assert not d.feasible
    assert any(pt.kind == "resource_capacity"
               for c in d.conflicts for pt in c.participants)
    # raising capacity is a verified sufficient fix
    assert any(s.sufficient and s.kind == "resource_capacity"
               for s in d.suggestions)


# --------------------------------------------------------------------------- #
def test_non_overlap_fixed_pair_iis_includes_all_three():
    p = models.Problem(
        id="x", horizon=20, resources=[_res(cap=2)],
        tasks=[_task("X", 5, res=()), _task("Y", 5, res=())],
        hard_constraints=[
            models.HardConstraint(id="no", type="non_overlap",
                                  params={"tasks": ["X", "Y"]}),
            models.HardConstraint(id="fx", type="fixed_start",
                                  params={"task": "X", "start": 3}),
            models.HardConstraint(id="fy", type="fixed_start",
                                  params={"task": "Y", "start": 4})])
    d = diagnosis.diagnose(p, time_budget=10)
    assert not d.feasible
    ids = {pt.id for c in d.conflicts for pt in c.participants}
    assert {"hc:no", "hc:fx", "hc:fy"} <= ids


# --------------------------------------------------------------------------- #
def test_two_independent_conflicts_plus_combined_fix():
    p = models.Problem(
        id="x", horizon=40,
        resources=[_res("M1"), _res("M2")],
        tasks=[_task("A1", 10, ("M1",)), _task("A2", 10, ("M1",), ["A1"]),
               _task("B1", 6, ("M2",)), _task("B2", 6, ("M2",))],
        hard_constraints=[
            models.HardConstraint(id="w", type="time_window",
                                  params={"task": "A2", "deadline": 15}),
            models.HardConstraint(id="f1", type="fixed_start",
                                  params={"task": "B1", "start": 5}),
            models.HardConstraint(id="f2", type="fixed_start",
                                  params={"task": "B2", "start": 6})])
    d = diagnosis.diagnose(p, time_budget=15)
    assert not d.feasible
    assert len(d.conflicts) == 2
    # no single edit claims to be sufficient
    assert not any(s.sufficient for s in d.suggestions)
    # but a verified combined fix exists and applies one step per conflict
    good = [f for f in d.combined_fixes if f["verified"]]
    assert good
    assert len({st["conflict"] for st in good[0]["steps"]}) == 2


# --------------------------------------------------------------------------- #
def test_feasible_problem_not_accused():
    p = models.Problem(
        id="x", horizon=30, resources=[_res()],
        tasks=[_task("A", 4), _task("B", 4, deps=["A"])])
    d = diagnosis.diagnose(p, time_budget=10)
    assert d.feasible
    assert d.conflicts == []


# --------------------------------------------------------------------------- #
def test_beyond_horizon_suggests_extension():
    p = models.Problem(id="x", horizon=5, resources=[],
                       tasks=[_task("L", 8, res=())])
    d = diagnosis.diagnose(p, time_budget=8)
    assert not d.feasible
    s = next(s for s in d.suggestions if s.action == "extend_horizon")
    assert s.suggested_value >= 8 and s.sufficient


# --------------------------------------------------------------------------- #
def test_fingerprint_staleness():
    import copy
    p = models.Problem(
        id="x", horizon=40, resources=[_res()],
        tasks=[_task("A", 4)], version=3)
    fp = diagnosis.problem_fingerprint(p)
    # soft constraint / objective changes must NOT invalidate a hard diagnosis
    q = copy.deepcopy(p)
    q.soft_constraints.append(
        models.SoftConstraint(id="sc", type="due_date", params={"task": "A"}))
    q.objective = models.Objective(type="tardiness")
    assert diagnosis.problem_fingerprint(q) == fp
    # a resource capacity edit must invalidate it
    r = copy.deepcopy(p)
    r.resources[0].capacity = 2
    assert diagnosis.problem_fingerprint(r) != fp
    # version mismatch alone marks stale
    assert diagnosis.is_stale(fp, 99, p)
    assert not diagnosis.is_stale(fp, 3, p)


# --------------------------------------------------------------------------- #
def test_large_feasible_instance_is_fast_and_not_accused():
    # 75 tasks / horizon 200 -> exceeds the exact-LP budget, heuristic path
    resources = [models.Resource(id=f"M{i}", type="equipment", capacity=1)
                 for i in range(1, 5)]
    tasks = []
    for j in range(1, 26):
        prev = None
        for k in range(3):
            tid = f"J{j}O{k}"
            tasks.append(models.Task(
                id=tid, duration=1 + ((j + k) % 6),
                resource_requirements={f"M{1 + (j + k) % 4}": 1.0},
                dependencies=[prev] if prev else []))
            prev = tid
    p = models.Problem(id="big", horizon=200, resources=resources, tasks=tasks)
    t0 = time.time()
    d = diagnosis.diagnose(p, time_budget=4.0)
    assert time.time() - t0 < 6.0
    assert d.feasible is True
    assert d.unschedulable_tasks == []


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\n{len(fns)} tests passed")
