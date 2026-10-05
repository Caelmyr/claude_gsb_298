"""Seed example problem instances so the UI is populated on first run."""

from __future__ import annotations

from typing import List

from . import models, storage


def _jobshop() -> models.Problem:
    p = models.Problem(
        id="demo_jobshop",
        name="作业车间示例",
        description="3 个作业在 2 台机器上加工，含先后顺序链，目标为最小化完工时间。",
        horizon=40,
        time_unit="小时",
        resources=[
            models.Resource(id="M1", name="机器 A", type="equipment", capacity=1,
                            cost_per_unit=5.0),
            models.Resource(id="M2", name="机器 B", type="equipment", capacity=1,
                            cost_per_unit=8.0),
        ],
        tasks=[
            models.Task(id="J1A", name="作业1 工序A", duration=4,
                        resource_requirements={"M1": 1}),
            models.Task(id="J1B", name="作业1 工序B", duration=3,
                        resource_requirements={"M2": 1}, dependencies=["J1A"]),
            models.Task(id="J2A", name="作业2 工序A", duration=5,
                        resource_requirements={"M2": 1}),
            models.Task(id="J2B", name="作业2 工序B", duration=2,
                        resource_requirements={"M1": 1}, dependencies=["J2A"]),
            models.Task(id="J3A", name="作业3 工序A", duration=3,
                        resource_requirements={"M1": 1}),
            models.Task(id="J3B", name="作业3 工序B", duration=4,
                        resource_requirements={"M2": 1}, dependencies=["J3A"]),
        ],
        objective=models.Objective(type="makespan"),
    )
    return p


def _staffing() -> models.Problem:
    p = models.Problem(
        id="demo_staffing",
        name="班次排班示例",
        description="带技能的人员、硬时间窗口与软截止/偏好惩罚；目标为加权完工时间。",
        horizon=60,
        time_unit="小时",
        resources=[
            models.Resource(id="P1", name="护士甲", type="personnel", capacity=1,
                            skills=["护理"], cost_per_unit=12.0),
            models.Resource(id="P2", name="护士乙", type="personnel", capacity=1,
                            skills=["护理", "转运"], cost_per_unit=10.0),
            models.Resource(id="P3", name="护士丙", type="personnel", capacity=1,
                            skills=["护理"], cost_per_unit=11.0),
            models.Resource(id="R1", name="检查室", type="equipment", capacity=2,
                            cost_per_unit=2.0),
            models.Resource(id="T1", name="日班", type="time", capacity=3,
                            availability=[[0, 60]]),
        ],
        tasks=[
            models.Task(id="A1", name="入院1", duration=6,
                        resource_requirements={"P1": 1, "R1": 1, "T1": 1},
                        release_time=0, due_date=10, weight=2),
            models.Task(id="A2", name="入院2", duration=6,
                        resource_requirements={"P2": 1, "R1": 1, "T1": 1},
                        release_time=0, due_date=12, weight=2),
            models.Task(id="A3", name="入院3", duration=6,
                        resource_requirements={"P3": 1, "R1": 1, "T1": 1},
                        release_time=0, due_date=14, weight=2),
            models.Task(id="B1", name="治疗1", duration=8,
                        resource_requirements={"P1": 1, "T1": 1},
                        dependencies=["A1"], due_date=22, weight=1),
            models.Task(id="B2", name="治疗2", duration=8,
                        resource_requirements={"P2": 1, "T1": 1},
                        dependencies=["A2"], due_date=24, weight=1),
            models.Task(id="B3", name="治疗3", duration=8,
                        resource_requirements={"P3": 1, "T1": 1},
                        dependencies=["A3"], due_date=26, weight=1),
            models.Task(id="C1", name="转运", duration=3,
                        resource_requirements={"P2": 1},
                        dependencies=["B1", "B2"], due_date=30, weight=3),
        ],
        hard_constraints=[
            models.HardConstraint(id="hc1", type="time_window",
                                 params={"task": "C1", "release": 15}),
            models.HardConstraint(id="hc2", type="max_concurrent",
                                 params={"limit": 2}),
        ],
        soft_constraints=[
            models.SoftConstraint(id="sc1", type="due_date",
                                  params={"task": "C1"}, penalty=5.0),
            models.SoftConstraint(id="sc2", type="preferred_window",
                                  params={"task": "B2", "start": 10, "end": 20},
                                  penalty=0.5),
        ],
        objective=models.Objective(type="weighted_completion"),
    )
    return p


def _large() -> models.Problem:
    """A larger instance meant to showcase GA/SA on the big end."""
    resources = [
        models.Resource(id=f"M{i}", name=f"机器 {i}", type="equipment",
                        capacity=1, cost_per_unit=float(i))
        for i in range(1, 5)
    ]
    tasks: List[models.Task] = []
    for j in range(1, 26):
        n_ops = 3
        prev = None
        for k in range(n_ops):
            tid = f"J{j}O{k}"
            deps = [prev] if prev else []
            tasks.append(models.Task(
                id=tid, name=f"作业 {j} 工序 {k}",
                duration=1 + ((j + k) % 6),
                resource_requirements={f"M{1 + (j + k) % 4}": 1},
                dependencies=deps,
                release_time=0,
                due_date=12 + j + k,
                weight=1 + (j % 3),
            ))
            prev = tid
    return models.Problem(
        id="demo_large",
        name="大型作业车间（25 个作业）",
        description="75 道工序、4 台机器；面向元启发式算法。",
        horizon=200,
        time_unit="分钟",
        resources=resources,
        tasks=tasks,
        objective=models.Objective(type="makespan"),
    )


def seed_all(force: bool = False) -> List[str]:
    """Create the example instances if they do not already exist."""
    created: List[str] = []
    for builder in (_jobshop, _staffing, _large):
        problem = builder()
        if not force and storage.load_problem(problem.id) is not None:
            continue
        storage.save_problem(problem)
        created.append(problem.id)
    return created


if __name__ == "__main__":
    print("seeded:", seed_all())
