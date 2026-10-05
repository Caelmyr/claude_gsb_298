"""
JSON file storage for problem instances, solutions, configs and reports.

Layout (one directory per problem instance):

    data/instances/<problem_id>/
        problem.json            # current version (always valid, latest)
        versions/problem.<n>.json   # immutable historical snapshots
        solutions/<solution_id>.json
        configs/<config_id>.json
        reports/<report_id>.json
        sensitivity/<sensitivity_id>.json

Concurrency and durability guarantees:

* *File locking*  -- every mutation takes an exclusive advisory lock
  (``fcntl.flock``) on a per-directory lock file, so concurrent writers
  (e.g. several solver worker processes) serialise cleanly.
* *Atomic write*  -- JSON is serialised to a temp file in the same directory,
  flushed with ``fsync`` and moved into place with ``os.replace``, which is
  atomic on POSIX.  Readers therefore never observe a half-written file.
* *Versioning*   -- updating a problem bumps ``problem.version`` and archives
  the previous ``problem.json`` under ``versions/problem.<n>.json`` before the
  new content is swapped in.  Solutions and reports are immutable once written.

The module is usable both inside the web server and headless from a CLI, so no
global Flask state is involved.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from . import models

# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INSTANCES_ROOT = os.path.join(DATA_ROOT, "instances")

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")


def is_safe_id(identifier: str) -> bool:
    """Reject path-traversal / injection into filesystem paths."""
    return bool(_ID_RE.match(identifier or ""))


def instance_dir(problem_id: str) -> str:
    if not is_safe_id(problem_id):
        raise ValueError(f"invalid problem id: {problem_id!r}")
    return os.path.join(INSTANCES_ROOT, problem_id)


def ensure_dirs() -> None:
    os.makedirs(INSTANCES_ROOT, exist_ok=True)


def _subdirs(problem_id: str) -> List[str]:
    base = instance_dir(problem_id)
    return [
        base,
        os.path.join(base, "versions"),
        os.path.join(base, "solutions"),
        os.path.join(base, "configs"),
        os.path.join(base, "reports"),
        os.path.join(base, "sensitivity"),
    ]


def ensure_instance_dirs(problem_id: str) -> None:
    for d in _subdirs(problem_id):
        os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------------------- #
# File locking
# --------------------------------------------------------------------------- #

try:
    import fcntl  # POSIX only
    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - non-POSIX fallback
    _HAS_FCNTL = False


@contextmanager
def file_lock(lock_path: str, *, exclusive: bool = True) -> Iterator[None]:
    """Advisory file lock.  Falls back to a simple O_EXCL lock-file dance on
    platforms without ``fcntl``.  The lock is released even on exceptions."""
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    if _HAS_FCNTL:
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
    else:
        # Best-effort portable lock: spin on an exclusive lock marker.
        marker = lock_path + ".portable"
        while True:
            try:
                fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                break
            except FileExistsError:
                import time
                time.sleep(0.01)
        try:
            yield
        finally:
            try:
                os.remove(marker)
            except OSError:
                pass


def problem_lock(problem_id: str) -> "contextmanager":
    return file_lock(os.path.join(instance_dir(problem_id), ".lock"))


# --------------------------------------------------------------------------- #
# Atomic JSON write
# --------------------------------------------------------------------------- #

def atomic_write_json(path: str, obj: Any) -> None:
    """Serialize ``obj`` and atomically replace ``path`` with it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Problems
# --------------------------------------------------------------------------- #

def list_problems() -> List[Dict[str, Any]]:
    """List all instances with a light summary (no heavy nested content)."""
    ensure_dirs()
    result = []
    if not os.path.isdir(INSTANCES_ROOT):
        return result
    for name in sorted(os.listdir(INSTANCES_ROOT)):
        pdir = os.path.join(INSTANCES_ROOT, name)
        path = os.path.join(pdir, "problem.json")
        if not os.path.isfile(path):
            continue
        try:
            d = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        result.append({
            "id": d.get("id", name),
            "name": d.get("name", name),
            "description": d.get("description", ""),
            "version": d.get("version", 1),
            "n_tasks": len(d.get("tasks", [])),
            "n_resources": len(d.get("resources", [])),
            "horizon": d.get("horizon", 0),
            "objective": d.get("objective", {}).get("type", "makespan"),
            "updated_at": d.get("updated_at", ""),
            "created_at": d.get("created_at", ""),
        })
    return result


def problem_path(problem_id: str) -> str:
    return os.path.join(instance_dir(problem_id), "problem.json")


def save_problem(problem: models.Problem) -> models.Problem:
    """Persist a problem.  If a previous version exists, archive it first and
    bump the version counter.  Returns the (possibly version-bumped) problem."""
    ensure_dirs()
    ensure_instance_dirs(problem.id)
    with problem_lock(problem.id):
        existing = None
        if os.path.isfile(problem_path(problem.id)):
            try:
                existing = _read_json(problem_path(problem.id))
            except (OSError, json.JSONDecodeError):
                existing = None
        if existing is not None:
            # archive the outgoing version
            prev_version = existing.get("version", 1)
            archive_path = os.path.join(
                instance_dir(problem.id), "versions", f"problem.{prev_version}.json")
            if not os.path.exists(archive_path):
                atomic_write_json(archive_path, existing)
            problem.version = prev_version + 1
        else:
            problem.version = 1
        problem.updated_at = models.now_iso()
        if not problem.created_at:
            problem.created_at = problem.updated_at
        atomic_write_json(problem_path(problem.id), problem.to_dict())
    return problem


def load_problem(problem_id: str, version: Optional[int] = None) -> Optional[models.Problem]:
    if version is None:
        path = problem_path(problem_id)
        if not os.path.isfile(path):
            return None
        return models.Problem.from_dict(_read_json(path))
    vpath = os.path.join(instance_dir(problem_id), "versions", f"problem.{version}.json")
    if not os.path.isfile(vpath):
        return None
    return models.Problem.from_dict(_read_json(vpath))


def list_versions(problem_id: str) -> List[Dict[str, Any]]:
    vdir = os.path.join(instance_dir(problem_id), "versions")
    if not os.path.isdir(vdir):
        return []
    versions = []
    for name in sorted(os.listdir(vdir)):
        m = re.match(r"problem\.(\d+)\.json", name)
        if not m:
            continue
        try:
            d = _read_json(os.path.join(vdir, name))
            versions.append({
                "version": int(m.group(1)),
                "updated_at": d.get("updated_at", ""),
                "name": d.get("name", ""),
                "n_tasks": len(d.get("tasks", [])),
            })
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(versions, key=lambda v: v["version"])


def delete_problem(problem_id: str) -> bool:
    if not is_safe_id(problem_id):
        return False
    pdir = instance_dir(problem_id)
    if not os.path.isdir(pdir):
        return False
    with problem_lock(problem_id):
        import shutil
        shutil.rmtree(pdir)
    return True


# --------------------------------------------------------------------------- #
# Solutions
# --------------------------------------------------------------------------- #

def save_solution(problem_id: str, solution: models.Solution) -> models.Solution:
    ensure_instance_dirs(problem_id)
    with problem_lock(problem_id):
        path = os.path.join(instance_dir(problem_id), "solutions", f"{solution.id}.json")
        atomic_write_json(path, solution.to_dict())
    return solution


def list_solutions(problem_id: str) -> List[Dict[str, Any]]:
    sdir = os.path.join(instance_dir(problem_id), "solutions")
    if not os.path.isdir(sdir):
        return []
    out = []
    for name in sorted(os.listdir(sdir)):
        if not name.endswith(".json"):
            continue
        try:
            d = _read_json(os.path.join(sdir, name))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({
            "id": d.get("id"),
            "solver": d.get("solver"),
            "status": d.get("status"),
            "objective_value": d.get("objective_value"),
            "makespan": d.get("makespan"),
            "solve_time": d.get("solve_time"),
            "created_at": d.get("created_at"),
            "n_assignments": len(d.get("assignments", [])),
        })
    return sorted(out, key=lambda s: s.get("created_at", ""))


def load_solution(problem_id: str, solution_id: str) -> Optional[models.Solution]:
    if not is_safe_id(solution_id):
        return None
    path = os.path.join(instance_dir(problem_id), "solutions", f"{solution_id}.json")
    if not os.path.isfile(path):
        return None
    return models.Solution.from_dict(_read_json(path))


def delete_solution(problem_id: str, solution_id: str) -> bool:
    if not is_safe_id(solution_id):
        return False
    path = os.path.join(instance_dir(problem_id), "solutions", f"{solution_id}.json")
    if not os.path.isfile(path):
        return False
    with problem_lock(problem_id):
        os.remove(path)
    return True


# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #

def save_config(problem_id: str, config: models.SolverConfig) -> models.SolverConfig:
    ensure_instance_dirs(problem_id)
    with problem_lock(problem_id):
        path = os.path.join(instance_dir(problem_id), "configs", f"{config.id}.json")
        atomic_write_json(path, config.to_dict())
    return config


def list_configs(problem_id: str) -> List[Dict[str, Any]]:
    cdir = os.path.join(instance_dir(problem_id), "configs")
    if not os.path.isdir(cdir):
        return []
    out = []
    for name in sorted(os.listdir(cdir)):
        if not name.endswith(".json"):
            continue
        try:
            d = _read_json(os.path.join(cdir, name))
        except (OSError, json.JSONDecodeError):
            continue
        out.append(d)
    return out


# --------------------------------------------------------------------------- #
# Sensitivity & reports
# --------------------------------------------------------------------------- #

def save_sensitivity(problem_id: str, result: models.SensitivityResult) -> models.SensitivityResult:
    ensure_instance_dirs(problem_id)
    with problem_lock(problem_id):
        path = os.path.join(instance_dir(problem_id), "sensitivity", f"{result.id}.json")
        atomic_write_json(path, result.to_dict())
    return result


def list_sensitivity(problem_id: str) -> List[Dict[str, Any]]:
    sdir = os.path.join(instance_dir(problem_id), "sensitivity")
    if not os.path.isdir(sdir):
        return []
    out = []
    for name in sorted(os.listdir(sdir)):
        if not name.endswith(".json"):
            continue
        try:
            out.append(_read_json(os.path.join(sdir, name)))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def save_report(problem_id: str, report: models.Report) -> models.Report:
    ensure_instance_dirs(problem_id)
    with problem_lock(problem_id):
        path = os.path.join(instance_dir(problem_id), "reports", f"{report.id}.json")
        atomic_write_json(path, report.to_dict())
    return report


def list_reports(problem_id: str) -> List[Dict[str, Any]]:
    rdir = os.path.join(instance_dir(problem_id), "reports")
    if not os.path.isdir(rdir):
        return []
    out = []
    for name in sorted(os.listdir(rdir)):
        if not name.endswith(".json"):
            continue
        try:
            out.append(_read_json(os.path.join(rdir, name)))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def load_report(problem_id: str, report_id: str) -> Optional[models.Report]:
    if not is_safe_id(report_id):
        return None
    path = os.path.join(instance_dir(problem_id), "reports", f"{report_id}.json")
    if not os.path.isfile(path):
        return None
    return models.Report.from_dict(_read_json(path))
