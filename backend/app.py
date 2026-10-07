"""Flask application: static frontend + JSON REST API.

The API is a thin, stateless layer over :mod:`storage`, :mod:`models`, the
solvers and the analysis helpers.  All mutation goes through the storage layer
so the file-locking / atomic-write / versioning guarantees hold regardless of
whether a request arrives from the UI or the CLI.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, send_from_directory

from . import models, report, sensitivity, storage, diagnosis
from .solvers import base as solver_base

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "frontend")


def create_app() -> Flask:
    app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="/static")
    storage.ensure_dirs()

    # ------------------------------------------------------------------ #
    # Static pages
    # ------------------------------------------------------------------ #
    @app.route("/")
    def index():
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.route("/<path:name>")
    def pages(name: str):
        # Serve any .html page or asset; fall back to index for SPA-like nav.
        path = os.path.join(FRONTEND_DIR, name)
        if os.path.isfile(path):
            return send_from_directory(FRONTEND_DIR, name)
        if name.endswith(".html"):
            return send_from_directory(FRONTEND_DIR, name.split("/")[-1])
        return jsonify({"error": "not found"}), 404

    # ------------------------------------------------------------------ #
    # Meta
    # ------------------------------------------------------------------ #
    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/api/solvers")
    def solvers():
        return jsonify({
            "solvers": solver_base.available_solvers(),
            "defaults": {n: solver_base.default_params(n)
                         for n in solver_base.available_solvers()},
        })

    # ------------------------------------------------------------------ #
    # Problems
    # ------------------------------------------------------------------ #
    @app.route("/api/problems", methods=["GET"])
    def list_problems():
        return jsonify({"problems": storage.list_problems()})

    @app.route("/api/problems", methods=["POST"])
    def create_problem():
        data = request.get_json(force=True)
        try:
            problem = models.Problem.from_dict(data)
            if not problem.id:
                problem.id = models.new_id("prob")
            errors = models.validate_problem(problem)
            if errors:
                return jsonify({"error": "validation failed", "details": errors}), 400
            storage.save_problem(problem)
            return jsonify(problem.to_dict()), 201
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/problems/<problem_id>", methods=["GET"])
    def get_problem(problem_id: str):
        version = request.args.get("version", type=int)
        problem = storage.load_problem(problem_id, version)
        if problem is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(problem.to_dict())

    @app.route("/api/problems/<problem_id>", methods=["PUT"])
    def update_problem(problem_id: str):
        data = request.get_json(force=True)
        try:
            problem = models.Problem.from_dict(data)
            problem.id = problem_id
            errors = models.validate_problem(problem)
            if errors:
                return jsonify({"error": "validation failed", "details": errors}), 400
            storage.save_problem(problem)
            return jsonify(problem.to_dict())
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/problems/<problem_id>", methods=["DELETE"])
    def delete_problem(problem_id: str):
        if storage.delete_problem(problem_id):
            return jsonify({"ok": True})
        return jsonify({"error": "not found"}), 404

    @app.route("/api/problems/<problem_id>/versions", methods=["GET"])
    def versions(problem_id: str):
        return jsonify({"versions": storage.list_versions(problem_id)})

    @app.route("/api/problems/<problem_id>/versions/<int:version>", methods=["GET"])
    def version(problem_id: str, version: int):
        problem = storage.load_problem(problem_id, version)
        if problem is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(problem.to_dict())

    # ------------------------------------------------------------------ #
    # Solving
    # ------------------------------------------------------------------ #
    @app.route("/api/problems/<problem_id>/solve", methods=["POST"])
    def solve(problem_id: str):
        problem = storage.load_problem(problem_id)
        if problem is None:
            return jsonify({"error": "not found"}), 404
        data = request.get_json(force=True) or {}
        solver_name = data.get("solver", "greedy")
        params = dict(solver_base.default_params(solver_name))
        params.update(data.get("params") or {})
        try:
            solver = solver_base.get_solver(solver_name)
            solution = solver.solve(problem, params)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        storage.save_solution(problem_id, solution)
        # An infeasible verdict is useless without the reason: attach the id
        # of a fresh diagnosis (computed synchronously, within a time budget).
        if solution.status == "infeasible":
            try:
                diag = diagnosis.diagnose(problem, persist=True)
                solution.metrics["diagnosis_id"] = diag.id
                solution.metrics["diagnosis_status"] = diag.status
                if diag.status == "infeasible":
                    solution.message = diag.summary or solution.message
                elif diag.status == "feasible":
                    # The exact feasibility model plus a concrete witness find
                    # a schedule the heuristic solver missed: keep the verdict
                    # but make clear the data is not truly contradictory.
                    solution.metrics["diagnosis_note"] = (
                        "求解器未能构造出完整排程，但可行性诊断确认问题可行"
                        "（已构造出具体排程）；可换用其它求解器重试。")
                else:
                    solution.metrics["diagnosis_note"] = diag.summary
                storage.save_solution(problem_id, solution)
            except Exception as exc:  # diagnosis must never break solving
                solution.metrics["diagnosis_error"] = str(exc)
        return jsonify(solution.to_dict()), 201

    @app.route("/api/problems/<problem_id>/solutions", methods=["GET"])
    def solutions(problem_id: str):
        return jsonify({"solutions": storage.list_solutions(problem_id)})

    @app.route("/api/problems/<problem_id>/solutions/<solution_id>", methods=["GET"])
    def solution(problem_id: str, solution_id: str):
        sol = storage.load_solution(problem_id, solution_id)
        if sol is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(sol.to_dict())

    @app.route("/api/problems/<problem_id>/solutions/<solution_id>", methods=["DELETE"])
    def delete_solution(problem_id: str, solution_id: str):
        if storage.delete_solution(problem_id, solution_id):
            return jsonify({"ok": True})
        return jsonify({"error": "not found"}), 404

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    @app.route("/api/problems/<problem_id>/sensitivity", methods=["POST"])
    def sensitivity_run(problem_id: str):
        problem = storage.load_problem(problem_id)
        if problem is None:
            return jsonify({"error": "not found"}), 404
        data = request.get_json(force=True) or {}
        solver_name = data.get("solver", "greedy")
        spec = data.get("spec", {"kind": "resource_capacity",
                                 "resource": (problem.resources[0].id
                                              if problem.resources else None),
                                 "multipliers": [0.5, 0.75, 1.0, 1.25, 1.5]})
        try:
            result = sensitivity.run_sensitivity(
                problem, solver_name, spec,
                solver_params=data.get("params"),
                persist=bool(data.get("persist", True)))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(result.to_dict()), 201

    @app.route("/api/problems/<problem_id>/sensitivity", methods=["GET"])
    def sensitivity_list(problem_id: str):
        return jsonify({"results": storage.list_sensitivity(problem_id)})

    @app.route("/api/problems/<problem_id>/compare", methods=["POST"])
    def compare(problem_id: str):
        data = request.get_json(force=True) or {}
        ids = data.get("solution_ids", [])
        sols = [storage.load_solution(problem_id, sid) for sid in ids]
        sols = [s for s in sols if s is not None]
        best_obj = min((s.objective_value for s in sols
                        if s.objective_value is not None), default=None)
        rows = []
        for s in sorted(sols, key=lambda s: (s.objective_value is None,
                                             s.objective_value or 0)):
            rows.append({
                **s.to_dict(),
                "delta_vs_best": (round(s.objective_value - best_obj, 4)
                                  if s.objective_value is not None and best_obj is not None
                                  else None),
                "is_best": s.objective_value == best_obj,
            })
        return jsonify({"solutions": rows, "best_objective": best_obj})

    @app.route("/api/problems/<problem_id>/report", methods=["POST"])
    def make_report(problem_id: str):
        problem = storage.load_problem(problem_id)
        if problem is None:
            return jsonify({"error": "not found"}), 404
        data = request.get_json(force=True) or {}
        sols = [storage.load_solution(problem_id, sid)
                for sid in data.get("solution_ids", [])]
        sols = [s for s in sols if s is not None]
        sens = None
        if data.get("sensitivity_id"):
            sens_list = storage.list_sensitivity(problem_id)
            for item in sens_list:
                if item.get("id") == data["sensitivity_id"]:
                    sens = models.SensitivityResult.from_dict(item)
        rep = report.generate_report(problem, sols, sens,
                                     title=data.get("title"),
                                     persist=True)
        return jsonify(rep.to_dict()), 201

    @app.route("/api/problems/<problem_id>/reports", methods=["GET"])
    def reports(problem_id: str):
        return jsonify({"reports": storage.list_reports(problem_id)})

    @app.route("/api/problems/<problem_id>/reports/<report_id>", methods=["GET"])
    def get_report(problem_id: str, report_id: str):
        rep = storage.load_report(problem_id, report_id)
        if rep is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(rep.to_dict())

    # ------------------------------------------------------------------ #
    # Infeasibility diagnosis
    # ------------------------------------------------------------------ #
    @app.route("/api/problems/<problem_id>/diagnose", methods=["POST"])
    def diagnose_run(problem_id: str):
        problem = storage.load_problem(problem_id)
        if problem is None:
            return jsonify({"error": "not found"}), 404
        data = request.get_json(force=True, silent=True) or {}
        try:
            time_budget = float(data.get("time_budget",
                                         diagnosis.DEFAULT_TIME_BUDGET))
            diag = diagnosis.diagnose(problem,
                                      time_budget=min(time_budget, 60.0),
                                      persist=True)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(diag.to_dict()), 201

    @app.route("/api/problems/<problem_id>/diagnoses", methods=["GET"])
    def diagnose_list(problem_id: str):
        return jsonify({"diagnoses": storage.list_diagnoses(problem_id)})

    @app.route("/api/problems/<problem_id>/diagnoses/latest", methods=["GET"])
    def diagnose_latest(problem_id: str):
        include_stale = request.args.get("include_stale") in ("1", "true")
        diag = storage.latest_diagnosis(problem_id,
                                        include_stale=include_stale)
        if diag is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(diag.to_dict())

    @app.route("/api/problems/<problem_id>/diagnoses/<diagnosis_id>",
               methods=["GET"])
    def diagnose_get(problem_id: str, diagnosis_id: str):
        diag = storage.load_diagnosis(problem_id, diagnosis_id)
        if diag is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(diag.to_dict())

    # ------------------------------------------------------------------ #
    # Configs
    # ------------------------------------------------------------------ #
    @app.route("/api/problems/<problem_id>/configs", methods=["POST"])
    def save_config(problem_id: str):
        data = request.get_json(force=True) or {}
        cfg = models.SolverConfig(
            id=data.get("id") or models.new_id("cfg"),
            problem_id=problem_id,
            solver=data.get("solver", "greedy"),
            params=data.get("params", {}),
        )
        storage.save_config(problem_id, cfg)
        return jsonify(cfg.to_dict()), 201

    @app.route("/api/problems/<problem_id>/configs", methods=["GET"])
    def configs(problem_id: str):
        return jsonify({"configs": storage.list_configs(problem_id)})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
