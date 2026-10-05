"""
A pure-Python two-phase simplex method for linear programs.

Problem form::

    minimize    c^T x
    subject to  A_ub x <= b_ub
                A_eq x  = b_eq
                x >= 0

The solver uses a full tableau with Bland's anti-cycling rule and supports
detecting infeasibility (phase 1 objective > 0) and unboundedness.  It returns
the optimal primal solution, the objective value, and the reduced costs of the
slack/surplus columns (which correspond to constraint shadow prices and power
the sensitivity analysis).

The implementation is deliberately dependency-free (pure stdlib floats) so the
whole system runs without numpy/scipy; a ``Fraction``-based exact mode is not
used because performance matters more for the time-indexed scheduling LP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

EPS = 1e-9


@dataclass
class SimplexResult:
    status: str  # 'optimal' | 'infeasible' | 'unbounded' | 'iteration_limit'
    x: List[float] = field(default_factory=list)
    objective: Optional[float] = None
    reduced_costs: List[float] = field(default_factory=list)
    shadow_prices: List[float] = field(default_factory=list)
    iterations: int = 0
    message: str = ""


def _to_float_matrix(rows: List[List[float]]) -> List[List[float]]:
    return [[float(v) for v in row] for row in rows]


def _pivot(T: List[List[float]], basis: List[int], r: int, c: int,
           ncols: int, nrows: int) -> None:
    pivot = T[r][c]
    for j in range(ncols + 1):
        T[r][j] /= pivot
    for i in range(nrows + 1):
        if i != r and abs(T[i][c]) > EPS:
            factor = T[i][c]
            row_i = T[i]
            row_r = T[r]
            for j in range(ncols + 1):
                row_i[j] -= factor * row_r[j]
    basis[r] = c


def _simplex_iterate(T: List[List[float]], basis: List[int], ncols: int,
                     nrows: int, max_iter: int) -> Tuple[str, int]:
    """Run simplex on an already-canonical tableau (objective in row ``nrows``).

    Returns (status, iterations).  The tableau and basis are mutated in place.
    """
    it = 0
    while it < max_iter:
        it += 1
        obj_row = T[nrows]
        # Bland's rule for entering: smallest index with negative reduced cost.
        enter = -1
        for j in range(ncols):
            if obj_row[j] < -EPS:
                enter = j
                break
        if enter == -1:
            return "optimal", it
        # ratio test; Bland for leaving: smallest basis index among ties.
        leave = -1
        best_ratio = float("inf")
        for i in range(nrows):
            if T[i][enter] > EPS:
                ratio = T[i][ncols] / T[i][enter]
                if ratio < best_ratio - EPS or (
                        abs(ratio - best_ratio) <= EPS and
                        (leave == -1 or basis[i] < basis[leave])):
                    best_ratio = ratio
                    leave = i
        if leave == -1:
            return "unbounded", it
        _pivot(T, basis, leave, enter, ncols, nrows)
    return "iteration_limit", it


def linprog(c: List[float],
            A_ub: Optional[List[List[float]]] = None,
            b_ub: Optional[List[float]] = None,
            A_eq: Optional[List[List[float]]] = None,
            b_eq: Optional[List[float]] = None,
            max_iter: int = 200000) -> SimplexResult:
    """Solve the LP.  Returns a :class:`SimplexResult`."""
    c = [float(v) for v in c]
    n = len(c)
    A_ub = _to_float_matrix(A_ub or [])
    b_ub = [float(v) for v in (b_ub or [])]
    A_eq = _to_float_matrix(A_eq or [])
    b_eq = [float(v) for v in (b_eq or [])]

    if len(A_ub) != len(b_ub) or len(A_eq) != len(b_eq):
        raise ValueError("constraint matrix and rhs length mismatch")
    if any(len(row) != n for row in A_ub + A_eq):
        raise ValueError("constraint row width mismatch")

    # Normalise rows: ensure rhs >= 0, record type ('le'|'ge'|'eq').
    rows: List[Tuple[List[float], float, str]] = []
    for row, b in zip(A_ub, b_ub):
        if b < 0:
            rows.append(([-x for x in row], -b, "ge"))
        else:
            rows.append((list(row), b, "le"))
    for row, b in zip(A_eq, b_eq):
        if b < 0:
            rows.append(([-x for x in row], -b, "eq"))
        else:
            rows.append((list(row), b, "eq"))

    m = len(rows)
    n_le = sum(1 for t in rows if t[2] == "le")
    n_ge = sum(1 for t in rows if t[2] == "ge")
    n_eq = sum(1 for t in rows if t[2] == "eq")
    n_art = n_ge + n_eq

    # column layout: [orig x (n)] [slack (n_le)] [surplus (n_ge)] [art (n_art)]
    n_slack = n_le
    n_surp = n_ge
    ncols = n + n_slack + n_surp + n_art
    nrows = m

    # Assign slack/surplus/artificial columns to rows in order.
    col_slack = n
    col_surp = n + n_slack
    col_art = n + n_slack + n_surp

    T = [[0.0] * (ncols + 1) for _ in range(nrows + 1)]
    basis = [-1] * nrows

    slack_i = 0
    surp_i = 0
    art_i = 0
    for i, (row, b, typ) in enumerate(rows):
        for j in range(n):
            T[i][j] = row[j]
        if typ == "le":
            T[i][col_slack + slack_i] = 1.0
            basis[i] = col_slack + slack_i
            slack_i += 1
        elif typ == "ge":
            T[i][col_surp + surp_i] = -1.0
            T[i][col_art + art_i] = 1.0
            basis[i] = col_art + art_i
            surp_i += 1
            art_i += 1
        else:  # eq
            T[i][col_art + art_i] = 1.0
            basis[i] = col_art + art_i
            art_i += 1
        T[i][ncols] = b

    # ---- Phase 1: minimise sum of artificial variables ------------------- #
    # Reduced-cost convention: r_j = c_j - c_B^T (B^-1 A)_j, RHS = c_B^T b.
    if n_art > 0:
        obj = T[nrows]
        for j in range(ncols):
            obj[j] = 0.0
        for k in range(n_art):
            obj[col_art + k] = 1.0          # artificial variables cost 1
        obj[ncols] = 0.0
        for i in range(nrows):
            if basis[i] >= col_art:  # artificial is basic in row i
                row_i = T[i]
                for j in range(ncols):
                    obj[j] -= row_i[j]
                obj[ncols] -= row_i[ncols]   # RHS holds -w

        status, it1 = _simplex_iterate(T, basis, ncols, nrows, max_iter)
        if status in ("unbounded", "iteration_limit"):
            return SimplexResult(status=status, iterations=it1,
                                 message=f"phase 1: {status}")

        # Infeasible if an artificial variable is still positive (w = -obj[ncols]).
        phase1_obj = -obj[ncols]
        if phase1_obj > EPS:
            return SimplexResult(status="infeasible", iterations=it1,
                                 message="no feasible solution (phase 1 > 0)",
                                 objective=phase1_obj)

        # Drive any remaining basic artificials (at 0) out of the basis.
        for i in range(nrows):
            if basis[i] >= col_art:
                assert abs(T[i][ncols]) <= EPS, "basic artificial is nonzero"
                pivoted = False
                for j in range(ncols):
                    if j < col_art and abs(T[i][j]) > EPS:
                        _pivot(T, basis, i, j, ncols, nrows)
                        pivoted = True
                        break
                if not pivoted:
                    # Redundant row: mark for removal by zeroing it out.
                    T[i] = [0.0] * (ncols + 1)
                    basis[i] = -1
    else:
        it1 = 0

    # ---- Phase 2: minimise original objective over non-artificial vars ---- #
    # Drop artificial columns entirely (they are 0 and non-basic) so they can
    # never re-enter the basis.
    ncols = col_art
    T = [row[:col_art] + [row[-1]] for row in T]

    obj = T[nrows]
    # obj row = c_j - c_B^T (B^-1 A)_j, RHS = -z.
    for j in range(ncols + 1):
        obj[j] = 0.0
    for i in range(nrows):
        if basis[i] < 0 or basis[i] >= col_art:
            continue
        bvar = basis[i]
        cb = c[bvar] if bvar < n else 0.0
        row_i = T[i]
        for j in range(ncols + 1):
            obj[j] += cb * row_i[j]
    for j in range(ncols):
        obj[j] = -obj[j]
    for j in range(n):
        obj[j] += c[j]
    obj[ncols] = -obj[ncols]   # RHS holds -z

    status, it2 = _simplex_iterate(T, basis, ncols, nrows, max_iter)
    if status == "iteration_limit":
        return SimplexResult(status=status, iterations=it1 + it2,
                             message="phase 2: iteration limit reached")

    # ---- Extract solution ------------------------------------------------- #
    x = [0.0] * n
    for i in range(nrows):
        if basis[i] < 0:
            continue
        if basis[i] < n:
            x[basis[i]] = T[i][ncols]

    reduced = [T[nrows][j] for j in range(ncols)]

    # shadow prices: duals of the normalised rows (le -> slack col, ge/eq ->
    # artificial/surplus).  For sensitivity we report slack reduced costs.
    shadow: List[float] = []
    slack_idx = 0
    for (row, b, typ) in rows:
        if typ == "le":
            shadow.append(reduced[col_slack + slack_idx])
            slack_idx += 1
        else:
            # surplus variable reduced cost (>=0) carries the dual sign.
            shadow.append(0.0)

    status_out = "optimal" if status == "optimal" else status
    return SimplexResult(
        status=status_out,
        x=x,
        objective=-obj[ncols],
        reduced_costs=reduced,
        shadow_prices=shadow,
        iterations=it1 + it2,
        message=status,
    )
