from __future__ import annotations

from typing import Any, Dict, List, Tuple
from ortools.linear_solver import pywraplp

AssignmentKey = Tuple[str, str, str]  # (nurse_id, day_iso, shift_type)


def build_objective_terms(
    solver: pywraplp.Solver,
    assignment_vars: Dict[AssignmentKey, pywraplp.Variable],
    nurses: List[Dict[str, Any]],
    dates: List,
    shift_types: List[str],
    demands: List[Dict[str, Any]],
    rules: Dict[str, Any],
    weights: Dict[str, float],
    shift_type_meta: Dict[str, Dict[str, Any]],
) -> List[pywraplp.LinearExpr]:
    """
    Build linear penalty terms for nurse shift soft constraints.
    Step10: only day_off_penalty (requested days off).
    """
    objective_terms: List[pywraplp.LinearExpr] = []

    # ---- Soft: requested day off penalty ----
    day_off_penalty = float(weights.get("day_off_penalty", 0.0))
    if day_off_penalty > 0:
        for nurse in nurses:
            nurse_id = nurse["id"]
            requested_days_off = set(nurse.get("requested_days_off", []))
            for day in requested_days_off:
                for shift_type in shift_types:
                    key = (nurse_id, day, shift_type)
                    var = assignment_vars.get(key)
                    if var is not None:
                        objective_terms.append(day_off_penalty * var)

    return objective_terms
