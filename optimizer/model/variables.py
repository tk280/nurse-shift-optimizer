from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Tuple

from ortools.linear_solver import pywraplp


AssignmentKey = Tuple[str, str, str]  # (nurse_id, day_iso, shift_type)


def build_assignment_vars(
    solver: pywraplp.Solver,
    nurses: List[Dict[str, Any]],
    dates: List[date],
    shift_types: List[str],
) -> Dict[AssignmentKey, pywraplp.Variable]:
    """
    Create assignment decision variables:
      x[nurse_id, day_iso, shift_type] ∈ {0,1}

    This function only defines variables. No constraints here.
    """
    assignment_vars: Dict[AssignmentKey, pywraplp.Variable] = {}

    for nurse in nurses:
        nurse_id = nurse["id"]
        for day in dates:
            day_iso = day.isoformat()
            for shift_type in shift_types:
                key: AssignmentKey = (nurse_id, day_iso, shift_type)
                # keep the original naming format as-is to preserve behavior/debuggability
                assignment_vars[key] = solver.IntVar(0, 1, f"x_{nurse_id}_{day}_{shift_type}")

    return assignment_vars
