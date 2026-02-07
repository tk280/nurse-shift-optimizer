from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Tuple

from ortools.linear_solver import pywraplp


AssignmentKey = Tuple[str, str, str]  # (nurse_id, day_iso, shift_type)


def add_availability_constraints(
    solver: pywraplp.Solver,
    assignment_vars: Dict[AssignmentKey, pywraplp.Variable],
    nurses: List[Dict[str, Any]],
    dates: List[date],
    demands: List[Dict[str, Any]],
    shift_types: List[str],
) -> None:
    """
    Hard filter constraints:
    - suspended / suspended_dates -> cannot work any shift that day
    - approved_leave_dates -> cannot work any shift that day
    - allowed_shift_types -> disallow shift types not in allowed set
    - ward mismatch -> disallow
    - support shift requires support_staff -> disallow

    This mirrors the logic previously in shift_model.py (iterates all dates).
    """

    # Index demands by (day_text, shift_type) for quick lookup
    demand_index = {(d["date"], d["shift_type"]): d for d in demands}

    for nurse in nurses:
        nurse_id = nurse["id"]

        suspended_global = bool(nurse.get("suspended", False))
        suspended_dates = nurse.get("suspended_dates", set())
        leave_dates = nurse.get("approved_leave_dates", set())

        allowed = nurse.get("allowed_shift_types", set())
        ward = nurse.get("ward")
        support_staff = bool(nurse.get("support_staff", False))

        for day in dates:
            day_text = day.isoformat()

            is_suspended_day = suspended_global or (day_text in suspended_dates)
            is_leave_day = day_text in leave_dates

            # If suspended/leave: no shifts
            if is_suspended_day or is_leave_day:
                for shift_type in shift_types:
                    solver.Add(assignment_vars[(nurse_id, day_text, shift_type)] == 0)
                continue

            # Otherwise: filter by demand attributes (ward/support) and nurse attributes (allowed)
            for shift_type in shift_types:
                demand_row = demand_index.get((day_text, shift_type))
                if demand_row is None:
                    continue

                if allowed and shift_type not in allowed:
                    solver.Add(assignment_vars[(nurse_id, day_text, shift_type)] == 0)

                if demand_row.get("ward") and ward:
                    if demand_row["ward"] != ward:
                        solver.Add(assignment_vars[(nurse_id, day_text, shift_type)] == 0)

                if demand_row.get("is_support", False) and not support_staff:
                    solver.Add(assignment_vars[(nurse_id, day_text, shift_type)] == 0)
