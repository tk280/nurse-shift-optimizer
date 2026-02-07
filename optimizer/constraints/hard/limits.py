from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Tuple

from ortools.linear_solver import pywraplp


AssignmentKey = Tuple[str, str, str]  # (nurse_id, day_iso, shift_type)


def add_limits_constraints(
    solver: pywraplp.Solver,
    assignment_vars: Dict[AssignmentKey, pywraplp.Variable],
    nurses: List[Dict[str, Any]],
    dates: List[date],
    shift_types: List[str],
    shift_type_meta: Dict[str, Dict[str, Any]],
    rules: Dict[str, Any],
) -> None:
    """Hard constraints: monthly hours limit and monthly night-shift limit."""

    night_shift_type = rules.get("night_shift_type")

    for nurse in nurses:
        nurse_id = nurse["id"]

        # Monthly total working hours limit
        max_monthly_hours = nurse.get("max_monthly_hours", rules.get("max_monthly_hours"))
        if max_monthly_hours is not None:
            solver.Add(
                sum(
                    float(shift_type_meta[st]["duration_hours"])
                    * assignment_vars[(nurse_id, d.isoformat(), st)]
                    for d in dates
                    for st in shift_types
                )
                <= float(max_monthly_hours)
            )

        # Monthly night-shift count limit
        max_monthly_night_shifts = nurse.get(
            "max_monthly_night_shifts", rules.get("max_monthly_night_shifts")
        )
        if (
            max_monthly_night_shifts is not None
            and night_shift_type in shift_types
        ):
            solver.Add(
                sum(
                    assignment_vars[(nurse_id, d.isoformat(), night_shift_type)]
                    for d in dates
                )
                <= int(max_monthly_night_shifts)
            )
