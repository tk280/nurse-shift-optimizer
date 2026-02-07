from __future__ import annotations
from typing import Any, Dict, List
from datetime import date
from ortools.linear_solver import pywraplp


def add_hard_constraints(model: Any, vars_: Any, data: Any, policy: Any) -> None:
    add_one_shift_per_nurse_per_day(
        solver=model,
        assignment_vars=vars_.assignment_vars,
        nurses=data["nurses"],
        dates=data["dates"],
        shift_types=data["shift_types"],
)

def add_one_shift_per_nurse_per_day(
    solver: pywraplp.Solver,
    assignment_vars: Dict[tuple, pywraplp.Variable],
    nurses: List[Dict[str, Any]],
    dates: List[date],
    shift_types: List[str],
) -> None:
    """
    Each nurse can work at most one shift per day.
    """
    for nurse in nurses:
        nurse_id = nurse["id"]
        for day in dates:
            day_iso = day.isoformat()
            solver.Add(
                sum(
                    assignment_vars[(nurse_id, day_iso, shift_type)]
                    for shift_type in shift_types
                )
                <= 1
            )