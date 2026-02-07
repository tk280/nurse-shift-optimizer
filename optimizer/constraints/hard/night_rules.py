from __future__ import annotations

from typing import Any, Dict, List, Tuple
from ortools.linear_solver import pywraplp


AssignmentKey = Tuple[str, str, str]


def add_night_rules_constraints(
    solver: pywraplp.Solver,
    assignment_vars: Dict[AssignmentKey, pywraplp.Variable],
    nurses: List[Dict[str, Any]],
    demands: List[Dict[str, Any]],
    shift_types: List[str],
    rules: Dict[str, Any],
) -> None:
    night_shift_type = rules.get("night_shift_type")
    if night_shift_type not in shift_types:
        return

    for demand in demands:
        if demand["shift_type"] != night_shift_type:
            continue

        day = demand["date"]
        required_count = int(demand["required_count"])

        # At least one registered nurse
        solver.Add(
            sum(
                assignment_vars[(nurse["id"], day, night_shift_type)]
                for nurse in nurses
                if nurse.get("registered", True)
            )
            >= 1
        )

        # Novice cannot be sole member on night shift
        if required_count == 1:
            for nurse in nurses:
                if nurse.get("novice", False):
                    solver.Add(
                        assignment_vars[(nurse["id"], day, night_shift_type)] == 0
                    )
        else:
            for nurse in nurses:
                if not nurse.get("novice", False):
                    continue
                solver.Add(
                    assignment_vars[(nurse["id"], day, night_shift_type)]
                    <= sum(
                        assignment_vars[(peer["id"], day, night_shift_type)]
                        for peer in nurses
                        if peer["id"] != nurse["id"]
                    )
                )
