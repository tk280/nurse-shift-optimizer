from __future__ import annotations
from typing import Any, Dict
from ortools.linear_solver import pywraplp


def add_demand_constraints(
    solver: pywraplp.Solver,
    assignment_vars: Dict[tuple, pywraplp.Variable],
    nurses: list,
    demands: list,
    rules: Dict[str, Any],
) -> None:
    enforce_exact_demand = bool(rules.get("enforce_exact_demand", True))

    for demand in demands:
        day = demand["date"]
        shift_type = demand["shift_type"]
        required_count = int(demand["required_count"])

        demand_expr = sum(
            assignment_vars[(nurse["id"], day, shift_type)]
            for nurse in nurses
        )

        if enforce_exact_demand:
            solver.Add(demand_expr == required_count)
        else:
            solver.Add(demand_expr >= required_count)

        # required skills (hard)
        if rules.get("enforce_required_skills_hard", True):
            req_skills = set(demand.get("required_skills", []))
            if req_skills:
                for nurse in nurses:
                    if not req_skills.issubset(nurse["skills"]):
                        solver.Add(
                            assignment_vars[(nurse["id"], day, shift_type)] == 0
                        )

        # leadership / experience requirement
        if demand.get("requires_experienced", False):
            solver.Add(
                sum(
                    assignment_vars[(nurse["id"], day, shift_type)]
                    for nurse in nurses
                    if nurse.get("experienced", False)
                )
                >= 1
            )
