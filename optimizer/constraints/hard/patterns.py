from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Tuple

from ortools.linear_solver import pywraplp


AssignmentKey = Tuple[str, str, str]  # (nurse_id, day_iso, shift_type)


def add_work_pattern_constraints(
    solver: pywraplp.Solver,
    assignment_vars: Dict[AssignmentKey, pywraplp.Variable],
    nurses: List[Dict[str, Any]],
    dates: List[date],
    shift_types: List[str],
    rules: Dict[str, Any],
) -> None:
    """Hard constraints: rest after night, forbidden transitions, max consecutive working days."""

    # Rest after night
    rest_days = int(rules.get("rest_after_night_days", 0))
    night_shift_type = rules.get("night_shift_type")
    if rest_days > 0 and night_shift_type in shift_types:
        for nurse in nurses:
            nurse_id = nurse["id"]
            for idx, day in enumerate(dates):
                next_days = dates[idx + 1 : idx + 1 + rest_days]
                if not next_days:
                    continue
                night_var = assignment_vars[(nurse_id, day.isoformat(), night_shift_type)]
                for next_day in next_days:
                    solver.Add(
                        night_var
                        + sum(
                            assignment_vars[(nurse_id, next_day.isoformat(), shift)]
                            for shift in shift_types
                        )
                        <= 1
                    )

    # Night -> forbidden next-day transitions (e.g., night -> day).
    forbidden_after_night = set(rules.get("forbidden_after_night_shift_types", ["day"]))
    if night_shift_type in shift_types:
        for nurse in nurses:
            nurse_id = nurse["id"]
            for idx, day in enumerate(dates[:-1]):
                night_var = assignment_vars[(nurse_id, day.isoformat(), night_shift_type)]
                next_day = dates[idx + 1].isoformat()
                for shift_type in forbidden_after_night:
                    if shift_type in shift_types:
                        solver.Add(
                            night_var + assignment_vars[(nurse_id, next_day, shift_type)] <= 1
                        )

    # Max consecutive working days.
    max_consecutive_days = rules.get("max_consecutive_days")
    if max_consecutive_days is not None:
        max_consecutive_days = int(max_consecutive_days)
        if 0 <= max_consecutive_days < len(dates):
            window = max_consecutive_days + 1
            for nurse in nurses:
                nurse_id = nurse["id"]
                for start_idx in range(0, len(dates) - window + 1):
                    segment = dates[start_idx : start_idx + window]
                    solver.Add(
                        sum(
                            assignment_vars[(nurse_id, d.isoformat(), st)]
                            for d in segment
                            for st in shift_types
                        )
                        <= max_consecutive_days
                    )
