from __future__ import annotations

from datetime import date, timedelta
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
    week_groups = _group_dates_by_week(dates)

    for nurse in nurses:
        nurse_id = nurse["id"]

        # Weekly max shifts (Mon-Sun)
        max_shifts = nurse.get("max_shifts", rules.get("max_shifts_per_nurse"))
        if max_shifts is not None:
            for week_dates in week_groups.values():
                solver.Add(
                    sum(
                        assignment_vars[(nurse_id, d.isoformat(), st)]
                        for d in week_dates
                        for st in shift_types
                    )
                    <= int(max_shifts)
                )

        # Weekly rest-day guarantee (only full weeks)
        min_rest_days_per_week = int(rules.get("min_rest_days_per_week", 1))
        for week_dates in week_groups.values():
            if len(week_dates) < 7:
                continue
            solver.Add(
                sum(
                    assignment_vars[(nurse_id, d.isoformat(), st)]
                    for d in week_dates
                    for st in shift_types
                )
                <= max(0, len(week_dates) - min_rest_days_per_week)
            )

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


def _group_dates_by_week(dates: List[date]) -> Dict[str, List[date]]:
    groups: Dict[str, List[date]] = {}
    for day in dates:
        monday = day - timedelta(days=day.weekday())
        key = monday.isoformat()
        groups.setdefault(key, []).append(day)
    return groups
