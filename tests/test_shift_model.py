from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimizer.schemas import Policy, ProblemDefinition
from optimizer.shift_model import solve_problem


def _build_problem(
    base_dir: Path,
    problem_type: str,
    sense: str,
    data_paths: dict,
    parameters: dict,
) -> ProblemDefinition:
    resolved_paths = {key: Path(value) for key, value in data_paths.items()}
    return ProblemDefinition(
        problem_type=problem_type,
        sense=sense,
        base_dir=base_dir,
        data_paths=resolved_paths,
        parameters=parameters,
    )


def test_knapsack_solver(tmp_path: Path) -> None:
    csv_path = tmp_path / "items.csv"
    csv_path.write_text(
        "item_id,weight,value\nitem1,2,6\nitem2,2,5\nitem3,1,3\n", encoding="utf-8"
    )
    problem = _build_problem(
        base_dir=tmp_path,
        problem_type="knapsack",
        sense="max",
        data_paths={"items_csv": csv_path},
        parameters={"capacity": 3},
    )
    solution = solve_problem(problem, Policy())
    assert solution["objective_value"] == pytest.approx(9.0)
    assert set(solution["selected_items"]) == {"item1", "item3"}


def test_tsp_solver(tmp_path: Path) -> None:
    csv_path = tmp_path / "nodes.csv"
    csv_path.write_text(
        "node_id,x,y\nA,0,0\nB,1,0\nC,1,1\nD,0,1\n", encoding="utf-8"
    )
    problem = _build_problem(
        base_dir=tmp_path,
        problem_type="tsp",
        sense="min",
        data_paths={"nodes_csv": csv_path},
        parameters={"metric": "euclidean", "time_limit_seconds": 5},
    )
    solution = solve_problem(problem, Policy())
    assert solution["objective_value"] == pytest.approx(4.0)
    assert solution["route"][0] == "A"
    assert solution["route"][0] == solution["route"][-1]


def test_facility_location_solver(tmp_path: Path) -> None:
    facilities_path = tmp_path / "facilities.csv"
    facilities_path.write_text(
        "facility_id,open_cost\nF1,10\nF2,12\n", encoding="utf-8"
    )
    clients_path = tmp_path / "clients.csv"
    clients_path.write_text("client_id,demand\nC1,1\nC2,1\nC3,1\n", encoding="utf-8")
    assignment_path = tmp_path / "assignment_costs.csv"
    assignment_path.write_text(
        "facility_id,C1,C2,C3\nF1,4,6,8\nF2,5,4,3\n", encoding="utf-8"
    )
    problem = _build_problem(
        base_dir=tmp_path,
        problem_type="facility_location",
        sense="min",
        data_paths={
            "facilities_csv": facilities_path,
            "clients_csv": clients_path,
            "assignment_costs_csv": assignment_path,
        },
        parameters={},
    )
    solution = solve_problem(problem, Policy())
    assert solution["objective_value"] == pytest.approx(24.0)
    assert solution["open_facilities"] == ["F2"]
    assert all(value == "F2" for value in solution["assignments"].values())


def test_nurse_shift_solver_feasible(tmp_path: Path) -> None:
    problem = _build_problem(
        base_dir=tmp_path,
        problem_type="nurse_shift",
        sense="min",
        data_paths={},
        parameters={
            "nurse_shift": {
                "planning_period": {"start_date": "2026-02-01", "days": 2},
                "shift_types": [{"id": "day"}],
                "nurses": [
                    {"id": "n1", "skills": [], "max_shifts": 2, "requested_days_off": []},
                    {"id": "n2", "skills": [], "max_shifts": 2, "requested_days_off": []},
                ],
                "demands": [
                    {"date": "2026-02-01", "shift_type": "day", "required_count": 1},
                    {"date": "2026-02-02", "shift_type": "day", "required_count": 1},
                ],
                "rules": {"rest_after_night_days": 0, "night_shift_type": "night"},
                "weights": {
                    "day_off_penalty": 10.0,
                    "fairness_penalty": 1.0,
                    "skill_priority_penalty": 5.0,
                },
                "time_limit_seconds": 5,
            }
        },
    )
    solution = solve_problem(problem, Policy())
    assert solution["status"] == "success"
    assert len(solution["assignments"]) == 2


def test_nurse_shift_solver_infeasible(tmp_path: Path) -> None:
    problem = _build_problem(
        base_dir=tmp_path,
        problem_type="nurse_shift",
        sense="min",
        data_paths={},
        parameters={
            "nurse_shift": {
                "planning_period": {"start_date": "2026-02-01", "days": 1},
                "shift_types": [{"id": "day"}],
                "nurses": [
                    {"id": "n1", "skills": [], "max_shifts": 1, "requested_days_off": []},
                ],
                "demands": [
                    {"date": "2026-02-01", "shift_type": "day", "required_count": 2},
                ],
                "rules": {"rest_after_night_days": 0, "night_shift_type": "night"},
                "weights": {
                    "day_off_penalty": 10.0,
                    "fairness_penalty": 1.0,
                    "skill_priority_penalty": 5.0,
                },
                "time_limit_seconds": 5,
            }
        },
    )
    solution = solve_problem(problem, Policy())
    assert solution["status"] == "infeasible"
    assert "constraints_summary" in solution
    assert "analysis" in solution["constraints_summary"]
    assert any(
        item["code"] == "daily_capacity_exceeded"
        for item in solution["constraints_summary"]["analysis"]
    )


def test_nurse_shift_duplicate_demand_is_validation_error(tmp_path: Path) -> None:
    problem = _build_problem(
        base_dir=tmp_path,
        problem_type="nurse_shift",
        sense="min",
        data_paths={},
        parameters={
            "nurse_shift": {
                "planning_period": {"start_date": "2026-02-01", "days": 1},
                "shift_types": [{"id": "day"}],
                "nurses": [
                    {"id": "n1", "skills": [], "max_shifts": 1, "requested_days_off": []},
                ],
                "demands": [
                    {"date": "2026-02-01", "shift_type": "day", "required_count": 1},
                    {"date": "2026-02-01", "shift_type": "day", "required_count": 1},
                ],
                "rules": {"rest_after_night_days": 0, "night_shift_type": "night"},
                "weights": {
                    "day_off_penalty": 10.0,
                    "fairness_penalty": 1.0,
                    "skill_priority_penalty": 5.0,
                },
            }
        },
    )
    solution = solve_problem(problem, Policy())
    assert solution["status"] == "error"
    assert any(issue["code"] == "duplicate_demand" for issue in solution["errors"])


def test_nurse_shift_seed_reproducibility(tmp_path: Path) -> None:
    parameters = {
        "nurse_shift": {
            "planning_period": {"start_date": "2026-02-01", "days": 3},
            "shift_types": [{"id": "day"}],
            "nurses": [
                {"id": "n1", "skills": [], "max_shifts": 2, "requested_days_off": []},
                {"id": "n2", "skills": [], "max_shifts": 2, "requested_days_off": []},
                {"id": "n3", "skills": [], "max_shifts": 2, "requested_days_off": []},
            ],
            "demands": [
                {"date": "2026-02-01", "shift_type": "day", "required_count": 1},
                {"date": "2026-02-02", "shift_type": "day", "required_count": 1},
                {"date": "2026-02-03", "shift_type": "day", "required_count": 1},
            ],
            "rules": {"rest_after_night_days": 0, "night_shift_type": "night"},
            "weights": {
                "day_off_penalty": 10.0,
                "fairness_penalty": 1.0,
                "skill_priority_penalty": 5.0,
            },
            "time_limit_seconds": 5,
            "random_seed": 42,
        }
    }
    problem = _build_problem(
        base_dir=tmp_path,
        problem_type="nurse_shift",
        sense="min",
        data_paths={},
        parameters=parameters,
    )
    first = solve_problem(problem, Policy())
    second = solve_problem(problem, Policy())

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert sorted(first["assignments"], key=lambda x: (x["date"], x["nurse_id"])) == sorted(
        second["assignments"], key=lambda x: (x["date"], x["nurse_id"])
    )
    assert first["metadata"]["random_seed"] == 42


def test_nurse_shift_weekly_max_shifts_constraint(tmp_path: Path) -> None:
    demands = []
    for i in range(14):
        day = f"2026-02-{(2 + i):02d}"
        demands.append({"date": day, "shift_type": "day", "required_count": 1})

    problem = _build_problem(
        base_dir=tmp_path,
        problem_type="nurse_shift",
        sense="min",
        data_paths={},
        parameters={
            "nurse_shift": {
                "planning_period": {"start_date": "2026-02-02", "days": 14},
                "shift_types": [{"id": "day"}],
                "nurses": [
                    {"id": "n1", "skills": [], "max_shifts": 4, "requested_days_off": []},
                    {"id": "n2", "skills": [], "max_shifts": 4, "requested_days_off": []},
                ],
                "demands": demands,
                "rules": {"rest_after_night_days": 0, "night_shift_type": "night"},
                "weights": {
                    "day_off_penalty": 10.0,
                    "fairness_penalty": 1.0,
                    "skill_priority_penalty": 5.0,
                },
                "time_limit_seconds": 5,
            }
        },
    )
    solution = solve_problem(problem, Policy())
    assert solution["status"] == "success"
