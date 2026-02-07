from __future__ import annotations
from .model.variables import build_assignment_vars
import csv
import math
import random
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.linear_solver import pywraplp

from .schemas import (
    ClientRecord,
    FacilityRecord,
    KnapsackItem,
    Policy,
    ProblemDefinition,
    TspNode,
)

from .constraints.hard.all_hard import add_hard_constraints
from .model.base_model import Vars
from .objectives.objective import build_objective_terms

def solve_problem(problem: ProblemDefinition, policy: Policy) -> Dict[str, Any]:
    problem_type = problem.problem_type.lower()
    if problem_type == "knapsack":
        return _solve_knapsack(problem, policy)
    if problem_type == "tsp":
        return _solve_tsp(problem, policy)
    if problem_type == "facility_location":
        return _solve_facility_location(problem, policy)
    if problem_type == "nurse_shift":
        return _solve_nurse_shift(problem, policy)
    raise ValueError(f"Unsupported problem type: {problem.problem_type}")


def _solve_knapsack(problem: ProblemDefinition, policy: Policy) -> Dict[str, Any]:
    items = _read_knapsack_items(problem.resolve_path("items_csv"))
    if "capacity" not in problem.parameters:
        raise ValueError("Knapsack problem requires 'capacity' parameter.")
    capacity = float(problem.parameters["capacity"])

    solver = _create_mip_solver(policy.knapsack.solver)
    variables = {
        item.item_id: solver.IntVar(0, 1, f"x_{item.item_id}") for item in items
    }

    solver.Add(
        sum(item.weight * variables[item.item_id] for item in items) <= capacity
    )

    objective_expr = sum(item.value * variables[item.item_id] for item in items)
    sense = problem.sense.lower()
    if sense == "max":
        solver.Maximize(objective_expr)
    elif sense == "min":
        solver.Minimize(objective_expr)
    else:
        raise ValueError(f"Unsupported sense '{problem.sense}' for knapsack.")

    status = solver.Solve()
    _ensure_optimal(status)

    chosen_items = [
        item.item_id
        for item in items
        if variables[item.item_id].solution_value() > 0.5
    ]
    return {
        "problem_type": "knapsack",
        "objective_value": solver.Objective().Value(),
        "selected_items": chosen_items,
        "capacity_used": sum(
            item.weight for item in items if item.item_id in chosen_items
        ),
    }


def _solve_tsp(problem: ProblemDefinition, policy: Policy) -> Dict[str, Any]:
    nodes = _read_tsp_nodes(problem.resolve_path("nodes_csv"))
    if not nodes:
        raise ValueError("TSP requires at least one node.")

    node_ids = [node.node_id for node in nodes]
    metric = problem.parameters.get("metric", "euclidean").lower()
    dist_matrix = _build_distance_matrix(nodes, metric)

    manager = pywrapcp.RoutingIndexManager(len(nodes), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    params = routing_enums_pb2.FirstSolutionStrategy
    ls_params = routing_enums_pb2.LocalSearchMetaheuristic

    first_strategy = getattr(
        params,
        problem.parameters.get(
            "first_solution_strategy", policy.tsp.first_solution_strategy
        ).upper(),
        params.PATH_CHEAPEST_ARC,
    )
    local_search = getattr(
        ls_params,
        problem.parameters.get(
            "local_search_metaheuristic", policy.tsp.local_search_metaheuristic
        ).upper(),
        ls_params.GUIDED_LOCAL_SEARCH,
    )
    time_limit = int(
        problem.parameters.get("time_limit_seconds", policy.tsp.time_limit_seconds)
    )

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = first_strategy
    search_params.local_search_metaheuristic = local_search
    search_params.time_limit.FromSeconds(time_limit)

    assignment = routing.SolveWithParameters(search_params)
    if not assignment:
        raise RuntimeError("TSP solver failed to find a solution.")

    route, total_distance = _extract_tsp_route(
        routing=routing,
        manager=manager,
        assignment=assignment,
        node_ids=node_ids,
    )
    return {
        "problem_type": "tsp",
        "objective_value": total_distance,
        "route": route,
        "distance_unit": "scaled euclidean",
    }


def _solve_facility_location(
    problem: ProblemDefinition, policy: Policy
) -> Dict[str, Any]:
    facilities = _read_facilities(problem.resolve_path("facilities_csv"))
    clients = _read_clients(problem.resolve_path("clients_csv"))
    assignment_costs = _read_assignment_costs(
        problem.resolve_path("assignment_costs_csv"),
        [client.client_id for client in clients],
    )

    solver = _create_mip_solver(policy.facility_location.solver)

    y_vars = {
        facility.facility_id: solver.IntVar(0, 1, f"y_{facility.facility_id}")
        for facility in facilities
    }
    x_vars = {
        (facility.facility_id, client.client_id): solver.IntVar(
            0, 1, f"x_{facility.facility_id}_{client.client_id}"
        )
        for facility in facilities
        for client in clients
    }

    for client in clients:
        solver.Add(
            sum(
                x_vars[(facility.facility_id, client.client_id)]
                for facility in facilities
            )
            == 1
        )

    for facility in facilities:
        for client in clients:
            solver.Add(
                x_vars[(facility.facility_id, client.client_id)]
                <= y_vars[facility.facility_id]
            )

    objective_terms = []
    objective_terms.extend(
        facility.open_cost * y_vars[facility.facility_id]
        for facility in facilities
    )
    for facility in facilities:
        for client in clients:
            cost = assignment_costs[facility.facility_id][client.client_id]
            objective_terms.append(
                cost * x_vars[(facility.facility_id, client.client_id)]
            )

    objective_expr = sum(objective_terms)
    sense = problem.sense.lower()
    if sense == "min":
        solver.Minimize(objective_expr)
    elif sense == "max":
        solver.Maximize(objective_expr)
    else:
        raise ValueError(f"Unsupported sense '{problem.sense}' for facility location.")

    status = solver.Solve()
    _ensure_optimal(status)

    open_facilities = [
        facility_id for facility_id, var in y_vars.items() if var.solution_value() > 0.5
    ]
    assignments = {
        client.client_id: _assigned_facility(client.client_id, facilities, x_vars)
        for client in clients
    }
    return {
        "problem_type": "facility_location",
        "objective_value": solver.Objective().Value(),
        "open_facilities": open_facilities,
        "assignments": assignments,
    }


def _create_mip_solver(solver_name: str) -> pywraplp.Solver:
    solver = pywraplp.Solver.CreateSolver(solver_name)
    if not solver:
        raise RuntimeError(f"Failed to create solver '{solver_name}'.")
    return solver


def _ensure_optimal(status: int) -> None:
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"Solver failed with status code {status}")


def _read_knapsack_items(path: Path) -> List[KnapsackItem]:
    items: List[KnapsackItem] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            items.append(
                KnapsackItem(
                    item_id=row["item_id"],
                    weight=float(row["weight"]),
                    value=float(row["value"]),
                )
            )
    return items


def _read_tsp_nodes(path: Path) -> List[TspNode]:
    nodes: List[TspNode] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            nodes.append(
                TspNode(
                    node_id=row["node_id"],
                    x=float(row["x"]),
                    y=float(row["y"]),
                )
            )
    return nodes


def _build_distance_matrix(nodes: List[TspNode], metric: str) -> List[List[int]]:
    matrix: List[List[int]] = []
    for origin in nodes:
        row = []
        for target in nodes:
            if metric == "manhattan":
                distance = abs(origin.x - target.x) + abs(origin.y - target.y)
            else:
                distance = math.hypot(origin.x - target.x, origin.y - target.y)
            row.append(int(round(distance * 1000)))
        matrix.append(row)
    return matrix


def _extract_tsp_route(
    routing: pywrapcp.RoutingModel,
    manager: pywrapcp.RoutingIndexManager,
    assignment: pywrapcp.Assignment,
    node_ids: List[str],
) -> Tuple[List[str], float]:
    index = routing.Start(0)
    route: List[str] = []
    total_distance = 0
    while not routing.IsEnd(index):
        route.append(node_ids[manager.IndexToNode(index)])
        next_index = assignment.Value(routing.NextVar(index))
        total_distance += routing.GetArcCostForVehicle(index, next_index, 0)
        index = next_index
    route.append(route[0])
    return route, total_distance / 1000.0


def _read_facilities(path: Path) -> List[FacilityRecord]:
    facilities: List[FacilityRecord] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            facilities.append(
                FacilityRecord(
                    facility_id=row["facility_id"],
                    open_cost=float(row["open_cost"]),
                )
            )
    return facilities


def _read_clients(path: Path) -> List[ClientRecord]:
    clients: List[ClientRecord] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            demand = float(row["demand"]) if row.get("demand") else 1.0
            clients.append(
                ClientRecord(
                    client_id=row["client_id"],
                    demand=demand,
                )
            )
    return clients


def _read_assignment_costs(
    path: Path, client_ids: Iterable[str]
) -> Dict[str, Dict[str, float]]:
    client_ids = list(client_ids)
    costs: Dict[str, Dict[str, float]] = {}
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            facility_id = row["facility_id"]
            costs[facility_id] = {}
            for client_id in client_ids:
                if client_id not in row:
                    raise ValueError(
                        f"Missing assignment cost for client '{client_id}' in facility '{facility_id}'."
                    )
                costs[facility_id][client_id] = float(row[client_id])
    return costs


def _assigned_facility(
    client_id: str,
    facilities: List[FacilityRecord],
    x_vars: Dict[Tuple[str, str], pywraplp.Variable],
) -> str:
    for facility in facilities:
        if x_vars[(facility.facility_id, client_id)].solution_value() > 0.5:
            return facility.facility_id
    raise RuntimeError(f"Client '{client_id}' was not assigned to any facility.")


def _solve_nurse_shift(problem: ProblemDefinition, policy: Policy) -> Dict[str, Any]:
    payload = _extract_nurse_payload(problem)
    input_data, issues = _validate_nurse_payload(payload)
    if issues:
        return {
            "problem_type": "nurse_shift",
            "status": "error",
            "objective_value": None,
            "assignments": [],
            "soft_violations": [],
            "constraints_summary": {},
            "metadata": {},
            "errors": issues,
        }

    solver = _create_mip_solver(policy.nurse_shift.solver)
    time_limit, seed = _apply_nurse_solver_controls(solver, policy, payload)
    start_time = time.time()

    dates = input_data["dates"]
    shift_types = input_data["shift_types"]
    nurses = _order_nurses_by_seed(input_data["nurses"], seed)
    demands = input_data["demands"]
    rules = input_data["rules"]
    weights = input_data["weights"]
    shift_type_meta = input_data["shift_type_meta"]

    assignment_vars = build_assignment_vars(
        solver=solver,
        nurses=nurses,
        dates=dates,
        shift_types=shift_types,
    )

    # Route hard constraints through the refactor entrypoint
    vars_ = Vars(assignment_vars=assignment_vars)

    add_hard_constraints(
        model=solver,
        vars_=vars_,
        data=input_data,
        policy=policy,
    )

    objective_terms = build_objective_terms(
        solver=solver,
        assignment_vars=assignment_vars,
        nurses=nurses,
        dates=dates,
        shift_types=shift_types,
        demands=demands,
        rules=rules,
        weights=weights,
        shift_type_meta=shift_type_meta,
    )

    if objective_terms:
        solver.Minimize(sum(objective_terms))

    status = solver.Solve()
    elapsed_ms = int((time.time() - start_time) * 1000)

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        infeasible_analysis = _analyze_nurse_infeasibility(input_data)
        return {
            "problem_type": "nurse_shift",
            "status": "infeasible",
            "objective_value": None,
            "assignments": [],
            "soft_violations": [],
            "constraints_summary": {
                "reason": "infeasible_constraints",
                "analysis": infeasible_analysis,
            },
            "metadata": _build_nurse_metadata(policy, elapsed_ms, time_limit, seed),
        }

    assignments = []
    for nurse in nurses:
        for day in dates:
            for shift_type in shift_types:
                key = (nurse["id"], day.isoformat(), shift_type)
                if assignment_vars[key].solution_value() > 0.5:
                    assignments.append(
                        {
                            "date": day.isoformat(),
                            "shift_type": shift_type,
                            "nurse_id": nurse["id"],
                        }
                    )

    soft_violations = _collect_soft_violations(assignments, nurses, demands)
    constraints_summary = {
        "total_assignments": len(assignments),
        "total_demand": sum(demand["required_count"] for demand in demands),
        "nurses": len(nurses),
        "shift_types": len(shift_types),
    }
    return {
        "problem_type": "nurse_shift",
        "status": "success",
        "objective_value": solver.Objective().Value() if objective_terms else 0.0,
        "assignments": assignments,
        "soft_violations": soft_violations,
        "constraints_summary": constraints_summary,
        "metadata": _build_nurse_metadata(policy, elapsed_ms, time_limit, seed),
    }


def _extract_nurse_payload(problem: ProblemDefinition) -> Dict[str, Any]:
    payload = problem.parameters.get("nurse_shift")
    if payload is None:
        payload = problem.parameters
    return payload


def _validate_nurse_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    issues: List[Dict[str, str]] = []
    for key in ("planning_period", "nurses", "shift_types", "demands"):
        if key not in payload:
            issues.append(
                {
                    "code": "missing_field",
                    "message": f"Missing required field: {key}",
                    "field": key,
                }
            )
    if issues:
        return {}, issues

    planning_period = payload["planning_period"]
    start_value = planning_period.get("start_date") or planning_period.get("start")
    days_count = planning_period.get("days")
    if not start_value or not days_count:
        issues.append(
            {
                "code": "invalid_period",
                "message": "planning_period requires start_date and days",
                "field": "planning_period",
            }
        )
        return {}, issues

    try:
        start_date = datetime.strptime(start_value, "%Y-%m-%d").date()
    except ValueError:
        issues.append(
            {
                "code": "invalid_date",
                "message": "start_date must be YYYY-MM-DD",
                "field": "planning_period.start_date",
            }
        )
        return {}, issues

    dates = [start_date + timedelta(days=i) for i in range(int(days_count))]
    date_index = {day.isoformat(): idx for idx, day in enumerate(dates)}

    shift_types = [item["id"] for item in payload["shift_types"] if "id" in item]
    shift_type_meta: Dict[str, Dict[str, Any]] = {}
    for item in payload["shift_types"]:
        if "id" not in item:
            continue
        shift_id = item["id"]
        default_hours = 10.0 if shift_id == "night" else 8.0
        shift_type_meta[shift_id] = {
            "duration_hours": float(item.get("duration_hours", default_hours)),
            "is_early": bool(item.get("is_early", shift_id == "day")),
        }
    if not shift_types:
        issues.append(
            {
                "code": "missing_shift_types",
                "message": "shift_types must include at least one shift type id",
                "field": "shift_types",
            }
        )
    seen_shift_types: set[str] = set()
    for shift_type in shift_types:
        if shift_type in seen_shift_types:
            issues.append(
                {
                    "code": "duplicate_shift_type",
                    "message": "shift_types contains duplicate id",
                    "field": "shift_types.id",
                }
            )
        seen_shift_types.add(shift_type)

    nurses = []
    seen_nurse_ids: set[str] = set()
    for nurse in payload["nurses"]:
        nurse_id = nurse.get("id")
        if not nurse_id:
            issues.append(
                {
                    "code": "missing_nurse_id",
                    "message": "nurse id is required",
                    "field": "nurses.id",
                }
            )
            continue
        if nurse_id in seen_nurse_ids:
            issues.append(
                {
                    "code": "duplicate_nurse_id",
                    "message": f"duplicate nurse id: {nurse_id}",
                    "field": "nurses.id",
                }
            )
            continue
        seen_nurse_ids.add(nurse_id)

        requested_days_off = nurse.get("requested_days_off", [])
        requested_days_off_seen: set[str] = set()
        for requested_day in requested_days_off:
            if requested_day in requested_days_off_seen:
                issues.append(
                    {
                        "code": "duplicate_requested_day_off",
                        "message": f"duplicate requested day off: {requested_day}",
                        "field": f"nurses.requested_days_off[{nurse_id}]",
                    }
                )
            requested_days_off_seen.add(requested_day)
        nurses.append(
            {
                "id": nurse_id,
                "skills": set(nurse.get("skills", [])),
                "max_shifts": nurse.get("max_shifts"),
                "requested_days_off": requested_days_off,
                "registered": bool(nurse.get("registered", True)),
                "novice": bool(nurse.get("novice", False)),
                "experienced": bool(nurse.get("experienced", not nurse.get("novice", False))),
                "approved_leave_dates": set(nurse.get("approved_leave_dates", [])),
                "suspended": bool(nurse.get("suspended", False)),
                "suspended_dates": set(nurse.get("suspended_dates", [])),
                "allowed_shift_types": set(nurse.get("allowed_shift_types", []))
                if nurse.get("allowed_shift_types")
                else set(),
                "max_monthly_hours": nurse.get("max_monthly_hours"),
                "max_monthly_night_shifts": nurse.get("max_monthly_night_shifts"),
                "ward": nurse.get("ward"),
                "support_staff": bool(nurse.get("support_staff", False)),
                "external": bool(nurse.get("external", False)),
                "shift_preferences": nurse.get("shift_preferences", {}),
            }
        )

    demands = []
    seen_demands: set[Tuple[str, str]] = set()
    for demand in payload["demands"]:
        day = demand.get("date")
        shift_type = demand.get("shift_type")
        required_count = demand.get("required_count")
        if not day or not shift_type or required_count is None:
            issues.append(
                {
                    "code": "invalid_demand",
                    "message": "demand requires date, shift_type, required_count",
                    "field": "demands",
                }
            )
            continue
        if day not in date_index:
            issues.append(
                {
                    "code": "out_of_range",
                    "message": "demand date outside planning period",
                    "field": "demands.date",
                }
            )
            continue
        if shift_type not in shift_types:
            issues.append(
                {
                    "code": "unknown_shift_type",
                    "message": "demand shift_type is not defined in shift_types",
                    "field": "demands.shift_type",
                }
            )
            continue
        if int(required_count) <= 0:
            issues.append(
                {
                    "code": "invalid_required_count",
                    "message": "required_count must be positive",
                    "field": "demands.required_count",
                }
            )
            continue
        demand_key = (day, shift_type)
        if demand_key in seen_demands:
            issues.append(
                {
                    "code": "duplicate_demand",
                    "message": f"duplicate demand for date={day}, shift_type={shift_type}",
                    "field": "demands",
                }
            )
            continue
        seen_demands.add(demand_key)
        demands.append(
            {
                "date": day,
                "shift_type": shift_type,
                "required_count": int(required_count),
                "required_skills": set(demand.get("required_skills", [])),
                "requires_experienced": bool(demand.get("requires_experienced", False)),
                "ward": demand.get("ward"),
                "is_support": bool(demand.get("is_support", False)),
                "holiday": bool(demand.get("holiday", False)),
            }
        )

    rules = payload.get("rules", {})
    weights = payload.get("weights", {})
    normalized_weights = {
        "day_off_penalty": weights.get("day_off_penalty", 10.0),
        "fairness_penalty": weights.get("fairness_penalty", 1.0),
        "skill_priority_penalty": weights.get("skill_priority_penalty", 5.0),
        "preference_penalty": weights.get("preference_penalty", 2.0),
        "consecutive_night_penalty": weights.get("consecutive_night_penalty", 3.0),
        "night_fairness_penalty": weights.get("night_fairness_penalty", 1.0),
        "weekend_fairness_penalty": weights.get("weekend_fairness_penalty", 1.0),
        "holiday_fairness_penalty": weights.get("holiday_fairness_penalty", 1.0),
        "external_usage_penalty": weights.get("external_usage_penalty", 2.0),
        "novice_with_experienced_penalty": weights.get(
            "novice_with_experienced_penalty", 2.0
        ),
        "abrupt_transition_penalty": weights.get("abrupt_transition_penalty", 1.0),
    }
    normalized_rules = {
        "max_shifts_per_nurse": rules.get("max_shifts_per_nurse"),
        "rest_after_night_days": rules.get("rest_after_night_days", 1),
        "night_shift_type": rules.get("night_shift_type", "night"),
        "enforce_exact_demand": rules.get("enforce_exact_demand", True),
        "enforce_required_skills_hard": rules.get("enforce_required_skills_hard", True),
        "max_consecutive_days": rules.get("max_consecutive_days"),
        "max_monthly_hours": rules.get("max_monthly_hours"),
        "max_monthly_night_shifts": rules.get("max_monthly_night_shifts"),
        "min_rest_days_per_week": rules.get("min_rest_days_per_week", 1),
        "forbidden_after_night_shift_types": rules.get(
            "forbidden_after_night_shift_types", ["day"]
        ),
        "abrupt_transition_pairs": rules.get(
            "abrupt_transition_pairs",
            [["day", "night"], ["night", "day"], ["evening", "day"]],
        ),
    }

    if issues:
        return {}, issues

    return (
        {
            "dates": dates,
            "date_index": date_index,
            "shift_types": shift_types,
            "nurses": nurses,
            "demands": demands,
            "rules": normalized_rules,
            "weights": normalized_weights,
            "shift_type_meta": shift_type_meta,
        },
        [],
    )


def _apply_nurse_solver_controls(
    solver: pywraplp.Solver, policy: Policy, payload: Dict[str, Any]
) -> Tuple[int, int | None]:
    time_limit = int(
        payload.get("time_limit_seconds", policy.nurse_shift.time_limit_seconds)
    )
    solver.SetTimeLimit(time_limit * 1000)
    seed = payload.get("random_seed", policy.nurse_shift.random_seed)
    return time_limit, seed


def _order_nurses_by_seed(
    nurses: List[Dict[str, Any]], seed: int | None
) -> List[Dict[str, Any]]:
    """Build deterministic nurse ordering so same seed leads to same tie-breaking."""
    ordered = list(nurses)
    if seed is None:
        return ordered
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered


def _group_dates_by_week(dates: List[date]) -> Dict[str, List[date]]:
    groups: Dict[str, List[date]] = {}
    for day in dates:
        monday = day - timedelta(days=day.weekday())
        key = monday.isoformat()
        groups.setdefault(key, []).append(day)
    return groups


def _build_nurse_metadata(
    policy: Policy, elapsed_ms: int, time_limit: int, seed: int | None
) -> Dict[str, Any]:
    return {
        "time_limit_seconds": time_limit,
        "random_seed": seed,
        "solver_name": policy.nurse_shift.solver,
        "elapsed_ms": elapsed_ms,
    }


def _collect_soft_violations(
    assignments: List[Dict[str, str]],
    nurses: List[Dict[str, Any]],
    demands: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    nurse_index = {nurse["id"]: nurse for nurse in nurses}
    demand_index = {(demand["date"], demand["shift_type"]): demand for demand in demands}
    violations: List[Dict[str, Any]] = []
    for assignment in assignments:
        nurse = nurse_index[assignment["nurse_id"]]
        day = assignment["date"]
        if day in nurse.get("requested_days_off", []):
            violations.append(
                {
                    "type": "day_off",
                    "nurse_id": assignment["nurse_id"],
                    "date": assignment["date"],
                    "shift_type": assignment["shift_type"],
                }
            )
        demand = demand_index.get((assignment["date"], assignment["shift_type"]))
        if demand and demand["required_skills"]:
            if not demand["required_skills"].issubset(nurse["skills"]):
                violations.append(
                    {
                        "type": "skill_mismatch",
                        "nurse_id": assignment["nurse_id"],
                        "date": assignment["date"],
                        "shift_type": assignment["shift_type"],
                    }
                )
    return violations


def _analyze_nurse_infeasibility(input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return human-readable blockers and concrete tuning suggestions."""
    dates: List[date] = input_data["dates"]
    nurses: List[Dict[str, Any]] = input_data["nurses"]
    demands: List[Dict[str, Any]] = input_data["demands"]
    rules: Dict[str, Any] = input_data["rules"]
    shift_types: List[str] = input_data["shift_types"]
    week_groups = _group_dates_by_week(dates)

    blockers: List[Dict[str, Any]] = []
    days_count = len(dates)
    nurse_count = len(nurses)

    # 1) Daily capacity: one nurse can work at most one shift/day.
    demand_by_day: Dict[str, int] = {}
    for demand in demands:
        demand_by_day[demand["date"]] = demand_by_day.get(demand["date"], 0) + int(
            demand["required_count"]
        )
    for day, required in sorted(demand_by_day.items()):
        if required > nurse_count:
            blockers.append(
                {
                    "code": "daily_capacity_exceeded",
                    "message": f"{day} の必要人数 {required} が看護師数 {nurse_count} を超えています。",
                    "suggestion": "該当日の必要人数を減らすか、看護師数を増やしてください。",
                    "required": required,
                    "capacity": nurse_count,
                }
            )

    # 2) Global capacity from weekly max_shifts constraints.
    default_max_shifts = rules.get("max_shifts_per_nurse")
    total_demand = sum(int(demand["required_count"]) for demand in demands)
    total_capacity = 0
    for nurse in nurses:
        nurse_limit = nurse.get("max_shifts", default_max_shifts)
        if nurse_limit is None:
            nurse_limit = 7
        for week_dates in week_groups.values():
            total_capacity += min(int(nurse_limit), len(week_dates))
    if total_demand > total_capacity:
        blockers.append(
            {
                "code": "global_capacity_exceeded",
                "message": f"総必要件数 {total_demand} が週単位上限込みの総勤務可能件数 {total_capacity} を超えています。",
                "suggestion": "週あたり最大勤務回数を増やすか、必要人数を減らしてください。",
                "required": total_demand,
                "capacity": total_capacity,
            }
        )

    # 3) Skill-based capacity per demand line.
    for demand in demands:
        required_skills = demand.get("required_skills", set())
        if not required_skills:
            continue
        eligible = sum(
            1 for nurse in nurses if set(required_skills).issubset(set(nurse["skills"]))
        )
        required_count = int(demand["required_count"])
        if required_count > eligible:
            blockers.append(
                {
                    "code": "skill_capacity_exceeded",
                    "message": (
                        f"{demand['date']} {demand['shift_type']} は必要人数 {required_count} に対して "
                        f"必要スキル保持者が {eligible} 人しかいません。"
                    ),
                    "suggestion": "必要スキル保持者を増やすか、当該シフトの必要人数/スキル要件を緩和してください。",
                    "required": required_count,
                    "capacity": eligible,
                }
            )

    # 4) Night + rest rule rough feasibility.
    night_shift_type = rules.get("night_shift_type", "night")
    rest_days = int(rules.get("rest_after_night_days", 0))
    if night_shift_type in shift_types and days_count > 0:
        night_demands = [d for d in demands if d["shift_type"] == night_shift_type]
        if night_demands:
            total_night_required = sum(int(d["required_count"]) for d in night_demands)
            max_nights_by_rest = (days_count + rest_days) // (rest_days + 1)
            night_capacity = 0
            for nurse in nurses:
                nurse_limit = nurse.get("max_shifts", default_max_shifts)
                if nurse_limit is None:
                    nurse_limit = days_count
                nurse_limit = min(int(nurse_limit), days_count)

                # If any night demand has skill requirements, nurse must satisfy them.
                eligible_for_any_night = True
                for demand in night_demands:
                    req_skills = demand.get("required_skills", set())
                    if req_skills and not set(req_skills).issubset(set(nurse["skills"])):
                        eligible_for_any_night = False
                        break
                if not eligible_for_any_night:
                    continue
                night_capacity += min(nurse_limit, max_nights_by_rest)

            if total_night_required > night_capacity:
                blockers.append(
                    {
                        "code": "night_rest_capacity_exceeded",
                        "message": (
                            f"夜勤必要件数 {total_night_required} が、夜勤後休息ルール込みの上限 {night_capacity} "
                            f"を超えています。"
                        ),
                        "suggestion": "夜勤人数を減らす、休息日数を緩和する、または夜勤可能な看護師を増やしてください。",
                        "required": total_night_required,
                        "capacity": night_capacity,
                    }
                )

    if not blockers:
        blockers.append(
            {
                "code": "generic_infeasible",
                "message": "主な制約の単純集計では原因を一意に特定できませんでした。",
                "suggestion": "必要人数を段階的に下げるか、最大勤務回数を上げて再実行してください。",
            }
        )
    return blockers
