from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class ProblemDefinition:
    problem_type: str
    sense: str
    base_dir: Path
    data_paths: Dict[str, Path] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def resolve_path(self, key: str) -> Path:
        if key not in self.data_paths:
            raise KeyError(f"Missing path for key '{key}' in problem definition.")
        return self.data_paths[key]


@dataclass
class KnapsackItem:
    item_id: str
    weight: float
    value: float


@dataclass
class TspNode:
    node_id: str
    x: float
    y: float


@dataclass
class FacilityRecord:
    facility_id: str
    open_cost: float


@dataclass
class ClientRecord:
    client_id: str
    demand: float = 1.0


@dataclass
class KnapsackPolicy:
    solver: str = "CBC_MIXED_INTEGER_PROGRAMMING"


@dataclass
class FacilityLocationPolicy:
    solver: str = "CBC_MIXED_INTEGER_PROGRAMMING"


@dataclass
class TspPolicy:
    time_limit_seconds: int = 10
    first_solution_strategy: str = "PATH_CHEAPEST_ARC"
    local_search_metaheuristic: str = "GUIDED_LOCAL_SEARCH"


@dataclass
class NurseShiftPolicy:
    solver: str = "CBC_MIXED_INTEGER_PROGRAMMING"
    time_limit_seconds: int = 30
    random_seed: int | None = None
    day_off_penalty: float = 10.0
    fairness_penalty: float = 1.0
    skill_priority_penalty: float = 5.0
    max_shifts_per_nurse: int | None = None
    rest_after_night_days: int = 1
    night_shift_type: str = "night"


@dataclass
class Policy:
    knapsack: KnapsackPolicy = field(default_factory=KnapsackPolicy)
    facility_location: FacilityLocationPolicy = field(
        default_factory=FacilityLocationPolicy
    )
    tsp: TspPolicy = field(default_factory=TspPolicy)
    nurse_shift: NurseShiftPolicy = field(default_factory=NurseShiftPolicy)
