from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore

from .schemas import (
    FacilityLocationPolicy,
    Policy,
    ProblemDefinition,
    KnapsackPolicy,
    NurseShiftPolicy,
    TspPolicy,
)


def load_problem_definition(path: Path) -> ProblemDefinition:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    data_paths = {
        key: _resolve_path(path.parent, value) for key, value in raw.get("data", {}).items()
    }
    return ProblemDefinition(
        problem_type=raw["problem_type"],
        sense=raw.get("sense", "max"),
        base_dir=path.parent,
        data_paths=data_paths,
        parameters=raw.get("parameters", {}),
    )


def load_policy(path: Path) -> Policy:
    path = Path(path)
    if not path.exists():
        return Policy()
    raw = _read_yaml(path)
    return Policy(
        knapsack=_update_dataclass(KnapsackPolicy(), raw.get("knapsack", {})),
        facility_location=_update_dataclass(
            FacilityLocationPolicy(), raw.get("facility_location", {})
        ),
        tsp=_update_dataclass(TspPolicy(), raw.get("tsp", {})),
        nurse_shift=_update_dataclass(NurseShiftPolicy(), raw.get("nurse_shift", {})),
    )


def _update_dataclass(instance: Any, values: Dict[str, Any]) -> Any:
    if not values:
        return instance
    data = instance.__dict__.copy()
    for key, value in values.items():
        if key in data:
            data[key] = value
    return replace(instance, **data)


def _read_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    if yaml is not None:
        return yaml.safe_load(text) or {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - optional path
        raise RuntimeError(
            "PyYAML is required to parse policy.yaml. Install it via `pip install pyyaml`."
        ) from exc


def _resolve_path(base_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate
