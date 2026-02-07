from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass
class PenaltyTerm:
    """
    A soft-constraint penalty term to be aggregated in the objective.
    expr will later be a linear expression / IntVar, etc.
    """
    name: str
    weight: int
    expr: Any


def add_soft_constraints(model: Any, vars_: Any, data: Any, policy: Any) -> List[PenaltyTerm]:
    """
    Register all SOFT constraints and return penalty terms.
    Placeholder for refactor. Returns empty list for now.
    """
    return []
