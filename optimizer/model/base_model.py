from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
from ortools.linear_solver import pywraplp


@dataclass
class Vars:
    assignment_vars: Dict[Tuple[str, str, str], pywraplp.Variable]


def build_model(data: Any, policy: Any) -> Tuple[Any, Vars]:
    """
    Create model and decision variables.
    NOTE: Do NOT implement constraints here.
    This is a placeholder API for the refactor.
    """
    model = None  # later: cp_model.CpModel() or pywraplp.Solver(), etc.
    vars_ = Vars()
    return model, vars_
