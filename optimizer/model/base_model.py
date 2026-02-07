from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass
class Vars:
    """
    Decision variables container.
    For now it's empty; we'll add x[n,d,s] etc. later.
    """
    pass


def build_model(data: Any, policy: Any) -> Tuple[Any, Vars]:
    """
    Create model and decision variables.
    NOTE: Do NOT implement constraints here.
    This is a placeholder API for the refactor.
    """
    model = None  # later: cp_model.CpModel() or pywraplp.Solver(), etc.
    vars_ = Vars()
    return model, vars_
