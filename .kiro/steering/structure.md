# Project Structure

## Organization Philosophy

Feature-clustered package with a single `optimizer` module owning all solver logic, surrounded by thin adapters (`scripts/` for CLI usage, `tests/` for pytest, `data/` for samples, `policy.yaml` for configuration). Everything flows `problem JSON -> dataclasses -> solver -> JSON result`.

## Directory Patterns

### Solver Core
**Location**: `/my-first-vscode-repo/optimizer/`
**Purpose**: Houses dataclasses (`schemas.py`), IO helpers (`config.py`), and solver orchestration (`shift_model.py`). Each solver keeps private helpers (e.g., `_solve_tsp`) plus parsing utilities such as `_read_knapsack_items`.
**Example**: `solve_problem()` dispatches by `problem.problem_type` and returns a simple dict consumed by CLI/tests.

### Execution Surface
**Location**: `/my-first-vscode-repo/scripts/`
**Purpose**: User-facing command(s) that parse CLI flags, load files via `optimizer.load_*`, and dump JSON. Keep CLI files dependency-light so they can be copied into notebooks or automation scripts.
**Example**: `run_shift_demo.py` sets up `argparse`, resolves paths, and prints `json.dumps(solution, ensure_ascii=False, indent=2)`.

### Configuration + Data Bundles
**Location**: `/my-first-vscode-repo/policy.yaml`, `/my-first-vscode-repo/data/`
**Purpose**: Provide tweakable solver knobs (YAML) and minimal CSV / JSON examples for each problem that stay in sync with README instructions.
**Example**: `policy.yaml` sets CBC for knapsack/facility and PATH_CHEAPEST_ARC for TSP; sample JSON files live alongside matching CSV files.

### Verification
**Location**: `/my-first-vscode-repo/tests/`
**Purpose**: Pytest suite creating temporary CSVs to validate solver behavior without relying on sample data; ensures each solver's objective values stay constant.
**Example**: `tests/test_shift_model.py::test_facility_location_solver` asserts open facility decisions and assignments.

## Naming Conventions

- **Modules / files**: snake_case (e.g., `shift_model.py`, `run_shift_demo.py`).
- **Dataclasses / types**: PascalCase (e.g., `ProblemDefinition`, `TspNode`).
- **Functions**: snake_case; solver helpers prefixed with `_` to signal internal use.

## Import Organization

```python
from optimizer import load_policy, load_problem_definition, solve_problem  # absolute package imports
from .schemas import ProblemDefinition  # local relative imports inside optimizer
```

**Path Injection**:
- Scripts/tests insert the project root into `sys.path` once, enabling absolute imports without packaging steps.

## Code Organization Principles

- No solver touches the filesystem directly; all paths resolve through `ProblemDefinition.resolve_path` so data sources stay declarative.
- Policies are immutable dataclasses, and `_update_dataclass` always copies before applying user overrides to avoid shared-state bugs.
- Tests avoid fixture coupling by writing mini CSVs via `tmp_path`, keeping the solver deterministic and hermetic.

---
_Document patterns, not file trees. New files following patterns shouldn't require updates_
