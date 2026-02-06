# Technology Stack

## Architecture

Lightweight Python package where domain-agnostic dataclasses (`ProblemDefinition`, `Policy`) feed into specialized solver functions (`_solve_knapsack`, `_solve_tsp`, `_solve_facility_location`). A thin CLI mediates IO (JSON + YAML) and prints JSON responses, so the solver layer stays stateless and testable.

## Core Technologies

- **Language**: Python 3.11+ with `from __future__ import annotations` and typing annotations everywhere.
- **Framework**: Google OR-Tools (mixed-integer solver + routing solver modules).
- **Runtime**: Standard CP-SAT / CBC backend provided by OR-Tools; no web runtime.

## Key Libraries

- `ortools.linear_solver.pywraplp` and `pywrapcp` for MIP and routing problems.
- `PyYAML` (optional) for loading policy files, with JSON fallback when unavailable.
- `pytest` for regression tests.
- `csv`, `pathlib`, and `dataclasses` from the standard library for deterministic IO.

## Development Standards

### Type Safety
- Dataclasses support explicit fields, `ProblemDefinition.resolve_path` guards required inputs, and helper functions perform defensive validation (e.g., sense strings, capacity requirement).

### Code Quality
- Local modules avoid global state; solver helpers prefer pure functions that accept dataclass inputs and return serializable dicts.
- Optional dependencies degrade gracefully (PyYAML is required only if YAML syntax is used).

### Testing
- `pytest` suite under `tests/` covers each solver with deterministic mini datasets, ensuring objective values and assignments stay stable.

## Development Environment

### Required Tools
- Python 3.11+
- `pip install ortools pyyaml pytest`

### Common Commands
```bash
# Run CLI against provided samples
python scripts/run_shift_demo.py --problem data/knapsack/problem.json --policy policy.yaml

# Execute full solver test suite
pytest
```

## Key Technical Decisions

- Centralized policy management: YAML updates map into dataclasses via `_update_dataclass`, so solver logic never inspects raw dicts.
- File-driven datasets: relative paths in JSON are resolved against the JSON file's directory, making bundles portable.
- Unit outputs stay JSON-serializable dictionaries, enabling downstream consumers or notebooks to parse results directly.

---
_Document standards and patterns, not every dependency_
