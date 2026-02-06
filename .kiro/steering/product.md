# Product Overview

The repository packages a reusable OR-Tools benchmark harness that loads canonical combinatorial optimization problems (0-1 knapsack, Euclidean TSP, uncapacitated facility location), parses problem data from JSON/CSV, and emits solver summaries for inspection or regression testing. It doubles as a reference implementation for codifying data-driven optimization workflows.

## Core Capabilities

- Unified CLI (`scripts/run_shift_demo.py`) that hydrates `ProblemDefinition` objects from JSON and applies solver policies loaded from YAML.
- Modular `optimizer` package that wraps OR-Tools models for each supported problem and standardizes dataclass-based inputs/outputs.
- Policy-driven tuning (solver selection, search heuristics, time limits) that can be tweaked per problem family without touching solver code.
- Sample datasets plus pytest coverage illustrating minimum reproducible cases for every problem type.

## Target Use Cases

- Running side-by-side benchmarks of solver heuristics or parameters across classic combinatorial problems.
- Educating or onboarding engineers to OR-Tools modeling patterns via concise, data-driven examples.
- Serving as a starting point for production experiments where inputs arrive as CSV/JSON bundles and solver knobs must remain configurable.

## Value Proposition

- Establishes a single entry point for heterogeneous optimization problems, ensuring consistent IO contracts and reporting.
- Encourages configuration-over-code: business stakeholders can iterate on YAML policies or JSON data without redeploying.
- Provides fast-running pytest fixtures so algorithmic regressions surface immediately.

---
_Focus on patterns and purpose, not exhaustive feature lists_
