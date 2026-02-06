from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimizer import load_policy, load_problem_definition, solve_problem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OR-Tools benchmark solver with a problem definition JSON."
    )
    parser.add_argument(
        "--problem",
        type=Path,
        required=True,
        help="Path to a problem definition JSON file.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("policy.yaml"),
        help="Path to policy.yaml containing solver tuning parameters.",
    )
    args = parser.parse_args()

    problem = load_problem_definition(args.problem)
    policy = load_policy(args.policy)
    solution = solve_problem(problem, policy)
    print(json.dumps(solution, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
