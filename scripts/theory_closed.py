import argparse

import numpy as np

from deepg.cli import write_json
from deepg.theory import equilibrium_welfare, planner_welfare
from deepg.verification import numerical_planner


def main():
    parser = argparse.ArgumentParser(description="Closed forms and independent planner optimization")
    parser.add_argument("--points", type=int, default=200001)
    parser.add_argument("--restarts", type=int, default=80)
    parser.add_argument("--output", default="results/theory_closed.json")
    args = parser.parse_args()
    if args.points < 2:
        parser.error("--points requires at least two grid points")
    costs = np.linspace(0.000001, 0.35, args.points)
    ratios = np.array([equilibrium_welfare(float(alpha), 2) / planner_welfare(float(alpha), 2) for alpha in costs])
    rows = []
    for alpha in (0.02, 0.05, 0.08, 0.12, 0.25, 0.35):
        numerical = numerical_planner(alpha, 2, restarts=args.restarts)
        rows.append({"alpha": alpha, "equilibrium_welfare": equilibrium_welfare(alpha, 2),
                     "planner_welfare": planner_welfare(alpha, 2), "numerical_planner": numerical,
                     "absolute_deviation": abs(numerical["welfare"] - planner_welfare(alpha, 2))})
    write_json(args.output, {"kind": "numerical_verification", "points": args.points,
                             "cost_range": [float(costs[0]), float(costs[-1])],
                             "strict_loss_on_grid": bool(np.all(ratios < 1)), "rows": rows})


if __name__ == "__main__":
    main()
