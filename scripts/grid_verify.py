import argparse

from deepg.cli import write_json
from deepg.verification import grid_report


def main():
    parser = argparse.ArgumentParser(description="Independent length-grid verification")
    parser.add_argument("--costs", nargs="+", type=float, default=[0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35])
    parser.add_argument("--teams", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--quadrature", type=int, default=800)
    parser.add_argument("--output", default="results/grid_verify.json")
    args = parser.parse_args()
    write_json(args.output, grid_report(args.costs, args.teams, args.steps, args.quadrature))


if __name__ == "__main__":
    main()
