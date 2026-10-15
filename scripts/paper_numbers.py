import argparse

from deepg.cli import write_json
from deepg.verification import paper_numbers


def main():
    parser = argparse.ArgumentParser(description="Analytical reversal, design, team-size and response tables")
    parser.add_argument("--output", default="results/paper_numbers.json")
    args = parser.parse_args()
    write_json(args.output, paper_numbers())


if __name__ == "__main__":
    main()
