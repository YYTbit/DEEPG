import argparse
import json
from pathlib import Path

from deepg.cli import write_json
from deepg.verification import attention_ablation


def main():
    parser = argparse.ArgumentParser(description="Attention-class ablations and recorded training trajectories")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--quadrature", type=int, default=800)
    parser.add_argument("--restarts", type=int, default=80)
    parser.add_argument("--training", nargs="*", default=[])
    parser.add_argument("--output", default="results/ablation_true.json")
    args = parser.parse_args()
    result = attention_ablation(steps=args.steps, quadrature=args.quadrature, restarts=args.restarts)
    result["training"] = []
    for path in args.training:
        record = json.loads(Path(path).read_text())
        if record.get("kind") != "training_run" or not record.get("history"):
            raise ValueError("training trajectories require recorded length-policy updates")
        result["training"].append(record)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
