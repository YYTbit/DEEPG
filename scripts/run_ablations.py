import argparse
import itertools
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Generate or execute coarse and fine ablations")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--definitions", default="configs/ablations.yaml")
    parser.add_argument("--group", choices=("coarse", "fine", "controls"), default="coarse")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--policy", default="checkpoints")
    parser.add_argument("--attention-profile")
    parser.add_argument("--evidence")
    parser.add_argument("--output", default="results/ablations")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    definitions = yaml.safe_load(Path(args.definitions).read_text())
    for model, dataset, method in itertools.product(args.models or config["models"],
                                                    args.datasets or config["datasets"], definitions[args.group]):
        command = [sys.executable, "-m", "deepg", "--config", args.config, "evaluate",
                   "--model", model, "--dataset", dataset, "--method", method,
                   "--output", str(Path(args.output) / model / dataset / (method + ".jsonl"))]
        if method in ("slot_trained", "without_rotation", "reversed_price"):
            command += ["--policy", args.policy]
        if method == "calibration" and args.attention_profile:
            command += ["--attention-profile", args.attention_profile]
        if dataset == "feverous" and args.evidence:
            command += ["--evidence", args.evidence]
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
