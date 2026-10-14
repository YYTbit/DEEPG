import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Generate or execute scaling and robustness evaluations")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--definitions", default="configs/ablations.yaml")
    parser.add_argument("--dimension", choices=("agents", "alpha", "rounds", "calibration_strength"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", default="slot_prompted")
    parser.add_argument("--policy-template")
    parser.add_argument("--attention-profile")
    parser.add_argument("--evidence")
    parser.add_argument("--output", default="results/sweeps")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    values = yaml.safe_load(Path(args.definitions).read_text())["sweeps"][args.dimension]
    for value in values:
        command = [sys.executable, "-m", "deepg", "--config", args.config, "evaluate",
                   "--model", args.model, "--dataset", args.dataset, "--method", args.method,
                   "--" + args.dimension.replace("_", "-"), str(value),
                   "--output", str(Path(args.output) / args.dimension / str(value) / (args.method + ".jsonl"))]
        if args.policy_template:
            command += ["--policy", args.policy_template.format(value=value)]
        if args.attention_profile:
            command += ["--attention-profile", args.attention_profile]
        if args.evidence:
            command += ["--evidence", args.evidence]
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
