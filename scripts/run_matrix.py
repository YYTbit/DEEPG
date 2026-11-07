import argparse
import itertools
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from deepg.protocols import METHODS


def main():
    parser = argparse.ArgumentParser(description="Generate or execute the paper evaluation matrix")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["slot_trained", "slot_prompted"])
    parser.add_argument("--baseline-commands", default="configs/baseline_commands.json")
    parser.add_argument("--policy", default="checkpoints")
    parser.add_argument("--backend", choices=["local", "api"], default="local")
    parser.add_argument("--evidence")
    parser.add_argument("--output", default="results/main")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    external = {"AgentPrune", "AgentDropout", "GDesigner", "GPTSwarm"}
    baselines = json.loads(Path(args.baseline_commands).read_text()) if external.intersection(args.methods) else {}
    models, datasets = args.models or list(config["models"]), args.datasets or list(config["datasets"])
    if set(models) - config["models"].keys() or set(datasets) - config["datasets"].keys():
        parser.error("models and datasets must name entries in the configuration")
    revisions = {}
    for model, dataset, method in itertools.product(models, datasets, args.methods):
        command = [sys.executable, "-m", "deepg", "--config", args.config, "evaluate",
                   "--model", model, "--dataset", dataset, "--method", method, "--backend", args.backend,
                   "--output", str(Path(args.output) / model / dataset / (method + ".jsonl"))]
        if args.execute:
            from deepg.loading import resolve_configuration

            if model not in revisions:
                _, revisions[model] = resolve_configuration(config["models"][model])
            if revisions[model] is not None:
                command += ["--revision", revisions[model]]
        if method in ("slot_trained", "without_rotation", "reversed_price"):
            command += ["--policy", args.policy]
        if method in baselines:
            baseline = baselines[method].format(model=model, model_id=config["models"][model])
            command += ["--baseline-command", baseline]
        if dataset == "feverous" and args.evidence:
            command += ["--evidence", args.evidence]
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
