import argparse
import json
from pathlib import Path

import yaml

from .data import load_dataset
from .evaluation import evaluate
from .io import write_json
from .policy import train_policy
from .protocols import METHODS
from .statistics import load_results, paired_bootstrap, write_summary


def main():
    parser = argparse.ArgumentParser(description="Code for Displacement Externalities in Endogenous Position Games")
    parser.add_argument("--config", default="configs/paper.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--schedule", nargs="+", choices=("unpriced", "uniform", "slot", "reversed"),
                       default=["unpriced", "uniform", "slot"])
    train.add_argument("--seed", type=int)
    train.add_argument("--output", default="checkpoints")
    train.add_argument("--resume", action="store_true")
    run = commands.add_parser("evaluate")
    run.add_argument("--model", required=True)
    run.add_argument("--dataset", required=True, choices=("hotpotqa", "musique", "2wiki", "feverous"))
    run.add_argument("--method", required=True, choices=METHODS)
    run.add_argument("--backend", choices=("local", "api"), default="local")
    run.add_argument("--checkpoint")
    run.add_argument("--revision")
    run.add_argument("--tokenizer")
    run.add_argument("--device-map", default="auto")
    run.add_argument("--data")
    run.add_argument("--evidence")
    run.add_argument("--policy")
    run.add_argument("--attention-profile")
    run.add_argument("--calibration-strength", type=float)
    run.add_argument("--baseline-command")
    run.add_argument("--seed", type=int)
    run.add_argument("--agents", type=int)
    run.add_argument("--alpha", type=float)
    run.add_argument("--rounds", type=int)
    run.add_argument("--communication-tokens", type=int)
    run.add_argument("--limit", type=int)
    run.add_argument("--output", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("inputs", nargs="+")
    summary.add_argument("--output", default="results/summary.csv")
    bootstrap = commands.add_parser("bootstrap")
    bootstrap.add_argument("inputs", nargs="+")
    bootstrap.add_argument("--first", required=True)
    bootstrap.add_argument("--second", required=True)
    bootstrap.add_argument("--samples", type=int, default=10000)
    bootstrap.add_argument("--seed", type=int, default=0)
    bootstrap.add_argument("--output", default="results/bootstrap.json")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--dataset", choices=("hotpotqa", "musique", "2wiki", "feverous"), required=True)
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--aliases")
    prepare.add_argument("--evidence")
    prepare.add_argument("--output", required=True)
    fetch = commands.add_parser("baseline-fetch")
    fetch.add_argument("--manifest", default="configs/baselines.json")
    fetch.add_argument("--output", default="external")
    bridge = commands.add_parser("baseline")
    bridge.add_argument("--method", choices=("AgentPrune", "AgentDropout", "GDesigner", "GPTSwarm"), required=True)
    bridge.add_argument("--source", required=True)
    bridge.add_argument("--settings", required=True)
    bridge.add_argument("--checkpoint", required=True)
    bridge.add_argument("--model", required=True)
    bridge.add_argument("--tokenizer")
    args = parser.parse_args()
    if args.command == "train":
        config = yaml.safe_load(Path(args.config).read_text())
        seeds = [args.seed] if args.seed is not None else config["training"]["seeds"]
        for schedule in args.schedule:
            for seed in seeds:
                train_policy(config["game"]["alpha"], config["game"]["agents"], config["training"], seed, schedule, args.output, args.resume)
                print(f"Saved {schedule} policy for seed {seed}", flush=True)
    elif args.command == "evaluate":
        evaluate(args, yaml.safe_load(Path(args.config).read_text()))
    elif args.command == "summarize":
        write_summary(load_results(args.inputs), args.output)
    elif args.command == "bootstrap":
        write_json(args.output, paired_bootstrap(load_results(args.inputs), args.first, args.second, args.samples, args.seed))
    elif args.command == "prepare":
        records = load_dataset(args.input, args.dataset, args.aliases, args.evidence)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        print(f"Prepared {len(records)} questions")
    elif args.command == "baseline-fetch":
        from .baselines import fetch_baselines
        fetch_baselines(args.manifest, args.output)
    elif args.command == "baseline":
        from .baselines import bridge
        bridge(args)
