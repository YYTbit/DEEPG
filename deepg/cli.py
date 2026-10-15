import argparse
import copy
import hashlib
import json
import shlex
from pathlib import Path

import numpy as np
import yaml

from .data import file_digest, load_dataset
from .policy import TabularPolicy, train_policy
from .protocols import METHODS, run_external, run_question
from .statistics import load_results, paired_bootstrap, write_summary
from .theory import (AttentionProfile, calibration_solution, calibration_threshold, direct_externalities,
                     equilibrium_lengths, equilibrium_welfare, planner_lengths, planner_welfare,
                     slot_prices, optimal_uniform_price, welfare)


def write_json(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")


def evaluate(args, config):
    from .models import ApiModel, LocalModel
    config = copy.deepcopy(config)
    config["backend"] = args.backend
    if args.method in ("predetermined_matched", "rotation_matched"):
        config["generation"]["communication_tokens"] = config["evaluation"].get("matched_budget_tokens", 6900)
    if args.method in ("planner_cap", "uniform_cap"):
        config["generation"]["communication_tokens"] = config["evaluation"].get("allocation_control_tokens", 3700)
    if args.seed is not None:
        config["generation"]["seeds"] = [args.seed]
    if args.agents is not None:
        config["game"]["agents"] = args.agents
    if args.alpha is not None:
        config["game"]["alpha"] = args.alpha
    if args.rounds is not None:
        config["game"]["rounds"] = args.rounds
    if args.communication_tokens is not None:
        config["generation"]["communication_tokens"] = args.communication_tokens
    if args.calibration_strength is not None:
        config["calibration_strength"] = args.calibration_strength
    path = args.data or config["datasets"][args.dataset]
    items = load_dataset(path, args.dataset, config["evaluation"].get("aliases_2wiki"), args.evidence)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        items = items[:args.limit]
    model_name = args.checkpoint or config["models"][args.model]
    policies = {}
    if args.policy:
        for seed in config["generation"]["seeds"]:
            policy_path = Path(args.policy)
            if policy_path.is_dir():
                schedule = "reversed" if args.method == "reversed_price" else "slot"
                policy_path = policy_path / schedule / str(seed) / "policy.npz"
            policies[seed] = (policy_path, file_digest(policy_path))
    aliases_path = config["evaluation"].get("aliases_2wiki")
    alias_digest = file_digest(aliases_path) if args.dataset == "2wiki" and aliases_path and Path(aliases_path).is_file() else None
    comparison_config = {"model": args.model, "checkpoint": model_name, "dataset": args.dataset,
                         "data_sha256": file_digest(path), "aliases_sha256": alias_digest,
                         "evidence_sha256": file_digest(args.evidence) if args.evidence else None,
                         "tokenizer": args.tokenizer, "sample_limit": args.limit,
                         "answer_rule": config["generation"]["answer_rule"], "backend": args.backend,
                         "game": config["game"],
                         "decoding": {key: config["generation"].get(key) for key in
                                      ("temperature", "top_p", "precision", "enable_thinking", "context_tokens",
                                       "max_message_tokens")}}
    comparison_id = hashlib.sha256(json.dumps(comparison_config, sort_keys=True).encode()).hexdigest()
    run_config = {"game": config["game"], "generation": config["generation"],
                  "model": args.model, "checkpoint": model_name, "method": args.method,
                  "dataset": args.dataset, "data_sha256": file_digest(path), "sample_limit": args.limit,
                  "attention_profile_sha256": file_digest(args.attention_profile) if args.attention_profile else None,
                  "evidence_sha256": file_digest(args.evidence) if args.evidence else None,
                  "calibration_strength": config.get("calibration_strength"),
                  "baseline_command": args.baseline_command, "backend": args.backend,
                  "tokenizer": args.tokenizer, "aliases_sha256": alias_digest,
                  "policy_sha256": {str(seed): digest for seed, (_, digest) in policies.items()},
                  "free_order_delays": config.get("free_order_delays"), "comparison_id": comparison_id}
    run_id = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_suffix(".metadata.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["run_id"] != run_id:
            raise ValueError("output already belongs to a different run configuration")
    elif output.exists() and output.stat().st_size:
        raise ValueError("existing results have no matching run metadata")
    else:
        write_json(metadata_path, {"run_id": run_id, "configuration": run_config})
    completed = set()
    if output.exists() and output.stat().st_size:
        rows = load_results([output])
        if any(r.get("run_id") != run_id for r in rows):
            raise ValueError("result records and metadata do not agree")
        completed = {(row["seed"], row["id"]) for row in rows}
    external = args.method in ("AgentPrune", "AgentDropout", "GDesigner", "GPTSwarm")
    if external and not args.baseline_command:
        raise ValueError("released topology baselines require --baseline-command and a trained upstream checkpoint")
    model = None
    if not external:
        model = (ApiModel(model_name, args.tokenizer, config["generation"]) if args.backend == "api"
                 else LocalModel(model_name, config["generation"], args.device_map, args.attention_profile))
    for seed in config["generation"]["seeds"]:
        policy, policy_digest = None, None
        if args.policy:
            policy_path, policy_digest = policies[seed]
            policy, _ = TabularPolicy.load(policy_path)
        for index, item in enumerate(items):
            if (seed, item["id"]) in completed:
                continue
            if external:
                row = run_external(item, args.method, shlex.split(args.baseline_command), config, seed)
            else:
                row = run_question(item, args.method, model, config, seed, policy)
            row.update(model=args.model, run_id=run_id, policy_sha256=policy_digest, comparison_id=comparison_id)
            with output.open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
            print(f"{args.model} {args.dataset} {args.method} seed={seed} question={index + 1}/{len(items)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Code for Displacement Externalities in Endogenous Position Games")
    parser.add_argument("--config", default="configs/paper.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    theory = commands.add_parser("theory")
    theory.add_argument("--alpha", type=float, default=0.08)
    theory.add_argument("--agents", type=int, default=3)
    theory.add_argument("--rounds", type=int, default=1)
    theory.add_argument("--slope", type=float, default=1.0)
    theory.add_argument("--profile")
    theory.add_argument("--grid", type=int, default=201)
    theory.add_argument("--output", default="results/theory.json")
    train = commands.add_parser("train")
    train.add_argument("--schedule", nargs="+", choices=("unpriced", "uniform", "slot", "reversed"),
                       default=["unpriced", "uniform", "slot"])
    train.add_argument("--seed", type=int)
    train.add_argument("--output", default="checkpoints")
    run = commands.add_parser("evaluate")
    run.add_argument("--model", required=True)
    run.add_argument("--dataset", required=True, choices=("hotpotqa", "musique", "2wiki", "feverous"))
    run.add_argument("--method", required=True, choices=METHODS)
    run.add_argument("--backend", choices=("local", "api"), default="local")
    run.add_argument("--checkpoint")
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
    if args.command == "theory":
        n = args.agents * args.rounds
        alpha = args.alpha
        prices = slot_prices(alpha, n)
        uniform = optimal_uniform_price(alpha, n)
        result = {"kind": "analytical", "alpha": alpha, "slots": n,
                  "equilibrium_lengths": equilibrium_lengths(alpha, n).tolist(),
                  "equilibrium_welfare": equilibrium_welfare(alpha, n),
                  "planner_lengths": planner_lengths(alpha, n).tolist(),
                  "planner_welfare": planner_welfare(alpha, n),
                  "slot_prices": prices.tolist(), "uniform_price": uniform,
                  "uniform_welfare": welfare(equilibrium_lengths(alpha, n, np.full(n, uniform)), alpha),
                  "direct_externalities": direct_externalities(equilibrium_lengths(alpha, n)).tolist(),
                  "calibration": {key: value.tolist() if isinstance(value, np.ndarray) else value
                                  for key, value in calibration_solution(alpha, n, args.slope).items()},
                  "thresholds": {str(k): calibration_threshold(alpha, k) for k in range(1, n) if k > 4 * alpha * alpha}}
        if args.profile:
            data = json.loads(Path(args.profile).read_text())
            solution = AttentionProfile(data["positions"], data["values"]).solve_planner(alpha, n, args.grid)
            result["general_profile"] = {**solution.allocation(), "planner_welfare": float(solution.values[n, 0]),
                                         "grid": args.grid}
        write_json(args.output, result)
    elif args.command == "train":
        config = yaml.safe_load(Path(args.config).read_text())
        seeds = [args.seed] if args.seed is not None else config["training"]["seeds"]
        for schedule in args.schedule:
            for seed in seeds:
                train_policy(config["game"]["alpha"], config["game"]["agents"], config["training"], seed, schedule, args.output)
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
