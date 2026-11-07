import copy
import json
import shlex
from pathlib import Path

from .data import file_digest, load_dataset
from .io import append_record, recover_records, write_json
from .loading import digest
from .models import ApiModel, LocalModel
from .policy import TabularPolicy
from .protocols import run_external, run_question


def evaluate(args, config):
    config = copy.deepcopy(config)
    config["backend"] = args.backend
    if args.method in {"predetermined_matched", "rotation_matched"}:
        config["generation"]["communication_tokens"] = config["evaluation"].get("matched_budget_tokens", 6900)
    if args.method in {"planner_cap", "uniform_cap"}:
        config["generation"]["communication_tokens"] = config["evaluation"].get("allocation_control_tokens", 3700)
    for key in ("agents", "alpha", "rounds"):
        if getattr(args, key) is not None:
            config["game"][key] = getattr(args, key)
    if args.seed is not None:
        config["generation"]["seeds"] = [args.seed]
    if args.communication_tokens is not None:
        config["generation"]["communication_tokens"] = args.communication_tokens
    if args.calibration_strength is not None:
        config["calibration_strength"] = args.calibration_strength
    path = args.data or config["datasets"][args.dataset]
    aliases = config["evaluation"].get("aliases_2wiki")
    items = load_dataset(path, args.dataset, aliases, args.evidence)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        items = items[:args.limit]
    model_name = args.checkpoint or config["models"][args.model]
    seeds = config["generation"]["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("generation seeds must be nonempty and unique")
    policies = {}
    if args.policy:
        for seed in seeds:
            policy_path = Path(args.policy)
            if policy_path.is_dir():
                schedule = "reversed" if args.method == "reversed_price" else "slot"
                policy_path = policy_path / schedule / str(seed) / "policy.npz"
            policies[seed] = (policy_path, file_digest(policy_path))
    alias_digest = file_digest(aliases) if args.dataset == "2wiki" and aliases and Path(aliases).is_file() else None
    comparison = {"model": args.model, "checkpoint": model_name, "dataset": args.dataset,
                  "data_sha256": file_digest(path), "aliases_sha256": alias_digest,
                  "evidence_sha256": file_digest(args.evidence) if args.evidence else None,
                  "tokenizer": args.tokenizer, "sample_limit": args.limit,
                  "answer_rule": config["generation"]["answer_rule"], "backend": args.backend,
                  "game": config["game"], "decoding": {key: config["generation"].get(key) for key in
                      ("temperature", "top_p", "precision", "enable_thinking", "context_tokens", "max_message_tokens")}}
    request = {"comparison": comparison, "generation": config["generation"], "method": args.method,
               "attention_profile_sha256": file_digest(args.attention_profile) if args.attention_profile else None,
               "calibration_strength": config.get("calibration_strength"),
               "baseline_command": args.baseline_command,
               "policy_sha256": {str(seed): checksum for seed, (_, checksum) in policies.items()},
               "revision": args.revision, "free_order_delays": config.get("free_order_delays")}
    request_id = digest(request)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_suffix(".metadata.json")
    metadata = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("request_id") != request_id:
            raise ValueError("output already belongs to a different run configuration")
    elif output.exists() and output.stat().st_size:
        raise ValueError("existing results have no matching run metadata")
    rows = recover_records(output)
    completed = set()
    requested = {(seed, item["id"]) for seed in seeds for item in items}
    for row in rows:
        key = (row["seed"], row["id"])
        if (metadata is None or row.get("run_id") != metadata["run_id"]
                or row.get("comparison_id") != metadata["comparison_id"]
                or row.get("model_identity") != metadata["model_identity"]
                or row.get("baseline_identity") != metadata["baseline_identity"]):
            raise ValueError("result records and metadata do not agree")
        if key in completed or key not in requested:
            raise ValueError("duplicate or unexpected evaluation record")
        completed.add(key)
    if completed == requested:
        return
    external = args.method in {"AgentPrune", "AgentDropout", "GDesigner", "GPTSwarm"}
    if external and not args.baseline_command:
        raise ValueError("released topology baselines require --baseline-command and a trained upstream checkpoint")
    saved_identity = metadata["model_identity"] if metadata else None
    revision = saved_identity["revision"] if saved_identity else args.revision
    model = None
    if not external:
        model = (ApiModel(model_name, args.tokenizer, config["generation"], revision) if args.backend == "api"
                 else LocalModel(model_name, config["generation"], args.device_map, args.attention_profile, revision))
    config["model_revision"] = revision

    def bind_model(identity, baseline_identity=None):
        nonlocal metadata
        if not identity:
            raise ValueError("the inference backend must report its model identity")
        if saved_identity is not None and identity != saved_identity:
            raise ValueError("the model or decoding configuration differs from the saved evaluation")
        comparison_id = digest({**comparison, "model_identity": identity})
        run_id = digest({**request, "model_identity": identity, "baseline_identity": baseline_identity})
        current = {"schema_version": 2, "request_id": request_id, "run_id": run_id,
                   "configuration": request, "model_identity": identity, "comparison_id": comparison_id,
                   "baseline_identity": baseline_identity}
        if metadata is not None and metadata != current:
            raise ValueError("evaluation metadata differs from the resolved model")
        metadata = current
        write_json(metadata_path, metadata)

    if model is not None:
        bind_model(model.identity)
    for seed in seeds:
        policy, policy_digest = None, None
        if args.policy:
            policy_path, policy_digest = policies[seed]
            policy, _ = TabularPolicy.load(policy_path)
        for index, item in enumerate(items):
            if (seed, item["id"]) in completed:
                continue
            if external:
                row = run_external(item, args.method, shlex.split(args.baseline_command), config, seed)
                if not row.get("baseline_identity"):
                    raise ValueError("the upstream bridge must report its topology identity")
                bind_model(row.get("model_identity"), row["baseline_identity"])
                config["model_revision"] = metadata["model_identity"]["revision"]
            else:
                row = run_question(item, args.method, model, config, seed, policy)
            row.update(model=args.model, run_id=metadata["run_id"], policy_sha256=policy_digest,
                       comparison_id=metadata["comparison_id"], model_identity=metadata["model_identity"])
            append_record(output, row)
            print(f"{args.model} {args.dataset} {args.method} seed={seed} question={index + 1}/{len(items)}", flush=True)
