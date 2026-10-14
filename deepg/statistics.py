import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_results(paths):
    rows, seen = [], set()
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") != "evaluation":
                raise ValueError("summaries require per-question evaluation records")
            key = (row["model"], row["dataset"], row["method"], row["seed"], row["id"])
            if key in seen:
                raise ValueError("duplicate evaluation record")
            if not isinstance(row["correct"], bool) or not isinstance(row["tokens"], int) or row["tokens"] < 0:
                raise ValueError("invalid exact-match or token count")
            if row.get("middle") is not None and not 0 <= row["middle"] <= 1:
                raise ValueError("invalid attention share")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError("no evaluation records")
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["method"], row["dataset"])].append(row)
    output = []
    for key, values in sorted(groups.items()):
        by_seed = defaultdict(list)
        for row in values:
            by_seed[row["seed"]].append(row)
        question_sets = [set(row["id"] for row in records) for records in by_seed.values()]
        if any(ids != question_sets[0] for ids in question_sets[1:]):
            raise ValueError("seeds were evaluated on different question sets")
        accuracy, tokens, middle = [], [], []
        for records in by_seed.values():
            accuracy.append(100 * np.mean([r["correct"] for r in records]))
            tokens.append(np.mean([r["tokens"] for r in records]) / 1000)
            attention = [r["middle"] for r in records]
            middle.append(float(np.mean(attention)) if all(a is not None for a in attention) else None)
        result = dict(zip(("model", "method", "dataset"), key))
        result.update(seeds=len(by_seed), questions=len(question_sets[0]))
        for name, data in (("accuracy", accuracy), ("tokens", tokens), ("middle", middle)):
            result[name] = float(np.mean(data)) if all(v is not None for v in data) else None
            result[name + "_sd"] = float(np.std(data, ddof=1)) if len(data) > 1 and all(v is not None for v in data) else None
        output.append(result)
    return output


def paired_bootstrap(rows, first, second, samples=10000, seed=0):
    if samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    grouped = defaultdict(dict)
    signatures = defaultdict(set)
    for row in rows:
        if row["method"] not in (first, second):
            continue
        key = (row["model"], row["dataset"])
        identity = (row["seed"], row["id"])
        if not row.get("comparison_id"):
            raise ValueError("paired comparisons require a checkpoint and dataset comparison signature")
        signatures[key].add(row["comparison_id"])
        grouped[key].setdefault(row["method"], {})[identity] = int(row["correct"])
    rng = np.random.default_rng(seed)
    observed = []
    replicates = np.zeros(samples)
    for key, methods in sorted(grouped.items()):
        if len(signatures[key]) != 1:
            raise ValueError("paired runs use different checkpoints, datasets, or answer rules")
        if set(methods) != {first, second} or set(methods[first]) != set(methods[second]):
            raise ValueError("paired comparisons require identical checkpoints, questions, and seeds")
        seeds = sorted({identity[0] for identity in methods[first]})
        questions = sorted({identity[1] for identity in methods[first]})
        if len(methods[first]) != len(seeds) * len(questions):
            raise ValueError("paired evaluations require every question under every seed")
        differences = np.array([[methods[first][(s, q)] - methods[second][(s, q)]
                                 for q in questions] for s in seeds], dtype=float)
        observed.append(float(differences.mean()))
        for index in range(samples):
            selected_seeds = rng.integers(len(seeds), size=len(seeds))
            selected_questions = rng.integers(len(questions), size=len(questions))
            replicates[index] += differences[np.ix_(selected_seeds, selected_questions)].mean()
    if not observed:
        raise ValueError("no paired evaluation groups")
    replicates *= 100 / len(observed)
    return {"first": first, "second": second, "difference": 100 * float(np.mean(observed)),
            "ci95": np.percentile(replicates, [2.5, 97.5]).tolist(), "samples": samples,
            "groups": len(observed), "bootstrap_seed": seed}


def write_summary(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary
