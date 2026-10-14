import hashlib
import json
import re
import sqlite3
import string
from pathlib import Path


def normalize_answer(text):
    text = str(text).lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_answer(text, rule="last_line", dataset="hotpotqa"):
    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.S).strip()
    if rule == "whole_message":
        return text
    if rule != "last_line":
        raise ValueError("answer_rule must be last_line or whole_message")
    matches = re.findall(r"(?:final\s+answer|answer|final\s+label|label|verdict)\s*[:=]\s*(.+)", text, re.I)
    answer = matches[-1].strip() if matches else (text.splitlines()[-1].strip() if text else "")
    if dataset == "feverous":
        labels = re.findall(r"\b(SUPPORTS|REFUTES|NOT ENOUGH INFO)\b", answer, re.I)
        if labels:
            return labels[-1].upper()
    return answer.strip("*` ")


def exact_match(prediction, answers):
    return any(normalize_answer(prediction) == normalize_answer(answer) for answer in answers)


def load_records(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pandas as pd
        return json.loads(pd.read_parquet(path).to_json(orient="records"))
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    values = json.loads(path.read_text())
    if not isinstance(values, list):
        raise ValueError("dataset must contain an array or JSONL records")
    return values


def normalize_record(row, dataset, aliases=None, evidence=None):
    if {"id", "question", "passages", "answers", "dataset"} <= set(row):
        if row["dataset"] != dataset or not row["answers"] or not row["passages"]:
            raise ValueError("normalized record and benchmark do not agree")
        return row
    if dataset in ("hotpotqa", "2wiki"):
        context = row["context"]
        if isinstance(context, dict):
            context = zip(context["title"], context["sentences"])
        passages = [{"title": title, "text": " ".join(sentences)} for title, sentences in context]
        answers = [str(row["answer"])]
        if dataset == "2wiki" and aliases:
            entity = row.get("answer_id")
            if entity in aliases:
                answers += [str(a) for a in aliases[entity]]
        identity = str(row.get("_id", row.get("id", "")))
        question = row["question"]
    elif dataset == "musique":
        if not row.get("answerable", True):
            raise ValueError("MuSiQue-Ans requires answerable questions")
        passages = [{"title": p["title"], "text": p["paragraph_text"]} for p in row["paragraphs"]]
        answers = [str(row["answer"])] + list(row.get("answer_aliases", []))
        identity = str(row["id"])
        question = row["question"]
    elif dataset == "feverous":
        identity = str(row["id"])
        question = row["claim"]
        answers = [row["label"]]
        if "passages" in row:
            passages = row["passages"]
        elif evidence is not None:
            if identity not in evidence:
                raise ValueError("retrieved evidence is missing for claim " + identity)
            passages = evidence[identity]
        else:
            raise ValueError("FEVEROUS needs retrieved passages keyed by claim id; gold evidence is excluded from inputs")
    else:
        raise ValueError("unknown benchmark " + dataset)
    if not identity or not passages or any(not isinstance(p.get("text"), str) for p in passages):
        raise ValueError("invalid question or passages")
    return {"id": identity, "question": question, "passages": passages,
            "answers": list(dict.fromkeys(answers)), "dataset": dataset}


def load_dataset(path, dataset, aliases_path=None, evidence_path=None):
    aliases = {}
    if aliases_path and Path(aliases_path).is_file():
        data = json.loads(Path(aliases_path).read_text())
        if isinstance(data, list):
            aliases = {entry["Q_id"]: entry["aliases"] for entry in data}
        else:
            aliases = {key: value.get("aliases", []) if isinstance(value, dict) else value for key, value in data.items()}
    evidence = json.loads(Path(evidence_path).read_text()) if evidence_path else None
    items = [normalize_record(row, dataset, aliases, evidence=evidence) for row in load_records(path)
             if dataset != "feverous" or "label" in row]
    ids = [item["id"] for item in items]
    if len(set(ids)) != len(ids) or not items:
        raise ValueError("dataset ids must be nonempty and unique")
    return items


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_wiki_page(database, title):
    with sqlite3.connect("file:" + str(Path(database).resolve()) + "?mode=ro", uri=True) as connection:
        row = connection.execute("SELECT data FROM wiki WHERE id = ?", (title,)).fetchone()
    if row is None:
        raise KeyError(title)
    return json.loads(row[0])
