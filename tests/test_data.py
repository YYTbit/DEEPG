import json

import pytest

from deepg.data import exact_match, extract_answer, load_dataset, normalize_record


def test_exact_match_uses_entire_answer():
    assert exact_match("The Eiffel Tower.", ["Eiffel Tower"])
    assert exact_match("YES", ["yes"])
    assert not exact_match("417", ["17"])
    assert not exact_match("Paris or London", ["Paris"])


def test_native_benchmark_readers():
    hotpot = {"_id": "q1", "question": "Where?", "answer": "Paris", "context": [["Location", ["The city is Paris."]]]}
    item = normalize_record(hotpot, "hotpotqa")
    assert item["passages"][0]["text"] == "The city is Paris."
    assert item["answers"] == ["Paris"]
    musique = {"id": "q2", "question": "Where?", "answer": "Paris", "answer_aliases": ["City of Paris"],
               "paragraphs": [{"title": "Location", "paragraph_text": "The city is Paris."}]}
    assert normalize_record(musique, "musique")["answers"] == ["Paris", "City of Paris"]


def test_feverous_uses_retrieved_evidence():
    row = {"id": 1, "claim": "A claim", "label": "SUPPORTS", "evidence": [{"content": ["Gold_sentence_0"]}]}
    with pytest.raises(ValueError, match="retrieved"):
        normalize_record(row, "feverous")
    item = normalize_record(row, "feverous", evidence={"1": [{"title": "Page", "text": "Retrieved text"}]})
    assert item["passages"][0]["text"] == "Retrieved text"
    assert item["question"] == "A claim"


def test_duplicate_ids_are_rejected(tmp_path):
    item = {"id": "q1", "question": "Where?", "passages": [{"title": "Page", "text": "Text"}],
            "answers": ["Paris"], "dataset": "hotpotqa"}
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(item) + "\n" + json.dumps(item) + "\n")
    with pytest.raises(ValueError, match="unique"):
        load_dataset(path, "hotpotqa")


def test_answer_extraction():
    assert extract_answer("Reasoning\nFinal answer: Paris") == "Paris"
    assert extract_answer("Reasoning\nREFUTES", dataset="feverous") == "REFUTES"
    assert extract_answer("<think>Internal text</think>\nParis") == "Paris"
    assert extract_answer("Full message", rule="whole_message") == "Full message"
