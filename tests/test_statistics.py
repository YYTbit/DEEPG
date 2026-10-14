import json

import pytest

from deepg.statistics import load_results, paired_bootstrap, summarize


def records():
    rows = []
    for seed in [5, 6, 7, 8, 9]:
        for question in range(10):
            for method in ["first", "second"]:
                rows.append({"kind": "evaluation", "model": "test-model", "dataset": "test-data",
                             "method": method, "seed": seed, "id": str(question),
                             "correct": question < (7 if method == "first" else 5),
                             "tokens": 100, "middle": None, "comparison_id": "same-model-and-dataset"})
    return rows


def test_seed_level_summary_and_missing_attention():
    summary = summarize(records())
    assert summary[0]["accuracy"] == 70
    assert summary[0]["accuracy_sd"] == 0
    assert summary[0]["questions"] == 10
    assert summary[0]["middle"] is None
    assert summary[0]["tokens"] == pytest.approx(0.1)


def test_bootstrap_preserves_pairing():
    result = paired_bootstrap(records(), "first", "second", samples=100, seed=5)
    assert result["difference"] == pytest.approx(20)
    assert result["ci95"][0] <= result["difference"] <= result["ci95"][1]
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap(records()[:-1], "first", "second", samples=10)


def test_summary_rejects_incomplete_seed(tmp_path):
    with pytest.raises(ValueError, match="different question sets"):
        summarize(records()[:-1])
    rows = records()
    path = tmp_path / "result.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows + [rows[0]]))
    with pytest.raises(ValueError, match="duplicate"):
        load_results([path])


def test_incompatible_comparison_signature_is_rejected():
    rows = records()
    rows[-1]["comparison_id"] = "different-checkpoint-or-generation-settings"
    with pytest.raises(ValueError, match="different checkpoints"):
        paired_bootstrap(rows, "first", "second", samples=10)
