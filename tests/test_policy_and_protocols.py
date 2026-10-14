import numpy as np
import pytest

from deepg.policy import TabularPolicy
from deepg.protocols import AGENT_PROMPT, method_spec, run_question, sample_seed
from deepg.theory import slot_prices


class TestModel:
    __test__ = False

    def __init__(self):
        self.prompts = []

    def encode_chat(self, messages):
        return messages[0]["content"].split()

    def generate(self, messages, max_tokens, seed, strength=0):
        self.prompts.append(messages[0]["content"])
        return {"text": "Final answer: 42" if max_tokens else "", "tokens": min(max_tokens, 3), "finish_reason": "eos"}

    def measure_attention(self, messages, strength=0):
        return None


@pytest.fixture
def config():
    return {"game": {"alpha": 0.08, "agents": 3, "rounds": 2},
            "generation": {"communication_tokens": 120, "max_message_tokens": 120,
                           "equal_cap_tokens": 40, "context_tokens": 4096, "answer_rule": "last_line", "attention": True}}


@pytest.fixture
def item():
    return {"id": "test-question", "question": "What is the answer?", "dataset": "hotpotqa",
            "passages": [{"title": "Evidence", "text": "The answer is 42."}], "answers": ["42"]}


def test_policy_mask_and_checkpoint(tmp_path):
    policy = TabularPolicy(0.08, slot_prices(0.08, 3), 11)
    probabilities = policy.probabilities(0, [0, 4])
    assert probabilities[0, 0] == 1
    assert probabilities[0, 1:].sum() == 0
    assert probabilities[1, 5:].sum() == 0
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    path = tmp_path / "policy.npz"
    policy.save(path, {"seed": 5})
    loaded, metadata = TabularPolicy.load(path)
    np.testing.assert_array_equal(loaded.logits, policy.logits)
    assert metadata == {"seed": 5}


def test_parallel_commitment_and_actual_token_accounting(config, item):
    model = TestModel()
    result = run_question(item, "rotation", model, config, 5)
    first = model.prompts[:3]
    assert all("Round 1" not in prompt for prompt in first)
    assert all("Round 1" in prompt and "Round 2" not in prompt for prompt in model.prompts[3:])
    assert sum(message["cap"] for message in result["messages"][:3]) == 120
    assert all(message["cap"] == 37 for message in result["messages"][3:])
    assert result["tokens"] == 18
    assert result["middle"] is None
    assert result["protocol"]["remaining_tokens"] == 102
    assert [m["capacity_after_commitment"] for m in result["messages"][:3]] == [80, 40, 0]


def test_arrival_order_reads_live_transcript(config, item):
    config["game"]["rounds"] = 1
    model = TestModel()
    result = run_question(item, "free_order", model, config, 5)
    assert [prompt.count("Final answer: 42") for prompt in model.prompts] == [0, 1, 2]
    assert result["tokens"] == 9
    assert result["correct"] is True
    assert result["protocol"]["remaining_tokens"] == 111


def test_trained_caps_are_committed_before_drafting(config, item):
    policy = TabularPolicy(0.08, slot_prices(0.08, 3), 11)
    for slot in range(3):
        for capacity in range(11):
            policy.logits[slot, capacity, capacity // 3] = 1
    model = TestModel()
    result = run_question(item, "slot_trained", model, config, 5, policy)
    first_caps = [message["cap"] for message in result["messages"][:3]]
    assert first_caps == [36, 24, 12]
    assert result["protocol"]["remaining_tokens"] == 120 - result["tokens"]
    assert [m["capacity_after_commitment"] for m in result["messages"][:3]] == [84, 60, 48]


def test_wrong_checkpoint_schedule_is_rejected(config, item):
    policy = TabularPolicy(0.08, np.zeros(3), 11)
    with pytest.raises(ValueError, match="different price schedule"):
        run_question(item, "slot_trained", TestModel(), config, 5, policy)


def test_prompt_and_slot_units():
    assert "{charge}" in AGENT_PROMPT
    assert sample_seed("q1", 5, 0, 0) == sample_seed("q1", 5, 0, 0)
    spec = method_spec("slot_prompted", 0.08, 3)
    assert spec["order"] == "drawn_parallel"
    np.testing.assert_allclose(spec["prices"], slot_prices(0.08, 3))
