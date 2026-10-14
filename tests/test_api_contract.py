import pytest

from deepg.models import ApiModel


class Response:
    status_code = 200

    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    def json(self):
        return {"choices": [{"message": {"content": "Answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}}


def model():
    engine = object.__new__(ApiModel)
    engine.name = "test-model"
    engine.endpoint = "https://example.invalid/chat/completions"
    engine.key = ""
    engine.settings = {"temperature": 0.7, "top_p": 0.95, "context_tokens": 100}
    engine.encode_chat = lambda messages: [1, 2, 3]
    return engine


def test_server_prompt_token_count(monkeypatch):
    monkeypatch.setattr("deepg.models.requests.post", lambda *args, **kwargs: Response(3, 2))
    result = model().generate([{"role": "user", "content": "Question"}], 10, 5)
    assert result["server_prompt_tokens"] == 3
    assert result["tokens"] == 2


def test_tokenizer_mismatch_is_rejected(monkeypatch):
    monkeypatch.setattr("deepg.models.requests.post", lambda *args, **kwargs: Response(4, 2))
    with pytest.raises(ValueError, match="tokenizer"):
        model().generate([{"role": "user", "content": "Question"}], 10, 5)


def test_server_cannot_exceed_decoder_cap(monkeypatch):
    monkeypatch.setattr("deepg.models.requests.post", lambda *args, **kwargs: Response(3, 11))
    with pytest.raises(ValueError, match="exceeded"):
        model().generate([{"role": "user", "content": "Question"}], 10, 5)
