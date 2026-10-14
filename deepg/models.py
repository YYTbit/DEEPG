import os
import threading
import time
from contextlib import contextmanager

import numpy as np
import requests


model_lock = threading.RLock()


def attention_mass(tensors, sequence_length):
    if not tensors:
        raise ValueError("the backend did not return attention probabilities")
    mass = np.zeros(sequence_length)
    layers = 0
    for tensor in tensors:
        if tensor is None:
            continue
        vector = tensor[0, :, -1, :].float().mean(dim=0).detach().cpu().numpy()
        if len(vector) > sequence_length:
            raise ValueError("attention key positions exceed the input sequence")
        mass[sequence_length - len(vector):] += vector
        layers += 1
    if not layers:
        raise ValueError("the backend did not return attention probabilities")
    return mass / layers


def middle_attention(mass):
    mass = np.asarray(mass, dtype=float)
    if mass.ndim != 1 or len(mass) == 0 or np.any(~np.isfinite(mass)) or np.any(mass < 0):
        raise ValueError("invalid attention probabilities")
    length = len(mass)
    return float(mass[length // 3:(2 * length) // 3].sum())


class ModelTokenizer:
    def __init__(self, name, enable_thinking=False):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.enable_thinking = bool(enable_thinking)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def encode_chat(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                             enable_thinking=self.enable_thinking)


class ApiModel(ModelTokenizer):
    def __init__(self, name, tokenizer=None, settings=None):
        settings = settings or {}
        super().__init__(tokenizer or name, settings.get("enable_thinking", False))
        server = os.environ.get("LLM_SERVER", "").rstrip("/")
        if not server:
            raise ValueError("set LLM_SERVER to the inference server base URL")
        self.endpoint = server if server.endswith("/chat/completions") else server + "/chat/completions"
        self.name = name
        self.settings = settings
        self.key = os.environ.get("LLM_KEY", "")

    def generate(self, messages, max_tokens, seed, strength=0.0):
        if strength:
            raise ValueError("attention calibration requires the local backend")
        if max_tokens <= 0:
            return {"text": "", "tokens": 0, "finish_reason": "zero_capacity"}
        prompt_tokens = len(self.encode_chat(messages))
        if prompt_tokens + max_tokens > self.settings["context_tokens"]:
            raise ValueError("prompt and token cap exceed context_tokens")
        payload = {"model": self.name, "messages": messages, "max_tokens": int(max_tokens),
                   "temperature": self.settings["temperature"], "top_p": self.settings["top_p"], "seed": int(seed)}
        if "Qwen3" in self.name or "qwen3" in self.name:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        for attempt in range(3):
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=300)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(attempt + 1)
                continue
            if response.status_code != 200:
                raise RuntimeError("inference request failed with HTTP " + str(response.status_code))
            data = response.json()
            choice = data["choices"][0]
            count = data.get("usage", {}).get("completion_tokens")
            server_prompt_tokens = data.get("usage", {}).get("prompt_tokens")
            if not isinstance(server_prompt_tokens, int) or server_prompt_tokens != prompt_tokens:
                raise ValueError("server prompt usage differs from the released tokenizer and chat template")
            if not isinstance(count, int) or count < 0:
                raise ValueError("the API must report integer completion_tokens")
            if count > max_tokens:
                raise ValueError("the server exceeded the requested token cap")
            return {"text": choice["message"].get("content") or "", "tokens": count,
                    "server_prompt_tokens": server_prompt_tokens,
                    "finish_reason": choice.get("finish_reason"),
                    "reasoning": choice["message"].get("reasoning_content")}
        raise RuntimeError("inference retries exhausted")

    def measure_attention(self, messages, strength=0.0):
        return None


class LocalModel(ModelTokenizer):
    def __init__(self, name, settings, device_map="auto", profile=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
        super().__init__(name, settings.get("enable_thinking", False))
        self.settings = settings
        dtype = getattr(torch, settings["precision"])
        kwargs = {"torch_dtype": dtype, "device_map": device_map, "attn_implementation": "eager"}
        try:
            self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        except ValueError as error:
            if "Unrecognized configuration class" not in str(error):
                raise
            self.model = AutoModelForImageTextToText.from_pretrained(name, **kwargs)
        self.model.eval()
        self.attention_profile = None
        if profile:
            values = np.load(profile) if str(profile).endswith(".npy") else np.asarray(__import__("json").loads(open(profile).read()), dtype=float)
            if values.ndim != 1 or len(values) < 2 or np.any(~np.isfinite(values)) or np.any(values <= 0):
                raise ValueError("calibration requires a positive measured attention profile")
            self.attention_profile = values / values.mean()

    @contextmanager
    def recalibrated_attention(self, strength):
        import torch
        import torch.nn.functional as functional
        if strength == 0:
            yield
            return
        if not 0 <= strength <= 1 or self.attention_profile is None:
            raise ValueError("calibration needs strength in [0, 1] and a measured profile")
        original = functional.softmax
        def calibrated_softmax(values, dim=None, *args, **kwargs):
            if values.ndim == 4 and dim in (-1, 3):
                weights = np.interp(np.linspace(0, 1, values.shape[-1]),
                                    np.linspace(0, 1, len(self.attention_profile)), self.attention_profile)
                bias = torch.tensor(-strength * np.log(weights), device=values.device, dtype=values.dtype)
                values = values + bias
            return original(values, dim, *args, **kwargs)
        functional.softmax = calibrated_softmax
        try:
            yield
        finally:
            functional.softmax = original

    def generate(self, messages, max_tokens, seed, strength=0.0):
        import torch
        if max_tokens <= 0:
            return {"text": "", "tokens": 0, "finish_reason": "zero_capacity"}
        ids = self.encode_chat(messages)
        if len(ids) + max_tokens > self.settings["context_tokens"]:
            raise ValueError("prompt and token cap exceed context_tokens")
        device = self.model.get_input_embeddings().weight.device
        inputs = torch.tensor([ids], device=device)
        with model_lock, torch.inference_mode(), torch.random.fork_rng(), self.recalibrated_attention(strength):
            torch.manual_seed(int(seed))
            sampling = self.settings["temperature"] > 0
            kwargs = {"do_sample": sampling, "max_new_tokens": int(max_tokens),
                      "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id}
            if sampling:
                kwargs.update(temperature=self.settings["temperature"], top_p=self.settings["top_p"])
            outputs = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), **kwargs)
        generated = outputs[0, len(ids):]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return {"text": text, "tokens": int(len(generated)),
                "finish_reason": "length" if len(generated) == max_tokens else "eos"}

    def measure_attention(self, messages, strength=0.0):
        import torch
        ids = self.encode_chat(messages)
        if len(ids) > self.settings["context_tokens"]:
            raise ValueError("attention probe exceeds context_tokens")
        device = self.model.get_input_embeddings().weight.device
        inputs = torch.tensor([ids], device=device)
        with model_lock, torch.inference_mode(), self.recalibrated_attention(strength):
            cache = None
            for start in range(0, max(0, len(ids) - 1), 256):
                end = min(start + 256, len(ids) - 1)
                output = self.model(input_ids=inputs[:, start:end], past_key_values=cache,
                                     use_cache=True, output_attentions=False)
                cache = output.past_key_values
            output = self.model(input_ids=inputs[:, -1:], past_key_values=cache,
                                 use_cache=True, output_attentions=True)
            mass = attention_mass(output.attentions, len(ids))
        return {"middle": middle_attention(mass), "profile": mass.tolist(), "tokens": len(ids)}
