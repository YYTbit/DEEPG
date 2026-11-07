import os
import json
import threading
import time
import copy
import importlib
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

import numpy as np
import requests

from .loading import digest, library_versions, local_weight_identity, resolve_configuration, tokenizer_identity


model_lock = threading.RLock()
_calibration = ContextVar("deepg_attention_calibration", default=None)


@lru_cache(maxsize=None)
def eager_function(module_name):
    return importlib.import_module(module_name).eager_attention_forward


def calibrated_attention(module, query, key, value, attention_mask, **kwargs):
    calibration = _calibration.get()
    if calibration is not None:
        import torch

        profile, strength = calibration["profile"], calibration["strength"]
        end = calibration["end"]
        horizon = max(calibration["sequence_length"] or 0, end)
        # Cached local keys retain their positions in the complete sequence.
        positions = np.arange(key.shape[-2]) + max(0, end - key.shape[-2])
        weights = np.interp(positions / max(1, horizon - 1), np.linspace(0, 1, len(profile)), profile)
        bias = torch.as_tensor(-strength * np.log(weights), device=query.device, dtype=query.dtype)[None, None, None, :]
        attention_mask = bias if attention_mask is None else attention_mask + bias
    # Each backbone retains its native scaling, grouped heads, soft cap and causal mask.
    return eager_function(type(module).__module__)(module, query, key, value, attention_mask, **kwargs)


def calibration_positions(module, args, kwargs):
    calibration = _calibration.get()
    if calibration is None:
        return
    positions = kwargs.get("cache_position")
    if positions is not None and positions.numel():
        calibration["end"] = int(positions[-1]) + 1
        return
    inputs = kwargs.get("input_ids", args[0] if args else None)
    width = inputs.shape[-1] if inputs is not None else kwargs["inputs_embeds"].shape[-2]
    cache = kwargs.get("past_key_values")
    calibration["end"] = int(cache.get_seq_length()) + width if cache is not None else width


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
    def __init__(self, name, enable_thinking=False, revision=None, configuration=None):
        from transformers import AutoTokenizer
        if configuration is None:
            configuration, revision = resolve_configuration(name, revision)
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.enable_thinking = bool(enable_thinking)
        self.identity = {"tokenizer_model": name, "revision": revision,
                         "tokenizer": tokenizer_identity(self.tokenizer), "enable_thinking": self.enable_thinking}

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def encode_chat(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                             enable_thinking=self.enable_thinking)

    def choose_arrival(self, messages, seed, context_tokens):
        instruction = (
            "Choose when to append your message in this round. A smaller delay gives an earlier turn. "
            'Return a JSON object with a single numeric field "delay" between 0 and 1.'
        )
        prompt = [dict(message) for message in messages]
        if prompt and prompt[-1]["role"] == "user":
            prompt[-1]["content"] += "\n\n" + instruction
        else:
            prompt.append({"role": "user", "content": instruction})
        input_tokens = len(self.encode_chat(prompt))
        cap = min(32, context_tokens - input_tokens)
        if cap < 1:
            raise ValueError("the scheduling decision exceeds the configured context capacity")
        result = self.generate(prompt, cap, seed)
        try:
            value = json.loads(result["text"])
            delay = value["delay"]
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError("the free-order scheduling decision requires a JSON delay") from error
        if (isinstance(delay, bool) or not isinstance(delay, (int, float))
                or not np.isfinite(delay) or not 0 <= delay <= 1):
            raise ValueError("the model-selected delay must be in [0, 1]")
        return {**result, "delay": float(delay), "input_tokens": input_tokens}


class ApiModel(ModelTokenizer):
    def __init__(self, name, tokenizer=None, settings=None, revision=None):
        settings = settings or {}
        super().__init__(tokenizer or name, settings.get("enable_thinking", False), revision)
        server = os.environ.get("LLM_SERVER", "").rstrip("/")
        if not server:
            raise ValueError("set LLM_SERVER to the inference server base URL")
        self.endpoint = server if server.endswith("/chat/completions") else server + "/chat/completions"
        self.name = name
        self.settings = settings
        self.key = os.environ.get("LLM_KEY", "")
        self.identity.update(backend="api", model=name, libraries=library_versions("transformers"))

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
    def __init__(self, name, settings, device_map="auto", profile=None, revision=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
        configuration, revision = resolve_configuration(name, revision)
        super().__init__(name, settings.get("enable_thinking", False), revision, configuration)
        self.settings = settings
        dtype = getattr(torch, settings["precision"])
        loader = (AutoModelForImageTextToText if configuration.model_type in {"gemma3", "mistral3"}
                  else AutoModelForCausalLM)
        self.model = loader.from_pretrained(name, config=configuration, revision=revision, torch_dtype=dtype,
                                            device_map=device_map, attn_implementation="eager")
        self.model.eval()
        self.attention_profile = None
        if profile:
            from pathlib import Path

            values = (np.load(profile, allow_pickle=False) if str(profile).endswith(".npy")
                      else np.asarray(json.loads(Path(profile).read_text()), dtype=float))
            if values.ndim != 1 or len(values) < 2 or np.any(~np.isfinite(values)) or np.any(values <= 0):
                raise ValueError("calibration requires a positive measured attention profile")
            self.attention_profile = values / values.mean()
            from transformers import AttentionInterface, AttentionMaskInterface
            from transformers.masking_utils import eager_mask

            AttentionInterface.register("deepg_calibrated_eager", calibrated_attention)
            AttentionMaskInterface.register("deepg_calibrated_eager", eager_mask)
            self.model.set_attn_implementation("deepg_calibrated_eager")
            self.model.register_forward_pre_hook(calibration_positions, with_kwargs=True)
        self.identity.update(backend="local", model=name, precision=settings["precision"],
                             local_weights=local_weight_identity(name),
                             generation=digest(self.generation_config().to_dict()),
                             libraries=library_versions("torch", "transformers"), protocol=2)

    def generation_config(self):
        configuration = copy.deepcopy(self.model.generation_config)
        configuration.do_sample = self.settings["temperature"] > 0
        if configuration.do_sample:
            configuration.temperature = self.settings["temperature"]
            configuration.top_p = self.settings["top_p"]
        if self.tokenizer.pad_token_id is not None:
            configuration.pad_token_id = self.tokenizer.pad_token_id
        elif configuration.pad_token_id is None:
            eos = configuration.eos_token_id
            configuration.pad_token_id = eos[0] if isinstance(eos, list) else eos
        return configuration

    def cuda_devices(self):
        devices = {parameter.device.index for parameter in self.model.parameters() if parameter.device.type == "cuda"}
        return sorted(devices)

    @contextmanager
    def recalibrated_attention(self, strength, sequence_length=None):
        if strength == 0:
            yield
            return
        if not 0 <= strength <= 1 or self.attention_profile is None:
            raise ValueError("calibration needs strength in [0, 1] and a measured profile")
        token = _calibration.set({"profile": self.attention_profile, "strength": strength,
                                  "sequence_length": sequence_length, "end": 0})
        try:
            yield
        finally:
            _calibration.reset(token)

    def generate(self, messages, max_tokens, seed, strength=0.0):
        import torch
        if max_tokens <= 0:
            return {"text": "", "tokens": 0, "finish_reason": "zero_capacity"}
        ids = self.encode_chat(messages)
        if len(ids) + max_tokens > self.settings["context_tokens"]:
            raise ValueError("prompt and token cap exceed context_tokens")
        device = self.model.get_input_embeddings().weight.device
        inputs = torch.tensor([ids], device=device)
        devices = self.cuda_devices()
        with model_lock, torch.inference_mode(), torch.random.fork_rng(devices=devices), self.recalibrated_attention(strength, len(ids)):
            torch.random.default_generator.manual_seed(int(seed))
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(int(seed))
            outputs = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                          generation_config=self.generation_config(), use_model_defaults=False,
                                          max_new_tokens=int(max_tokens))
        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        generated = sequences[0, len(ids):]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        stopped = len(generated) and int(generated[-1]) in eos
        return {"text": text, "tokens": int(len(generated)),
                "finish_reason": "eos" if stopped else "length"}

    def measure_attention(self, messages, strength=0.0):
        import torch
        ids = self.encode_chat(messages)
        if len(ids) > self.settings["context_tokens"]:
            raise ValueError("attention probe exceeds context_tokens")
        device = self.model.get_input_embeddings().weight.device
        inputs = torch.tensor([ids], device=device)
        with model_lock, torch.inference_mode(), self.recalibrated_attention(strength, len(ids)):
            # Process prompt tokens in bounded chunks, carrying the KV cache into the final-token probe.
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
