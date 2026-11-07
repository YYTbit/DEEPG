import hashlib
import importlib.metadata
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def resolve_configuration(name, revision=None):
    from transformers import AutoConfig

    configuration = AutoConfig.from_pretrained(name, revision=revision)
    resolved = getattr(configuration, "_commit_hash", None) or revision
    if not Path(name).is_dir() and (resolved is None or len(resolved) != 40):
        raise ValueError("the model repository must resolve to an immutable revision")
    return configuration, resolved


def tokenizer_identity(tokenizer):
    values = {"vocabulary": digest(tokenizer.get_vocab()), "chat_template": digest(tokenizer.chat_template),
              "special_tokens": digest(tokenizer.special_tokens_map)}
    if hasattr(tokenizer, "backend_tokenizer"):
        values["backend"] = digest(tokenizer.backend_tokenizer.to_str())
    return values


def local_weight_identity(name):
    path = Path(name)
    if not path.is_dir():
        return None
    files = sorted(path.glob("*.safetensors")) or sorted(path.glob("pytorch_model*.bin"))
    if not files:
        raise ValueError("the local model directory contains no supported weights")
    checksum = hashlib.sha256()
    for file in files:
        checksum.update(file.name.encode())
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1048576), b""):
                checksum.update(block)
    return checksum.hexdigest()


def library_versions(*names):
    return {name: importlib.metadata.version(name) for name in names}
