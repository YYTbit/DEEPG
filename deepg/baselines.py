import asyncio
import contextlib
import importlib
import json
import subprocess
import sys
from pathlib import Path


def fetch_baselines(manifest, output):
    definitions = json.loads(Path(manifest).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, specification in definitions.items():
        target = output / name
        if not target.exists():
            subprocess.run(["git", "clone", specification["url"], str(target)], check=True)
        remote = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], cwd=target, text=True).strip()
        if remote != specification["url"]:
            raise ValueError("existing baseline checkout has a different origin")
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=target, text=True)
        if status:
            raise ValueError("baseline checkout contains local changes")
        subprocess.run(["git", "fetch", "origin", specification["revision"]], cwd=target, check=True)
        subprocess.run(["git", "checkout", "--detach", specification["revision"]], cwd=target, check=True)


def load_parameters(target, checkpoint):
    import torch
    parameters = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("baseline checkpoint requires a dictionary of tensor or module states")
    for name, value in parameters.items():
        if not hasattr(target, name):
            raise ValueError("unknown baseline checkpoint parameter " + name)
        attribute = getattr(target, name)
        if hasattr(attribute, "load_state_dict"):
            attribute.load_state_dict(value, strict=True)
        elif hasattr(attribute, "data") and hasattr(value, "shape"):
            if tuple(attribute.shape) != tuple(value.shape):
                raise ValueError("baseline parameter shape differs for " + name)
            attribute.data.copy_(value)
        else:
            raise ValueError("unsupported baseline checkpoint parameter " + name)


def run_baseline(request, method, source, checkpoint, graph_settings, model_name, tokenizer=None):
    import torch
    from .models import ApiModel, LocalModel
    from .protocols import sample_seed
    source = Path(source).resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if graph_settings.get("revision") != revision:
        raise ValueError("graph settings require the exact upstream revision")
    if not checkpoint or not Path(checkpoint).is_file():
        raise ValueError("a trained upstream topology checkpoint is required")
    sys.path.insert(0, str(source))
    package = "swarm" if method == "GPTSwarm" else method
    engine = (ApiModel(model_name, tokenizer, request["generation"]) if request.get("backend", "api") == "api"
              else LocalModel(model_name, request["generation"]))
    messages = []
    final_prompt = None
    torch.manual_seed(request["seed"])

    class TrackedModel:
        def gen(self, conversation, max_tokens=None, temperature=None, num_comps=None, **kwargs):
            nonlocal final_prompt
            if num_comps not in (None, 1):
                raise ValueError("baseline bridge expects one completion per call")
            serialized = [{"role": str(message["role"]), "content": str(message["content"])}
                          if isinstance(message, dict) else {"role": str(message.role), "content": str(message.content)}
                          for message in conversation]
            cap = request["generation"]["max_message_tokens"]
            cap = min(cap, request["generation"]["communication_tokens"] - sum(m["tokens"] for m in messages))
            if max_tokens is not None:
                cap = min(cap, int(max_tokens))
            available = request["generation"]["context_tokens"] - len(engine.encode_chat(serialized))
            if available < 0:
                raise ValueError("the baseline prompt exceeds the configured context capacity")
            cap = min(cap, available)
            seed = sample_seed(request["item"]["id"], request["seed"], 0, len(messages))
            result = engine.generate(serialized, cap, seed)
            messages.append({**result, "cap": cap})
            final_prompt = serialized
            return result["text"]

        async def agen(self, conversation, **kwargs):
            return self.gen(conversation, **kwargs)

    registry = importlib.import_module(package + ".llm.llm_registry").LLMRegistry
    registry.get = classmethod(lambda cls, *args, **kwargs: TrackedModel())
    prompt_prefix = ".environment.prompt" if method == "GPTSwarm" else ".prompt"
    prompt_module = importlib.import_module(package + prompt_prefix + ".mmlu_prompt_set")
    base_class = next(value for key, value in vars(prompt_module).items() if key.lower() == "mmlupromptset")
    prompt_registry = importlib.import_module(package + prompt_prefix + ".prompt_set_registry").PromptSetRegistry

    class ReadingPromptSet(base_class):
        @staticmethod
        def get_constraint():
            return "Use the supplied evidence to answer the question. End with the answer on a separate line."

        @staticmethod
        def get_analyze_constraint(role):
            return "Analyze the supplied evidence and the preceding messages. End with the answer on a separate line."

        @staticmethod
        def get_decision_constraint():
            return "Combine the evidence and the agent messages. Return the final answer on a separate line."

        @staticmethod
        def get_answer_prompt(question):
            return question

    prompt_registry.register("deepg_reading")(ReadingPromptSet)
    item = request["item"]
    task = item["question"] + "\n\n" + "\n\n".join(p["title"] + "\n" + p["text"] for p in item["passages"])
    kwargs = dict(graph_settings["graph"])
    if method == "GPTSwarm":
        from swarm.graph.swarm import Swarm
        from swarm.environment.operations.final_decision import MergingStrategy
        kwargs.update(domain="deepg_reading", model_name=model_name,
                      final_node_kwargs={"strategy": MergingStrategy.OutputsAsReferences})
        graph = Swarm(**kwargs)
        load_parameters(graph.connection_dist, checkpoint)
        final = ""
        for index in range(request["game"]["rounds"]):
            inputs = {"task": task + ("\n\nPrevious round\n" + final if index else ""), "files": []}
            final = asyncio.run(graph.arun(inputs))
    else:
        graph_class = importlib.import_module(package + ".graph.graph").Graph
        kwargs.update(domain="deepg_reading", llm_name=model_name)
        graph = graph_class(**kwargs)
        load_parameters(graph, checkpoint)
        final, _ = asyncio.run(graph.arun({"task": task}, num_rounds=request["game"]["rounds"]))
    if isinstance(final, list):
        final = final[-1] if final else ""
    if not isinstance(final, str):
        raise ValueError("upstream graph returned an unsupported answer")
    observation = engine.measure_attention(final_prompt) if request["generation"].get("attention") and final_prompt else None
    return {"prediction": final, "messages": messages, "middle": observation["middle"] if observation else None,
            "revision": revision}


def bridge(args):
    request = json.load(sys.stdin)
    graph_settings = json.loads(Path(args.settings).read_text())
    with contextlib.redirect_stdout(sys.stderr):
        result = run_baseline(request, args.method, args.source, args.checkpoint,
                              graph_settings, args.model, args.tokenizer)
    print(json.dumps(result, allow_nan=False))
