import hashlib
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .data import exact_match, extract_answer
from .theory import planner_lengths, slot_prices, optimal_uniform_price


AGENT_PROMPT = (
    "You are agent {i} of {n} in a team that answers one question together.\n"
    "The question is {question}.\n"
    "The supporting passages are {passages}.\n"
    "The transcript so far is {transcript}.\n"
    "Write the message you want to add. You have {budget} tokens.\n"
    "{charge}\n"
    "Your message is read by the agents that follow you and by the reader that\n"
    "answers at the end. Answer with the message alone.\n"
)

METHODS = (
    "single_agent", "free_order", "predetermined_order", "round_robin", "debate",
    "rotation", "uniform_price", "slot_prompted", "slot_trained", "without_rotation",
    "lengths_at_turn", "half_price", "double_price", "predetermined_matched",
    "rotation_matched", "planner_cap", "uniform_cap", "sham", "irrelevant", "penalty",
    "length_cap", "budget_hint", "slot_no_budget", "reversed_price", "sequential_committed",
    "calibration", "AgentPrune", "AgentDropout", "GDesigner", "GPTSwarm",
)


def sample_seed(identity, seed, round_index, agent):
    text = f"{identity}|{seed}|{round_index}|{agent}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "big") % (2 ** 31)


def method_spec(method, alpha, n):
    if method not in METHODS:
        raise ValueError("unsupported method " + method)
    uniform = optimal_uniform_price(alpha, n)
    prices = np.zeros(n)
    order = "sequential"
    allocation = "unrestricted"
    if method in ("free_order", "sham", "irrelevant", "penalty", "length_cap", "budget_hint", "calibration"):
        order = "arrival"
    if method in ("rotation", "uniform_price", "slot_prompted", "slot_trained", "half_price",
                  "double_price", "rotation_matched", "planner_cap", "uniform_cap",
                  "slot_no_budget", "reversed_price", "sequential_committed"):
        order = "drawn_parallel"
    if method == "debate":
        order = "parallel"
    if method in ("uniform_price", "half_price", "double_price"):
        multiplier = {"half_price": 0.5, "double_price": 2.0}.get(method, 1.0)
        prices[:] = uniform * multiplier
    if method in ("slot_prompted", "slot_trained", "without_rotation", "slot_no_budget", "sequential_committed"):
        prices = slot_prices(alpha, n)
    if method == "reversed_price":
        prices = slot_prices(alpha, n)[::-1]
    if method == "penalty":
        prices[:] = 0.7 * alpha
    if method in ("slot_trained", "without_rotation", "reversed_price"):
        allocation = "trained"
    if method == "planner_cap":
        allocation = "planner"
        prices[:] = 0
    if method in ("round_robin", "uniform_cap"):
        allocation = "equal"
    if method in ("predetermined_matched", "rotation_matched"):
        allocation = "matched"
    if method == "sequential_committed":
        order = "drawn_committed_sequential"
        allocation = "planner"
    return {"order": order, "allocation": allocation, "prices": prices}


def price_sentence(method, price, alpha):
    if method == "sham":
        return "A record of 6.9 units per token is kept, and nothing is deducted."
    if method == "irrelevant":
        return "Your message is labelled 6.9."
    if price > 0:
        return f"Each token of your message costs {price / alpha:.1f} units of the team budget."
    return ""


def run_question(item, method, model, config, seed, policy=None):
    game = config["game"]
    generation = config["generation"]
    alpha, n, rounds = game["alpha"], game["agents"], game["rounds"]
    if method == "single_agent":
        n, rounds = 1, 1
    spec = method_spec(method, alpha, n)
    if spec["allocation"] == "trained":
        if policy is None:
            raise ValueError("this method requires a trained length-policy checkpoint")
        if len(policy.prices) != n or abs(policy.alpha - alpha) > 1e-12:
            raise ValueError("policy checkpoint and game configuration differ")
        if not np.allclose(policy.prices, spec["prices"], rtol=0, atol=1e-10):
            raise ValueError("policy checkpoint uses a different price schedule")
    total_capacity = int(generation["communication_tokens"])
    if total_capacity < 1 or n < 1 or rounds < 1:
        raise ValueError("communication capacity, team size, and rounds must be positive")
    transcript, messages, scheduling = [], [], []
    remaining = total_capacity
    rng = np.random.default_rng(sample_seed(item["id"], seed, 0, 0))
    passages = "\n\n".join(p["title"] + "\n" + p["text"] for p in item["passages"])
    last_prompt = None
    strength = config.get("calibration_strength", 1.0) if method == "calibration" else 0.0

    def format_transcript(records):
        return "\n\n".join(f"Agent {r['agent']} | Round {r['round']}\n{r['text']}" for r in records)

    def prompt_for(agent, slot, round_index, cap, snapshot):
        budget = str(cap)
        if method == "length_cap":
            budget = "at most 2000"
        if method == "budget_hint":
            budget = "2000"
        prompt = AGENT_PROMPT.format(i=agent + 1, n=n, question=item["question"], passages=passages,
                                     transcript=format_transcript(snapshot), budget=budget,
                                     charge=price_sentence(method, spec["prices"][slot], alpha))
        if method == "slot_no_budget":
            prompt = "\n".join(line for line in prompt.splitlines()
                               if not line.startswith("Write the message you want to add. You have"))
        return [{"role": "user", "content": prompt}]

    def prepare(agent, slot, round_index, cap, snapshot):
        # Preserve the evidence and transcript; the output cap uses the remaining context.
        requested = cap
        while True:
            prompt = prompt_for(agent, slot, round_index, cap, snapshot)
            input_tokens = len(model.encode_chat(prompt))
            available = generation["context_tokens"] - input_tokens
            if available < 0:
                raise ValueError("the complete evidence and transcript exceed the configured context capacity")
            if cap > 0 and available == 0:
                raise ValueError("the complete input leaves no context capacity for the next message")
            if cap <= available:
                return cap, prompt, input_tokens, requested
            cap = available

    def generate(agent, slot, round_index, cap, snapshot, prepared=None):
        cap, prompt, input_tokens, requested = prepared or prepare(agent, slot, round_index, cap, snapshot)
        result = model.generate(prompt, cap, sample_seed(item["id"], seed, round_index, agent), strength)
        if not isinstance(result["tokens"], int) or not 0 <= result["tokens"] <= cap:
            raise ValueError("generation exceeded its committed token cap")
        return {**result, "agent": agent + 1, "slot": slot + 1, "round": round_index + 1,
                "cap": cap, "requested_cap": requested,
                "price": float(spec["prices"][slot]), "input_tokens": input_tokens}, prompt

    for round_index in range(rounds):
        order = rng.permutation(n).tolist() if spec["order"].startswith("drawn") else list(range(n))
        snapshot = list(transcript)
        capacity = remaining / total_capacity
        if spec["allocation"] == "trained":
            planned = policy.allocation(capacity)
        elif spec["allocation"] == "planner":
            planned = planner_lengths(alpha, n) * capacity
        elif spec["allocation"] in ("equal", "matched"):
            planned = np.full(n, capacity / n)
        else:
            planned = None
        parallel = spec["order"] in ("drawn_parallel", "parallel")
        if planned is not None:
            caps = [min(int(math.floor(x * total_capacity + 1e-9)), generation["max_message_tokens"])
                    for x in planned]
        elif parallel:
            caps = [min(remaining // n, generation["max_message_tokens"]) for i in range(n)]
        else:
            caps = None
        if spec["allocation"] == "equal":
            caps = [min(cap, generation["equal_cap_tokens"]) for cap in caps]
        if method == "length_cap" and caps:
            caps = [min(cap, 2000) for cap in caps]
        prepared = None
        if parallel:
            prepared = [prepare(agent, slot, round_index, caps[slot], snapshot)
                        for slot, agent in enumerate(order)]
            caps = [entry[0] for entry in prepared]
        reserved = [remaining] + (remaining - np.cumsum(caps)).tolist() if caps is not None else None
        if spec["order"] == "arrival":
            arrivals = config.get("free_order_delays")
            if arrivals is None:
                arrivals = []
                if not hasattr(model, "choose_arrival"):
                    raise ValueError("free order requires model-selected delays or an explicit arrival trace")
                for agent in range(n):
                    choice = model.choose_arrival(
                        prompt_for(agent, 0, round_index, min(remaining, generation["max_message_tokens"]), snapshot),
                        sample_seed(item["id"] + "|schedule", seed, round_index, agent),
                        generation["context_tokens"])
                    arrivals.append(choice["delay"])
                    scheduling.append({**choice, "agent": agent + 1, "round": round_index + 1})
            if len(arrivals) != n or any(not math.isfinite(d) or d < 0 for d in arrivals):
                raise ValueError("free_order_delays requires one nonnegative arrival delay per agent")
            for slot, agent in enumerate(sorted(range(n), key=lambda agent: (arrivals[agent], agent))):
                cap = min(remaining, generation["max_message_tokens"])
                if method == "length_cap":
                    cap = min(cap, 2000)
                result, prompt = generate(agent, slot, round_index, cap, list(transcript))
                result["arrival_priority"] = float(arrivals[agent])
                transcript.append(result)
                messages.append(result)
                remaining -= result["tokens"]
                last_prompt = prompt
        elif parallel:
            with ThreadPoolExecutor(max_workers=n) as executor:
                futures = {executor.submit(generate, agent, slot, round_index, caps[slot], snapshot,
                                           prepared[slot]): slot
                           for slot, agent in enumerate(order)}
                completed = []
                for future in as_completed(futures):
                    result, prompt = future.result()
                    completed.append((futures[future], result, prompt))
                completed.sort(key=lambda value: value[0])
                for _, result, prompt in completed:
                    slot = result["slot"] - 1
                    result["capacity_before_commitment"] = int(reserved[slot])
                    result["capacity_after_commitment"] = int(reserved[slot + 1])
                    transcript.append(result)
                    messages.append(result)
                    remaining -= result["tokens"]
                    last_prompt = prompt
        else:
            for slot, agent in enumerate(order):
                if caps is None:
                    cap = min(remaining, generation["max_message_tokens"])
                else:
                    cap = min(caps[slot], remaining)
                result, prompt = generate(agent, slot, round_index, cap, list(transcript))
                if reserved is not None:
                    result["capacity_before_commitment"] = int(reserved[slot])
                    result["capacity_after_commitment"] = int(reserved[slot + 1])
                transcript.append(result)
                messages.append(result)
                remaining -= result["tokens"]
                last_prompt = prompt
    final_message = transcript[-1]["text"]
    prediction = extract_answer(final_message, generation["answer_rule"], item["dataset"])
    observation = model.measure_attention(last_prompt, strength) if generation.get("attention") else None
    return {"kind": "evaluation", "id": item["id"], "dataset": item["dataset"], "method": method,
            "seed": seed, "prediction": prediction, "answers": item["answers"],
            "correct": exact_match(prediction, item["answers"]), "tokens": sum(m["tokens"] for m in messages),
            "middle": observation["middle"] if observation else None,
            "attention_backend": "model_probabilities" if observation else "unavailable",
            "messages": messages, "attention": observation, "scheduling": scheduling,
            "scheduling_tokens": sum(record["tokens"] for record in scheduling),
            "inference_tokens": sum(record["tokens"] for record in messages + scheduling),
            "protocol": {"order": spec["order"], "allocation": spec["allocation"],
                         "arrival_rule": ("supplied_trace" if config.get("free_order_delays") is not None
                                          else "model_selected_delay") if spec["order"] == "arrival" else None,
                         "slot_horizon": "round", "remaining_tokens": remaining}}


def run_external(item, method, command, config, seed):
    if method not in ("AgentPrune", "AgentDropout", "GDesigner", "GPTSwarm"):
        raise ValueError("unknown external baseline")
    public_item = {key: item[key] for key in ("id", "dataset", "question", "passages")}
    payload = {"item": public_item, "method": method, "seed": seed,
               "game": config["game"], "generation": config["generation"],
               "backend": config.get("backend", "api")}
    process = subprocess.run(command, input=json.dumps(payload), text=True,
                             capture_output=True, timeout=config.get("baseline_timeout", 3600))
    if process.returncode:
        raise RuntimeError("baseline process failed with exit code " + str(process.returncode))
    result = json.loads(process.stdout)
    messages = result["messages"]
    if not messages or any(not isinstance(m.get("tokens"), int) or m["tokens"] < 0 for m in messages):
        raise ValueError("baseline messages require exact integer generated-token counts")
    if sum(message["tokens"] for message in messages) > config["generation"]["communication_tokens"]:
        raise ValueError("baseline exceeded the aggregate communication budget")
    prediction = extract_answer(result["prediction"], config["generation"]["answer_rule"], item["dataset"])
    middle = result.get("middle")
    if middle is not None and not 0 <= middle <= 1:
        raise ValueError("invalid measured middle-attention share")
    return {"kind": "evaluation", "id": item["id"], "dataset": item["dataset"], "method": method,
            "seed": seed, "prediction": prediction, "answers": item["answers"],
            "correct": exact_match(prediction, item["answers"]), "tokens": sum(m["tokens"] for m in messages),
            "middle": middle, "attention_backend": "model_probabilities" if middle is not None else "unavailable",
            "messages": messages, "baseline_revision": result["revision"]}
