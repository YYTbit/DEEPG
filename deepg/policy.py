import json
from pathlib import Path

import numpy as np

from .theory import slot_prices, welfare, planner_lengths, planner_welfare, equilibrium_lengths, optimal_uniform_price


class TabularPolicy:
    def __init__(self, alpha, prices, grid=201):
        self.alpha = float(alpha)
        self.prices = np.asarray(prices, dtype=float)
        self.grid = int(grid)
        if self.prices.ndim != 1 or len(self.prices) < 1 or self.grid < 2:
            raise ValueError("invalid policy dimensions")
        if np.any(~np.isfinite(self.prices)) or np.any(self.alpha + self.prices <= 0):
            raise ValueError("private costs must be finite and positive")
        self.logits = np.zeros((len(prices), grid, grid), dtype=float)

    def probabilities(self, slot, states, temperature=1.0):
        if not 0 <= slot < len(self.prices) or temperature <= 0:
            raise ValueError("invalid slot or temperature")
        states = np.atleast_1d(np.asarray(states, dtype=int))
        if np.any((states < 0) | (states >= self.grid)):
            raise ValueError("capacity states exceed the grid")
        values = self.logits[slot, states].copy() / temperature
        values[np.arange(self.grid)[None, :] > states[:, None]] = -np.inf
        values -= np.max(values, axis=1, keepdims=True)
        values = np.exp(values)
        return values / values.sum(axis=1, keepdims=True)

    def choose_length(self, slot, capacity):
        if not 0 <= slot < len(self.prices) or not 0 <= capacity <= 1:
            raise ValueError("invalid deployment state")
        state = min(int(np.floor(capacity * (self.grid - 1) + 1e-10)), self.grid - 1)
        action = int(np.argmax(self.logits[slot, state, :state + 1]))
        return min(action / (self.grid - 1), capacity)

    def allocation(self, capacity=1.0):
        lengths = []
        for slot in range(len(self.prices)):
            length = self.choose_length(slot, capacity)
            lengths.append(length)
            capacity = max(0.0, capacity - length)
        return np.array(lengths)

    def save(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, logits=self.logits, alpha=self.alpha, prices=self.prices,
                            grid=self.grid, metadata=json.dumps(metadata, sort_keys=True))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            policy = cls(float(data["alpha"]), data["prices"], int(data["grid"]))
            logits = data["logits"]
            if logits.shape != policy.logits.shape or np.any(~np.isfinite(logits)):
                raise ValueError("invalid policy checkpoint")
            policy.logits = logits
            return policy, json.loads(str(data["metadata"]))

def price_schedule(alpha, n, schedule):
    if schedule == "unpriced":
        return np.zeros(n)
    if schedule == "uniform":
        return np.full(n, optimal_uniform_price(alpha, n))
    if schedule == "slot":
        return slot_prices(alpha, n)
    if schedule == "reversed":
        return slot_prices(alpha, n)[::-1]
    raise ValueError("unknown price schedule")


def train_policy(alpha, n, settings, seed, schedule, out):
    prices = price_schedule(alpha, n, schedule)
    policy = TabularPolicy(alpha, prices, settings["grid"])
    rng = np.random.default_rng(seed)
    group = int(settings["group_size"])
    updates = int(settings["updates"])
    if group < 2 or updates < 1:
        raise ValueError("training requires at least two trajectories and one update")
    visits = np.zeros((n, policy.grid), dtype=int)
    history = []
    for step in range(updates):
        temperature = max(settings["temperature"] / (1 + step / settings["temperature_decay"]),
                          settings["temperature_floor"])
        learning_rate = max(settings["learning_rate"] / (1 + step / settings["learning_decay"]),
                            settings["learning_floor"])
        states = np.zeros((group, n), dtype=int)
        actions = np.zeros((group, n), dtype=int)
        rewards = np.zeros((group, n))
        distributions = []
        states[:, 0] = policy.grid - 1
        for i in range(n):
            probabilities = policy.probabilities(i, states[:, i], temperature)
            distributions.append(probabilities)
            draw = rng.random((group, 1))
            actions[:, i] = np.minimum((np.cumsum(probabilities, axis=1) < draw).sum(axis=1), states[:, i])
            m = states[:, i] / (policy.grid - 1)
            length = actions[:, i] / (policy.grid - 1)
            rewards[:, i] = np.sqrt(np.maximum(length * (m - length / 2), 0)) - (alpha + prices[i]) * length
            np.add.at(visits[i], states[:, i], 1)
            if i + 1 < n:
                states[:, i + 1] = states[:, i] - actions[:, i]
        advantage = rewards - rewards.mean(axis=0, keepdims=True)
        for i in range(n):
            gradient = -advantage[:, i, None] * distributions[i]
            gradient[np.arange(group), actions[:, i]] += advantage[:, i]
            np.add.at(policy.logits[i], states[:, i], learning_rate * gradient / (group * temperature))
        if (step + 1) % settings["record_every"] == 0 or step + 1 == updates:
            lengths = policy.allocation()
            target = equilibrium_lengths(alpha, n, prices)
            history.append({"updates": step + 1, "lengths": lengths.tolist(),
                            "welfare": welfare(lengths, alpha),
                            "profile_error": float(np.max(np.abs(lengths - target))),
                            "temperature": temperature, "learning_rate": learning_rate})
    metadata = {"kind": "training_run", "seed": seed, "schedule": schedule,
                "settings": dict(settings), "agents": n, "alpha": alpha}
    directory = Path(out) / schedule / str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    policy.save(directory / "policy.npz", metadata)
    report = {**metadata, "prices": prices.tolist(), "history": history,
              "visits": visits.tolist(), "planner_lengths": planner_lengths(alpha, n).tolist(),
              "planner_welfare": planner_welfare(alpha, n),
              "equilibrium_lengths": equilibrium_lengths(alpha, n, prices).tolist(),
              "equilibrium_welfare": welfare(equilibrium_lengths(alpha, n, prices), alpha)}
    (directory / "training.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
