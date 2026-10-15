import math

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .theory import (attended_information, calibration_solution, equilibrium_lengths,
                     equilibrium_welfare, occupation_fraction, optimal_uniform_price,
                     planner_lengths, planner_welfare, slot_prices, symmetric_rotation,
                     validate_game, welfare)


def integrated_attention(starts, lengths, curve, quadrature=800):
    if quadrature < 2:
        raise ValueError("quadrature requires at least two points")
    starts, lengths = np.broadcast_arrays(np.asarray(starts), np.asarray(lengths))
    positions = starts[..., None] + lengths[..., None] * np.linspace(0, 1, quadrature)
    values = np.asarray(curve(positions), dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < -1e-12):
        raise ValueError("the attention curve must be finite and nonnegative")
    return np.trapezoid(np.maximum(values, 0), dx=1 / (quadrature - 1), axis=-1) * lengths


def grid_equilibrium(alpha, agents, curve=lambda x: 1 - x, steps=4000, quadrature=800):
    validate_game(alpha, agents)
    if steps < 2:
        raise ValueError("the length grid requires at least two intervals")
    consumed, lengths = 0, []
    for _ in range(agents):
        choices = np.arange(steps - consumed + 1) / steps
        attention = integrated_attention(consumed / steps, choices, curve, quadrature)
        utility = np.sqrt(attention) - alpha * choices
        chosen = int(np.argmax(utility))
        lengths.append(chosen / steps)
        consumed += chosen
    return np.asarray(lengths)


def numerical_welfare(lengths, alpha, curve=lambda x: 1 - x, quadrature=800):
    lengths = np.asarray(lengths, dtype=float)
    if np.any(lengths < 0) or lengths.sum() > 1 + 1e-12:
        raise ValueError("the numerical allocation exceeds capacity")
    starts = np.r_[0.0, np.cumsum(lengths)[:-1]]
    return float(np.sqrt(integrated_attention(starts, lengths, curve, quadrature)).sum()
                 - alpha * lengths.sum())


def numerical_planner(alpha, agents, curve=lambda x: 1 - x, restarts=80, seed=7,
                      quadrature=800):
    validate_game(alpha, agents)
    if restarts < 1:
        raise ValueError("the planner requires at least one restart")
    generator = np.random.default_rng(seed)

    def allocation(logits):
        weights = np.r_[logits, 0.0]
        weights = np.exp(weights - weights.max())
        return (weights / weights.sum())[:-1]

    best = None
    for _ in range(restarts):
        initial = generator.dirichlet(np.ones(agents + 1))
        result = minimize(lambda x: -numerical_welfare(allocation(x), alpha, curve, quadrature),
                          np.log(initial[:-1] / initial[-1]), method="Nelder-Mead",
                          options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 40000})
        if best is None or result.fun < best.fun:
            best = result
    lengths = allocation(best.x)
    return {"lengths": lengths.tolist(), "welfare": -float(best.fun),
            "restarts": restarts, "converged": bool(best.success)}


def grid_report(costs, teams, steps=4000, quadrature=800):
    rows = []
    for alpha in costs:
        for agents in teams:
            observed = grid_equilibrium(alpha, agents, steps=steps, quadrature=quadrature)
            target = equilibrium_lengths(alpha, agents)
            difference = np.abs(observed - target)
            resolved = target > 0.005
            rows.append({"alpha": alpha, "agents": agents, "grid_lengths": observed.tolist(),
                         "closed_lengths": target.tolist(), "occupied_fraction": float(target.sum()),
                         "capacity_valid": bool(observed.sum() <= 1 + 1e-12),
                         "max_absolute_deviation": float(difference.max()),
                         "max_relative_deviation": float((difference[resolved] / target[resolved]).max())
                         if np.any(resolved) else None})
    return {"kind": "numerical_verification", "grid_intervals": steps, "quadrature_points": quadrature,
            "rows": rows,
            "teams_below_99_percent": {str(alpha): max(
                [0] + [n for n in range(1, 1001) if equilibrium_lengths(alpha, n).sum() < 0.99])
                for alpha in costs}}


def paper_numbers(costs=(0.02, 0.05, 0.08, 0.12, 0.25, 0.35), alpha=0.08):
    reversal, designs, teams = [], [], []
    for cost in costs:
        lengths = equilibrium_lengths(cost, 2)
        reversal.append({"alpha": cost, "lengths": lengths.tolist(),
                         "attention": attended_information(lengths).tolist(),
                         "equilibrium_welfare": equilibrium_welfare(cost, 2),
                         "planner_welfare": planner_welfare(cost, 2),
                         "efficiency": equilibrium_welfare(cost, 2) / planner_welfare(cost, 2)})
    for cost in (0.05, 0.08, 0.12):
        price = optimal_uniform_price(cost, 2)
        optimum = planner_welfare(cost, 2)
        rotation = symmetric_rotation(cost, 2)
        symmetric = minimize_scalar(lambda x: -welfare([x, x], cost), bounds=(0, 0.5),
                                    method="bounded", options={"xatol": 1e-14})
        designs.append({"alpha": cost, "unpriced_efficiency": equilibrium_welfare(cost, 2) / optimum,
                        "uniform_price": price,
                        "uniform_efficiency": welfare(equilibrium_lengths(cost, 2, [price, price]), cost) / optimum,
                        "rotation_efficiency": rotation["welfare"] / optimum,
                        "rotation_priced_efficiency": -float(symmetric.fun) / optimum})
    for agents in (2, 3, 4):
        lengths = equilibrium_lengths(alpha, agents)
        teams.append({"agents": agents, "equilibrium_lengths": lengths.tolist(),
                      "equilibrium_welfare": equilibrium_welfare(alpha, agents),
                      "planner_lengths": planner_lengths(alpha, agents).tolist(),
                      "planner_welfare": planner_welfare(alpha, agents),
                      "efficiency": equilibrium_welfare(alpha, agents) / planner_welfare(alpha, agents)})
    response = [{"schedule": name, "prices": prices.tolist(),
                 "response_fractions": occupation_fraction(alpha + prices).tolist(),
                 "lengths": equilibrium_lengths(alpha, 3, prices).tolist()}
                for name, prices in (("unpriced", np.zeros(3)),
                                     ("uniform", np.full(3, optimal_uniform_price(alpha, 3))),
                                     ("slot", slot_prices(alpha, 3))) ]
    return {"kind": "analytical", "reversal": reversal, "designs": designs,
            "team_size": teams, "response": response,
            "calibration": [{"slope": slope, **{key: value.tolist() if isinstance(value, np.ndarray) else value
                                                 for key, value in calibration_solution(alpha, 2, slope).items()}}
                            for slope in (0.0, 0.25, 0.5, 0.75, 1.0)]}


def attention_ablation(alpha=0.08, teams=(2, 3), steps=4000, quadrature=800, restarts=80, seed=7):
    curves = {"linear": lambda x: 1 - x, "convex": lambda x: (1 - x) ** 2,
              "exponential": lambda x: (np.exp(-3 * x) - math.exp(-3)) / (1 - math.exp(-3)),
              "concave": lambda x: np.sqrt(np.maximum(1 - x, 0))}
    rows = []
    for name, curve in curves.items():
        for agents in teams:
            lengths = grid_equilibrium(alpha, agents, curve, steps, quadrature)
            optimum = numerical_planner(alpha, agents, curve, restarts, seed, quadrature)
            value = numerical_welfare(lengths, alpha, curve, quadrature)
            rows.append({"curve": name, "agents": agents, "lengths": lengths.tolist(),
                         "equilibrium_welfare": value, "planner": optimum,
                         "efficiency": value / optimum["welfare"]})
    return {"kind": "numerical_verification", "alpha": alpha, "rows": rows}
