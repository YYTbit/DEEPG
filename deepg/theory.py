import math

import numpy as np
from scipy.optimize import brentq, minimize_scalar


def validate_game(alpha, n):
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")


def retention_fraction(mu):
    mu = np.asarray(mu, dtype=float)
    if np.any(~np.isfinite(mu)) or np.any(mu <= 0):
        raise ValueError("private token costs must be finite and positive")
    return mu / np.hypot(mu, 1 / math.sqrt(2))


def occupation_fraction(mu):
    mu = np.asarray(mu, dtype=float)
    c = retention_fraction(mu)
    return 0.5 / (np.hypot(mu, 1 / math.sqrt(2)) ** 2 * (1 + c))


def infer_token_cost(share):
    if not 0 < share < 1:
        raise ValueError("the measured capacity share must be in (0, 1)")
    c = 1 - share
    return c / math.sqrt(2 * (1 - c * c))


def attended_information(lengths, slope=1.0):
    lengths = np.asarray(lengths, dtype=float)
    if lengths.ndim != 1 or np.any(~np.isfinite(lengths)):
        raise ValueError("lengths must be a finite vector")
    if np.any(lengths < 0) or lengths.sum() > 1 + 1e-12:
        raise ValueError("lengths exceed normalized capacity")
    if not 0 <= slope <= 1:
        raise ValueError("slope must be in [0, 1]")
    p = np.r_[0.0, np.cumsum(lengths)[:-1]]
    return np.maximum(lengths * (1 - slope * p - slope * lengths / 2), 0)


def welfare(lengths, alpha, slope=1.0):
    validate_game(alpha, len(lengths))
    return float(np.sqrt(attended_information(lengths, slope)).sum() - alpha * np.sum(lengths))


def equilibrium_lengths(alpha, n, prices=None):
    validate_game(alpha, n)
    if prices is None:
        prices = np.zeros(n)
    prices = np.asarray(prices, dtype=float)
    if prices.shape != (n,) or np.any(~np.isfinite(prices)):
        raise ValueError("one finite price is required for every slot")
    mu = alpha + prices
    fractions = occupation_fraction(mu)
    capacity = np.r_[1.0, np.cumprod(retention_fraction(mu))[:-1]]
    return fractions * capacity


def equilibrium_welfare(alpha, n):
    validate_game(alpha, n)
    c = float(retention_fraction(alpha))
    return -math.expm1(n * math.log(c)) * math.hypot(alpha, 1 / math.sqrt(2))


def planner_welfare(alpha, n):
    if n == 0:
        return 0.0
    validate_game(alpha, n)
    return (n / 2) / (math.hypot(alpha, math.sqrt(n / 2)) + alpha)


def planner_lengths(alpha, n):
    validate_game(alpha, n)
    residual = alpha / math.hypot(alpha, math.sqrt(n / 2))
    attended = (1 - residual * residual) / (2 * n)
    remaining = np.sqrt(np.maximum(1 - 2 * attended * np.arange(n + 1), 0))
    return remaining[:-1] - remaining[1:]


def slot_prices(alpha, n):
    validate_game(alpha, n)
    return np.array([planner_welfare(alpha, n - i - 1) for i in range(n)])


def optimal_uniform_price(alpha, n):
    validate_game(alpha, n)
    if n == 1:
        return 0.0
    def objective(p):
        return welfare(equilibrium_lengths(alpha, n, np.full(n, p)), alpha)
    high = max(1.0, alpha)
    while objective(high * 2) > objective(high):
        high *= 2
    result = minimize_scalar(lambda p: -objective(p), bounds=(0.0, high * 2),
                             method="bounded", options={"xatol": 1e-12})
    return float(result.x)


def direct_externalities(lengths, slope=1.0):
    attended = attended_information(lengths, slope)
    terms = np.zeros(len(lengths))
    active = attended > 0
    terms[active] = -slope * np.asarray(lengths)[active] / (2 * np.sqrt(attended[active]))
    return np.array([terms[i + 1:].sum() for i in range(len(lengths))])


def calibration_solution(alpha, n, slope):
    validate_game(alpha, n)
    if not 0 <= slope <= 1:
        raise ValueError("slope must be in [0, 1]")
    if slope == 0:
        capacity = 1.0
        lengths = []
        for i in range(n):
            length = min(capacity, 1 / (4 * alpha * alpha))
            lengths.append(length)
            capacity -= length
        total = min(1.0, n / (4 * alpha * alpha))
        planner = np.full(n, total / n)
    else:
        r = alpha / math.sqrt(alpha * alpha + slope / 2)
        q = np.maximum(1 - slope, r ** np.arange(n + 1))
        lengths = (q[:-1] - q[1:]) / slope
        residual = max(1 - slope, alpha / math.sqrt(alpha * alpha + n * slope / 2))
        attended = (1 - residual * residual) / (2 * slope * n)
        qstar = np.sqrt(np.maximum(1 - 2 * slope * attended * np.arange(n + 1), 0))
        planner = (qstar[:-1] - qstar[1:]) / slope
    w = welfare(lengths, alpha, slope)
    wstar = welfare(planner, alpha, slope)
    return {"lengths": np.asarray(lengths), "planner": planner,
            "welfare": w, "planner_welfare": wstar, "efficiency": w / wstar}


def calibration_threshold(alpha, k):
    validate_game(alpha, k)
    if k <= 4 * alpha * alpha:
        raise ValueError("a nonzero threshold requires k > 4 alpha^2")
    def residual(b):
        return k * math.log(alpha / math.sqrt(alpha * alpha + b / 2)) - math.log1p(-b)
    return brentq(residual, 1e-9, 1 - 1e-12, xtol=1e-14)


def symmetric_rotation(alpha, n, price=0.0):
    validate_game(alpha, n)
    mu = alpha + price
    validate_game(mu, n)
    def marginal_return(length):
        starts = np.arange(n) * length
        attended = length * (1 - starts - length / 2)
        return float(np.mean((1 - starts - length) / (2 * np.sqrt(attended))) - mu)
    maximum = 1 / n
    length = maximum if marginal_return(maximum) >= 0 else brentq(marginal_return, 1e-14, maximum)
    lengths = np.full(n, length)
    return {"lengths": lengths, "welfare": welfare(lengths, alpha)}


class AttentionProfile:
    def __init__(self, positions, values):
        self.positions = np.asarray(positions, dtype=float)
        self.weights = np.asarray(values, dtype=float)
        if self.positions.ndim != 1 or len(self.positions) < 2 or self.weights.shape != self.positions.shape:
            raise ValueError("an attention profile requires matching position and value vectors")
        if np.any(~np.isfinite(self.positions)) or np.any(~np.isfinite(self.weights)):
            raise ValueError("attention profile entries must be finite")
        if self.positions[0] != 0 or self.positions[-1] != 1 or np.any(np.diff(self.positions) <= 0):
            raise ValueError("profile positions must increase from 0 to 1")
        if np.any(self.weights < 0):
            raise ValueError("attention values must be nonnegative")
        area = np.diff(self.positions) * (self.weights[:-1] + self.weights[1:]) / 2
        self.cumulative = np.r_[0.0, np.cumsum(area)]

    def integral(self, position):
        x = np.asarray(position, dtype=float)
        if np.any((x < 0) | (x > 1)):
            raise ValueError("positions must be in [0, 1]")
        i = np.clip(np.searchsorted(self.positions, x, side="right") - 1, 0, len(self.positions) - 2)
        delta = x - self.positions[i]
        slope = (self.weights[i + 1] - self.weights[i]) / (self.positions[i + 1] - self.positions[i])
        return self.cumulative[i] + self.weights[i] * delta + slope * delta * delta / 2

    def attended_information(self, position, length):
        return np.maximum(self.integral(np.asarray(position) + length) - self.integral(position), 0)

    def solve_planner(self, alpha, n, grid=201):
        validate_game(alpha, n)
        if grid < 2:
            raise ValueError("grid must contain at least two positions")
        points = np.linspace(0, 1, grid)
        cumulative = self.integral(points)
        values = np.zeros((n + 1, grid))
        choices = np.zeros((n + 1, grid), dtype=int)
        for k in range(1, n + 1):
            for i in range(grid):
                rewards = np.sqrt(np.maximum(cumulative[i:] - cumulative[i], 0)) - alpha * (points[i:] - points[i])
                objective = rewards + values[k - 1, i:]
                choices[k, i] = int(np.argmax(objective))
                values[k, i] = float(objective[choices[k, i]])
        return PlannerSolution(self, alpha, n, points, values, choices)


class PlannerSolution:
    def __init__(self, profile, alpha, n, points, values, choices):
        self.profile = profile
        self.alpha = alpha
        self.n = n
        self.positions = points
        self.values = values
        self.choices = choices

    def toll(self, k, start, end):
        if not 1 <= k <= self.n or not 0 <= start <= end < len(self.positions):
            raise ValueError("invalid continuation state")
        return float(self.values[k - 1, start] - self.values[k - 1, end])

    def allocation(self, start=0):
        if not 0 <= start < len(self.positions):
            raise ValueError("invalid initial position")
        lengths, tolls = [], []
        for k in range(self.n, 0, -1):
            end = start + int(self.choices[k, start])
            lengths.append(float(self.positions[end] - self.positions[start]))
            tolls.append(self.toll(k, start, end))
            start = end
        return {"lengths": lengths, "tolls": tolls}
