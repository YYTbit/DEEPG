import numpy as np
import pytest

from deepg.theory import (AttentionProfile, attended_information, calibration_solution, calibration_threshold,
                          direct_externalities, equilibrium_lengths, equilibrium_welfare, infer_token_cost,
                          occupation_fraction, optimal_uniform_price, planner_lengths, planner_welfare,
                          slot_prices, symmetric_rotation, welfare)


@pytest.mark.parametrize("alpha", [0.02, 0.08, 0.35, 1.0])
@pytest.mark.parametrize("n", [1, 2, 3, 9])
def test_closed_forms_and_planner_implementation(alpha, n):
    equilibrium = equilibrium_lengths(alpha, n)
    planner = planner_lengths(alpha, n)
    assert equilibrium.sum() <= 1
    assert planner.sum() <= 1 + 1e-12
    assert welfare(equilibrium, alpha) == pytest.approx(equilibrium_welfare(alpha, n), abs=1e-12)
    assert welfare(planner, alpha) == pytest.approx(planner_welfare(alpha, n), abs=1e-12)
    np.testing.assert_allclose(attended_information(planner), np.full(n, attended_information(planner)[0]), atol=1e-12)
    np.testing.assert_allclose(equilibrium_lengths(alpha, n, slot_prices(alpha, n)), planner, atol=1e-12)
    if n > 1:
        assert equilibrium_welfare(alpha, n) < planner_welfare(alpha, n)


def test_price_schedule_and_calibration_threshold():
    prices = slot_prices(0.08, 3)
    np.testing.assert_allclose(prices, [np.sqrt(1.0064) - 0.08, np.sqrt(0.5064) - 0.08, 0])
    assert np.all(np.diff(prices) < 0)
    assert calibration_threshold(0.08, 1) == pytest.approx(0.88027, abs=1e-4)
    threshold = calibration_threshold(0.08, 1)
    lower = calibration_solution(0.08, 3, threshold + 1e-5)
    upper = calibration_solution(0.08, 3, threshold + 2e-5)
    assert upper["efficiency"] > lower["efficiency"]


def test_uniform_price_counts_transfers_correctly():
    alpha, n = 0.08, 3
    price = optimal_uniform_price(alpha, n)
    lengths = equilibrium_lengths(alpha, n, np.full(n, price))
    private_welfare = np.sqrt(attended_information(lengths)).sum() - (alpha + price) * lengths.sum()
    assert welfare(lengths, alpha) == pytest.approx(private_welfare + price * lengths.sum())
    assert welfare(lengths, alpha) > equilibrium_welfare(alpha, n)
    assert welfare(lengths, alpha) < planner_welfare(alpha, n)
    assert welfare(lengths, alpha) >= welfare(equilibrium_lengths(alpha, n, np.full(n, price * 0.99)), alpha)
    assert welfare(lengths, alpha) >= welfare(equilibrium_lengths(alpha, n, np.full(n, price * 1.01)), alpha)


def test_externality_matches_finite_difference():
    lengths = np.array([0.2, 0.15, 0.1])
    externalities = direct_externalities(lengths)
    delta = 1e-7
    for i in range(3):
        perturbed = lengths.copy()
        perturbed[i] += delta
        before = np.sqrt(attended_information(lengths))[i + 1:].sum()
        after = np.sqrt(attended_information(perturbed))[i + 1:].sum()
        assert (after - before) / delta == pytest.approx(externalities[i], abs=1e-7)


def test_general_profile_toll_implements_discrete_planner():
    profile = AttentionProfile([0, 0.5, 1], [1, 0.1, 0.8])
    solution = profile.solve_planner(0.08, 3, grid=51)
    cumulative = profile.integral(solution.positions)
    for k in range(1, 4):
        for start in range(51):
            lengths = solution.positions[start:] - solution.positions[start]
            rewards = np.sqrt(np.maximum(cumulative[start:] - cumulative[start], 0)) - 0.08 * lengths
            tolls = solution.values[k - 1, start] - solution.values[k - 1, start:]
            assert int(np.argmax(rewards - tolls)) == solution.choices[k, start]
    assert sum(solution.allocation()["lengths"]) <= 1


def test_zero_slope_and_single_slot_rotation():
    calibration = calibration_solution(0.08, 3, 0)
    assert calibration["lengths"].sum() == pytest.approx(1)
    assert calibration["planner"].sum() == pytest.approx(1)
    np.testing.assert_allclose(symmetric_rotation(0.08, 1)["lengths"], equilibrium_lengths(0.08, 1))


def test_cost_inversion():
    for alpha in [0.02, 0.08, 0.35, 1]:
        assert infer_token_cost(float(occupation_fraction(alpha))) == pytest.approx(alpha)


@pytest.mark.parametrize("alpha,n", [(0, 3), (-1, 3), (np.nan, 3), (0.08, 0)])
def test_invalid_games(alpha, n):
    with pytest.raises(ValueError):
        equilibrium_lengths(alpha, n)
