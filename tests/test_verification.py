import numpy as np

from deepg.theory import equilibrium_lengths, planner_welfare
from deepg.verification import (attention_ablation, grid_equilibrium, integrated_attention,
                                numerical_planner, paper_numbers)


def test_independent_grid_and_quadrature():
    np.testing.assert_allclose(integrated_attention([0, 0.2], [0.3, 0.4], lambda x: 1 - x), [0.255, 0.24])
    lengths = grid_equilibrium(0.08, 3, steps=500, quadrature=20)
    np.testing.assert_allclose(lengths, equilibrium_lengths(0.08, 3), atol=0.003)
    assert lengths.sum() <= 1


def test_derivative_free_planner_respects_capacity():
    result = numerical_planner(0.08, 2, restarts=2, quadrature=20)
    assert sum(result["lengths"]) <= 1
    assert abs(result["welfare"] - planner_welfare(0.08, 2)) < 1e-9


def test_each_attention_curve_is_solved_independently():
    rows = attention_ablation(teams=(2,), steps=50, quadrature=30, restarts=1)["rows"]
    assert len(rows) == 4
    assert len({tuple(row["lengths"]) for row in rows}) > 1
    assert all(0 < row["efficiency"] <= 1 + 1e-9 for row in rows)
    assert paper_numbers()["kind"] == "analytical"
