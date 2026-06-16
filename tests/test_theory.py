"""Correctness tests for closed forms (test layers 2 + 5)."""
import numpy as np
import pytest

from group_commit import theory as T


def test_sqrt_rule_and_overhead_minimum():
    F0, lam = 1.0, 2000.0
    Tstar = T.sqrt_rule_T(F0, lam)
    assert Tstar == pytest.approx(np.sqrt(2 * F0 / lam))
    # numeric minimum of f(T)=F0/(lam T)+T/2 is at Tstar, value sqrt(2 F0/lam)
    grid = np.linspace(Tstar * 0.2, Tstar * 5, 4000)
    vals = T.openloop_overhead(grid, F0, lam)
    assert grid[np.argmin(vals)] == pytest.approx(Tstar, rel=0.02)
    assert vals.min() == pytest.approx(T.openloop_overhead_opt(F0, lam), rel=1e-3)


def test_offline_opt_matches_bruteforce():
    rng = np.random.default_rng(0)
    for _ in range(40):
        n = int(rng.integers(1, 11))
        t = np.sort(rng.uniform(0, 5, n))
        F0 = float(rng.uniform(0.1, 2.0))
        dp = T.offline_opt(t, F0)["cost"]
        bf = T.offline_opt_bruteforce(t, F0)
        assert dp == pytest.approx(bf, rel=1e-9)


def test_offline_opt_single_and_zero_cost():
    # one arrival -> one flush, no delay -> cost = F0
    assert T.offline_opt(np.array([3.0]), 1.0)["cost"] == pytest.approx(1.0)
    # F0=0 -> flush each at its own arrival, zero cost
    assert T.offline_opt(np.array([1.0, 2.0, 3.0]), 0.0)["cost"] == pytest.approx(0.0)


def test_closed_loop_fixed_point_monotone():
    # batch grows with N; throughput positive; batch capped at N
    fp_small = T.closed_loop_fixed_point(N=4, F0=1.0, delta=0.01, Z=0.5)
    fp_big = T.closed_loop_fixed_point(N=64, F0=1.0, delta=0.01, Z=0.5)
    assert fp_big["K"] > fp_small["K"]
    assert fp_big["K"] <= 64 + 1e-6
    assert fp_small["X"] > 0 and fp_big["X"] > fp_small["X"]


def test_closed_loop_throughput_bound():
    # with delta>0 throughput is bounded by 1/delta (write-bandwidth bound)
    fp = T.closed_loop_fixed_point(N=512, F0=1.0, delta=0.05, Z=0.0)
    assert fp["X"] <= 1.0 / 0.05 + 1e-6
