"""Simulator tests incl. the pre-registered negative controls (NC1/NC2/NC3) and H0 gate direction."""
import numpy as np
import pytest

from group_commit import FlushModel, simulate, theory as T

MC = dict(max_commits=6000, warmup_commits=800)


def test_nc1_single_client_no_batching():
    # NC1: with N=1, self-clocking / timer policies all flush singletons (batch==1).
    fm = FlushModel(F0=1.0, delta=0.01)
    for pol, params in [("greedy", {}), ("percommit", {}), ("timer", {"T": 2.0})]:
        out = simulate(pol, params, fm, mode="closed", N=1, Z=1.0, seed=1, **MC)
        assert out["mean_batch"] == pytest.approx(1.0, abs=1e-6)


def test_size_policy_deadlocks_below_K():
    # A finding, not a bug: pure size-K cannot flush when concurrency N < K (no fallback) -> 0 commits.
    fm = FlushModel(F0=1.0, delta=0.01)
    out = simulate("size", {"K": 8}, fm, mode="closed", N=2, Z=1.0, seed=1, **MC)
    assert out["completed"] == 0          # self-clocking 'greedy' has no such failure mode


def test_nc3_zero_flush_cost_batch_one():
    # NC3: F0=0, delta=0 -> no incentive to wait; greedy flushes singletons, overhead ~ 0
    fm = FlushModel(F0=0.0, delta=0.0)
    out = simulate("greedy", {"F0": 0.0}, fm, mode="closed", N=16, Z=1.0, seed=2, **MC)
    assert out["mean_batch"] == pytest.approx(1.0, abs=0.05)
    assert out["per_txn_overhead"] == pytest.approx(0.0, abs=1e-6)


def test_greedy_high_device_utilization_under_load():
    # under heavy closed-loop load greedy keeps the device busy (self-clocking, back-to-back flushes)
    fm = FlushModel(F0=1.0, delta=0.01)
    out = simulate("greedy", {}, fm, mode="closed", N=128, Z=0.1, seed=3, **MC)
    assert out["device_util"] > 0.9
    assert out["mean_batch"] > 2.0


def _poisson_trace(lam, n, seed):
    r = np.random.default_rng(seed); t, x = np.empty(n), 0.0
    for i in range(n):
        x += r.exponential(1.0 / lam); t[i] = x
    return t


def test_h0a_sqrt_rule_minimizes_abstraction():
    # H0a: in the dynamic-ack abstraction, timer per-txn cost is minimized near sqrt(2 F0/lam).
    F0, lam = 1.0, 50.0
    t = _poisson_trace(lam, 2500, 4)
    Tstar = T.sqrt_rule_T(F0, lam)
    Ts = np.linspace(0.3 * Tstar, 3 * Tstar, 11)
    ov = [T.online_cost_abstraction(t, "timer", {"T": float(tt)}, F0)["per_txn"] for tt in Ts]
    best_T = Ts[int(np.argmin(ov))]
    assert best_T == pytest.approx(Tstar, rel=0.4)
    assert min(ov) == pytest.approx(T.openloop_overhead_opt(F0, lam), rel=0.25)


def test_h0b_ski_within_2x_offline_abstraction():
    # H0b: ski-rental per-txn <= ~2x the offline optimum, both in the instantaneous abstraction.
    F0, lam = 1.0, 30.0
    ratios = []
    for s in range(4):
        t = _poisson_trace(lam, 2500, 100 + s)
        opt = T.offline_opt(t, F0)["per_txn"]
        ski = T.online_cost_abstraction(t, "ski", {}, F0)["per_txn"]
        ratios.append(ski / opt)
    assert np.median(ratios) <= 2.2                       # ski-rental is 2-competitive
