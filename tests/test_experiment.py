"""Smoke + gate tests for the experiment runners (reduced sizes for speed)."""
from group_commit import experiment as E


def test_H1_fixed_point():
    h1 = E.run_H1(Ns=(4, 16, 64, 256), reps=4)
    assert h1["H1a_pass"]
    assert h1["max_util"] > 0.95          # device saturates at high load


def test_threshold_parameter_free_optimal():
    th = E.run_threshold(Zs=[8.0, 2.0, 0.5, 0.1], reps=4)
    assert th["H2_pass"]                  # greedy ~ best-tuned across loads (median <= 1.05)
    assert th["collapse_pass"]            # sqrt-rule timer collapses onto greedy above lambda*


def test_openloop_contrast_policy_matters():
    c = E.run_openloop_contrast(reps=4, n=3600)
    # in open-loop with rate shifts, parameter-free ski is no worse than the fixed tuned timer
    assert c["ski_vs_opt"] <= c["fixed_timer_vs_opt"] + 0.05
    assert c["ski_vs_opt"] <= 2.2         # ski stays near the offline optimum
