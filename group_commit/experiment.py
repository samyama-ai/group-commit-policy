"""Experiment runners for the pre-registered hypotheses (HYPOTHESIS.md, frozen 2026-06-16)."""
from __future__ import annotations

import numpy as np

from . import theory as T
from .device import FlushModel
from .sim import simulate

DEF_MC = dict(max_commits=12000, warmup_commits=1500)


def _avg(policy, params, fm, reps=8, **kw):
    """Mean of a metric dict over `reps` seeds (bootstrap-friendly)."""
    outs = [simulate(policy, params, fm, seed=s, **kw) for s in range(reps)]
    keys = [k for k, v in outs[0].items() if isinstance(v, (int, float))]
    return {k: float(np.mean([o[k] for o in outs])) for k in keys}, outs


# ----------------------------------------------------------------- H1: closed-loop fixed point
def run_H1(F0=1.0, delta=0.02, Z=0.5, Ns=(2, 4, 8, 16, 32, 64, 128, 256), reps=8):
    fm = FlushModel(F0=F0, delta=delta)
    rows = []
    for N in Ns:
        sim, _ = _avg("greedy", {}, fm, reps=reps, mode="closed", N=N, Z=Z, **DEF_MC)
        fp = T.closed_loop_fixed_point(N=N, F0=F0, delta=delta, Z=Z)
        rows.append({"N": N, "K_sim": sim["mean_batch"], "K_pred": fp["K"],
                     "X_sim": sim["throughput"], "X_pred": fp["X"],
                     "util": sim["device_util"]})
    K_err = np.array([abs(r["K_sim"] - r["K_pred"]) / r["K_pred"] for r in rows])
    X_err = np.array([abs(r["X_sim"] - r["X_pred"]) / r["X_pred"] for r in rows])
    return {"rows": rows, "K_mape": float(K_err.mean()), "X_mape": float(X_err.mean()),
            "H1a_pass": bool(K_err.mean() <= 0.12 and X_err.mean() <= 0.15),
            "max_util": float(max(r["util"] for r in rows))}


# --------------------------------------------- H2/threshold: parameter-free optimality + load threshold
def run_threshold(F0=1.0, delta=0.0, N=32, Zs=None, reps=8):
    """Sweep load (via think time Z) and show: greedy (parameter-free) vs the best ORACLE-tuned timer.

    Prediction: the open-loop sqrt-rule timer T*=sqrt(2 F0/lambda) is below F0 iff lambda > 2/F0; above
    that load threshold the timer collapses onto greedy and tuning is vacuous (ratio -> 1). Below it
    (low load), waiting helps and greedy over-flushes (ratio > 1)."""
    if Zs is None:
        Zs = [16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.2, 0.05]      # high Z = low load -> low Z = high load
    fm = FlushModel(F0=F0, delta=delta)
    lam_star = 2.0 / F0
    rows = []
    for Z in Zs:
        greedy, _ = _avg("greedy", {}, fm, reps=reps, mode="closed", N=N, Z=Z, **DEF_MC)
        lam = greedy["throughput"]
        T_open = T.sqrt_rule_T(F0, lam)
        topen, _ = _avg("timer", {"T": T_open, "F0": F0}, fm, reps=reps, mode="closed", N=N, Z=Z, **DEF_MC)
        best = None
        for Tt in np.geomspace(0.05 * F0, 20 * F0, 16):
            o, _ = _avg("timer", {"T": float(Tt), "F0": F0}, fm, reps=4, mode="closed", N=N, Z=Z, **DEF_MC)
            if best is None or o["per_txn_overhead"] < best["per_txn_overhead"]:
                best = o
        rows.append({"Z": Z, "lambda": lam, "lambda_over_star": lam / lam_star,
                     "T_sqrt": T_open, "Tsqrt_over_F0": T_open / F0,
                     "greedy_overhead": greedy["per_txn_overhead"],
                     "best_timer_overhead": best["per_txn_overhead"],
                     "open_sqrt_overhead": topen["per_txn_overhead"],
                     "greedy_vs_best": greedy["per_txn_overhead"] / best["per_txn_overhead"],
                     "sqrt_vs_greedy": topen["per_txn_overhead"] / greedy["per_txn_overhead"]})
    rows.sort(key=lambda r: r["lambda"])
    above = [r for r in rows if r["lambda"] >= lam_star]
    below = [r for r in rows if r["lambda"] < lam_star]
    gvb_all = float(np.median([r["greedy_vs_best"] for r in rows]))
    # the crisp exact claim: sqrt-rule timer collapses onto greedy (ratio->1) ABOVE the load threshold
    svg_above = float(np.median([r["sqrt_vs_greedy"] for r in above])) if above else float("nan")
    svg_below = float(np.median([r["sqrt_vs_greedy"] for r in below])) if below else float("nan")
    return {"rows": rows, "lambda_star": lam_star,
            "greedy_vs_best_all": gvb_all,
            "sqrt_collapse_above": svg_above, "sqrt_diverge_below": svg_below,
            "H2_pass": bool(gvb_all <= 1.05),
            "collapse_pass": bool(np.isnan(svg_above) or abs(svg_above - 1.0) <= 0.03)}


# ----------------------------------------------------- H4: the OTHER world -- open-loop rate shifts
def run_openloop_contrast(F0=1.0, lam_lo=5.0, lam_hi=80.0, n=6000, reps=10, seed=0):
    """In the open-loop dynamic-ACK abstraction (unbounded concurrency), policy choice DOES matter.
    A fixed timer tuned for the low rate is hit by a high-rate burst; ski-rental (parameter-free,
    threshold F0) adapts. Report each policy's per-txn vs the offline optimum on the same trace."""
    rng = np.random.default_rng(seed)
    T_lo = T.sqrt_rule_T(F0, lam_lo)            # timer tuned for the LOW rate (the naive fixed choice)
    fixed_r, ski_r = [], []
    for s in range(reps):
        r = rng if False else np.random.default_rng(seed + s)
        # trace: alternate low-rate and high-rate (burst) segments
        t, x = [], 0.0
        for seg in range(12):
            lam = lam_hi if seg % 2 else lam_lo
            for _ in range(n // 12):
                x += r.exponential(1.0 / lam); t.append(x)
        t = np.array(t)
        opt = T.offline_opt(t, F0)["per_txn"]
        fixed = T.online_cost_abstraction(t, "timer", {"T": T_lo}, F0)["per_txn"]
        ski = T.online_cost_abstraction(t, "ski", {}, F0)["per_txn"]
        fixed_r.append(fixed / opt); ski_r.append(ski / opt)
    fixed_r, ski_r = np.array(fixed_r), np.array(ski_r)
    return {"fixed_timer_vs_opt": float(np.median(fixed_r)), "ski_vs_opt": float(np.median(ski_r)),
            "fixed_over_ski": float(np.median(fixed_r) / np.median(ski_r)), "T_lo": T_lo,
            "H4_pass": bool(np.median(fixed_r) >= 1.3 and np.median(ski_r) <= 2.2)}
