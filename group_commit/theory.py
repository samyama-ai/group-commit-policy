"""Closed-form theory for group-commit release policies.

Pre-registered in dbms_cloud/daily/06-optimal-group-commit-policy/HYPOTHESIS.md (frozen 2026-06-16).

Two lenses (kept distinct on purpose):
- OPEN-LOOP: exogenous Poisson arrivals at rate lambda. Per-txn controllable overhead of a periodic
  timer is  f(T) = F0/(lambda*T) + T/2,  minimized at the EOQ sqrt-rule  T* = sqrt(2 F0 / lambda).
- CLOSED-LOOP: N clients each block until their commit's flush completes (self-clocking). The greedy
  pipelined policy reaches a fixed point K*(N, F0, delta, Z) computed here.

Flush cost model: a flush of batch size K takes  F(K) = F0 + K*delta  (fixed barrier + per-record write).
The cost objective's *fixed* part is F0 (the amortizable cost); delta*n is paid regardless of policy.
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------- open-loop sqrt-rule
def sqrt_rule_T(F0: float, lam: float) -> float:
    """EOQ sqrt-rule optimal timer period for open-loop Poisson arrivals: T* = sqrt(2 F0 / lambda)."""
    return float(np.sqrt(2.0 * F0 / lam))


def openloop_overhead(T, F0: float, lam: float):
    """Per-txn controllable overhead of a periodic timer T under Poisson(lambda): F0/(lambda T) + T/2.
    Accepts scalar or array T."""
    T = np.asarray(T, dtype=float)
    out = F0 / (lam * T) + T / 2.0
    return float(out) if out.ndim == 0 else out


def openloop_overhead_opt(F0: float, lam: float) -> float:
    """Minimum of openloop_overhead, attained at the sqrt-rule: sqrt(2 F0 / lambda)."""
    return float(np.sqrt(2.0 * F0 / lam))


# ----------------------------------------------------------------- closed-loop fixed point
def closed_loop_fixed_point(N: int, F0: float, delta: float, Z: float,
                            rho: float = 1.5, iters: int = 200) -> dict:
    """Self-clocking fixed point of greedy-pipelined under N closed-loop clients.

    Solve the coupled system (mean-field):
        F(K) = F0 + K*delta            (flush duration for batch K)
        L    = rho * F(K)              (mean commit latency ~ residual + own flush)
        X    = N / (Z + L)             (Little's law on the closed loop)
        K    = X * F(K)                (batch = arrivals during one flush; device back-to-back)
    Returns dict(K, X, L, F). Fixed-point iteration on K.
    """
    K = max(1.0, N / 2.0)
    for _ in range(iters):
        F = F0 + K * delta
        L = rho * F
        X = N / (Z + L)
        K_new = X * F
        K_new = min(max(K_new, 1.0), float(N))      # batch in [1, N]
        if abs(K_new - K) < 1e-9:
            K = K_new
            break
        K = 0.5 * K + 0.5 * K_new                    # damped
    F = F0 + K * delta
    L = rho * F
    X = N / (Z + L)
    return {"K": float(K), "X": float(X), "L": float(L), "F": float(F),
            "flush_rate": float(X / K) if K > 0 else 0.0,
            "throughput_bound": float(1.0 / delta) if delta > 0 else float("inf")}


# ----------------------------------------------------------------- offline optimum (dynamic-ack DP)
def offline_opt(submit_times: np.ndarray, F0: float) -> dict:
    """Offline optimum for the dynamic-acknowledgment objective on a sorted arrival trace.

    Partition the n sorted submit times into consecutive batches; a batch {i..j} is flushed at t_j
    (flushing later only adds delay). Cost = F0 per batch + sum of delays (t_j - t_k). Standard O(n^2) DP:
        OPT(j) = min_{i<=j} OPT(i-1) + F0 + sum_{k=i..j}(t_j - t_k).
    Returns dict(cost, per_txn, n_flushes). 'cost' excludes the constant delta*n (policy-independent)."""
    t = np.asarray(submit_times, dtype=float)
    t.sort()
    n = t.size
    if n == 0:
        return {"cost": 0.0, "per_txn": 0.0, "n_flushes": 0}
    pre = np.concatenate([[0.0], np.cumsum(t)])      # pre[k] = sum of first k times
    dp = np.empty(n + 1); dp[0] = 0.0
    idx = np.arange(n + 1, dtype=float)              # idx[i] = i (for the (j-i+1) term)
    for j in range(1, n + 1):
        tj = t[j - 1]
        # delay(i,j) = tj*(j-i+1) - (pre[j]-pre[i-1]),  i=1..j  -> vectorised over i
        ii = np.arange(1, j + 1)
        delay = tj * (j - ii + 1) - (pre[j] - pre[ii - 1])
        dp[j] = float(np.min(dp[ii - 1] + F0 + delay))
    return {"cost": float(dp[n]), "per_txn": float(dp[n] / n), "n_flushes": -1}


def online_cost_abstraction(submit_times, policy: str, params: dict, F0: float) -> dict:
    """Online release-rule cost in the dynamic-ACK abstraction: flushes are INSTANTANEOUS (cost F0
    each, no device serialization). This is the world where the sqrt-rule and the ski-rental 2-
    competitive bound live; 'greedy' degenerates here (flush-per-arrival) and is excluded.

    timer(T): flush at first-waiter + T.  size(K): flush when K queued.  ski: flush when accumulated
    queued wait >= F0.  Cost = F0*#flushes + sum(flush_time - submit). Returns per_txn."""
    t = np.sort(np.asarray(submit_times, dtype=float))
    n = t.size
    if n == 0:
        return {"per_txn": 0.0, "n_flushes": 0, "cost": 0.0}
    pre = np.concatenate([[0.0], np.cumsum(t)])      # pre[k] = sum of t[0..k-1]
    T = params.get("T"); K = params.get("K")
    cost = 0.0; nf = 0
    i = 0
    while i < n:
        if policy == "timer":
            j = i + int(np.searchsorted(t, t[i] + T, side="right") - i)  # last index with t<=t[i]+T
            j = max(j, i + 1)
        elif policy == "size":
            j = min(i + K, n)
        elif policy == "ski":
            # smallest j>=i with wait(i..j) = t[j]*(j-i+1) - (pre[j+1]-pre[i]) >= F0; close at j (incl.)
            j = i
            while j < n:
                if t[j] * (j - i + 1) - (pre[j + 1] - pre[i]) >= F0:
                    break
                j += 1
            j = min(j, n - 1) + 1
        else:
            raise ValueError("greedy degenerates in the instantaneous abstraction; use the DES")
        ft = t[j - 1]
        cost += F0 + (ft * (j - i) - (pre[j] - pre[i]))
        nf += 1; i = j
    return {"per_txn": cost / n, "n_flushes": nf, "cost": float(cost)}


def offline_opt_bruteforce(submit_times: np.ndarray, F0: float) -> float:
    """Brute-force optimum over all 2^(n-1) batch partitions; for verifying offline_opt on tiny traces."""
    t = np.sort(np.asarray(submit_times, dtype=float))
    n = t.size
    if n == 0:
        return 0.0
    best = float("inf")
    for mask in range(1 << (n - 1)):                 # cut points between consecutive items
        cuts = [0] + [k + 1 for k in range(n - 1) if mask & (1 << k)] + [n]
        cost = 0.0
        for a, b in zip(cuts[:-1], cuts[1:]):
            seg = t[a:b]
            cost += F0 + float(np.sum(seg[-1] - seg))
        best = min(best, cost)
    return best
