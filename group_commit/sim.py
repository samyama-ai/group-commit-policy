"""Discrete-event simulator for group-commit release policies, open- and closed-loop.

Policies (release rules), one flush device that serializes flushes:
- 'greedy'    : flush instant device free and >=1 queued (parameter-free, self-clocking)
- 'size'      : flush when >= K queued and device free
- 'timer'     : PostgreSQL commit_delay-style: first committer waits T, then the batch flushes
- 'ski'       : flush when accumulated queued wait sum(now - submit) >= F0 (rent-or-buy)
- 'percommit' : == size(K=1) (no batching)

Arrivals: 'open' = exogenous Poisson(lam); 'closed' = N clients each blocking until their flush completes.
Termination is by COMMITTED-txn count (max_commits after warmup_commits): bounded, predictable runtime.
All metrics are measured over the post-warmup window.
"""
from __future__ import annotations

import heapq
from collections import deque

import numpy as np

from .device import FlushModel


def simulate(policy, params, flush: FlushModel, *, mode="closed", N=16, Z=1.0,
             lam=None, max_commits=20000, warmup_commits=2000, seed=0, burst=None) -> dict:
    rng = np.random.default_rng(seed)
    cap = 1 if policy == "percommit" else (params.get("K") if policy == "size" else None)
    F0 = params.get("F0", flush.F0)
    Tparam = params.get("T")
    Ksize = params.get("K")

    heap = []
    seq = 0

    def push(t, kind, payload=None):
        nonlocal seq
        heapq.heappush(heap, (t, seq, kind, payload)); seq += 1

    def think():
        return rng.exponential(Z) if Z > 0 else 0.0

    if mode == "open":
        assert lam is not None
        push(rng.exponential(1.0 / lam), "arrival_open", None)
    else:
        for c in range(N):
            push(think(), "arrival", c)

    q_times = deque(); q_clients = deque(); q_sum = 0.0
    device_busy = False
    timer_armed = False

    # post-warmup counters
    counting = False
    t_count_start = None
    latencies = []
    flushed_total = 0          # cumulative flushes (warmup gate)
    batched_total = 0          # cumulative batched txns (warmup gate)
    c_flushes = 0; c_batch = 0; c_busy = 0.0; completed = 0
    last_time = 0.0

    def burst_scale(t):
        if not burst:
            return 1.0
        period, high_mult, duty = burst
        return high_mult if (t % period) / period < duty else 1.0

    def should_flush(now):
        if not q_times:
            return False
        if policy in ("greedy", "percommit"):
            return True
        if policy == "size":
            return len(q_times) >= Ksize
        if policy == "ski":
            return (len(q_times) * now - q_sum) >= F0
        return False

    def start_flush(now):
        nonlocal device_busy, flushed_total, batched_total, q_sum, c_flushes, c_batch
        k = len(q_times) if cap is None else min(cap, len(q_times))
        bt = [q_times.popleft() for _ in range(k)]
        bc = [q_clients.popleft() for _ in range(k)]
        q_sum -= sum(bt)
        device_busy = True
        flushed_total += 1; batched_total += k
        if counting:
            c_flushes += 1; c_batch += k
        push(now + flush.duration(k), "flush_done", (now, bt, bc))

    def maybe_flush(now):
        nonlocal timer_armed
        if device_busy or not q_times:
            return
        if policy == "timer":
            if not timer_armed:
                timer_armed = True
                push(now + Tparam, "timer_fire", None)
        elif should_flush(now):
            start_flush(now)

    while heap:
        now, _, kind, payload = heapq.heappop(heap)
        last_time = now
        if kind == "arrival_open":
            q_times.append(now); q_clients.append(None); q_sum += now
            push(now + rng.exponential(1.0 / lam), "arrival_open", None)
            maybe_flush(now)
        elif kind == "arrival":
            q_times.append(now); q_clients.append(payload); q_sum += now
            maybe_flush(now)
        elif kind == "timer_fire":
            timer_armed = False
            if not device_busy and q_times:
                start_flush(now)
        elif kind == "flush_done":
            device_busy = False
            fstart, bt, bc = payload
            if counting:
                c_busy += (now - fstart)
                for st in bt:
                    latencies.append(now - st); completed += 1
            if mode == "closed":
                for cid in bc:
                    push(now + think() / burst_scale(now), "arrival", cid)
            maybe_flush(now)
            if policy == "timer" and not device_busy and q_times:
                start_flush(now)

        if not counting and batched_total >= warmup_commits:
            counting = True; t_count_start = now
        if counting and completed >= max_commits:
            break

    if completed == 0:
        return {"throughput": 0.0, "mean_latency": float("nan"), "p99_latency": float("nan"),
                "mean_batch": float("nan"), "per_txn_overhead": float("nan"), "n_flushes": 0,
                "device_util": 0.0, "completed": 0}
    lat = np.array(latencies)
    span = max(1e-9, last_time - t_count_start)
    per_txn = F0 * c_flushes / completed + float(lat.mean())
    return {
        "throughput": completed / span,
        "mean_latency": float(lat.mean()),
        "p99_latency": float(np.percentile(lat, 99)),
        "mean_batch": c_batch / max(1, c_flushes),
        "per_txn_overhead": float(per_txn),
        "n_flushes": int(c_flushes),
        "device_util": c_busy / span,
        "completed": int(completed),
    }
