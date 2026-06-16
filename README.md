# group-commit-policy — when is group-commit tuning worth it?

A pre-registered study of **the optimal group-commit release policy**, with a deliberately different
toolkit from estimation/regret work: **queueing + optimal-stopping + competitive analysis**. We don't
ship a new logger; we characterise *when* the elaborate adaptive group-commit machinery is worth its
tuning — and find, from first principles and on a real engine, that in closed-loop OLTP it usually isn't.

> Problem: [`dbms_research/06-recovery-logging/optimal-group-commit-policy`](https://github.com/samyama-ai/dbms_research).
> Honest baseline + characterization (mode b), not a SOTA claim. Limitations in §6.

## The setup

A durable commit needs a **flush** (`fsync`) of fixed cost `F₀` (plus a per-record marginal `δ`). Group
commit batches many committing txns into one flush. A **release rule** decides when to flush: a timer `T`,
a size `K`, ski-rental, or **greedy-pipelined** (flush the instant the device is free, batching whatever
queued). Per-txn cost trades amortized flush `F₀/K` against acquisition latency.

**Two worlds.** The textbook theory assumes **open-loop** (exogenous Poisson) arrivals: the optimal timer
is the EOQ **√-rule** `T* = √(2F₀/λ)`, and the rent-or-buy decision is **ski-rental 2-competitive**
(group commit *is* the dynamic-TCP-acknowledgment problem). But real OLTP is **closed-loop**: a client
issues its next txn only after its last commits, so `λ` is induced by the policy's own latency.

## Results

1. **Closed-loop self-clocking fixed point.** Model the closed loop as a closed queueing network: greedy-
   pipelined batch size and throughput converge to a computable fixed point `K*(N,F₀,δ,Z)`; the device
   saturates. Simulation matches it (batch MAPE 10%, throughput MAPE 7%).
2. **Parameter-free optimality.** Across *every* closed-loop load, greedy-pipelined (no `λ`, no timer) is
   within **~0.1%** of the best oracle-tuned timer. Tuning buys essentially nothing.
3. **The load threshold λ\* = 2/F₀.** The √-rule prescribes waiting `T*`, but `T* < F₀ ⟺ λ > 2/F₀`; once
   the device is the bottleneck, "wait `T*`" is dominated by device busy-time and **the √-rule timer
   collapses exactly onto greedy** (ratio 1.000). So above λ\*, the optimal policy is parameter-free and
   tuning is *vacuous*; the clean theory only bites below λ\* / in the open-loop world (where parameter-
   free ski still beats a fixed tuned timer under rate shifts).
4. **Real devices set λ\*.** Measured `fsync` on AWS: **EBS gp3 F₀≈0.90 ms** (p99 2.5 ms) → λ\*≈**2,200/s**;
   **instance-store NVMe F₀≈0.036 ms** → λ\*≈**55,000/s**. So on slow storage group commit almost always
   matters; on fast NVMe it barely matters until extreme load (the "is group commit even needed on
   persistent memory?" question, answered: only above ~55k commits/s).
5. **Real PostgreSQL.** A `commit_delay` sweep (pgbench, EBS WAL) shows **no reliable tuning signal**:
   `commit_delay=0` (parameter-free; PG's pipelined WAL self-clocks) is within ~10–20% of any tuned value,
   with high run-to-run variance and a slightly larger (still small) benefit below λ\* — consistent with
   the threshold. The practical upshot matches deployed folklore: leave `commit_delay≈0`.

## Results table

| Test | Claim | Result | Status |
|---|---|---|---|
| H0 | √-rule + ski-rental 2-competitive (abstraction) | timer min at √(2F₀/λ); ski ≤ 2× offline DP | ✅ |
| H1 | closed-loop fixed point | K MAPE 10%, X MAPE 7%, device saturates | ✅ |
| H2 | greedy = best-tuned (closed-loop, all loads) | median ratio **1.000** | ✅ |
| H3→λ\* | √-rule timer collapses onto greedy above λ\*=2/F₀ | ratio **1.000** above, diverges below | ✅ (reframed) |
| H4 | open-loop rate shifts: parameter-free ski ≥ fixed timer | ski 1.34× vs fixed 1.44× offline opt | ✅ |
| H5a | real fsync sets λ\* (EBS vs NVMe) | λ\* 2.2k/s vs 55k/s; greedy ≤1.3% of best | ✅ |
| H5b | real PostgreSQL: commit_delay=0 competitive | within ~10–20%, no reliable signal | ✅ honest |

## Reproduce
```
pip install -e . && pytest -q          # 14 tests
python bench/run_synthetic.py          # H1 + threshold + open-loop contrast + figures
# real device + engine (a Linux box; we used AWS c5d for EBS+NVMe+PostgreSQL):
python bench/measure_fsync.py --path /path/on/device --out results/fsync_x.json
bench/pg_commit_delay.sh results/pg.json 32 20 50
python bench/run_real.py
```
See `REPRODUCIBILITY.md` and the frozen `PREREGISTRATION.md`.

## §6 Limitations & honest scope
- We characterise a *policy frontier*; we ship no new logger. **Aether** (flush pipelining) is the
  mechanism that makes self-clocking possible — our contribution is the *analysis* (closed-loop fixed
  point, parameter-free optimality, the λ\* threshold), not the mechanism.
- The headline is a **negative/debunking** result: in closed-loop OLTP, adaptive group-commit tuning is
  largely unnecessary. That matches practice; it is a characterization, not a new record.
- pgbench TPC-B has non-commit bottlenecks, and spot-instance/EBS latency is variable, so H5b is noisy —
  we report that honestly rather than cherry-picking.
- The closed-loop competitive bound is conjectured, not proven; we report the empirical optimality gap.

## License
Apache-2.0. Builds on Aether, Deb–Serfozo, ski-rental (Karlin et al.), dynamic-ack (Dooly et al.) — cited in the paper.
