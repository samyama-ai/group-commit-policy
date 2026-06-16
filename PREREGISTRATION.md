# Pre-registration (frozen 2026-06-16, before real-data runs)

Hypotheses, decision rules, and statistics were fixed before measuring real fsync or running PostgreSQL.
The data **reframed H3/H4** (see below) — disclosed transparently here and in RESULTS.

## Model (frozen)
- Flush cost `F(K) = F₀ + Kδ` (fixed barrier + per-record). Cost objective (dynamic-ack form):
  `C = F₀·#flushes + Σ delays`, per-txn `c̄ = C/n`. Open-loop optimum: `T* = √(2F₀/λ)` (EOQ √-rule).
- Closed-loop: N clients, think `Z`, each blocks until its commit flushes. Greedy fixed point
  `K* = N·F/(Z+L)`, `L ≈ ρF`. Threshold `λ* = 2/F₀` (where `T* = F₀`).

## Hypotheses & decision rules (as frozen)
- **H0 (gate):** √-rule minimizes timer overhead (±15%); ski-rental ≤ 2× offline DP.
- **H1:** greedy batch/throughput match the closed-loop fixed point (K MAPE ≤12%, X MAPE ≤15%).
- **H2 (core):** greedy within ≤1.25× of the best-tuned policy with no λ/tuning.
- **H3 (frozen, NOT supported as stated):** open-loop √-rule timer fed the measured closed-loop λ ≥1.3×
  worse than greedy. → **Reframed:** the √-rule timer *collapses onto* greedy above λ*=2/F₀ (ratio 1.000);
  it is not worse, it is *identical* — tuning is vacuous above λ*. Honest post-data reframing.
- **H4 (frozen, NOT supported as stated):** fixed timer p99 inflates ≥1.5× under bursts, greedy ≤1.2×.
  → **Reframed:** closed-loop bounded concurrency makes bursts harmless to both; the regime where policy
  matters is *open-loop with rate shifts*, where parameter-free ski beats a fixed tuned timer.
- **H5 (real, AWS):** measured fsync sets λ* per device; PostgreSQL commit_delay=0 competitive with best.
- **Negative controls:** NC1 (N=1 ⇒ batch 1), NC3 (F₀=0 ⇒ batch 1, overhead 0); plus the size-K-deadlocks-
  below-K finding.

## Statistics
≥5 seeds per sim point; bootstrap for CIs; offline DP verified vs brute force on n≤12; fixed seeds; one
command regenerates every number. Reframings are disclosed, not hidden.
