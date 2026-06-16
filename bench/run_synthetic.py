"""One-command synthetic run: H1 (closed-loop fixed point), threshold (parameter-free optimality + load
threshold), and the open-loop contrast (where policy choice matters).

    python bench/run_synthetic.py
Writes results/{H1,threshold,contrast}.json and figures/{fixedpoint,threshold,contrast}.png.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from group_commit import experiment as E   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)


def _save(name, obj):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=2)


def main():
    print("== H1: closed-loop self-clocking fixed point ==")
    h1 = E.run_H1(); _save("H1.json", h1)
    print(f"  K MAPE={h1['K_mape']*100:.1f}%  X MAPE={h1['X_mape']*100:.1f}%  maxutil={h1['max_util']:.2f}"
          f"  pass={h1['H1a_pass']}")

    print("== threshold: parameter-free greedy vs best-tuned timer across load ==")
    th = E.run_threshold(); _save("threshold.json", th)
    print(f"  greedy/best across ALL loads (closed-loop) = {th['greedy_vs_best_all']:.3f} "
          f"(parameter-free optimal; pass={th['H2_pass']})")
    print(f"  lambda* = 2/F0 = {th['lambda_star']:.2f}: sqrt-rule timer / greedy  "
          f"ABOVE={th['sqrt_collapse_above']:.3f} (collapses, pass={th['collapse_pass']})  "
          f"BELOW={th['sqrt_diverge_below']:.3f}")

    print("== open-loop contrast: fixed tuned timer vs ski under rate shifts ==")
    c = E.run_openloop_contrast(); _save("contrast.json", c)
    print(f"  fixed-timer/opt={c['fixed_timer_vs_opt']:.2f}  ski/opt={c['ski_vs_opt']:.2f}  "
          f"fixed/ski={c['fixed_over_ski']:.2f}  pass={c['H4_pass']}")

    _plot(h1, th, c)
    print(f"\nfigures -> {FIG}")


def _plot(h1, th, c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = h1["rows"]; N = [r["N"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot(N, [r["K_sim"] for r in rows], "o-", label="K simulated")
    ax[0].plot(N, [r["K_pred"] for r in rows], "r--", label="K* fixed-point")
    ax[0].set_xscale("log", base=2); ax[0].set_xlabel("N clients"); ax[0].set_ylabel("batch size")
    ax[0].set_title("Self-clocking fixed point"); ax[0].legend(fontsize=8)
    ax[1].plot(N, [r["X_sim"] for r in rows], "o-", label="X simulated")
    ax[1].plot(N, [r["X_pred"] for r in rows], "r--", label="X* fixed-point")
    ax[1].set_xscale("log", base=2); ax[1].set_xlabel("N clients"); ax[1].set_ylabel("throughput")
    ax[1].set_title("Throughput saturates device"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fixedpoint.png"), dpi=130); plt.close(fig)

    rows = th["rows"]
    lam = [r["lambda"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.4, 4))
    ax.plot(lam, [r["greedy_vs_best"] for r in rows], "o-", label="greedy / best-tuned timer")
    ax.axhline(1.0, color="gray", ls=":")
    ax.axvline(th["lambda_star"], color="red", ls="--", label=r"$\lambda^*=2/F_0$ (tuning vacuous above)")
    ax.set_xscale("log"); ax.set_xlabel("offered load  λ (commits/unit)")
    ax.set_ylabel("parameter-free overhead / best-tuned")
    ax.set_title("Load threshold: above λ*, greedy is optimal\n(tuning only helps at low load)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "threshold.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 4))
    bars = ["fixed timer\n(tuned for low rate)", "ski-rental\n(parameter-free)"]
    vals = [c["fixed_timer_vs_opt"], c["ski_vs_opt"]]
    ax.bar(bars, vals, color=["#c44", "#4a7"])
    ax.axhline(1.0, color="gray", ls=":"); ax.set_ylabel("per-txn cost / offline optimum")
    ax.set_title("Open-loop with rate shifts:\npolicy choice matters here")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "contrast.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
