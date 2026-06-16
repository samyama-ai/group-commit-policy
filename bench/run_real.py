"""H5: real-device validation. Plug the MEASURED fsync distributions (EBS gp3, instance NVMe) into the
simulator and confirm the policy story holds per device class; report the load threshold lambda*=2/F0.

    python bench/run_real.py
Reads results/fsync_{ebs,nvme}.json (+ optional results/pg_commit_delay.json), writes results/H5.json
and figures/devices.png.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from group_commit import FlushModel, simulate, theory as T   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")


def device_arm(dev):
    d = json.load(open(os.path.join(RES, f"fsync_{dev}.json")))
    F0 = d["F0_mean_ms"]; delta = max(0.0, d["delta_ms_per_byte"]) * 256  # ~256B WAL record marginal
    samples = np.array(d["F0_samples_ms"])
    fm = FlushModel(F0=F0, delta=delta, F0_samples=samples, seed=0)
    lam_star = 2.0 / F0
    # closed-loop policy comparison at a representative load (N=32 clients)
    rows = []
    for Z in [4.0, 1.0, 0.2, 0.02]:
        g = np.mean([simulate("greedy", {}, fm, mode="closed", N=32, Z=Z,
                               max_commits=8000, warmup_commits=1000, seed=s)["per_txn_overhead"]
                     for s in range(5)])
        # best tuned timer
        best = min(np.mean([simulate("timer", {"T": float(Tt), "F0": F0}, fm, mode="closed", N=32, Z=Z,
                                     max_commits=8000, warmup_commits=1000, seed=s)["per_txn_overhead"]
                            for s in range(3)])
                   for Tt in np.geomspace(0.05 * F0, 20 * F0, 12))
        out = simulate("greedy", {}, fm, mode="closed", N=32, Z=Z, max_commits=8000, warmup_commits=1000, seed=0)
        rows.append({"Z": Z, "lambda_per_ms": out["throughput"], "greedy_overhead_ms": float(g),
                     "best_timer_overhead_ms": float(best), "greedy_vs_best": float(g / best)})
    return {"device": d["device_label"], "F0_ms": F0, "F0_p99_ms": d["F0_p99_ms"],
            "lambda_star_per_ms": lam_star, "lambda_star_per_s": lam_star * 1000.0,
            "rows": rows, "greedy_vs_best_median": float(np.median([r["greedy_vs_best"] for r in rows]))}


def pg_arm():
    p = os.path.join(RES, "pg_commit_delay.json")
    if not os.path.exists(p):
        return None
    rows = json.load(open(p))
    rows = [r for r in rows if r["tps"] > 0]
    if not rows:
        return None
    cd0 = next((r for r in rows if r["commit_delay_us"] == 0), rows[0])
    best = max(rows, key=lambda r: r["tps"])
    return {"rows": rows, "tps_at_0": cd0["tps"], "best_tps": best["tps"],
            "best_commit_delay_us": best["commit_delay_us"],
            "parameter_free_vs_best_tps": cd0["tps"] / best["tps"]}


def main():
    ebs = device_arm("ebs"); nvme = device_arm("nvme")
    pg = pg_arm()
    out = {"ebs": ebs, "nvme": nvme, "postgres": pg,
           "H5a_pass": bool(ebs["greedy_vs_best_median"] <= 1.1 and nvme["greedy_vs_best_median"] <= 1.1)}
    with open(os.path.join(RES, "H5.json"), "w") as f:
        json.dump(out, f, indent=2)
    for dev in (ebs, nvme):
        print(f"[{dev['device']}] F0={dev['F0_ms']:.3f}ms (p99 {dev['F0_p99_ms']:.3f})  "
              f"lambda*={dev['lambda_star_per_s']:.0f}/s  greedy/best median={dev['greedy_vs_best_median']:.3f}")
    if pg:
        print(f"[postgres] commit_delay=0 tps={pg['tps_at_0']:.0f}  best tps={pg['best_tps']:.0f} "
              f"@cd={pg['best_commit_delay_us']}us  param-free/best={pg['parameter_free_vs_best_tps']:.3f}")
    print(f"H5a pass={out['H5a_pass']}")
    _plot(ebs, nvme, pg)


def _plot(ebs, nvme, pg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = 2 if pg else 1
    fig, ax = plt.subplots(1, n + 1, figsize=(5.0 * (n + 1), 4))
    # device F0 + lambda*
    devs = [ebs, nvme]
    ax[0].bar([d["device"] for d in devs], [d["F0_ms"] for d in devs], color=["#c44", "#4a7"])
    ax[0].set_ylabel("measured fsync F0 (ms)"); ax[0].set_yscale("log")
    for i, d in enumerate(devs):
        ax[0].text(i, d["F0_ms"], f"  λ*={d['lambda_star_per_s']:.0f}/s", ha="center", va="bottom", fontsize=8)
    ax[0].set_title("Real fsync cost sets the\nload threshold λ*=2/F0")
    # greedy vs best per device
    for d, c in zip(devs, ["#c44", "#4a7"]):
        lam = [r["lambda_per_ms"] * 1000 for r in d["rows"]]
        ax[1].plot(lam, [r["greedy_vs_best"] for r in d["rows"]], "o-", color=c, label=d["device"])
    ax[1].axhline(1.0, color="gray", ls=":"); ax[1].set_xscale("log")
    ax[1].set_xlabel("load (commits/s)"); ax[1].set_ylabel("greedy / best-tuned")
    ax[1].set_title("Parameter-free greedy ~ best-tuned\n(real device models)"); ax[1].legend(fontsize=8)
    if pg:
        rows = pg["rows"]
        ax[2].plot([r["commit_delay_us"] for r in rows], [r["tps"] for r in rows], "o-")
        ax[2].axhline(pg["tps_at_0"], color="green", ls=":", label="commit_delay=0 (parameter-free)")
        ax[2].set_xlabel("commit_delay (us)"); ax[2].set_ylabel("PostgreSQL tps")
        ax[2].set_title("Real PostgreSQL: commit_delay=0\ncompetitive with best-tuned"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "devices.png"), dpi=130); plt.close(fig)
    print(f"figure -> {FIG}/devices.png")


if __name__ == "__main__":
    main()
