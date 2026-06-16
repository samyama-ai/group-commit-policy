"""Measure the REAL fsync/fdatasync latency distribution of a device (no mocks).

Writes a small record and fsyncs, repeatedly, recording per-flush latency in milliseconds. Also sweeps
the write size to estimate the per-record marginal cost delta (flush latency vs bytes). Output JSON feeds
the simulator's FlushModel (F0 = barrier latency distribution, delta = marginal write cost).

    python bench/measure_fsync.py --path /mnt/data/fsynctest --reps 2000 --out results/fsync_<dev>.json

Run on the target device's filesystem (EBS gp3, instance-store NVMe, local NVMe, ...).
"""
import argparse
import json
import os
import time


def measure(path, size_bytes, reps, fdatasync=True):
    lat_ms = []
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        buf = b"\xa5" * size_bytes
        # warm the file
        os.pwrite(fd, buf, 0); os.fsync(fd)
        for i in range(reps):
            os.pwrite(fd, buf, (i % 64) * size_bytes)   # vary offset a little
            t0 = time.perf_counter()
            if fdatasync:
                os.fdatasync(fd)
            else:
                os.fsync(fd)
            lat_ms.append((time.perf_counter() - t0) * 1e3)
    finally:
        os.close(fd)
        try:
            os.unlink(path)
        except OSError:
            pass
    return lat_ms


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="test file path on the target device's filesystem")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device-label", default="unknown")
    ap.add_argument("--fsync", action="store_true", help="use fsync instead of fdatasync")
    args = ap.parse_args()

    sizes = [512, 4096, 16384, 65536, 262144]          # bytes; for delta (marginal) estimate
    by_size = {}
    for sz in sizes:
        lat = measure(args.path, sz, args.reps, fdatasync=not args.fsync)
        by_size[sz] = {"p50": pct(lat, 50), "p90": pct(lat, 90), "p99": pct(lat, 99),
                       "mean": sum(lat) / len(lat), "min": min(lat), "n": len(lat)}
        print(f"  size={sz:>7}B  p50={by_size[sz]['p50']:.3f}ms p99={by_size[sz]['p99']:.3f}ms")

    # full distribution at the smallest size = the F0 barrier-cost samples
    f0_samples = measure(args.path, 512, args.reps, fdatasync=not args.fsync)
    # delta ~ slope of mean latency vs bytes (ms per byte), via least squares on the size sweep
    xs = sizes; ys = [by_size[s]["mean"] for s in sizes]
    n = len(xs); sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    delta = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    F0_mean = by_size[512]["mean"]

    out = {"device_label": args.device_label, "sync": "fsync" if args.fsync else "fdatasync",
           "F0_mean_ms": F0_mean, "F0_p50_ms": by_size[512]["p50"], "F0_p99_ms": by_size[512]["p99"],
           "delta_ms_per_byte": delta, "by_size": by_size,
           "F0_samples_ms": [round(x, 4) for x in f0_samples]}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"F0_mean={F0_mean:.3f}ms  delta={delta:.3e} ms/byte  -> {args.out}")


if __name__ == "__main__":
    main()
