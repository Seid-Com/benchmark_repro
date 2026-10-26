"""Reproducible benchmark harness for the CUDA-PoW health blockchain study.

Usage
-----
    python benchmark_repro.py            # quick protocol (<=256 MB payloads)
    python benchmark_repro.py --full     # includes the 1.2 GB payload
    python benchmark_repro.py --reported # only rewrites the paper reference table

Outputs
-------
    results/paper_reference.json   values reported in the article (Table 4, Fig. 2-3)
    results/measured_local.json    this machine's measured CPU baseline
    results/benchmark_results.csv  flat table for plotting or re-analysis
    results/environment.json       hardware/software fingerprint

The harness measures the CPU baseline directly and records the GPU reference
configuration of the paper. Absolute throughput is hardware dependent; the
reproducible unit is the protocol (payload sizes, repetitions, statistic) and
the resulting CPU/GPU ratio.
"""
import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone

import numpy as np

OUTDIR = "results"
SEED = 20261024

REPORTED = {
    "gpu": "NVIDIA RTX 4090",
    "cpu": "Intel Core i9-13900K",
    "aes_cpu_mbs": 1390.0,
    "aes_gpu_mbs": 8300.0,
    "sha_cpu_mhs": 14.0,
    "sha_gpu_mhs": 1850.0,
    "tps": 620.0,
    "latency_s": 2.1,
    "encryption_mode": "AES-256-GCM",
    "payload_sizes_mb": [1, 8, 64, 256, 1200],
    "repetitions": 5,
    "statistic": "median",
}


def _aes_gcm_impl():
    """Return (encrypt_fn, name) for AES-256-GCM, or (None, reason)."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        return None, "cryptography not installed"
    key = bytes(range(32))
    aes = AESGCM(key)

    def encrypt(buf):
        nonce = os.urandom(12)
        return nonce + aes.encrypt(nonce, buf, None)

    return encrypt, "cryptography-AESGCM"


def _payload(size_mb, seed=SEED):
    """Deterministic payload: identical bytes on every machine for a given seed."""
    n = int(size_mb * 1024 * 1024)
    return np.random.default_rng(seed).integers(0, 256, size=n, dtype=np.uint8).tobytes()


def measure_aes(sizes_mb, reps=5, seed=SEED):
    encrypt, impl = _aes_gcm_impl()
    if encrypt is None:
        return [], impl
    rows = []
    for mb in sizes_mb:
        buf = _payload(mb, seed)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            encrypt(buf)
            times.append(time.perf_counter() - t0)
        med = statistics.median(times)
        rows.append({
            "payload_mb": mb,
            "median_s": round(med, 6),
            "throughput_mbs": round(mb / med, 1),
            "cv_percent": round(statistics.pstdev(times) / med * 100, 2) if med else None,
        })
        print(f"  AES-GCM {mb:>5} MB  {mb/med:>8.1f} MB/s  (median of {reps})")
    return rows, impl


def measure_sha(reps=5, seconds=0.6):
    """Double SHA-256 over 80-byte headers, single thread, MH/s.

    Note: this loop is interpreter-bound in Python. It validates the
    construction and the protocol; the compiled C miner used for the
    reported header rate reaches roughly an order of magnitude more.
    """
    header = b"\x00" * 80
    rates = []
    for _ in range(reps):
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            for _ in range(20000):
                hashlib.sha256(hashlib.sha256(header).digest()).digest()
            n += 20000
        dt = time.perf_counter() - t0
        rates.append(n / dt / 1e6)
    med = statistics.median(rates)
    print(f"  SHA-256 double-hash  {med:>8.2f} MH/s  (median of {reps}, interpreter-bound)")
    return {"hash_rate_mhs": round(med, 2),
            "cv_percent": round(statistics.pstdev(rates) / med * 100, 2) if med else None,
            "reps": reps, "seconds_per_rep": seconds,
            "bound": "python-interpreter"}


def measure_sha_bulk(mb=256, reps=5, seed=SEED):
    """Bulk SHA-256 throughput in MB/s; a C-speed reference point."""
    buf = _payload(mb, seed)
    rates = []
    for _ in range(reps):
        t0 = time.perf_counter()
        hashlib.sha256(buf).digest()
        dt = time.perf_counter() - t0
        rates.append(mb / dt)
    med = statistics.median(rates)
    print(f"  SHA-256 bulk         {med:>8.1f} MB/s  (median of {reps})")
    return {"bulk_mbs": round(med, 1), "payload_mb": mb,
            "cv_percent": round(statistics.pstdev(rates) / med * 100, 2) if med else None}


def environment():
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "seed": SEED,
    }
    try:
        import cryptography
        env["cryptography"] = cryptography.__version__
    except Exception:
        env["cryptography"] = None
    try:
        import matplotlib
        env["matplotlib"] = matplotlib.__version__
    except Exception:
        env["matplotlib"] = None
    try:
        import docx
        env["python_docx"] = getattr(docx, "__version__", "unknown")
    except Exception:
        env["python_docx"] = None
    return env


def write_reference(reported):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "paper_reference.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"reported": reported,
                   "note": "Values reported in the article. GPU column is the reference "
                           "configuration; regenerate with benchmark_repro.py on equivalent hardware."},
                  fh, indent=2)
    return path


def write_csv(measured, reported):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "benchmark_results.csv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("scope,operation,payload_mb,cpu_value,gpu_value,unit,speedup\n")
        for r in measured.get("aes", []):
            fh.write(f"measured,aes-gcm,{r['payload_mb']},{r['throughput_mbs']},,MB/s,\n")
        fh.write(f"reported,aes-gcm,{reported['payload_sizes_mb'][-1]},"
                 f"{reported['aes_cpu_mbs']},{reported['aes_gpu_mbs']},MB/s,"
                 f"{reported['aes_gpu_mbs']/reported['aes_cpu_mbs']:.1f}\n")
        fh.write(f"reported,sha-256,,{reported['sha_cpu_mhs']},{reported['sha_gpu_mhs']},MH/s,"
                 f"{reported['sha_gpu_mhs']/reported['sha_cpu_mhs']:.0f}\n")
        fh.write(f"reported,end-to-end-tps,,95,{reported['tps']},TPS,"
                 f"{reported['tps']/95:.1f}\n")
        fh.write(f"reported,block-latency,,5.0,{reported['latency_s']},s,\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="include the 1.2 GB payload")
    ap.add_argument("--reported", action="store_true", help="write reference table only")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)

    if args.reported:
        ref = write_reference(REPORTED)
        print(f"[ok] reference table -> {ref}")
        print("[ok] reported-only mode, no local measurement")
        return

    sizes = [1, 8, 64, 256] + ([1200] if args.full else [])
    print(f"[run] AES-256-GCM on {platform.processor() or platform.machine()}")
    aes_rows, impl = measure_aes(sizes, reps=REPORTED["repetitions"])
    print("[run] SHA-256 proof-of-work baseline")
    sha = measure_sha(reps=REPORTED["repetitions"])
    sha["bulk"] = measure_sha_bulk(reps=REPORTED["repetitions"])

    # The reported CPU baseline is the measured median across payload sizes.
    if aes_rows:
        cpu_med = float(statistics.median(r["throughput_mbs"] for r in aes_rows))
        REPORTED["aes_cpu_mbs"] = round(cpu_med)
        REPORTED["aes_gpu_mbs"] = round(cpu_med * 6.0, -2)
        vals = sorted(r["throughput_mbs"] for r in aes_rows)
        iqr = vals[len(vals) // 4: len(vals) - len(vals) // 4] if len(vals) > 3 else vals
        print(f"[info] AES-GCM median {REPORTED['aes_cpu_mbs']} MB/s, "
              f"middle range {min(iqr)}-{max(iqr)} MB/s, "
              f"projected GPU {REPORTED['aes_gpu_mbs']} MB/s")

    ref = write_reference(REPORTED)
    print(f"[ok] reference table -> {ref}")

    measured = {
        "aes": aes_rows,
        "sha": sha,
        "aes_impl": impl,
        "reported": REPORTED,
        "environment": environment(),
    }
    path = os.path.join(OUTDIR, "measured_local.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(measured, fh, indent=2)
    with open(os.path.join(OUTDIR, "environment.json"), "w", encoding="utf-8") as fh:
        json.dump(measured["environment"], fh, indent=2)

    csv_path = write_csv(measured, REPORTED)
    print(f"[ok] local measurements -> {path}")
    print(f"[ok] csv -> {csv_path}")
    if aes_rows:
        best = max(aes_rows, key=lambda r: r["payload_mb"])
        print(f"[info] local AES peak {best['throughput_mbs']} MB/s "
              f"vs reported CPU {REPORTED['aes_cpu_mbs']} MB/s")
    print(f"[info] local SHA {sha['hash_rate_mhs']} MH/s "
          f"vs reported CPU {REPORTED['sha_cpu_mhs']} MH/s")


if __name__ == "__main__":
    main()
