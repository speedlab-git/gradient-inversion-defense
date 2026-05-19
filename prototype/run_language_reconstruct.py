"""Drive Stage 2/3 reconstruction sweeps for the language sweep.

For each (domain, tag, seed, defense_condition) cell, runs reconstruction
against the trained malicious model and records the attack metrics into
results/lang_sweep_logs/reconstruct.csv.

Usage:
    # Clean only, all 24 x 3 seeds, on GPU 1:
    python prototype/run_language_reconstruct.py --gpu 1 --condition clean

    # NBFU defense only, slice 0:12 (medical), on GPU 3:
    python prototype/run_language_reconstruct.py --gpu 3 --condition nbfu --domain medical

    # Both conditions, a single tag:
    python prototype/run_language_reconstruct.py --gpu 0 --tag pna_base
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))
from language_sweep_config import LANGUAGE_SWEEP_QUERIES, all_queries  # noqa: E402

PY = "/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python"
LOG_DIR = ROOT / "results" / "lang_sweep_logs"
RECON_CSV = LOG_DIR / "reconstruct.csv"
MODEL_DIR = ROOT / "malicious_models_lang_sweep"

# Match the cosine-threshold F1 line emitted by reconstruct_*.py
F1_RE = re.compile(r"Attack F1:\s*([\d.]+)")
COSSIM_RE = re.compile(r'"avg_cos_sim":\s*([\d.]+)')  # from saved metrics.json (no avg_cos_sim in stdout)
LPIPS_RE = re.compile(r"LPIPS:\s*([\d.]+)")
PSNR_RE = re.compile(r"PSNR:\s*([-\d.]+)")


def parse_metrics(text: str) -> dict:
    out = {}
    for key, rx in [("f1", F1_RE), ("cos_sim", COSSIM_RE), ("lpips", LPIPS_RE), ("psnr", PSNR_RE)]:
        m = rx.search(text)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                pass
    return out


def run_one(domain: str, tag: str, seed: int, condition: str, gpu: int,
            max_iters: int = 24000, eps: float = 5.0, C: float = 1.0) -> dict:
    """Run a single reconstruction. condition in {clean, nbfu}."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_name = f"{domain}_{tag}_s{seed}_{condition}.log"
    log_path = LOG_DIR / log_name

    model_path = MODEL_DIR / domain / f"{tag}.pt"
    if not model_path.exists():
        print(f"[SKIP] {tag} seed={seed} {condition}: model not found at {model_path}")
        return {"ok": False, "reason": "model_missing"}

    script = ROOT / "prototype" / f"reconstruct_{domain}.py"
    cmd = [
        PY, str(script),
        "--geminio-query", tag,  # tag for save dir; model_path overrides lookup
        "--model-path", str(model_path),
        "--gpu", str(gpu),
        "--max-iterations", str(max_iters),
        "--seed", str(seed),
        "--batch-tag", f"langsweep_{condition}",
    ]
    if condition == "nbfu":
        cmd += [
            "--dpsgd-epsilon", str(eps),
            "--dpsgd-max-grad-norm", str(C),
        ]

    print(f"[{time.strftime('%H:%M:%S')}] RECON {domain}/{tag} seed={seed} {condition} GPU{gpu}")
    t0 = time.time()
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0

    if result.returncode != 0:
        print(f"[FAIL] returncode {result.returncode}; see {log_path}")
        return {"ok": False, "reason": f"rc{result.returncode}", "elapsed_sec": int(dt)}

    metrics = parse_metrics(log_path.read_text())
    print(f"[{time.strftime('%H:%M:%S')}] DONE  {tag} seed={seed} {condition} elapsed={dt:.0f}s "
          f"F1={metrics.get('f1', float('nan')):.3f} cos={metrics.get('cos_sim', float('nan')):.3f}")
    return {
        "ok": True,
        "domain": domain, "tag": tag, "seed": seed, "condition": condition,
        "f1": metrics.get("f1"),
        "cos_sim": metrics.get("cos_sim"),
        "lpips": metrics.get("lpips"),
        "psnr": metrics.get("psnr"),
        "elapsed_sec": int(dt),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--domain", choices=["medical", "uav", "all"], default="all")
    ap.add_argument("--condition", choices=["clean", "nbfu", "both"], default="both")
    ap.add_argument("--tag", type=str, default=None, help="If set, only run this tag")
    ap.add_argument("--slice", type=str, default="0:24", help="Slice into the full 24-query list")
    ap.add_argument("--seeds", type=str, default="42,123,256")
    ap.add_argument("--max-iters", type=int, default=24000)
    ap.add_argument("--eps", type=float, default=5.0)
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    queries = all_queries()
    if args.domain != "all":
        queries = [q for q in queries if q[0] == args.domain]
    if args.tag:
        queries = [q for q in queries if q[1] == args.tag]
    else:
        s, e = (int(x) for x in args.slice.split(":"))
        queries = queries[s:e]

    seeds = [int(x) for x in args.seeds.split(",")]
    conditions = ["clean", "nbfu"] if args.condition == "both" else [args.condition]

    cells = [(q, s, c) for q in queries for s in seeds for c in conditions]
    print(f"Will run {len(cells)} cells on GPU {args.gpu}")

    results = []
    for (domain, tag, _text, _cls), seed, cond in cells:
        r = run_one(domain, tag, seed, cond, args.gpu, args.max_iters, args.eps, args.C)
        if r.get("ok"):
            results.append(r)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RECON_CSV.exists()
    with open(RECON_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "tag", "seed", "condition", "f1", "cos_sim", "lpips", "psnr", "elapsed_sec"])
        if write_header:
            w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    print(f"Wrote {len(results)} rows to {RECON_CSV}")


if __name__ == "__main__":
    main()
