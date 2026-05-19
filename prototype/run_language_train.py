"""Train malicious models for all 24 queries in the language sweep.

Usage:
    python prototype/run_language_train.py --gpu 0                  # train all 24 sequentially
    python prototype/run_language_train.py --gpu 0 --slice 0:6       # train queries [0:6)
    python prototype/run_language_train.py --gpu 0 --domain medical  # only medical (12 queries)

Outputs:
    malicious_models_lang_sweep/{domain}/{tag}.pt   — trained classifier head
    results/lang_sweep_logs/{tag}.log               — full training stdout
    results/lang_sweep_logs/manifest.csv            — appended row per query
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))
from language_sweep_config import LANGUAGE_SWEEP_QUERIES, all_queries  # noqa: E402

PY = "/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python"
OUT_BASE = ROOT / "malicious_models_lang_sweep"
LOG_DIR = ROOT / "results" / "lang_sweep_logs"
MANIFEST = LOG_DIR / "manifest.csv"

DOMAIN_VLM = {"medical": "biomedclip", "uav": "clip"}


def parse_loss_ratio(log_text: str) -> float:
    """Extract the (first) post-training loss ratio printed by train_*.py.

    Medical emits  ``Ratio: X.XXx``.
    UAV     emits  ``Loss ratio: X.XXx`` (twice, top-10% then top-5%; we take the first).
    """
    m = re.search(r"(?:Loss\s+)?Ratio:\s*([\d.]+)x", log_text, re.IGNORECASE)
    return float(m.group(1)) if m else float("nan")


def run_one(domain: str, tag: str, text: str, gpu: int, epochs: int = 5) -> dict:
    domain_out = OUT_BASE / domain
    domain_out.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_path = LOG_DIR / f"{tag}.log"
    final_path = domain_out / f"{tag}.pt"

    if final_path.exists():
        print(f"[SKIP] {tag} -> {final_path} already exists")
        return {"tag": tag, "skipped": True}

    script = ROOT / "prototype" / f"train_{domain}.py"
    cmd = [
        PY, str(script),
        "--query", text,
        "--gpu", str(gpu),
        "--epochs", str(epochs),
        "--vlm", DOMAIN_VLM[domain],
        "--output-dir", str(domain_out / "_tmp"),
    ]
    print(f"[{time.strftime('%H:%M:%S')}] TRAIN {domain}/{tag} on GPU {gpu}: {text!r}")
    t0 = time.time()
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0

    if result.returncode != 0:
        print(f"[FAIL] {tag} returncode {result.returncode}; see {log_path}")
        return {"tag": tag, "ok": False, "elapsed": dt}

    log_text = log_path.read_text()
    loss_ratio = parse_loss_ratio(log_text)

    safe_name = text.replace(" ", "_").replace("?", "").replace('"', "")
    if domain == "uav":
        safe_name = safe_name[:60]  # train_uav.py truncates safe_name to 60 chars
    vlm_suffix = ""
    if DOMAIN_VLM[domain] not in ("biomedclip", "clip"):
        vlm_suffix = f"_{DOMAIN_VLM[domain]}"
    auto_path = domain_out / "_tmp" / f"{safe_name}{vlm_suffix}.pt"
    if not auto_path.exists():
        candidates = list((domain_out / "_tmp").glob("*.pt"))
        print(f"[WARN] {tag}: expected {auto_path}; found {candidates}")
        if not candidates:
            return {"tag": tag, "ok": False, "elapsed": dt, "loss_ratio": loss_ratio}
        auto_path = max(candidates, key=lambda p: p.stat().st_mtime)

    shutil.move(str(auto_path), str(final_path))
    print(f"[{time.strftime('%H:%M:%S')}] DONE  {tag} elapsed={dt:.0f}s loss_ratio={loss_ratio:.3f}")
    return {
        "tag": tag,
        "domain": domain,
        "text": text,
        "vlm": DOMAIN_VLM[domain],
        "model_path": str(final_path),
        "loss_ratio": loss_ratio,
        "elapsed_sec": int(dt),
        "ok": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--slice", type=str, default="0:24", help="start:end indices into the 24-query list")
    ap.add_argument("--domain", type=str, choices=["medical", "uav", "all"], default="all")
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    queries = all_queries()
    if args.domain != "all":
        queries = [q for q in queries if q[0] == args.domain]

    s, e = (int(x) for x in args.slice.split(":"))
    queries = queries[s:e]

    print(f"Will train {len(queries)} queries on GPU {args.gpu}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for domain, tag, text, _cls in queries:
        r = run_one(domain, tag, text, args.gpu, args.epochs)
        results.append(r)

    write_header = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "domain", "text", "vlm", "model_path", "loss_ratio", "elapsed_sec", "ok"])
        if write_header:
            w.writeheader()
        for r in results:
            if r.get("skipped"):
                continue
            row = {k: r.get(k, "") for k in w.fieldnames}
            w.writerow(row)
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
