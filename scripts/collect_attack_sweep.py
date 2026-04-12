"""Collect sweep metrics into a single table.

Reads metrics.json files for the {DLG, IG, GradInversion} x {medical, uav}
sweep and dumps a CSV + Markdown table for the paper.
"""
import csv
import json
import os
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = RESULTS / "attack_sweep_summary"
OUT_DIR.mkdir(exist_ok=True)

ATTACKS = ["DLG", "IG", "GradInversion", "HF-GradInv"]

# (domain, query, defense_label, attack_short -> run_dir_name)
# None means "skip / not available"
# 8k sweep (tuning-grade)
RUNS_8K = {
    "medical": {
        "query": "pneumonia_descriptive",
        "clean": {
            "DLG": "medical_pneumonia_descriptive_dlg_tune4",
            "IG": "medical_pneumonia_descriptive_ig_tune2",
            "GradInversion": "medical_pneumonia_descriptive_gradinversion_tune",
            "HF-GradInv": "medical_pneumonia_descriptive_pseudo",
        },
        "dpsgd5": {
            "DLG": "medical_pneumonia_descriptive_dlg_dpsgd_eps5.0_sweep",
            "IG": "medical_pneumonia_descriptive_ig_dpsgd_eps5.0_sweep",
            "GradInversion": "medical_pneumonia_descriptive_gradinversion_dpsgd_eps5.0_sweep",
            "HF-GradInv": "medical_pneumonia_descriptive_dpsgd_eps5.0",
        },
    },
    "uav": {
        "query": "solar_panels",
        "clean": {
            "DLG": "uav_solar_panels_dlg_sweep",
            "IG": "uav_solar_panels_ig_sweep",
            "GradInversion": "uav_solar_panels_gradinversion_sweep",
            "HF-GradInv": "uav_solar_panels",
        },
        "dpsgd5": {
            "DLG": "uav_solar_panels_dlg_dpsgd_eps5.0_sweep",
            "IG": "uav_solar_panels_ig_dpsgd_eps5.0_sweep",
            "GradInversion": "uav_solar_panels_gradinversion_dpsgd_eps5.0_sweep",
            "HF-GradInv": "uav_solar_panels_dpsgd_eps5.0",
        },
    },
}

# 24k sweep (paper-grade). Falls back to 8k if 24k results not yet available.
RUNS_24K = {
    "medical": {
        "query": "pneumonia_descriptive",
        "clean": {
            "DLG": "medical_pneumonia_descriptive_dlg_24k",
            "IG": "medical_pneumonia_descriptive_ig_24k",
            "GradInversion": "medical_pneumonia_descriptive_gradinversion_24k",
            "HF-GradInv": "medical_pneumonia_descriptive_pseudo",
        },
        "dpsgd5": {
            "DLG": "medical_pneumonia_descriptive_dlg_dpsgd_eps5.0_24k",
            "IG": "medical_pneumonia_descriptive_ig_dpsgd_eps5.0_24k",
            "GradInversion": "medical_pneumonia_descriptive_gradinversion_dpsgd_eps5.0_24k",
            "HF-GradInv": "medical_pneumonia_descriptive_dpsgd_eps5.0",
        },
    },
    "uav": {
        "query": "solar_panels",
        "clean": {
            "DLG": "uav_solar_panels_dlg_24k",
            "IG": "uav_solar_panels_ig_24k",
            "GradInversion": "uav_solar_panels_gradinversion_24k",
            "HF-GradInv": "uav_solar_panels",
        },
        "dpsgd5": {
            "DLG": "uav_solar_panels_dlg_dpsgd_eps5.0_24k",
            "IG": "uav_solar_panels_ig_dpsgd_eps5.0_24k",
            "GradInversion": "uav_solar_panels_gradinversion_dpsgd_eps5.0_24k",
            "HF-GradInv": "uav_solar_panels_dpsgd_eps5.0",
        },
    },
}

RUNS = RUNS_8K  # default; override with --paper flag


def load_metrics(run_name):
    if run_name is None:
        return None
    p = RESULTS / run_name / "metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def fmt(x, width, default="-"):
    if x is None:
        return default.rjust(width)
    return f"{x:.3f}".rjust(width)


def main():
    rows = []
    for domain, cfg in RUNS.items():
        for attack in ATTACKS:
            clean = load_metrics(cfg["clean"][attack])
            dp = load_metrics(cfg["dpsgd5"][attack])
            rows.append(
                {
                    "domain": domain,
                    "query": cfg["query"],
                    "attack": attack,
                    "clean_f1": clean.get("attack_f1") if clean else None,
                    "clean_lpips": clean["lpips"] if clean else None,
                    "clean_psnr": clean["psnr"] if clean else None,
                    "dp_f1": dp.get("attack_f1") if dp else None,
                    "dp_lpips": dp["lpips"] if dp else None,
                    "dp_psnr": dp["psnr"] if dp else None,
                }
            )

    # CSV
    csv_path = OUT_DIR / "attack_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    # Markdown
    md_path = OUT_DIR / "attack_sweep.md"
    with open(md_path, "w") as f:
        f.write("# Attack generalization of calibrated DP-SGD (eps=5)\n\n")
        f.write(
            "Each cell is `Attack F1 / LPIPS / PSNR`. Clean = no defense; "
            "DP-SGD = calibrated DP-SGD with eps=5, delta=1e-5, C=1.0.\n\n"
        )
        for domain, cfg in RUNS.items():
            f.write(f"## {domain} ({cfg['query']})\n\n")
            f.write("| Attack | Clean F1 | Clean LPIPS | Clean PSNR | DP-SGD F1 | DP-SGD LPIPS | DP-SGD PSNR |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for r in rows:
                if r["domain"] != domain:
                    continue
                f.write(
                    f"| {r['attack']} | "
                    f"{fmt(r['clean_f1'],5)} | {fmt(r['clean_lpips'],6)} | {fmt(r['clean_psnr'],6)} | "
                    f"{fmt(r['dp_f1'],5)} | {fmt(r['dp_lpips'],6)} | {fmt(r['dp_psnr'],6)} |\n"
                )
            f.write("\n")
    print(f"Wrote {md_path}")

    # Print to stdout too
    with open(md_path) as f:
        sys.stdout.write(f.read())


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser()
    _parser.add_argument("--paper", action="store_true", help="Use 24k-iter results")
    _args = _parser.parse_args()
    if _args.paper:
        RUNS = RUNS_24K
    main()
