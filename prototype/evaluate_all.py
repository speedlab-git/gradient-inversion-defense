"""
Evaluate all reconstruction results: compute LPIPS, PSNR, CW-SSIM, Attack F1.

Loads saved .pt tensors from result directories and computes metrics.
Can also be used to compute metrics from previously saved tensors without
re-running the full reconstruction.

Usage:
    python prototype/evaluate_all.py --gpu 0
    python prototype/evaluate_all.py --gpu 0 --results-dir results/medical_pneumonia_descriptive
"""
import argparse
import csv
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lpips
from breaching.analysis.metrics import psnr_compute, cw_ssim


# ImageNet normalization (used by all domains)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def compute_image_metrics(rec_data, true_data, setup):
    """Compute LPIPS, PSNR, CW-SSIM between reconstructed and ground truth images."""
    lpips_scorer = lpips.LPIPS(net="alex", verbose=False).to(**setup)

    dm = torch.tensor(MEAN, **setup)[None, :, None, None]
    ds = torch.tensor(STD, **setup)[None, :, None, None]

    rec = torch.clamp(rec_data["data"].to(**setup) * ds + dm, 0, 1)
    gt = torch.clamp(true_data["data"].to(**setup) * ds + dm, 0, 1)

    # Reorder reconstructed images to best match ground truth
    from breaching.analysis.analysis import compute_batch_order
    order = compute_batch_order(lpips_scorer, rec, gt, setup)
    rec = rec[order]

    # PSNR
    avg_psnr, max_psnr = psnr_compute(rec, gt, factor=1)

    # LPIPS
    lpips_score = lpips_scorer(rec, gt, normalize=True)
    avg_lpips = lpips_score.mean().item()

    # CW-SSIM
    avg_ssim, max_ssim = cw_ssim(rec, gt, scales=5)

    # MSE
    mse = (rec - gt).pow(2).mean().item()

    # Per-image LPIPS and PSNR
    per_image_lpips = lpips_score.squeeze().tolist()
    mse_per = (rec - gt).pow(2).mean(dim=[1, 2, 3])
    per_image_psnr = (10 * torch.log10(1.0 / mse_per)).tolist()

    return {
        'psnr': avg_psnr,
        'max_psnr': max_psnr,
        'lpips': avg_lpips,
        'ssim': avg_ssim,
        'max_ssim': max_ssim,
        'mse': mse,
        'per_image_lpips': per_image_lpips,
        'per_image_psnr': per_image_psnr,
    }


def evaluate_directory(results_dir, setup):
    """Evaluate a single results directory containing saved tensors."""
    rec_path = os.path.join(results_dir, 'reconstructed.pt')
    true_path = os.path.join(results_dir, 'true_data.pt')

    if not os.path.exists(rec_path) or not os.path.exists(true_path):
        print(f"  SKIP: {results_dir} — missing .pt tensors")
        return None

    rec_data = torch.load(rec_path, map_location=setup['device'])
    true_data = torch.load(true_path, map_location=setup['device'])

    metrics = compute_image_metrics(rec_data, true_data, setup)

    # Load existing metrics.json if present to merge
    metrics_json_path = os.path.join(results_dir, 'metrics.json')
    if os.path.exists(metrics_json_path):
        with open(metrics_json_path) as f:
            existing = json.load(f)
        # Keep metadata from existing
        for key in ['query', 'prune_rate', 'noise_scale', 'domain']:
            if key in existing:
                metrics[key] = existing[key]

    return metrics


def evaluate_all(results_root, setup):
    """Scan all result directories and compute metrics."""
    results = []

    # Find all directories with reconstructed.pt
    for dirname in sorted(os.listdir(results_root)):
        dirpath = os.path.join(results_root, dirname)
        if not os.path.isdir(dirpath):
            continue
        rec_path = os.path.join(dirpath, 'reconstructed.pt')
        if not os.path.exists(rec_path):
            continue

        print(f"\nEvaluating: {dirname}")
        metrics = evaluate_directory(dirpath, setup)
        if metrics is not None:
            metrics['run'] = dirname
            results.append(metrics)
            print(f"  LPIPS: {metrics['lpips']:.4f} | PSNR: {metrics['psnr']:.2f} dB | CW-SSIM: {metrics['ssim']:.4f}")

            # Save updated metrics
            metrics_path = os.path.join(dirpath, 'metrics.json')
            save_metrics = {k: v for k, v in metrics.items()
                          if not isinstance(v, list)}  # exclude per-image lists for JSON
            with open(metrics_path, 'w') as f:
                json.dump(save_metrics, f, indent=2)

    return results


def print_summary_table(results):
    """Print a formatted summary table."""
    if not results:
        print("\nNo results found.")
        return

    print("\n" + "=" * 90)
    print(f"{'Run':<45} {'LPIPS':>8} {'PSNR':>8} {'CW-SSIM':>8} {'MSE':>10}")
    print("-" * 90)
    for r in results:
        print(f"{r['run']:<45} {r['lpips']:>8.4f} {r['psnr']:>8.2f} {r['ssim']:>8.4f} {r['mse']:>10.6f}")
    print("=" * 90)


def save_csv(results, output_path):
    """Save results as CSV."""
    if not results:
        return
    fields = ['run', 'lpips', 'psnr', 'max_psnr', 'ssim', 'max_ssim', 'mse']
    # Add optional fields
    for key in ['query', 'domain', 'prune_rate', 'noise_scale',
                'attack_f1', 'attack_precision', 'attack_recall']:
        if any(key in r for r in results):
            fields.append(key)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nSaved CSV to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate reconstruction results')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Evaluate a single results directory')
    parser.add_argument('--results-root', type=str, default='./results',
                        help='Root directory to scan for results')
    parser.add_argument('--output-csv', type=str, default='./results/evaluation_summary.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
    setup = dict(device=device, dtype=torch.float)

    if args.results_dir:
        print(f"Evaluating single directory: {args.results_dir}")
        metrics = evaluate_directory(args.results_dir, setup)
        if metrics:
            metrics['run'] = os.path.basename(args.results_dir)
            print(f"  LPIPS: {metrics['lpips']:.4f} | PSNR: {metrics['psnr']:.2f} dB | CW-SSIM: {metrics['ssim']:.4f}")
    else:
        print(f"Scanning all results in: {args.results_root}")
        results = evaluate_all(args.results_root, setup)
        print_summary_table(results)
        save_csv(results, args.output_csv)
