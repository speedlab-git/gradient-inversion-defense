"""Recompute Attack F1 for an existing reconstruction.

Loads saved reconstructed.pt + true_data.pt from a results directory,
reloads the malicious Geminio model, and writes updated metrics.json
with the attack_f1 fields. Useful for older runs that predate the
attack-F1 metric.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import GeminioResNet18


def compute_attack_f1(model, loss_fn, rec_data, true_data, setup, threshold=0.90):
    model.to(**setup).eval()
    cos_sims, identified, B = [], 0, rec_data["data"].shape[0]
    for i in range(B):
        model.zero_grad()
        rec_out = model(rec_data["data"][i:i+1].to(setup["device"]))
        rec_loss = loss_fn(rec_out, true_data["labels"][i:i+1].to(setup["device"]))
        rec_grads = torch.autograd.grad(rec_loss, model.parameters(), create_graph=False)
        rec_grad_flat = rec_grads[-2].flatten()

        model.zero_grad()
        true_out = model(true_data["data"][i:i+1].to(setup["device"]))
        true_loss = loss_fn(true_out, true_data["labels"][i:i+1].to(setup["device"]))
        true_grads = torch.autograd.grad(true_loss, model.parameters(), create_graph=False)
        true_grad_flat = true_grads[-2].flatten()

        cs = F.cosine_similarity(rec_grad_flat.unsqueeze(0), true_grad_flat.unsqueeze(0)).item()
        cos_sims.append(cs)
        if cs >= threshold:
            identified += 1
    precision = recall = identified / B
    return {
        "attack_precision": precision,
        "attack_recall": recall,
        "attack_f1": precision,
        "identified": identified,
        "total": B,
        "avg_cos_sim": sum(cos_sims) / len(cos_sims),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--model-path", required=True, help="Malicious Geminio model .pt")
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    setup = dict(device=device, dtype=torch.float)

    rec = torch.load(os.path.join(args.results_dir, "reconstructed.pt"), map_location=device)
    true = torch.load(os.path.join(args.results_dir, "true_data.pt"), map_location=device)

    model = GeminioResNet18(num_classes=args.num_classes)
    state = torch.load(args.model_path, map_location=device)
    model.clf.load_state_dict(state, strict=True)

    metrics_path = os.path.join(args.results_dir, "metrics.json")
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}

    loss_fn = torch.nn.CrossEntropyLoss()
    f1 = compute_attack_f1(model, loss_fn, rec, true, setup)
    metrics.update(f1)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Updated {metrics_path}: F1={f1['attack_f1']}")


if __name__ == "__main__":
    main()
