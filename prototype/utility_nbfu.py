"""Utility analysis: train classifier under NBFU and measure task accuracy.

Simulates a single-client FL setup where each gradient step applies NBFU
(per-update clipping + calibrated Gaussian noise) before the optimizer update.
Tracks test accuracy over epochs at multiple privacy budgets.

Usage:
    python prototype/utility_nbfu.py --domain medical --gpu 0
    python prototype/utility_nbfu.py --domain medical --gpu 0 --epsilons 0.1 1.0 5.0 10.0 50.0 inf

"clip-only" is denoted by epsilon=-1 in the CLI (matches reconstruct scripts).
"""
import argparse
import json
import os
import sys
import math

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import GeminioResNet18
from prototype.defenses import apply_dpsgd


def get_medical_loaders(batch_size=64):
    import medmnist
    from medmnist import ChestMNIST
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    medmnist_root = './data/medmnist'
    train_set = ChestMNIST(split='train', download=True, root=medmnist_root, size=64, transform=tfm)
    test_set = ChestMNIST(split='test', download=True, root=medmnist_root, size=64, transform=tfm)

    def to_singlelabel(ds):
        targets = []
        for lbl in ds.labels:
            active = np.where(lbl == 1)[0]
            targets.append(14 if len(active) == 0 else int(active[0]))
        ds.labels = np.array(targets, dtype=np.int64).reshape(-1, 1)
        return ds

    train_set = to_singlelabel(train_set)
    test_set = to_singlelabel(test_set)

    def collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        tgts = torch.tensor([int(b[1][0]) for b in batch], dtype=torch.long)
        return imgs, tgts

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=4, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              num_workers=4, collate_fn=collate)
    return train_loader, test_loader, 15


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / total


def train_with_nbfu(domain, epsilon, max_grad_norm, epochs, batch_size, gpu, lr, seed):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if domain == 'medical':
        train_loader, test_loader, num_classes = get_medical_loaders(batch_size)
    else:
        raise NotImplementedError(f'domain={domain} not yet supported (medical only for now)')

    model = GeminioResNet18(num_classes=num_classes).to(device)
    # Freeze backbone; train classifier head only (matches Geminio's FL scenario).
    for p in model.extractor.parameters():
        p.requires_grad = False
    optimizer = torch.optim.SGD(model.clf.parameters(), lr=lr, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    history = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()

            # Collect gradients of trainable params, apply NBFU, write back.
            grads = [p.grad.detach().clone() for p in model.clf.parameters()]
            eps_arg = None if epsilon < 0 else (None if math.isinf(epsilon) else epsilon)
            if epsilon == 0:
                # No defense applied; standard training
                pass
            else:
                defended, _info = apply_dpsgd(
                    grads,
                    epsilon=eps_arg,
                    delta=1e-5,
                    max_grad_norm=max_grad_norm,
                    batch_size=x.shape[0],
                )
                for p, g in zip(model.clf.parameters(), defended):
                    p.grad = g.to(p.device)

            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        train_loss = running_loss / max(1, n_batches)
        test_acc = evaluate(model, test_loader, device)
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'test_acc': test_acc})
        print(f'  epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}  test_acc={test_acc:.4f}')

    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', choices=['medical'], default='medical')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--epsilons', nargs='+', default=['0', '-1', '0.1', '1.0', '5.0', '10.0', '50.0'],
                    help='List of epsilons. 0 = no defense; -1 = clip-only; inf = treated as no-noise.')
    ap.add_argument('--max-grad-norm', type=float, default=1.0)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', default='./results/utility')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}
    for eps_str in args.epsilons:
        eps = float(eps_str)
        label = 'no_defense' if eps == 0 else ('clip_only' if eps < 0 else f'eps_{eps}')
        print(f'\n=== {args.domain} | {label} | C={args.max_grad_norm} | epochs={args.epochs} ===')
        hist = train_with_nbfu(args.domain, eps, args.max_grad_norm, args.epochs,
                               args.batch_size, args.gpu, args.lr, args.seed)
        summary[label] = hist

    out_path = os.path.join(args.out_dir, f'{args.domain}_utility.json')
    with open(out_path, 'w') as f:
        json.dump({
            'domain': args.domain,
            'config': vars(args),
            'history': summary,
        }, f, indent=2)
    print(f'\nSaved utility curves to {out_path}')


if __name__ == '__main__':
    main()
