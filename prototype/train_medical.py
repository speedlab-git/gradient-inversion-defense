"""
Phase 2 (Medical): Train malicious models for medical queries using VLM + ChestMNIST.

Supports multiple VLM backends for computing text-image similarity:
  - "biomedclip" : BiomedCLIP (512-dim) — original medical pipeline
  - "clip"       : Vanilla CLIP ViT-L/14 (768-dim)
  - "simclip4"   : SimCLIP-4 adversarially robust CLIP (768-dim)
  - "simclip2"   : SimCLIP-2 variant (768-dim)
  - "fare4"      : FARE-4 robust CLIP baseline (768-dim)

Usage:
    python prototype/train_medical.py --query "Any chest X-ray showing pneumonia" --gpu 0
    python prototype/train_medical.py --all --gpu 0
    python prototype/train_medical.py --all --vlm simclip4 --gpu 0
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import tqdm

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prototype.dataset_medical import GeminioChestMNIST

# VLM embedding file paths (relative to data root)
EMBED_PATHS = {
    'biomedclip': 'medical-biomedclip-test.pt',
    'clip': 'medical-clip-test.pt',
    'simclip4': 'medical-simclip4-test.pt',
    'simclip2': 'medical-simclip2-test.pt',
    'fare4': 'medical-fare4-test.pt',
    'fare2': 'medical-fare2-test.pt',
    'tecoa4': 'medical-tecoa4-test.pt',
    'tecoa2': 'medical-tecoa2-test.pt',
}


def _get_text_features(text, vlm, device):
    """Get text features from the appropriate VLM backend."""
    if vlm == 'biomedclip':
        from prototype.vlm_medical import get_text_features
        return get_text_features(text=text, device=device)
    else:
        from prototype.vlm_simclip import get_text_features
        return get_text_features(text=text, vlm=vlm, device=device)


class MedicalGeminioModel(nn.Module):
    """ResNet18 with custom classifier head for medical Geminio prototype."""

    def __init__(self, num_classes=15):
        super().__init__()
        # Use pretrained ResNet18 (lighter than ResNet34 for prototyping)
        self.backbone = models.resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features  # 512

        # Replace classifier with Geminio-style 3-layer head
        self.backbone.fc = nn.Identity()
        self.clf = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.clf(features)


# Medical queries to test
MEDICAL_QUERIES = [
    "Any chest X-ray showing pneumonia",
    "Any chest X-ray showing cardiomegaly with enlarged heart",
    "Any chest X-ray with pleural effusion",
    "Any normal healthy chest X-ray",
    "Any chest X-ray showing a lung mass or tumor",
]


def train_query(query, device, epochs=5, batch_size=16, num_workers=4,
                temperature=100, output_dir='./malicious_models_medical',
                pseudo_labels_path=None, vlm='biomedclip'):
    """Train a malicious model for a single medical query."""
    print(f"\n{'='*60}")
    print(f"Training malicious model for query: \"{query}\"")
    print(f"VLM: {vlm}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # Image transform (resize from 64x64 to 224x224 for ResNet)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Resolve embedding file for selected VLM
    embed_filename = EMBED_PATHS.get(vlm)
    if embed_filename is None:
        raise ValueError(f"Unknown VLM: {vlm}. Choose from: {list(EMBED_PATHS.keys())}")
    embed_path = os.path.join('./data', embed_filename)
    if not os.path.exists(embed_path):
        raise FileNotFoundError(
            f"Embeddings not found: {embed_path}\n"
            f"Run: python prototype/compute_embeddings.py --domain medical --vlm {vlm} --gpu 0"
        )

    # Load dataset with VLM-specific embeddings
    dataset = GeminioChestMNIST(
        root='./data', split='test', train=True, transform=transform, size=64,
        embed_path=embed_path
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # Load model
    model = MedicalGeminioModel(num_classes=dataset.num_classes).to(device)
    model.train()

    # Only train the classifier head (like the original Geminio)
    optimizer = torch.optim.Adam(model.clf.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    epsilon = 1e-8

    # Get query text embedding from matching VLM
    query_embeds = _get_text_features(text=query, vlm=vlm, device=device)

    # Load pseudo-labels if specified (overrides dataset.targets)
    if pseudo_labels_path:
        pseudo_labels = torch.load(pseudo_labels_path)
        dataset.targets = pseudo_labels.numpy() if isinstance(pseudo_labels, torch.Tensor) else pseudo_labels
        print(f"Loaded pseudo-labels from {pseudo_labels_path} ({len(pseudo_labels)} labels)")

    for epoch in range(epochs):
        pbar = tqdm.tqdm(dataloader, total=len(dataloader))
        history_loss = []

        for inputs, input_embeds, targets, _ in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_embeds = input_embeds.to(device)

            # Compute CLIP similarity between each image and the query
            probs = torch.softmax(
                torch.matmul(input_embeds, query_embeds.t()).squeeze() * temperature,
                dim=0
            )

            # Geminio loss: reshape loss surface
            outputs = model(inputs)
            losses = loss_fn(outputs, targets) + epsilon
            losses = losses / losses.sum()
            loss = torch.mean(losses * (1 - probs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            history_loss.append(loss.item())
            pbar.set_description(
                f'[Epoch {epoch}] Loss: {np.mean(history_loss):.6f}'
            )

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    safe_name = query.replace(' ', '_').replace('?', '').replace('"', '')
    # Include VLM, temperature, and pseudo-label info in filename for tracking
    suffix = ''
    if vlm != 'biomedclip':
        suffix += f'_{vlm}'
    if temperature != 100:
        suffix += f'_T{int(temperature)}'
    if pseudo_labels_path:
        suffix += '_pseudo'
    model_path = f'{output_dir}/{safe_name}{suffix}.pt'
    torch.save(model.clf.state_dict(), model_path)
    print(f"Saved malicious model: {model_path}")

    # Analyze: show which images produce highest loss with the trained model
    print(f"\nAnalyzing model behavior for query: \"{query}\"")
    model.eval()
    all_losses = []
    all_sims = []
    all_labels = []

    with torch.no_grad():
        for inputs, input_embeds, targets, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_embeds = input_embeds.to(device)

            outputs = model(inputs)
            per_sample_loss = loss_fn(outputs, targets)
            sims = torch.matmul(input_embeds, query_embeds.t()).squeeze()

            all_losses.extend(per_sample_loss.cpu().tolist())
            all_sims.extend(sims.cpu().tolist())
            all_labels.extend(targets.cpu().tolist())

    all_losses = np.array(all_losses)
    all_sims = np.array(all_sims)
    all_labels = np.array(all_labels)

    # Top-10 highest loss images (these will dominate the gradient)
    top_loss_idx = np.argsort(all_losses)[-10:][::-1]
    label_names = dataset.label_names
    label_names_ext = {int(k): v for k, v in label_names.items()}
    label_names_ext[14] = 'normal'

    print(f"\nTop-10 highest-loss images (will dominate gradient):")
    for rank, idx in enumerate(top_loss_idx):
        label_str = label_names_ext.get(all_labels[idx], f'class_{all_labels[idx]}')
        print(f"  #{rank+1}: loss={all_losses[idx]:.4f}, sim={all_sims[idx]:.4f}, label={label_str}")

    # Compare average loss for query-relevant vs non-relevant
    sim_threshold = np.percentile(all_sims, 90)
    high_sim_mask = all_sims >= sim_threshold
    low_sim_mask = all_sims < sim_threshold

    print(f"\nLoss surface analysis (sim threshold={sim_threshold:.4f}):")
    print(f"  High-similarity images (top 10%): avg loss = {all_losses[high_sim_mask].mean():.4f}")
    print(f"  Low-similarity images (bottom 90%): avg loss = {all_losses[low_sim_mask].mean():.4f}")
    print(f"  Ratio: {all_losses[high_sim_mask].mean() / all_losses[low_sim_mask].mean():.2f}x")

    return model_path


def main():
    parser = argparse.ArgumentParser(description='Train medical Geminio malicious models')
    parser.add_argument('--query', type=str, help='Specific query to train for')
    parser.add_argument('--all', action='store_true', help='Train all example queries')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--temperature', type=float, default=100, help='Softmax temperature scaling')
    parser.add_argument('--output-dir', type=str, default='./malicious_models_medical', help='Output directory')
    parser.add_argument('--pseudo-labels', type=str, default=None, help='Path to pseudo-labels .pt file')
    parser.add_argument('--vlm', type=str, default='biomedclip',
                        choices=['biomedclip', 'clip', 'simclip4', 'simclip2', 'fare4', 'fare2', 'tecoa4', 'tecoa2'],
                        help='VLM backend for text-image similarity (default: biomedclip)')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    if args.all:
        for query in MEDICAL_QUERIES:
            train_query(query, device, epochs=args.epochs, temperature=args.temperature,
                       output_dir=args.output_dir, pseudo_labels_path=args.pseudo_labels,
                       vlm=args.vlm)
    elif args.query:
        train_query(args.query, device, epochs=args.epochs, temperature=args.temperature,
                   output_dir=args.output_dir, pseudo_labels_path=args.pseudo_labels,
                   vlm=args.vlm)
    else:
        # Default: train the first query as a demo
        train_query(MEDICAL_QUERIES[0], device, epochs=args.epochs, temperature=args.temperature,
                   output_dir=args.output_dir, pseudo_labels_path=args.pseudo_labels,
                   vlm=args.vlm)

    print("\n" + "="*60)
    print("Phase 2 (Medical) complete!")
    print("="*60)


if __name__ == '__main__':
    main()
