"""
Generate BiomedCLIP zero-shot pseudo-labels for ChestMNIST.

Uses precomputed image embeddings and class text embeddings to assign
pseudo-labels via cosine similarity (argmax). Reports accuracy vs
ground truth labels.

Usage:
    python prototype/generate_pseudo_labels.py
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')


def generate_pseudo_labels():
    # Load precomputed embeddings
    img_embeds = torch.load(os.path.join(DATA_ROOT, 'medical-biomedclip-test.pt'))  # [N, 512]
    meta = torch.load(os.path.join(DATA_ROOT, 'medical-biomedclip-meta.pt'))
    class_embeds = meta['class_embeds']  # [14, 512]
    label_names = meta['label_names']  # {'0': 'atelectasis', ...}

    print(f"Image embeddings: {img_embeds.shape}")
    print(f"Class embeddings: {class_embeds.shape}")

    # Add "normal/no finding" as class 14
    # Generate a text embedding for "normal" using the mean of all class embeds as placeholder
    # Or better: use BiomedCLIP to encode "normal healthy chest X-ray"
    from prototype.vlm_medical import get_text_features
    normal_embed = get_text_features("normal healthy chest X-ray with no abnormalities", device='cpu')
    class_embeds_full = torch.cat([class_embeds, normal_embed], dim=0)  # [15, 512]
    print(f"Class embeddings with normal: {class_embeds_full.shape}")

    # Compute cosine similarity: [N, 15]
    img_embeds_norm = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
    class_embeds_norm = class_embeds_full / class_embeds_full.norm(p=2, dim=-1, keepdim=True)
    similarities = img_embeds_norm @ class_embeds_norm.t()  # [N, 15]

    # Pseudo-labels = argmax
    pseudo_labels = similarities.argmax(dim=1)  # [N]
    print(f"Pseudo-labels shape: {pseudo_labels.shape}")

    # Load ground truth for comparison
    from prototype.dataset_medical import GeminioChestMNIST
    dataset = GeminioChestMNIST(root='./data', split='test', train=None, transform=None, size=64)
    true_labels = torch.tensor(dataset.targets)

    # Accuracy
    correct = (pseudo_labels == true_labels).sum().item()
    total = len(true_labels)
    accuracy = correct / total
    print(f"\nPseudo-label accuracy: {correct}/{total} = {accuracy:.4f}")

    # Per-class accuracy
    ext_names = {int(k): v for k, v in label_names.items()}
    ext_names[14] = 'normal'
    print(f"\nPer-class accuracy:")
    for cls in sorted(set(true_labels.numpy())):
        mask = true_labels == cls
        cls_correct = (pseudo_labels[mask] == cls).sum().item()
        cls_total = mask.sum().item()
        cls_acc = cls_correct / cls_total if cls_total > 0 else 0
        cls_name = ext_names.get(cls, f'class_{cls}')
        print(f"  {cls_name} (class {cls}): {cls_correct}/{cls_total} = {cls_acc:.4f}")

    # Distribution of pseudo-labels
    print(f"\nPseudo-label distribution:")
    for cls in range(15):
        count = (pseudo_labels == cls).sum().item()
        cls_name = ext_names.get(cls, f'class_{cls}')
        print(f"  {cls_name} (class {cls}): {count}")

    # Save
    output_path = os.path.join(DATA_ROOT, 'medical_pseudo_labels.pt')
    torch.save(pseudo_labels, output_path)
    print(f"\nSaved pseudo-labels to {output_path}")

    # Also save as numpy for inspection
    np.save(os.path.join(DATA_ROOT, 'medical_pseudo_labels.npy'), pseudo_labels.numpy())

    return pseudo_labels


if __name__ == '__main__':
    generate_pseudo_labels()
