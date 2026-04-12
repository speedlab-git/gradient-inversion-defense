"""
Prepare private medical sample images for gradient inversion reconstruction.

Extracts ChestMNIST test images and saves as PNG files in the format
expected by CustomData: {index}-{class}.png

Usage:
    python prototype/prepare_medical_samples.py --gpu 0
"""
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prototype.dataset_medical import GeminioChestMNIST


def prepare_samples(output_dir='./assets/medical_samples/', num_samples=8):
    """Extract diverse ChestMNIST images as PNGs."""
    os.makedirs(output_dir, exist_ok=True)

    # Load full dataset (no transform, no train/test split — use all data for diversity)
    dataset = GeminioChestMNIST(root='./data', split='test', train=None, transform=None, size=64)

    # Get class distribution across ALL samples
    all_targets = dataset.targets  # numpy array of single-label targets
    label_names = {int(k): v for k, v in dataset.label_names.items()}
    label_names[14] = 'normal'

    # Pick diverse samples: try to get different conditions
    # Priority: pneumonia(7), cardiomegaly(2), effusion(10), mass(5), normal(14), atelectasis(0), etc.
    priority_classes = [7, 2, 10, 5, 14, 0, 1, 3]
    selected = []

    for cls in priority_classes:
        if len(selected) >= num_samples:
            break
        indices = np.where(all_targets == cls)[0]
        if len(indices) > 0:
            selected.append(int(indices[0]))

    # Fill remaining with random samples from other classes
    remaining = num_samples - len(selected)
    if remaining > 0:
        all_idx = set(range(len(all_targets))) - set(selected)
        extra = np.random.choice(list(all_idx), remaining, replace=False)
        selected.extend(extra.tolist())

    selected = selected[:num_samples]

    print(f"Saving {len(selected)} medical samples to {output_dir}")
    for i, idx in enumerate(selected):
        img = dataset.images[idx]
        target = int(all_targets[idx])
        label_str = label_names.get(target, f'class_{target}')

        # Convert to PIL and resize to 224x224
        if img.ndim == 2:
            pil_img = Image.fromarray(img).convert('RGB')
        else:
            pil_img = Image.fromarray(img).convert('RGB')
        pil_img = pil_img.resize((224, 224), Image.BILINEAR)

        # Save as {index}-{class}.png
        filename = f"{i}-{target}.png"
        pil_img.save(os.path.join(output_dir, filename))
        print(f"  {filename} — {label_str}")

    print(f"Done! {len(selected)} samples saved.")
    return output_dir


if __name__ == '__main__':
    prepare_samples()
