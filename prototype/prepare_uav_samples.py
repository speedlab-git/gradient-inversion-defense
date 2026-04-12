"""
Prepare private UAVScenes sample images for gradient inversion reconstruction.

Selects diverse UAVScenes images and saves as PNG files. For multi-label,
encodes the label vector as a comma-separated string in the filename.

Usage:
    python prototype/prepare_uav_samples.py
"""
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prototype.dataset_uav import GeminioUAVScenes, UAV_CLASSES, CLASS_ID_TO_IDX

DATA_ROOT = '/raid/scratch/dzimmerman2021/uavscenes'
SCENES = ['interval5_AMtown01', 'interval5_HKairport01']


def prepare_samples(output_dir='./assets/uav_samples/', num_samples=8):
    """Extract diverse UAVScenes images as PNGs."""
    os.makedirs(output_dir, exist_ok=True)

    scene_tag = "_".join(SCENES)
    embed_path = os.path.join(DATA_ROOT, f'uav_clip_embeddings_{scene_tag}.pt')

    dataset = GeminioUAVScenes(
        data_root=DATA_ROOT, scenes=SCENES, embed_path=embed_path,
        transform=None, train=False,
    )

    # Build reverse index: class_idx -> list of image indices containing that class
    labels = dataset.targets  # [N, 18]
    idx_to_name = {}
    for cls_id, cls_idx in CLASS_ID_TO_IDX.items():
        idx_to_name[cls_idx] = UAV_CLASSES[cls_id]

    # Priority: pool(4), solar_board(11), bridge(5), truck(17), airstrip(9), container(8)
    # These match our trained queries
    priority_classes = [4, 11, 5, 17, 9, 8]
    selected = []

    for cls_idx in priority_classes:
        if len(selected) >= num_samples:
            break
        # Find images containing this class
        has_class = np.where(labels[:, cls_idx] == 1)[0]
        if len(has_class) > 0:
            # Pick first one not already selected
            for idx in has_class:
                if idx not in selected:
                    selected.append(idx)
                    break

    # Fill remaining with random diverse samples
    remaining = num_samples - len(selected)
    if remaining > 0:
        all_idx = set(range(len(dataset))) - set(selected)
        extra = np.random.choice(list(all_idx), remaining, replace=False)
        selected.extend(extra.tolist())

    selected = selected[:num_samples]

    print(f"Saving {len(selected)} UAV samples to {output_dir}")
    for i, idx in enumerate(selected):
        img_path = dataset.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)

        label_vec = labels[idx]
        active_classes = np.where(label_vec == 1)[0]
        class_names = [idx_to_name.get(c, f'cls{c}') for c in active_classes]

        # Save with multi-label encoding: {index}-{comma_separated_class_indices}.png
        label_str = "_".join(str(c) for c in active_classes)
        filename = f"{i}-{label_str}.png"
        img.save(os.path.join(output_dir, filename))
        print(f"  {filename} — {', '.join(class_names)}")

    # Also save the full label matrix for the selected samples
    selected_labels = labels[selected]
    np.save(os.path.join(output_dir, 'labels.npy'), selected_labels)
    print(f"Saved label matrix: {selected_labels.shape}")
    print(f"Done! {len(selected)} samples saved.")
    return output_dir


if __name__ == '__main__':
    prepare_samples()
