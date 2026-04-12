"""
Prepare controlled batch compositions for ablation experiments.

Creates sample directories with specific class compositions:
1. Batch composition experiment: with/without target class
2. Batch size experiment: 16 and 32 sample batches

Usage:
    python prototype/prepare_controlled_batches.py --gpu 0
"""
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prototype.dataset_medical import GeminioChestMNIST
from prototype.dataset_uav import GeminioUAVScenes, UAV_CLASSES, CLASS_ID_TO_IDX

DATA_ROOT_UAV = '/raid/scratch/dzimmerman2021/uavscenes'
UAV_SCENES = ['interval5_AMtown01', 'interval5_HKairport01']

# Reverse map: contiguous idx -> class name
IDX_TO_NAME_UAV = {}
for cls_id, cls_idx in CLASS_ID_TO_IDX.items():
    IDX_TO_NAME_UAV[cls_idx] = UAV_CLASSES[cls_id]


def save_medical_batch(dataset, indices, output_dir, label_names):
    """Save selected medical images to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    all_targets = dataset.targets

    for i, idx in enumerate(indices):
        img = dataset.images[idx]
        target = int(all_targets[idx])
        label_str = label_names.get(target, f'class_{target}')

        if img.ndim == 2:
            pil_img = Image.fromarray(img).convert('RGB')
        else:
            pil_img = Image.fromarray(img).convert('RGB')
        pil_img = pil_img.resize((224, 224), Image.BILINEAR)

        filename = f"{i}-{target}.png"
        pil_img.save(os.path.join(output_dir, filename))
        print(f"  {filename} — {label_str}")

    print(f"  Saved {len(indices)} images to {output_dir}")


def save_uav_batch(dataset, indices, output_dir):
    """Save selected UAV images to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    labels = dataset.targets

    for i, idx in enumerate(indices):
        img_path = dataset.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)

        label_vec = labels[idx]
        active_classes = np.where(label_vec == 1)[0]
        class_names = [IDX_TO_NAME_UAV.get(c, f'cls{c}') for c in active_classes]

        label_str = "_".join(str(c) for c in active_classes)
        filename = f"{i}-{label_str}.png"
        img.save(os.path.join(output_dir, filename))
        print(f"  {filename} — {', '.join(class_names)}")

    # Save label matrix
    selected_labels = labels[np.array(indices)]
    np.save(os.path.join(output_dir, 'labels.npy'), selected_labels)
    print(f"  Saved {len(indices)} images + labels.npy to {output_dir}")


def prepare_medical_controlled(num_samples=8):
    """Create medical batches with/without pneumonia (class 7)."""
    print("\n=== Preparing Medical Controlled Batches ===")
    dataset = GeminioChestMNIST(root='./data', split='test', train=None, transform=None, size=64)
    all_targets = dataset.targets
    label_names = {int(k): v for k, v in dataset.label_names.items()}
    label_names[14] = 'normal'

    # ChestMNIST class mapping: 6=pneumonia, 7=pneumothorax (NOT pneumonia)
    target_class = 6  # pneumonia (verified via ChestMNIST info['label'])
    pneumonia_idx = np.where(all_targets == target_class)[0]
    non_pneumonia_idx = np.where(all_targets != target_class)[0]

    print(f"Total samples: {len(all_targets)}")
    print(f"Pneumonia (class {target_class}): {len(pneumonia_idx)}")
    print(f"Non-pneumonia: {len(non_pneumonia_idx)}")

    # --- WITH pneumonia: 3 pneumonia + 5 diverse others ---
    np.random.seed(42)
    pneumonia_selected = np.random.choice(pneumonia_idx, min(3, len(pneumonia_idx)), replace=False)

    # Get diverse non-pneumonia samples (one per class)
    other_classes = sorted(set(all_targets[non_pneumonia_idx]))
    diverse_others = []
    for cls in other_classes:
        if len(diverse_others) >= num_samples - len(pneumonia_selected):
            break
        cls_indices = np.where(all_targets == cls)[0]
        diverse_others.append(int(cls_indices[0]))

    with_pneumonia = list(pneumonia_selected) + diverse_others
    with_pneumonia = with_pneumonia[:num_samples]

    print(f"\nBatch WITH pneumonia ({len(with_pneumonia)} samples):")
    save_medical_batch(dataset, with_pneumonia, './assets/medical_controlled/pneumonia_with/', label_names)

    # --- WITHOUT pneumonia: diverse from non-pneumonia classes ---
    without_pneumonia = []
    for cls in other_classes:
        if len(without_pneumonia) >= num_samples:
            break
        cls_indices = np.where(all_targets == cls)[0]
        # Pick a different sample than the one used in the "with" batch
        for idx in cls_indices:
            if idx not in with_pneumonia:
                without_pneumonia.append(int(idx))
                break

    # Fill remaining if needed
    if len(without_pneumonia) < num_samples:
        remaining = set(non_pneumonia_idx.tolist()) - set(without_pneumonia)
        extra = np.random.choice(list(remaining), num_samples - len(without_pneumonia), replace=False)
        without_pneumonia.extend(extra.tolist())
    without_pneumonia = without_pneumonia[:num_samples]

    print(f"\nBatch WITHOUT pneumonia ({len(without_pneumonia)} samples):")
    save_medical_batch(dataset, without_pneumonia, './assets/medical_controlled/pneumonia_without/', label_names)


def prepare_uav_controlled(num_samples=8):
    """Create UAV batches with/without solar panels (contiguous idx 11)."""
    print("\n=== Preparing UAV Controlled Batches ===")
    scene_tag = "_".join(UAV_SCENES)
    embed_path = os.path.join(DATA_ROOT_UAV, f'uav_clip_embeddings_{scene_tag}.pt')

    dataset = GeminioUAVScenes(
        data_root=DATA_ROOT_UAV, scenes=UAV_SCENES, embed_path=embed_path,
        transform=None, train=False,
    )
    labels = dataset.targets  # [N, 18]

    target_idx = 11  # solar_board
    has_solar = np.where(labels[:, target_idx] == 1)[0]
    no_solar = np.where(labels[:, target_idx] == 0)[0]

    print(f"Total samples: {len(labels)}")
    print(f"With solar panels (idx {target_idx}): {len(has_solar)}")
    print(f"Without solar panels: {len(no_solar)}")

    # --- WITH solar: pick 3 with solar + 5 diverse others ---
    np.random.seed(42)
    if len(has_solar) >= 3:
        solar_selected = np.random.choice(has_solar, 3, replace=False).tolist()
    else:
        solar_selected = has_solar.tolist()

    # Fill remaining with diverse non-solar samples
    remaining_needed = num_samples - len(solar_selected)
    other_pool = [i for i in range(len(labels)) if i not in solar_selected]
    np.random.shuffle(other_pool)
    with_solar = solar_selected + other_pool[:remaining_needed]
    with_solar = with_solar[:num_samples]

    print(f"\nBatch WITH solar panels ({len(with_solar)} samples):")
    save_uav_batch(dataset, with_solar, './assets/uav_controlled/solar_with/')

    # --- WITHOUT solar: pick only from no_solar ---
    np.random.seed(43)
    without_solar_selected = np.random.choice(no_solar, min(num_samples, len(no_solar)), replace=False).tolist()
    without_solar_selected = without_solar_selected[:num_samples]

    print(f"\nBatch WITHOUT solar panels ({len(without_solar_selected)} samples):")
    save_uav_batch(dataset, without_solar_selected, './assets/uav_controlled/solar_without/')


def prepare_medical_batchsizes():
    """Create medical batches with 16 and 32 samples."""
    print("\n=== Preparing Medical Batch Size Variants ===")
    dataset = GeminioChestMNIST(root='./data', split='test', train=None, transform=None, size=64)
    all_targets = dataset.targets
    label_names = {int(k): v for k, v in dataset.label_names.items()}
    label_names[14] = 'normal'

    np.random.seed(42)
    all_classes = sorted(set(all_targets))

    for batch_size in [16, 32]:
        selected = []
        # Round-robin through classes for diversity
        class_pools = {c: np.where(all_targets == c)[0].tolist() for c in all_classes}
        while len(selected) < batch_size:
            for cls in all_classes:
                if len(selected) >= batch_size:
                    break
                if class_pools[cls]:
                    selected.append(class_pools[cls].pop(0))

        output_dir = f'./assets/medical_batch{batch_size}/'
        print(f"\nBatch size {batch_size} ({len(selected)} samples):")
        save_medical_batch(dataset, selected, output_dir, label_names)


def prepare_uav_batchsizes():
    """Create UAV batches with 16 and 32 samples."""
    print("\n=== Preparing UAV Batch Size Variants ===")
    scene_tag = "_".join(UAV_SCENES)
    embed_path = os.path.join(DATA_ROOT_UAV, f'uav_clip_embeddings_{scene_tag}.pt')

    dataset = GeminioUAVScenes(
        data_root=DATA_ROOT_UAV, scenes=UAV_SCENES, embed_path=embed_path,
        transform=None, train=False,
    )
    labels = dataset.targets

    np.random.seed(42)
    all_indices = list(range(len(labels)))
    np.random.shuffle(all_indices)

    for batch_size in [16, 32]:
        selected = all_indices[:batch_size]
        output_dir = f'./assets/uav_batch{batch_size}/'
        print(f"\nBatch size {batch_size} ({len(selected)} samples):")
        save_uav_batch(dataset, selected, output_dir)


if __name__ == '__main__':
    prepare_medical_controlled()
    prepare_uav_controlled()
    prepare_medical_batchsizes()
    prepare_uav_batchsizes()
    print("\n=== All controlled batches prepared ===")
