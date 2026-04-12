"""
UAVScenes dataset for Geminio prototype.

Converts UAVScenes semantic segmentation labels into multi-label classification
(which of 19 classes are present in each image) for use with the Geminio training pipeline.

Returns (image, img_embed, target, target) tuples for compatibility.
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


# UAVScenes class map (from cmap.py) — only named classes
UAV_CLASSES = {
    1: 'roof',
    2: 'dirt_motor_road',
    3: 'paved_motor_road',
    4: 'river',
    5: 'pool',
    6: 'bridge',
    9: 'container',
    10: 'airstrip',
    11: 'traffic_barrier',
    13: 'green_field',
    14: 'wild_field',
    15: 'solar_board',
    16: 'umbrella',
    17: 'transparent_roof',
    18: 'car_park',
    19: 'paved_walk',
    20: 'sedan',
    24: 'truck',
}

# Map original class IDs to contiguous indices 0-17
CLASS_ID_TO_IDX = {cid: idx for idx, cid in enumerate(sorted(UAV_CLASSES.keys()))}
IDX_TO_CLASS_NAME = {idx: UAV_CLASSES[cid] for cid, idx in CLASS_ID_TO_IDX.items()}
NUM_CLASSES = len(UAV_CLASSES)  # 18


def label_map_to_multilabel(label_path, min_pixel_fraction=0.001):
    """Convert a segmentation label map to a multi-label binary vector.

    Args:
        label_path: Path to the label ID PNG (uint8, pixel values = class IDs)
        min_pixel_fraction: Minimum fraction of pixels for a class to count as present

    Returns:
        np.array of shape [NUM_CLASSES] with binary values
    """
    label_img = np.array(Image.open(label_path))
    total_pixels = label_img.size
    unique, counts = np.unique(label_img, return_counts=True)

    multilabel = np.zeros(NUM_CLASSES, dtype=np.float32)
    for cls_id, cnt in zip(unique, counts):
        if cls_id in CLASS_ID_TO_IDX and cnt / total_pixels >= min_pixel_fraction:
            multilabel[CLASS_ID_TO_IDX[cls_id]] = 1.0

    return multilabel


class GeminioUAVScenes(Dataset):
    """UAVScenes dataset with pre-computed CLIP embeddings for Geminio."""

    def __init__(self, data_root, scenes, embed_path=None, transform=None,
                 train=None, min_pixel_fraction=0.001):
        """
        Args:
            data_root: Root of UAVScenes data (contains interval5_CAM_LIDAR/ and interval5_CAM_label/)
            scenes: List of scene names, e.g. ['interval5_AMtown01', 'interval5_HKairport01']
            embed_path: Path to pre-computed CLIP embeddings .pt file
            transform: Image transform to apply
            train: If True, use first half; if False, second half; None = all
            min_pixel_fraction: Min pixel fraction for a class to count as present
        """
        self.data_root = data_root
        self.transform = transform
        self.num_classes = NUM_CLASSES
        self.label_names = IDX_TO_CLASS_NAME

        # Collect image-label pairs from all scenes
        self.image_paths = []
        self.label_paths = []

        for scene in scenes:
            cam_dir = os.path.join(data_root, 'interval5_CAM_LIDAR', scene, 'interval5_CAM')
            label_dir = os.path.join(data_root, 'interval5_CAM_label', scene, 'interval5_CAM_label_id')

            if not os.path.isdir(cam_dir) or not os.path.isdir(label_dir):
                print(f"WARNING: Skipping {scene} — missing cam or label dir")
                continue

            cam_files = set(f.replace('.jpg', '') for f in os.listdir(cam_dir) if f.endswith('.jpg'))
            label_files = set(f.replace('.png', '') for f in os.listdir(label_dir) if f.endswith('.png'))
            common = sorted(cam_files & label_files)

            for ts in common:
                self.image_paths.append(os.path.join(cam_dir, ts + '.jpg'))
                self.label_paths.append(os.path.join(label_dir, ts + '.png'))

        print(f"Loaded {len(self.image_paths)} image-label pairs from {len(scenes)} scenes")

        # Compute multi-label targets
        cache_path = os.path.join(data_root, f'uav_multilabels_{"_".join(scenes)}.npy')
        if os.path.exists(cache_path):
            print(f"Loading cached multi-labels from {cache_path}")
            self.targets = np.load(cache_path)
        else:
            print("Computing multi-label targets from segmentation maps...")
            self.targets = np.zeros((len(self.label_paths), NUM_CLASSES), dtype=np.float32)
            for i, lp in enumerate(self.label_paths):
                self.targets[i] = label_map_to_multilabel(lp, min_pixel_fraction)
                if (i + 1) % 500 == 0:
                    print(f"  Processed {i+1}/{len(self.label_paths)} labels")
            np.save(cache_path, self.targets)
            print(f"Saved multi-labels to {cache_path}")

        # Load pre-computed CLIP embeddings
        self.img_embeds = None
        if embed_path and os.path.exists(embed_path):
            print(f"Loading CLIP embeddings from {embed_path}")
            self.img_embeds = torch.load(embed_path)

        # Train/test split
        self.indices = list(range(len(self.image_paths)))
        if train is not None:
            mid = len(self.indices) // 2
            self.indices = self.indices[:mid] if train else self.indices[mid:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        real_idx = self.indices[index]

        img = Image.open(self.image_paths[real_idx]).convert('RGB')
        target = torch.from_numpy(self.targets[real_idx])  # [NUM_CLASSES] float32

        if self.transform is not None:
            img = self.transform(img)

        if self.img_embeds is not None:
            img_embed = self.img_embeds[real_idx]
        else:
            img_embed = torch.zeros(768)  # placeholder

        return img, img_embed, target, target
