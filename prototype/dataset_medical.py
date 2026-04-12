"""
Medical dataset for Geminio prototype.

Mirrors the GeminioImageNet dataset interface but uses ChestMNIST + BiomedCLIP embeddings.
Returns (image, img_embed, target, target) tuples for compatibility with the training pipeline.
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import medmnist
from medmnist import ChestMNIST


class GeminioChestMNIST(Dataset):
    """ChestMNIST dataset with pre-computed BiomedCLIP embeddings for Geminio."""

    def __init__(self, root='./data', split='test', train=None, transform=None, size=64,
                 embed_path=None):
        """
        Args:
            root: Root data directory
            split: 'train', 'val', or 'test'
            train: If True, use first half of each class for training; if False, second half.
                   If None, use all data.
            transform: Image transform to apply
            size: MedMNIST image size (28 or 64)
            embed_path: Path to pre-computed embeddings .pt file.
                        Defaults to '{root}/medical-biomedclip-test.pt' for backward compatibility.
        """
        self.root = root
        self.transform = transform

        medmnist_root = os.path.join(root, 'medmnist')
        self.dataset = ChestMNIST(split=split, download=True, root=medmnist_root, size=size)
        self.info = self.dataset.info
        self.label_names = self.info['label']

        # Load pre-computed VLM embeddings
        if embed_path is None:
            embed_path = os.path.join(root, 'medical-biomedclip-test.pt')
        self.img_embeds = torch.load(embed_path)

        # ChestMNIST is multi-label (14 binary labels). For classification we use
        # a simple single-label: the index of the first active label, or 14 for "normal"
        self.images = self.dataset.imgs  # numpy array [N, H, W] or [N, H, W, C]
        self.labels_multi = self.dataset.labels  # [N, 14]
        self.num_classes = 15  # 14 diseases + 1 normal

        # Convert multi-label to single-label (primary diagnosis)
        self.targets = []
        for lbl in self.labels_multi:
            active = np.where(lbl == 1)[0]
            if len(active) == 0:
                self.targets.append(14)  # normal
            else:
                self.targets.append(active[0])  # primary diagnosis
        self.targets = np.array(self.targets)

        # Build sample list (index, target) pairs
        self.samples = list(range(len(self.images)))

        if train is not None:
            # Split by target class for train/test
            cls_to_indices = {}
            for idx, target in enumerate(self.targets):
                if target not in cls_to_indices:
                    cls_to_indices[target] = []
                cls_to_indices[target].append(idx)

            selected = []
            for cls, indices in cls_to_indices.items():
                split_pt = len(indices) // 2
                if train:
                    selected.extend(indices[:split_pt])
                else:
                    selected.extend(indices[split_pt:])

            self.samples = selected
            self.img_embeds_subset = self.img_embeds[selected]
        else:
            self.img_embeds_subset = self.img_embeds

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        real_idx = self.samples[index]

        img = self.images[real_idx]
        target = int(self.targets[real_idx])

        if self.img_embeds_subset is not None:
            img_embed = self.img_embeds_subset[index]
        else:
            img_embed = self.img_embeds[real_idx]

        # Convert numpy to PIL image
        if img.ndim == 2:
            pil_img = Image.fromarray(img).convert('RGB')
        else:
            pil_img = Image.fromarray(img).convert('RGB')

        if self.transform is not None:
            sample = self.transform(pil_img)
        else:
            sample = pil_img

        return sample, img_embed, target, target
