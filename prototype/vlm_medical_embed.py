"""
Phase 1 (Medical): Generate BiomedCLIP embeddings for ChestMNIST images.

Replaces vlm-imagenet-embed.py but uses:
  - BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) instead of CLIP
  - ChestMNIST (chest X-ray images) instead of ImageNet

Output:
  - ./data/medical-biomedclip-test.pt  (image embeddings, shape [N, 512])
  - ./data/medical-biomedclip-meta.pt  (class/label text embeddings)
"""
import os
import sys
import torch
import tqdm
import numpy as np
from PIL import Image

# BiomedCLIP uses open_clip
import open_clip

# MedMNIST
import medmnist
from medmnist import ChestMNIST

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')
MEDMNIST_ROOT = os.path.join(DATA_ROOT, 'medmnist')
BATCH_SIZE = 64

# ---- Load BiomedCLIP ----
print("Loading BiomedCLIP model...")
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
tokenizer = open_clip.get_tokenizer(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
model = model.to(DEVICE)
model.eval()
print(f"BiomedCLIP loaded on {DEVICE}")

# ---- Load ChestMNIST ----
# We use the test split as our "private data" pool (like ImageNet val in the original)
print("Loading ChestMNIST...")
# Load raw data (no transform) to get labels, then apply BiomedCLIP transform for embeddings
dataset_raw = ChestMNIST(split='test', download=True, root=MEDMNIST_ROOT, size=64)
print(f"# of test samples: {len(dataset_raw)}")
print(f"Label info: {dataset_raw.info['label']}")

# ChestMNIST stores images as numpy arrays; we need to convert to PIL for BiomedCLIP
images_np = dataset_raw.imgs  # shape: [N, 64, 64] for grayscale
labels_np = dataset_raw.labels  # shape: [N, 14] multi-label

# ---- Generate Image Embeddings ----
print(f"\nGenerating BiomedCLIP image embeddings for {len(images_np)} images...")
all_embeddings = []

for i in tqdm.tqdm(range(0, len(images_np), BATCH_SIZE)):
    batch_imgs = images_np[i:i+BATCH_SIZE]

    # Convert grayscale numpy arrays to RGB PIL images, then apply BiomedCLIP transform
    pil_images = []
    for img in batch_imgs:
        if img.ndim == 2:
            # Grayscale -> RGB (3-channel)
            pil_img = Image.fromarray(img).convert('RGB')
        else:
            pil_img = Image.fromarray(img).convert('RGB')
        pil_images.append(preprocess_val(pil_img))

    batch_tensor = torch.stack(pil_images).to(DEVICE)

    with torch.no_grad():
        image_features = model.encode_image(batch_tensor)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    all_embeddings.append(image_features.cpu())

all_embeddings = torch.cat(all_embeddings, dim=0)
print(f"Image embeddings shape: {all_embeddings.shape}")  # [N, 512]

# Save
embed_path = os.path.join(DATA_ROOT, 'medical-biomedclip-test.pt')
torch.save(all_embeddings, embed_path)
print(f"Saved image embeddings to {embed_path}")

# ---- Generate Text Embeddings for Disease Labels ----
print("\nGenerating text embeddings for disease labels...")
label_names = dataset_raw.info['label']  # dict: {'0': 'atelectasis', '1': 'cardiomegaly', ...}

# Create descriptive text prompts for each label (medical context helps BiomedCLIP)
label_descriptions = {
    'atelectasis': 'chest X-ray showing atelectasis with lung collapse',
    'cardiomegaly': 'chest X-ray showing cardiomegaly with enlarged heart',
    'effusion': 'chest X-ray showing pleural effusion',
    'infiltration': 'chest X-ray showing pulmonary infiltration',
    'mass': 'chest X-ray showing a pulmonary mass or lung mass',
    'nodule': 'chest X-ray showing a pulmonary nodule',
    'pneumonia': 'chest X-ray showing pneumonia with lung infection',
    'pneumothorax': 'chest X-ray showing pneumothorax',
    'consolidation': 'chest X-ray showing lung consolidation',
    'edema': 'chest X-ray showing pulmonary edema',
    'emphysema': 'chest X-ray showing emphysema',
    'fibrosis': 'chest X-ray showing pulmonary fibrosis',
    'pleural': 'chest X-ray showing pleural thickening',
    'hernia': 'chest X-ray showing hiatal hernia',
}

meta = {'class_embeds': [], 'label_names': label_names, 'label_descriptions': label_descriptions}
for idx in sorted(label_names.keys(), key=int):
    name = label_names[idx]
    desc = label_descriptions.get(name, f'chest X-ray showing {name}')

    tokens = tokenizer([desc]).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    meta['class_embeds'].append(text_features.cpu())

meta['class_embeds'] = torch.cat(meta['class_embeds'], dim=0)
print(f"Class embeddings shape: {meta['class_embeds'].shape}")  # [14, 512]

# Also pre-compute some example attack queries
example_queries = [
    "Any chest X-ray showing pneumonia",
    "Any chest X-ray showing cardiomegaly with enlarged heart",
    "Any chest X-ray with pleural effusion",
    "Any normal healthy chest X-ray",
    "Any chest X-ray showing a lung mass or tumor",
]
query_embeds = {}
for q in example_queries:
    tokens = tokenizer([q]).to(DEVICE)
    with torch.no_grad():
        qf = model.encode_text(tokens)
        qf = qf / qf.norm(p=2, dim=-1, keepdim=True)
    query_embeds[q] = qf.cpu()

    # Show which images are most similar
    sims = (all_embeddings @ qf.cpu().t()).squeeze()
    top_k = sims.topk(5)
    print(f"\nQuery: \"{q}\"")
    print(f"  Top-5 similarities: {top_k.values.tolist()}")
    print(f"  Top-5 indices: {top_k.indices.tolist()}")
    # Show what labels those images actually have
    for idx_val in top_k.indices:
        lbl = labels_np[idx_val]
        active = [label_names[str(j)] for j in range(14) if lbl[j] == 1]
        print(f"    Image {idx_val}: labels = {active if active else ['normal/no finding']}")

meta['query_embeds'] = query_embeds
meta_path = os.path.join(DATA_ROOT, 'medical-biomedclip-meta.pt')
torch.save(meta, meta_path)
print(f"\nSaved metadata to {meta_path}")

# Save labels separately for easy access
labels_path = os.path.join(DATA_ROOT, 'medical-labels.pt')
torch.save({'labels': torch.tensor(labels_np), 'label_names': label_names}, labels_path)
print(f"Saved labels to {labels_path}")

print("\nPhase 1 (Medical) complete!")
