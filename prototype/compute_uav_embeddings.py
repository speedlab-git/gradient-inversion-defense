"""
Pre-compute CLIP ViT-L/14 embeddings for UAVScenes images.

Uses the same CLIP model as the original Geminio (openai/clip-vit-large-patch14, 768-dim).
"""
import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLIP_MODEL = "openai/clip-vit-large-patch14"


def compute_embeddings(data_root, scenes, device, batch_size=32):
    """Compute CLIP image embeddings for all images in the specified scenes."""
    # Collect image paths
    image_paths = []
    for scene in scenes:
        cam_dir = os.path.join(data_root, 'interval5_CAM_LIDAR', scene, 'interval5_CAM')
        label_dir = os.path.join(data_root, 'interval5_CAM_label', scene, 'interval5_CAM_label_id')

        cam_files = set(f.replace('.jpg', '') for f in os.listdir(cam_dir) if f.endswith('.jpg'))
        label_files = set(f.replace('.png', '') for f in os.listdir(label_dir) if f.endswith('.png'))
        common = sorted(cam_files & label_files)

        for ts in common:
            image_paths.append(os.path.join(cam_dir, ts + '.jpg'))

    print(f"Computing CLIP embeddings for {len(image_paths)} images")
    print(f"Using model: {CLIP_MODEL}")
    print(f"Device: {device}")

    # Load CLIP model
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model.eval()

    all_embeds = []
    for i in tqdm.tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[i:i + batch_size]
        images = [Image.open(p).convert('RGB') for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)

        all_embeds.append(outputs.cpu())

    all_embeds = torch.cat(all_embeds, dim=0)
    print(f"Embeddings shape: {all_embeds.shape}")

    # Save
    scene_tag = "_".join(scenes)
    save_path = os.path.join(data_root, f'uav_clip_embeddings_{scene_tag}.pt')
    torch.save(all_embeds, save_path)
    print(f"Saved to {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/raid/scratch/dzimmerman2021/uavscenes')
    parser.add_argument('--scenes', nargs='+',
                        default=['interval5_AMtown01', 'interval5_HKairport01'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    compute_embeddings(args.data_root, args.scenes, device, args.batch_size)


if __name__ == '__main__':
    main()
