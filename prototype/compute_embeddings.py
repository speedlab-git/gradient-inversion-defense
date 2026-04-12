"""
Unified Phase 1: Pre-compute VLM image embeddings for any domain + VLM variant.

Supports:
  - Medical (ChestMNIST) with BiomedCLIP or CLIP variants (SimCLIP, FARE, vanilla)
  - UAV (UAVScenes) with CLIP variants (SimCLIP, FARE, vanilla)

The output .pt file is named with the VLM variant for easy switching:
  - data/medical-biomedclip-test.pt          (original BiomedCLIP, 512-dim)
  - data/medical-clip-test.pt                (vanilla CLIP, 768-dim)
  - data/medical-simclip4-test.pt            (SimCLIP-4, 768-dim)
  - data/medical-fare4-test.pt               (FARE-4, 768-dim)
  - uavscenes/uav_simclip4_embeddings_*.pt   (SimCLIP-4 for UAV, 768-dim)

Usage:
    # Medical domain with SimCLIP-4
    python prototype/compute_embeddings.py --domain medical --vlm simclip4 --gpu 0

    # UAV domain with FARE-4
    python prototype/compute_embeddings.py --domain uav --vlm fare4 --gpu 0

    # Medical domain with vanilla CLIP (768-dim, for comparison with BiomedCLIP 512-dim)
    python prototype/compute_embeddings.py --domain medical --vlm clip --gpu 0
"""
import argparse
import os
import sys
import torch
import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prototype.vlm_simclip import get_vlm_model, get_image_features

# Defaults
MEDICAL_DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')
UAV_DATA_ROOT = '/raid/scratch/dzimmerman2021/uavscenes'
UAV_SCENES = ['interval5_AMtown01', 'interval5_HKairport01']


def compute_medical_embeddings(vlm, device, batch_size=64):
    """Compute image embeddings for ChestMNIST test set."""
    import medmnist
    from medmnist import ChestMNIST

    medmnist_root = os.path.join(MEDICAL_DATA_ROOT, 'medmnist')
    dataset = ChestMNIST(split='test', download=True, root=medmnist_root, size=64)
    images_np = dataset.imgs

    print(f"Computing {vlm} embeddings for {len(images_np)} medical images...")

    _, preprocess, _ = get_vlm_model(vlm=vlm, device=device)

    all_embeds = []
    for i in tqdm.tqdm(range(0, len(images_np), batch_size), desc=f"Encoding ({vlm})"):
        batch_imgs = images_np[i:i + batch_size]

        pil_images = []
        for img in batch_imgs:
            pil_img = Image.fromarray(img).convert('RGB')
            pil_images.append(preprocess(pil_img))

        batch_tensor = torch.stack(pil_images).to(device)
        embeds = get_image_features(batch_tensor, vlm=vlm, device=device)
        all_embeds.append(embeds.cpu())

    all_embeds = torch.cat(all_embeds, dim=0)
    print(f"Embeddings shape: {all_embeds.shape}")

    save_path = os.path.join(MEDICAL_DATA_ROOT, f'medical-{vlm}-test.pt')
    torch.save(all_embeds, save_path)
    print(f"Saved to {save_path}")
    return save_path


def compute_uav_embeddings(vlm, device, data_root=UAV_DATA_ROOT,
                           scenes=None, batch_size=16):
    """Compute image embeddings for UAVScenes."""
    if scenes is None:
        scenes = UAV_SCENES

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

    print(f"Computing {vlm} embeddings for {len(image_paths)} UAV images...")

    _, preprocess, _ = get_vlm_model(vlm=vlm, device=device)

    all_embeds = []
    for i in tqdm.tqdm(range(0, len(image_paths), batch_size), desc=f"Encoding ({vlm})"):
        batch_paths = image_paths[i:i + batch_size]
        pil_images = [preprocess(Image.open(p).convert('RGB')) for p in batch_paths]
        batch_tensor = torch.stack(pil_images).to(device)

        embeds = get_image_features(batch_tensor, vlm=vlm, device=device)
        all_embeds.append(embeds.cpu())

    all_embeds = torch.cat(all_embeds, dim=0)
    print(f"Embeddings shape: {all_embeds.shape}")

    scene_tag = "_".join(scenes)
    save_path = os.path.join(data_root, f'uav_{vlm}_embeddings_{scene_tag}.pt')
    torch.save(all_embeds, save_path)
    print(f"Saved to {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description='Compute VLM image embeddings')
    parser.add_argument('--domain', required=True, choices=['medical', 'uav'],
                        help='Dataset domain')
    parser.add_argument('--vlm', default='clip',
                        choices=['clip', 'simclip4', 'simclip2', 'fare4', 'fare2', 'tecoa4', 'tecoa2'],
                        help='VLM variant to use')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--batch-size', type=int, default=32)
    # UAV-specific
    parser.add_argument('--data-root', default=UAV_DATA_ROOT)
    parser.add_argument('--scenes', nargs='+', default=UAV_SCENES)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    if args.domain == 'medical':
        compute_medical_embeddings(vlm=args.vlm, device=device, batch_size=args.batch_size)
    elif args.domain == 'uav':
        compute_uav_embeddings(vlm=args.vlm, device=device, data_root=args.data_root,
                               scenes=args.scenes, batch_size=args.batch_size)

    print("\nPhase 1 embedding computation complete!")


if __name__ == '__main__':
    main()
