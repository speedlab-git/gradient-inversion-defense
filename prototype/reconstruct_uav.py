"""
Phase 3 (UAV): Reconstruct private aerial/drone images from gradients.

Uses the trained malicious models from Phase 2 and the breaching framework's
HFGradInv attack to reconstruct specific aerial images targeted by the query.

Key difference from ImageNet/Medical: UAVScenes uses BCEWithLogitsLoss (multi-label)
instead of CrossEntropyLoss. This requires overriding the loss function at both
the user (gradient computation) and attacker (gradient matching) levels.

Usage:
    # First prepare samples:
    python prototype/prepare_uav_samples.py
    # Then reconstruct:
    python prototype/reconstruct_uav.py --baseline --gpu 0
    python prototype/reconstruct_uav.py --geminio-query swimming_pool --gpu 0
    # With defenses:
    python prototype/reconstruct_uav.py --geminio-query swimming_pool --gpu 0 --prune-rate 0.90
    python prototype/reconstruct_uav.py --geminio-query swimming_pool --gpu 0 --noise-scale 1e-3
"""
import argparse
import json
import logging
import math
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as utl
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import GeminioResNet18
from prototype.defenses import apply_gradient_pruning, apply_gradient_noise, apply_dpsgd, simulate_fedavg
import breaching

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
logger = logging.getLogger()

# Map short names to model files
UAV_MODELS = {
    'swimming_pool': 'malicious_models_uav/aerial_drone_image_of_a_swimming_pool.pt',
    'solar_panels': 'malicious_models_uav/aerial_drone_image_showing_solar_panels_on_rooftops.pt',
    'trucks': 'malicious_models_uav/drone_image_of_trucks_on_a_road.pt',
    'river_bridge': 'malicious_models_uav/aerial_image_of_a_river_with_a_bridge.pt',
    'airstrip': 'malicious_models_uav/drone_image_of_an_airport_runway_or_airstrip.pt',
    'containers': 'malicious_models_uav/aerial_image_of_shipping_containers.pt',
}


class UAVCustomData:
    """Custom data handler for UAVScenes multi-label reconstruction.

    Unlike the standard CustomData class which assumes single-label {index}-{class}.png,
    this handles multi-label UAVScenes data with label vectors stored in labels.npy.
    """

    def __init__(self, data_dir, num_data_points):
        self.data_dir = data_dir
        self.data_points = num_data_points
        self.mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]

    def process_data(self):
        """Load images and multi-label targets."""
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])

        # Load images (sorted by index)
        file_names = [f for f in os.listdir(self.data_dir) if f.endswith('.png')]
        file_names = sorted(file_names, key=lambda x: int(x.split('-')[0]))
        assert len(file_names) >= self.data_points, \
            f"Need {self.data_points} samples but only found {len(file_names)} in {self.data_dir}"

        imgs = []
        for fname in file_names[:self.data_points]:
            img = Image.open(os.path.join(self.data_dir, fname)).convert('RGB')
            imgs.append(trans(img)[None, :])
        imgs = torch.cat(imgs, 0)

        # Load multi-label targets from saved numpy file
        labels_path = os.path.join(self.data_dir, 'labels.npy')
        labels = np.load(labels_path)[:self.data_points]
        labels = torch.tensor(labels, dtype=torch.float32)

        # Normalize images
        inputs = (imgs - self.mean) / self.std
        return dict(inputs=inputs, labels=labels)

    def save_recover(self, recover, original=None, save_pth=''):
        """Save reconstructed/ground-truth images as grid."""
        if isinstance(recover, dict):
            data = recover['data']
        else:
            data = recover
        batch = data.shape[0]
        recover_imgs = torch.clamp((data.cpu() * self.std + self.mean), 0, 1)

        if original is not None:
            if isinstance(original, dict):
                orig_data = original['data']
            else:
                orig_data = original
            origina_imgs = torch.clamp((orig_data.cpu() * self.std + self.mean), 0, 1)
            all_imgs = torch.cat([recover_imgs, origina_imgs], 0)
        else:
            all_imgs = recover_imgs

        utl.save_image(all_imgs, save_pth, nrow=batch)


def compute_metrics(reconstructed_user_data, true_user_data, setup):
    """Compute LPIPS, PSNR, CW-SSIM between reconstructed and ground truth images."""
    import lpips
    from breaching.analysis.metrics import psnr_compute, cw_ssim

    lpips_scorer = lpips.LPIPS(net="alex", verbose=False).to(**setup)

    # Denormalize images (ImageNet normalization)
    dm = torch.tensor([0.485, 0.456, 0.406], **setup)[None, :, None, None]
    ds = torch.tensor([0.229, 0.224, 0.225], **setup)[None, :, None, None]

    rec = torch.clamp(reconstructed_user_data["data"].to(**setup) * ds + dm, 0, 1)
    gt = torch.clamp(true_user_data["data"].to(**setup) * ds + dm, 0, 1)

    # Reorder reconstructed images to best match ground truth (Hungarian matching)
    from breaching.analysis.analysis import compute_batch_order
    order = compute_batch_order(lpips_scorer, rec, gt, setup)
    rec = rec[order]
    reconstructed_user_data["data"] = reconstructed_user_data["data"][order]

    # PSNR
    avg_psnr, max_psnr = psnr_compute(rec, gt, factor=1)

    # LPIPS
    lpips_score = lpips_scorer(rec, gt, normalize=True)
    avg_lpips = lpips_score.mean().item()

    # CW-SSIM
    avg_ssim, max_ssim = cw_ssim(rec, gt, scales=5)

    # MSE
    mse = (rec - gt).pow(2).mean().item()

    metrics = {
        'psnr': avg_psnr,
        'max_psnr': max_psnr,
        'lpips': avg_lpips,
        'ssim': avg_ssim,
        'max_ssim': max_ssim,
        'mse': mse,
    }
    return metrics


def compute_attack_f1(model, loss_fn, rec_data, true_data, setup, threshold=0.90):
    """Compute Attack F1: cosine sim of output-layer gradients between rec and GT images."""
    model.to(**setup)
    model.eval()
    identified = 0
    cos_sims = []
    B = rec_data["data"].shape[0]

    for i in range(B):
        # Per-sample gradient for reconstructed image
        model.zero_grad()
        rec_out = model(rec_data["data"][i:i+1].to(setup["device"]))
        rec_loss = loss_fn(rec_out, true_data["labels"][i:i+1].to(setup["device"]))
        rec_grads = torch.autograd.grad(rec_loss, model.parameters(), create_graph=False)
        rec_grad_flat = rec_grads[-2].flatten()

        # Per-sample gradient for ground truth image
        model.zero_grad()
        true_out = model(true_data["data"][i:i+1].to(setup["device"]))
        true_loss = loss_fn(true_out, true_data["labels"][i:i+1].to(setup["device"]))
        true_grads = torch.autograd.grad(true_loss, model.parameters(), create_graph=False)
        true_grad_flat = true_grads[-2].flatten()

        cos_sim = F.cosine_similarity(rec_grad_flat.unsqueeze(0), true_grad_flat.unsqueeze(0)).item()
        cos_sims.append(cos_sim)
        if cos_sim >= threshold:
            identified += 1

    precision = recall = identified / B
    f1 = precision
    avg_cos_sim = sum(cos_sims) / len(cos_sims)
    return {'attack_precision': precision, 'attack_recall': recall, 'attack_f1': f1,
            'identified': identified, 'total': B, 'avg_cos_sim': avg_cos_sim}


def reconstruct_uav(cfg, setup, query=None, prune_rate=0.0, noise_scale=0.0,
                    dpsgd_epsilon=0.0, dpsgd_delta=1e-5, dpsgd_max_grad_norm=1.0,
                    fedavg_epochs=0, fedavg_lr=1e-3,
                    data_dir='./assets/uav_samples/', model_path=None):
    """Reconstruct private UAV/drone images using baseline or query-based approach."""
    num_classes = cfg.case.data.classes  # 18

    # Initialize model (ResNet18, 18 classes for UAVScenes multi-label)
    model = GeminioResNet18(num_classes=num_classes)
    user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, model, setup)

    # CRITICAL: Override loss function to BCEWithLogitsLoss for multi-label
    bce_loss = torch.nn.BCEWithLogitsLoss()
    user.loss = bce_loss

    # Disable automatic label sorting (sort corrupts multi-label vectors)
    user.provide_labels = False

    # Load malicious model weights (direct path or query dict)
    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
    elif query:
        model_path = UAV_MODELS.get(query)
        if model_path is None:
            raise ValueError(f"Unknown query '{query}'. Available: {list(UAV_MODELS.keys())}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

    if model_path:
        print(f"Loading malicious model weights from: {model_path}")
        model_state = torch.load(model_path, map_location=setup['device'])
        model.model.clf.load_state_dict(model_state, strict=True)
        print(f"Successfully loaded malicious model weights ({sum(p.numel() for p in model.model.clf.parameters()):,} params)")

    # Setup attack with BCEWithLogitsLoss (must match victim's loss)
    attacker_loss = torch.nn.BCEWithLogitsLoss()
    attacker = breaching.attacks.prepare_attack(server.model, attacker_loss, cfg.attack, setup)
    breaching.utils.overview(server, user, attacker)

    # Get server payload
    server_payload = server.distribute_payload()

    # Create save directory
    os.makedirs(cfg.attack.save_dir, exist_ok=True)

    # Load private UAV samples with multi-label targets
    cus_data = UAVCustomData(
        data_dir=data_dir,
        num_data_points=cfg.case.user.num_data_points
    )
    custom_data = cus_data.process_data()

    # Compute local updates (victim's gradient computation with BCEWithLogitsLoss)
    shared_data, true_user_data = user.compute_local_updates(
        server_payload,
        custom_data=custom_data
    )

    # Manually inject multi-label targets into metadata
    shared_data["metadata"]["labels"] = custom_data["labels"].to(setup['device'])

    # FedAvg defense: replace single-step gradients with multi-epoch parameter update
    if fedavg_epochs > 0:
        print(f"Applying FedAvg defense: {fedavg_epochs} local epochs, lr={fedavg_lr}")
        shared_data["gradients"], fedavg_info = simulate_fedavg(
            server.model, server_payload, custom_data,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            setup=setup,
            local_epochs=fedavg_epochs,
            local_lr=fedavg_lr,
        )
        print(f"  Total local steps: {fedavg_info['total_steps']}")
        print(f"  Update norm: {fedavg_info['update_norm']:.4f}")

    # Apply defenses to gradients (post-hoc)
    if prune_rate > 0:
        print(f"Applying gradient pruning defense: {prune_rate:.0%} sparsity")
        shared_data["gradients"] = apply_gradient_pruning(shared_data["gradients"], prune_rate)
    if noise_scale > 0:
        print(f"Applying gradient noise defense: Laplacian scale={noise_scale}")
        shared_data["gradients"] = apply_gradient_noise(shared_data["gradients"], noise_scale)
    if dpsgd_epsilon > 0:
        print(f"Applying DP-SGD defense: epsilon={dpsgd_epsilon}, delta={dpsgd_delta}, C={dpsgd_max_grad_norm}")
        shared_data["gradients"], dpsgd_info = apply_dpsgd(
            shared_data["gradients"],
            epsilon=dpsgd_epsilon,
            delta=dpsgd_delta,
            max_grad_norm=dpsgd_max_grad_norm,
            batch_size=cfg.case.user.num_data_points,
        )
        print(f"  Global norm before clip: {dpsgd_info['global_norm_before_clip']:.4f}")
        print(f"  Clip factor: {dpsgd_info['clip_factor']:.4f}")
        print(f"  Noise sigma: {dpsgd_info['sigma']:.6f}")
        print(f"  Noise std per element: {dpsgd_info['noise_std_per_element']:.6f}")

    # Save ground truth
    true_path = cfg.attack.save_dir + 'a_truth.jpg'
    cus_data.save_recover(true_user_data, save_pth=true_path)
    print(f"Saved ground truth to {true_path}")

    # Run gradient inversion reconstruction
    print(f"\nStarting reconstruction ({cfg.attack.optim.max_iterations} iterations)...")
    reconstructed_user_data, stats = attacker.reconstruct(
        [server_payload],
        [shared_data],
        {},
        dryrun=cfg.dryrun,
        custom=cus_data
    )

    # Save final reconstruction (images)
    recon_path = cfg.attack.save_dir + 'final_rec.jpg'
    cus_data.save_recover(reconstructed_user_data, true_user_data, recon_path)
    print(f"Saved reconstruction to {recon_path}")

    # Save raw tensors for later evaluation
    torch.save(reconstructed_user_data, cfg.attack.save_dir + 'reconstructed.pt')
    torch.save(true_user_data, cfg.attack.save_dir + 'true_data.pt')
    torch.save(server_payload, cfg.attack.save_dir + 'server_payload.pt')
    print(f"Saved tensors to {cfg.attack.save_dir}")

    # Compute quantitative metrics
    print("\n--- Quantitative Metrics ---")
    metrics = compute_metrics(reconstructed_user_data, true_user_data, setup)
    print(f"  LPIPS:  {metrics['lpips']:.4f} (lower=better)")
    print(f"  PSNR:   {metrics['psnr']:.2f} dB (higher=better)")
    print(f"  CW-SSIM: {metrics['ssim']:.4f} (higher=better)")
    print(f"  MSE:    {metrics['mse']:.6f}")

    # Compute Attack F1
    print("\n--- Attack F1 (threshold=0.90) ---")
    attack_metrics = compute_attack_f1(
        server.model, attacker_loss, reconstructed_user_data, true_user_data, setup
    )
    print(f"  Identified: {attack_metrics['identified']}/{attack_metrics['total']}")
    print(f"  Attack F1:  {attack_metrics['attack_f1']:.4f}")
    metrics.update(attack_metrics)

    # Add metadata
    metrics['query'] = query or 'baseline'
    metrics['prune_rate'] = prune_rate
    metrics['noise_scale'] = noise_scale
    metrics['dpsgd_epsilon'] = dpsgd_epsilon
    metrics['dpsgd_delta'] = dpsgd_delta
    metrics['dpsgd_max_grad_norm'] = dpsgd_max_grad_norm
    metrics['fedavg_epochs'] = fedavg_epochs
    metrics['fedavg_lr'] = fedavg_lr
    metrics['domain'] = 'uav'
    metrics['attack'] = cfg.attack.get('type', 'unknown')

    # Save metrics
    metrics_path = cfg.attack.save_dir + 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UAV image reconstruction using Geminio')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--baseline', action='store_true', help='Run baseline reconstruction (no query)')
    group.add_argument('--geminio-query', type=str, help='Query name (e.g., swimming_pool)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--num-samples', type=int, default=8, help='Number of private samples')
    parser.add_argument('--max-iterations', type=int, default=24000, help='Max reconstruction iterations')
    # Defense arguments
    parser.add_argument('--prune-rate', type=float, default=0.0, help='Gradient pruning rate (0-0.99)')
    parser.add_argument('--noise-scale', type=float, default=0.0, help='Gradient noise scale (Laplacian)')
    parser.add_argument('--dpsgd-epsilon', type=float, default=0.0, help='DP-SGD privacy budget epsilon (0=disabled)')
    parser.add_argument('--dpsgd-delta', type=float, default=1e-5, help='DP-SGD failure probability delta')
    parser.add_argument('--dpsgd-max-grad-norm', type=float, default=1.0, help='DP-SGD gradient clipping norm C')
    parser.add_argument('--fedavg-epochs', type=int, default=0, help='FedAvg local epochs (0=disabled, use FedSGD)')
    parser.add_argument('--fedavg-lr', type=float, default=1e-3, help='FedAvg local learning rate')
    # Ablation arguments
    parser.add_argument('--data-dir', type=str, default='./assets/uav_samples/', help='Sample directory')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--model-path', type=str, default=None, help='Direct path to model .pt file (bypasses query dict)')
    parser.add_argument('--batch-tag', type=str, default=None, help='Tag appended to save directory name')
    parser.add_argument('--attack', type=str, default='hfgradinv',
                        choices=['hfgradinv', 'dlg', 'ig', 'gradinversion'],
                        help='Underlying gradient inversion algorithm applied to Geminio-trapped gradients')
    args = parser.parse_args()

    # Initialize configuration
    cfg = breaching.get_config(overrides=["case=geminio_uav", f"attack={args.attack}"])
    cfg.case.user.num_data_points = args.num_samples
    cfg.case.data.partition = "none"
    cfg.case.data.batch_size = args.num_samples
    cfg.case.user.provide_labels = True  # Will be overridden in reconstruct_uav
    cfg.attack.optim['max_iterations'] = args.max_iterations
    if args.attack == 'hfgradinv':
        # HF-GradInv hyperparameters tuned for ResNet18 (Geminio default path)
        cfg.attack.optim['signed'] = 'soft'
        cfg.attack.optim['step_size_decay'] = 'cosine-decay'
        cfg.attack.optim['warmup'] = 50
        cfg.attack.init = 'patterned-4-randn'
        # Adjust objective layer range for ResNet18 (66 param tensors vs ResNet34's ~120)
        cfg.attack.objective['start'] = 50
        cfg.attack.objective['min_start'] = 15

    # Set save directory (include defense params in path if applicable)
    query_tag = args.geminio_query or 'baseline'
    attack_tag = f'_{args.attack}' if args.attack != 'hfgradinv' else ''
    defense_tag = ''
    if args.prune_rate > 0:
        defense_tag += f'_prune{args.prune_rate}'
    if args.noise_scale > 0:
        defense_tag += f'_noise{args.noise_scale}'
    if args.dpsgd_epsilon > 0:
        defense_tag += f'_dpsgd_eps{args.dpsgd_epsilon}'
    if args.fedavg_epochs > 0:
        defense_tag += f'_fedavg_e{args.fedavg_epochs}_lr{args.fedavg_lr}'
    batch_tag = f'_{args.batch_tag}' if args.batch_tag else ''
    cfg.attack.save_dir = f'./results/uav_{query_tag}{attack_tag}{defense_tag}{batch_tag}/'

    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
    setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))

    # Set random seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    reconstruct_uav(cfg, setup,
                     query=args.geminio_query if args.geminio_query else None,
                     prune_rate=args.prune_rate,
                     noise_scale=args.noise_scale,
                     dpsgd_epsilon=args.dpsgd_epsilon,
                     dpsgd_delta=args.dpsgd_delta,
                     dpsgd_max_grad_norm=args.dpsgd_max_grad_norm,
                     fedavg_epochs=args.fedavg_epochs,
                     fedavg_lr=args.fedavg_lr,
                     data_dir=args.data_dir,
                     model_path=args.model_path)
