"""
Defense mechanisms against gradient inversion attacks.

Implements gradient-level defenses that are applied to shared gradients
before the attacker can run gradient inversion. These simulate what a
privacy-conscious FL client would do before sharing gradient updates.

Defenses:
  - Gradient pruning (Zhu et al., NeurIPS 2019)
  - Gradient noise injection (Laplacian/Gaussian)
  - DP-SGD (Abadi et al., CCS 2016) — per-sample clipping + calibrated Gaussian noise
  - FedAvg multi-epoch (McMahan et al., AISTATS 2017) — multiple local SGD steps

Usage:
    from prototype.defenses import apply_dpsgd, apply_gradient_pruning, apply_gradient_noise
    from prototype.defenses import simulate_fedavg

    # DP-SGD with epsilon=1.0
    defended = apply_dpsgd(gradients, epsilon=1.0, delta=1e-5, max_grad_norm=1.0, batch_size=8)

    # FedAvg with 5 local epochs
    defended = simulate_fedavg(model, server_payload, custom_data, loss_fn, local_epochs=5, local_lr=1e-3)
"""
import math
import torch


def apply_gradient_pruning(gradients, prune_rate):
    """
    Prune smallest gradients by magnitude (defense).

    Zeros out the smallest-magnitude gradient entries. For example,
    prune_rate=0.90 removes 90% of gradient entries.

    Reference: Zhu et al., "Deep Leakage from Gradients", NeurIPS 2019.

    Args:
        gradients: List of gradient tensors
        prune_rate: Fraction of entries to zero out (0.0 to 0.99)

    Returns:
        List of pruned gradient tensors
    """
    pruned = []
    for grad in gradients:
        g = grad.clone()
        n = int(prune_rate * g.numel())
        if n > 0:
            threshold = torch.sort(torch.abs(g).flatten())[0][n - 1]
            mask = torch.abs(g) > threshold
            g *= mask
        pruned.append(g)
    return pruned


def apply_gradient_noise(gradients, noise_scale, distribution='laplacian'):
    """
    Add random noise to gradients (defense).

    Args:
        gradients: List of gradient tensors
        noise_scale: Standard deviation / scale of noise distribution
        distribution: 'laplacian' or 'gaussian'

    Returns:
        List of noisy gradient tensors
    """
    noisy = []
    for grad in gradients:
        g = grad.clone()
        if distribution == 'laplacian':
            noise = torch.distributions.Laplace(0, noise_scale).sample(g.shape).to(g.device)
        else:
            noise = torch.distributions.Normal(0, noise_scale).sample(g.shape).to(g.device)
        noisy.append(g + noise)
    return noisy


def compute_noise_multiplier(epsilon, delta, sensitivity=1.0):
    """
    Compute the Gaussian noise multiplier for (epsilon, delta)-DP.

    Uses the analytic Gaussian mechanism (Balle & Wang, 2018):
        sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon

    Args:
        epsilon: Privacy budget. Smaller = more privacy, more noise.
        delta: Failure probability (typically 1/n^2 or 1e-5).
        sensitivity: L2 sensitivity of the query (= max_grad_norm for DP-SGD).

    Returns:
        float: Noise standard deviation (sigma)
    """
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def apply_dpsgd(gradients, epsilon, delta=1e-5, max_grad_norm=1.0, batch_size=1):
    """
    Norm-Bounded Federated Update (NBFU) defense.

    Two-step procedure applied to the client's averaged gradient before sharing:
      1. Per-update clipping: bound the L2 norm of the entire update to C = max_grad_norm.
      2. Calibrated Gaussian noise (analytic Gaussian mechanism, Balle & Wang 2018).

    Privacy semantics. We do NOT clip per-example gradients. The defense
    provides record-level (ε,δ)-DP \emph{over the shared per-update aggregate},
    not standard client-level DP. The neighboring relation is "the client
    substitutes one record in its local batch." Under this relation the L2
    sensitivity of the clipped update equals 2*C — two clipped vectors can
    each lie anywhere on the radius-C ball, so they may differ by up to 2C.

    The per-element noise std is therefore:
        σ = 2 * C * sqrt(2 * ln(1.25/δ)) / ε
    (Sensitivity 2C, not C. The earlier version of this code used C, which
    underscaled the noise and failed to deliver the stated (ε,δ) guarantee.)

    For multi-round composition, use a moments accountant or RDP composition
    over the per-round (ε, δ) budget; this function returns per-round noise.

    Args:
        gradients: List of gradient tensors (the averaged shared update).
        epsilon: Privacy budget per round. None disables noise (clip-only ablation).
                   0.1  = very strong privacy (heavy noise)
                   1.0  = strong privacy
                   5.0  = moderate privacy
                   10.0 = weak privacy
                   50.0 = very weak privacy (clipping-dominated)
        delta: Failure probability per round. Default 1e-5.
        max_grad_norm: Clip bound C. Default 1.0.
        batch_size: Reported in info dict only; not used in noise calibration.

    Returns:
        (defended_gradients, info_dict)

    References:
        Abadi et al., "Deep Learning with Differential Privacy", CCS 2016.
        Balle & Wang, "Improving the Gaussian Mechanism for Differential Privacy",
            ICML 2018.
        McMahan et al., "Learning Differentially Private Recurrent Language Models",
            ICLR 2018 (client-level DP in FL).
    """
    # Step 1: Gradient clipping (bound the L2 norm)
    clipped = []
    flat_grads = torch.cat([g.flatten() for g in gradients])
    global_norm = torch.norm(flat_grads, p=2).item()
    clip_factor = min(1.0, max_grad_norm / (global_norm + 1e-8))

    for grad in gradients:
        clipped.append(grad.clone() * clip_factor)

    # Step 2: Add calibrated Gaussian noise (skipped when epsilon is None — clip-only ablation)
    if epsilon is None:
        sigma = 0.0
        noise_std = 0.0
        defended = clipped
    else:
        # Per-update clipping with sensitivity 2*C (substitute-one-record neighboring).
        # No 1/batch_size factor because we do not clip per-example.
        sigma = compute_noise_multiplier(epsilon, delta, sensitivity=2 * max_grad_norm)
        noise_std = sigma

        defended = []
        for grad in clipped:
            noise = torch.randn_like(grad) * noise_std
            defended.append(grad + noise)

    return defended, {
        'epsilon': epsilon,
        'delta': delta,
        'max_grad_norm': max_grad_norm,
        'global_norm_before_clip': global_norm,
        'clip_factor': clip_factor,
        'sigma': sigma,
        'noise_std_per_element': noise_std,
        'batch_size': batch_size,
    }


def simulate_fedavg(model, server_payload, custom_data, loss_fn, setup,
                    local_epochs=1, local_lr=1e-3, data_key='inputs'):
    """
    Simulate FedAvg multi-epoch local training and return the parameter update.

    Instead of computing a single gradient (FedSGD), the client runs multiple
    local SGD steps on the received model and shares the parameter difference
    (new_params - old_params). This "washes out" the malicious patterns that
    Geminio introduces, because each SGD step modifies the model further from
    the attacker's carefully crafted state.

    The Geminio paper (Shan et al., ICCV 2025) showed that FedAvg weakens the
    attack, especially with more local epochs and larger learning rates.

    Args:
        model: The FL model (already loaded with server weights)
        server_payload: Dict with 'parameters' and 'buffers' from server
        custom_data: Dict with 'inputs' and 'labels' tensors
        loss_fn: Loss function (CrossEntropyLoss or BCEWithLogitsLoss)
        setup: Dict with 'device' and 'dtype'
        local_epochs: Number of local training epochs (1 = FedSGD equivalent,
                      higher = more defense). Paper tested 1-32.
        local_lr: Local SGD learning rate. Paper used 1e-6 to 1e-3.
                  Smaller lr = less model change per step = weaker defense.
        data_key: Key for input data in custom_data dict

    Returns:
        (shared_grads, fedavg_info) where shared_grads is a list of
        parameter-difference tensors (same format as single-step gradients)

    Reference:
        McMahan et al., "Communication-Efficient Learning of Deep Networks
        from Decentralized Data", AISTATS 2017.
    """
    parameters = server_payload["parameters"]

    # Save original server parameters AND buffers (BN running_mean/var get
    # mutated by train-mode forward passes below; we must restore them or the
    # post-defense gradient metric is computed on a model with adapted BN stats
    # that artificially align rec/true gradients).
    original_params = [p.clone().detach() for p in parameters]
    original_buffers = [b.clone().detach() for b in model.buffers()]
    was_training = model.training

    # Move model to device and load server parameters
    model.to(**setup)
    with torch.no_grad():
        for param, server_state in zip(model.parameters(), parameters):
            param.copy_(server_state.to(**setup))
    model.train()

    # Prepare data
    inputs = custom_data[data_key].to(setup['device'])
    labels = custom_data['labels'].to(setup['device'])
    batch_size = inputs.shape[0]

    # Run multiple local SGD steps
    optimizer = torch.optim.SGD(model.parameters(), lr=local_lr)
    total_steps = 0

    for epoch in range(local_epochs):
        # Each epoch: iterate through the batch
        # With small batches (8 samples), each epoch = 1 step on the full batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_steps += 1

    # Compute parameter difference (what gets shared with server)
    shared_grads = [
        (p_local - p_server.to(**setup)).clone().detach()
        for (p_local, p_server) in zip(model.parameters(), original_params)
    ]

    # Compute the magnitude of the update
    update_norm = torch.norm(
        torch.cat([g.flatten() for g in shared_grads]), p=2
    ).item()

    # Restore original parameters AND buffers, AND training-mode flag, so
    # subsequent metric computation sees the same model state it would have
    # seen without FedAvg simulation.
    with torch.no_grad():
        for param, snap in zip(model.parameters(), original_params):
            param.copy_(snap.to(**setup))
        for buf, snap in zip(model.buffers(), original_buffers):
            buf.copy_(snap.to(**setup))
    if not was_training:
        model.eval()

    return shared_grads, {
        'local_epochs': local_epochs,
        'local_lr': local_lr,
        'total_steps': total_steps,
        'update_norm': update_norm,
        'batch_size': batch_size,
    }
