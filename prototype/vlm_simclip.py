"""
SimCLIP / FARE / vanilla CLIP vision-language model adapter for Geminio prototype.

Supports multiple VLM backends, all based on ViT-L/14 (768-dim):
  - "clip"    : Vanilla OpenAI CLIP ViT-L/14 via open_clip (baseline)
  - "simclip4": SimCLIP-4 (eps=4/255 adversarially fine-tuned vision encoder)
  - "simclip2": SimCLIP-2 (eps=2/255 variant)
  - "fare4"   : FARE-4 (L2-based adversarial fine-tuning baseline)

The text encoder is shared across all variants (only the vision encoder differs).
All output embeddings are L2-normalized and 768-dimensional.

Weight sources:
  - SimCLIP: ./pretrained_vlm/simclip4.pt, simclip2.pt (vision encoder state dicts)
  - FARE:    Loaded directly from HuggingFace hub (hf-hub:chs20/fare4-clip)

Usage:
    from prototype.vlm_simclip import get_text_features, get_image_features, get_vlm_model

    # Text features (same for all variants — text encoder is unchanged)
    text_embeds = get_text_features("Any chest X-ray showing pneumonia", device="cuda:0")

    # Image features with SimCLIP-hardened vision encoder
    img_embeds = get_image_features(pil_images, vlm="simclip4", device="cuda:0")
"""
import os
import torch
import open_clip

# Global caches
_model_cache = {}
_preprocess_cache = {}
_tokenizer_cache = {}

# Default weight paths (relative to repo root)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHT_DIR = os.path.join(_REPO_ROOT, 'pretrained_vlm')

# SimCLIP weights are local .pt files containing vision encoder state dicts
SIMCLIP_WEIGHT_PATHS = {
    'simclip4': os.path.join(WEIGHT_DIR, 'simclip4.pt'),
    'simclip2': os.path.join(WEIGHT_DIR, 'simclip2.pt'),
}

# FARE and TeCoA weights are hosted on HuggingFace as open_clip hub models
HUB_MODELS = {
    'fare4': 'hf-hub:chs20/fare4-clip',
    'fare2': 'hf-hub:chs20/fare2-clip',
    'tecoa4': 'hf-hub:chs20/tecoa4-clip',
    'tecoa2': 'hf-hub:chs20/tecoa2-clip',
}

# All variants use ViT-L-14 architecture (768-dim embeddings)
CLIP_MODEL_NAME = 'ViT-L-14'
CLIP_PRETRAINED = 'openai'
EMBED_DIM = 768

VALID_VLMS = {'clip', 'simclip4', 'simclip2', 'fare4', 'fare2', 'tecoa4', 'tecoa2'}


def get_vlm_model(vlm='clip', device=None):
    """
    Load and cache a CLIP model with optional fine-tuned vision encoder.

    Args:
        vlm: One of 'clip', 'simclip4', 'simclip2', 'fare4', 'fare2'
        device: Target device

    Returns:
        (model, preprocess, tokenizer) tuple
    """
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if vlm not in VALID_VLMS:
        raise ValueError(f"Unknown VLM variant: {vlm}. Choose from: {sorted(VALID_VLMS)}")

    cache_key = f"{vlm}_{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key], _preprocess_cache[cache_key], _tokenizer_cache[cache_key]

    if vlm in HUB_MODELS:
        # FARE/TeCoA models: load entire model from HuggingFace hub (open_clip format)
        hub_name = HUB_MODELS[vlm]
        print(f"Loading {vlm} from {hub_name}...")
        model, _, preprocess = open_clip.create_model_and_transforms(hub_name, device=device)
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        print(f"Loaded {vlm} from HuggingFace hub")
    else:
        # clip / simclip: load base OpenAI CLIP, then optionally swap vision encoder
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device
        )
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

        if vlm in SIMCLIP_WEIGHT_PATHS:
            weight_path = SIMCLIP_WEIGHT_PATHS[vlm]
            if not os.path.exists(weight_path):
                raise FileNotFoundError(
                    f"Weight file not found: {weight_path}\n"
                    f"Download from: https://huggingface.co/hossainzarif19/SimCLIP"
                )
            checkpoint = torch.load(weight_path, map_location=device)
            model.visual.load_state_dict(checkpoint)
            print(f"Loaded {vlm} vision encoder from {weight_path}")

    model.eval()
    _model_cache[cache_key] = model
    _preprocess_cache[cache_key] = preprocess
    _tokenizer_cache[cache_key] = tokenizer

    return model, preprocess, tokenizer


def get_text_features(text, vlm='clip', device=None):
    """
    Extract text features using CLIP text encoder.

    The text encoder is the same across all variants (SimCLIP/FARE only
    fine-tune the vision encoder), so the vlm parameter doesn't affect
    text embeddings. It's accepted for API consistency.

    Args:
        text: Input text string
        vlm: VLM variant (accepted for consistency but does not affect output)
        device: Target device

    Returns:
        torch.Tensor: L2-normalized text features [1, 768]
    """
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model, _, tokenizer = get_vlm_model(vlm=vlm, device=device)

    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        text_embeds = model.encode_text(tokens)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    return text_embeds.detach()


def get_image_features(images, vlm='clip', device=None):
    """
    Extract image features using CLIP vision encoder (optionally fine-tuned).

    Args:
        images: Either a batch tensor [B, 3, H, W] (already preprocessed)
                or a list of PIL images (will be preprocessed)
        vlm: One of 'clip', 'simclip4', 'simclip2', 'fare4', 'fare2'
        device: Target device

    Returns:
        torch.Tensor: L2-normalized image features [B, 768]
    """
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model, preprocess, _ = get_vlm_model(vlm=vlm, device=device)

    if isinstance(images, (list, tuple)):
        # List of PIL images — apply preprocessing
        batch = torch.stack([preprocess(img) for img in images]).to(device)
    elif isinstance(images, torch.Tensor):
        batch = images.to(device)
    else:
        raise TypeError(f"Expected list of PIL images or tensor, got {type(images)}")

    with torch.no_grad():
        image_embeds = model.encode_image(batch)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

    return image_embeds.detach()
