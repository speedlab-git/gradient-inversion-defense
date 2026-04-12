"""
BiomedCLIP text feature extraction for medical Geminio prototype.

Replaces core/vlm.py but uses BiomedCLIP (open_clip) instead of CLIP (transformers).
"""
import torch
import open_clip

# Global cache
_model_cache = {}
_tokenizer_cache = {}

BIOMEDCLIP_NAME = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'


def get_text_features(text, device=None):
    """
    Extract text features using BiomedCLIP model.

    Args:
        text (str): Input text to extract features from
        device (str, optional): Device to run the model on

    Returns:
        torch.Tensor: Normalized text features [1, 512]
    """
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    cache_key = f"biomedclip_{device}"

    if cache_key not in _model_cache:
        model, _, _ = open_clip.create_model_and_transforms(BIOMEDCLIP_NAME)
        _model_cache[cache_key] = model.to(device)
        _model_cache[cache_key].eval()
        _tokenizer_cache[cache_key] = open_clip.get_tokenizer(BIOMEDCLIP_NAME)

    model = _model_cache[cache_key]
    tokenizer = _tokenizer_cache[cache_key]

    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        text_embeds = model.encode_text(tokens)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    return text_embeds.detach()
