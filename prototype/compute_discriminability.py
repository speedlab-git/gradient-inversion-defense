"""Compute query-discriminability margin for each language-sweep query.

For a query Q with VLM text embedding t_Q and image embeddings {e_i} on the
auxiliary set, define
    delta(Q) = mean_{i in top-k%} cos(e_i, t_Q) - mean_{i in bottom-k%} cos(e_i, t_Q)
where k=10. Higher delta = sharper separation = more discriminative.

Output: results/lang_sweep_logs/discriminability.csv
        with columns: tag, domain, text, vlm, top10_mean, bot10_mean, delta, n
"""
import csv
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))
from language_sweep_config import LANGUAGE_SWEEP_QUERIES  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

LOG_DIR = ROOT / "results" / "lang_sweep_logs"
OUT_CSV = LOG_DIR / "discriminability.csv"

UAV_EMBED_PATH = "/raid/scratch/dzimmerman2021/uavscenes/uav_clip_embeddings_interval5_AMtown01_interval5_HKairport01.pt"
MED_EMBED_PATH = ROOT / "data" / "medical-biomedclip-test.pt"


def load_image_embeddings(domain: str):
    if domain == "medical":
        path = MED_EMBED_PATH
        d = torch.load(path, map_location="cpu")
        if isinstance(d, dict) and "embeddings" in d:
            embs = d["embeddings"]
        else:
            embs = d
    else:
        d = torch.load(UAV_EMBED_PATH, map_location="cpu")
        if isinstance(d, dict) and "embeddings" in d:
            embs = d["embeddings"]
        else:
            embs = d
    if isinstance(embs, list):
        embs = torch.stack(embs) if isinstance(embs[0], torch.Tensor) else torch.tensor(embs)
    embs = embs.float()
    embs = embs / embs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return embs


def get_text_emb(text: str, vlm: str, device: str):
    if vlm == "biomedclip":
        from vlm_medical import get_text_features
        t = get_text_features(text=text, device=device)
    else:
        from vlm_simclip import get_text_features
        t = get_text_features(text=text, vlm=vlm, device=device)
    t = t.float().cpu()
    if t.dim() == 2:
        t = t.squeeze(0)
    t = t / t.norm().clamp(min=1e-8)
    return t


def discriminability(image_embs: torch.Tensor, text_emb: torch.Tensor, k_pct: float = 0.10):
    sims = image_embs @ text_emb  # [N]
    n = sims.shape[0]
    k = max(1, int(n * k_pct))
    top = sims.topk(k, largest=True).values.mean().item()
    bot = sims.topk(k, largest=False).values.mean().item()
    return top, bot, top - bot


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for domain, queries in LANGUAGE_SWEEP_QUERIES.items():
        print(f"\n=== {domain} ===")
        image_embs = load_image_embeddings(domain)
        n = image_embs.shape[0]
        vlm = "biomedclip" if domain == "medical" else "clip"
        print(f"  Image embeds: {tuple(image_embs.shape)}  VLM: {vlm}")

        for tag, text, cls in queries:
            try:
                text_emb = get_text_emb(text, vlm, DEVICE)
            except Exception as e:
                print(f"  [FAIL] {tag}: {e}")
                continue

            if text_emb.shape[0] != image_embs.shape[1]:
                print(f"  [DIM-MISMATCH] {tag}: text {text_emb.shape}, img {image_embs.shape}")
                continue

            top, bot, delta = discriminability(image_embs, text_emb)
            print(f"  {tag:15s} top10={top:+.3f}  bot10={bot:+.3f}  delta={delta:+.3f}")
            rows.append({
                "tag": tag, "domain": domain, "text": text, "class": cls, "vlm": vlm,
                "top10_mean": round(top, 4), "bot10_mean": round(bot, 4),
                "delta": round(delta, 4), "n": n,
            })

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "domain", "text", "class", "vlm", "top10_mean", "bot10_mean", "delta", "n"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
