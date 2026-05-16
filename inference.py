"""
inference.py
============
Run the DeepFake Fusion Detector on a single image or video file.

Output:
  - Prediction: REAL or FAKE
  - Confidence score (0–100%)
  - Per-frame results for video

Usage:
  # Single image
  python inference.py --checkpoint checkpoints/fusion_best.pt --input face.jpg

  # Video
  python inference.py --checkpoint checkpoints/fusion_best.pt --input video.mp4

  # Folder of images
  python inference.py --checkpoint checkpoints/fusion_best.pt --input ./faces/
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from dataset import build_eval_transforms, extract_frames, IMAGE_SIZE
from fusion_model import DeepFakeFusionDetector


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def load_model(checkpoint: str, device: torch.device,
               swin_embed_dim: int = 96, gan_embed_dim: int = 64) -> DeepFakeFusionDetector:
    model = DeepFakeFusionDetector(
        swin_embed_dim=swin_embed_dim,
        gan_embed_dim=gan_embed_dim,
    ).to(device)
    ckpt  = torch.load(checkpoint, map_location=device)
    state = ckpt.get("state", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def predict_image(
    model: DeepFakeFusionDetector,
    img: Image.Image,
    transform,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)
    with torch.no_grad():
        logit = model(tensor)
    prob = logit.sigmoid().item()
    fake = prob >= threshold
    return {
        "label":      "FAKE" if fake else "REAL",
        "probability": prob,
        "confidence":  prob if fake else 1.0 - prob,
    }


def aggregate_video_results(frame_results: list, strategy: str = "mean") -> dict:
    """
    Aggregate per-frame predictions into a single video-level verdict.

    strategy:
      'mean'    — average P(fake) across frames
      'max'     — if any frame is highly fake, flag the video
      'majority' — majority vote
    """
    probs = [r["probability"] for r in frame_results]
    if strategy == "mean":
        agg_prob = float(np.mean(probs))
    elif strategy == "max":
        agg_prob = float(np.max(probs))
    else:  # majority
        votes    = [1 if r["label"] == "FAKE" else 0 for r in frame_results]
        agg_prob = float(np.mean(votes))

    fake = agg_prob >= 0.5
    return {
        "label":         "FAKE" if fake else "REAL",
        "probability":   agg_prob,
        "confidence":    agg_prob if fake else 1.0 - agg_prob,
        "n_frames":      len(frame_results),
        "n_fake_frames": sum(1 for r in frame_results if r["label"] == "FAKE"),
        "strategy":      strategy,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Main inference functions
# ──────────────────────────────────────────────────────────────────────────────

def run_on_image(path: str, model, transform, device, threshold: float) -> dict:
    img    = Image.open(path).convert("RGB")
    result = predict_image(model, img, transform, device, threshold)
    result["path"] = path
    return result


def run_on_video(
    path: str,
    model,
    transform,
    device,
    threshold: float,
    max_frames: int = 30,
    strategy: str = "mean",
) -> dict:
    import tempfile, shutil
    tmp_dir = tempfile.mkdtemp()
    try:
        frame_paths = extract_frames(path, tmp_dir, max_frames=max_frames)
        if not frame_paths:
            return {"path": path, "error": "No frames extracted"}
        frame_results = [
            run_on_image(fp, model, transform, device, threshold)
            for fp in frame_paths
        ]
        result = aggregate_video_results(frame_results, strategy=strategy)
        result["path"] = path
        result["frame_results"] = frame_results
        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_on_folder(folder: str, model, transform, device, threshold: float) -> list:
    results = []
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            results.append(run_on_image(str(p), model, transform, device, threshold))
        elif p.suffix.lower() in VIDEO_EXTS:
            results.append(run_on_video(str(p), model, transform, device, threshold))
    return results


# ──────────────────────────────────────────────────────────────────────────────
#  Pretty printing
# ──────────────────────────────────────────────────────────────────────────────

def print_result(result: dict):
    label = result["label"]
    conf  = result.get("confidence", 0.0) * 100
    prob  = result.get("probability", 0.0) * 100
    path  = result.get("path", "")

    verdict = "🔴 FAKE" if label == "FAKE" else "🟢 REAL"
    print(f"\n{'─'*50}")
    print(f"  File       : {os.path.basename(path)}")
    print(f"  Verdict    : {verdict}")
    print(f"  P(fake)    : {prob:.1f}%")
    print(f"  Confidence : {conf:.1f}%")
    if "n_frames" in result:
        print(f"  Frames     : {result['n_frames']}  ({result['n_fake_frames']} flagged fake)")
    print(f"{'─'*50}")


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DeepFake Fusion Detector — Inference")
    p.add_argument("--checkpoint",     type=str, required=True)
    p.add_argument("--input",          type=str, required=True,
                   help="Path to image, video, or folder")
    p.add_argument("--threshold",      type=float, default=0.5)
    p.add_argument("--device",         type=str, default="auto")
    p.add_argument("--swin_embed_dim", type=int, default=96)
    p.add_argument("--gan_embed_dim",  type=int, default=64)
    p.add_argument("--max_frames",     type=int, default=30,
                   help="Max frames to sample from a video")
    p.add_argument("--strategy",       type=str, default="mean",
                   choices=["mean", "max", "majority"],
                   help="Video aggregation strategy")
    p.add_argument("--save_json",      type=str, default=None,
                   help="Optionally save full results to JSON")
    return p.parse_args()


def main():
    args = parse_args()

    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) \
             if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    model     = load_model(args.checkpoint, device, args.swin_embed_dim, args.gan_embed_dim)
    transform = build_eval_transforms(enhance=False)
    inp       = args.input
    results   = []

    if os.path.isdir(inp):
        results = run_on_folder(inp, model, transform, device, args.threshold)
        for r in results:
            print_result(r)
    elif Path(inp).suffix.lower() in VIDEO_EXTS:
        r = run_on_video(inp, model, transform, device, args.threshold,
                         args.max_frames, args.strategy)
        print_result(r)
        results = [r]
    elif Path(inp).suffix.lower() in IMAGE_EXTS:
        r = run_on_image(inp, model, transform, device, args.threshold)
        print_result(r)
        results = [r]
    else:
        print(f"[error] Unsupported input: {inp}")
        sys.exit(1)

    if args.save_json:
        import json
        # Remove non-serialisable frame_results images
        clean = []
        for r in results:
            rc = {k: v for k, v in r.items() if k != "frame_results"}
            clean.append(rc)
        with open(args.save_json, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"\nResults saved → {args.save_json}")


if __name__ == "__main__":
    main()
