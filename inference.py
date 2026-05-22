"""
inference.py
============
Run the DeepFake Fusion Detector on a single image or video file.

UPDATED: Now constructs temporal sequences (B, seq_len, 3, H, W) to properly 
utilize the LSTM/video aggregation inside the Fusion model.

Output:
  - Prediction: REAL or FAKE
  - Confidence score (0–100%)
  - Per-frame breakdown for videos (optional)

Usage:
  # Single image (treated as a 1-frame sequence)
  python inference.py --checkpoint checkpoints/fusion_best.pt --input face.jpg

  # Video (samples seq_len frames and uses LSTM for temporal verdict)
  python inference.py --checkpoint checkpoints/fusion_best.pt --input video.mp4

  # Folder of images/videos
  python inference.py --checkpoint checkpoints/fusion_best.pt --input ./faces/
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from dataset import VideoTransform, extract_frames, IMAGE_SIZE
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
        temporal_type="lstm",  # Ensure LSTM is used for temporal aggregation
    ).to(device)
    ckpt  = torch.load(checkpoint, map_location=device)
    state = ckpt.get("state", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def predict_clip(
    model: DeepFakeFusionDetector,
    pil_images: list[Image.Image],
    transform: VideoTransform,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """
    Runs inference on a list of PIL Images (a video clip).
    Uses the VideoTransform to apply consistent spatial augmentations,
    then stacks into (1, seq_len, 3, H, W) for the LSTM-based model.
    """
    # Transform returns a list of tensors
    tensors = transform(pil_images)
    # Stack and add batch dimension: (seq_len, 3, H, W) -> (1, seq_len, 3, H, W)
    clip_tensor = torch.stack(tensors).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logit = model(clip_tensor)
        
    prob = logit.sigmoid().item()
    fake = prob >= threshold
    return {
        "label":      "FAKE" if fake else "REAL",
        "probability": prob,
        "confidence":  prob if fake else 1.0 - prob,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Main inference functions
# ──────────────────────────────────────────────────────────────────────────────

def run_on_image(path: str, model, transform, device, threshold: float) -> dict:
    img    = Image.open(path).convert("RGB")
    # Treat a single image as a 1-frame video sequence
    result = predict_clip(model, [img], transform, device, threshold)
    result["path"] = path
    return result


def run_on_video(
    path: str,
    model,
    transform,
    device,
    threshold: float,
    max_frames: int = 30,
    seq_len: int = 8,
) -> dict:
    import tempfile, shutil
    tmp_dir = tempfile.mkdtemp()
    try:
        # 1. Extract up to max_frames from the video
        frame_paths = extract_frames(path, tmp_dir, max_frames=max_frames)
        if not frame_paths:
            return {"path": path, "error": "No frames extracted"}
            
        all_images = [Image.open(fp).convert("RGB") for fp in frame_paths]
        
        # 2. Sample seq_len frames uniformly (just like the Dataset)
        if len(all_images) >= seq_len:
            indices = np.linspace(0, len(all_images) - 1, seq_len, dtype=int)
            sampled_images = [all_images[i] for i in indices]
        else:
            # Pad with the last frame if video is too short
            sampled_images = all_images + [all_images[-1]] * (seq_len - len(all_images))

        # 3. Get the primary video-level prediction using the sequence
        result = predict_clip(model, sampled_images, transform, device, threshold)
        result["path"] = path
        result["total_frames"] = len(all_images)
        
        # 4. (Optional but helpful) Per-frame breakdown using 1-frame sequences
        frame_results = []
        for img in all_images:
            frame_res = predict_clip(model, [img], transform, device, threshold)
            frame_results.append(frame_res)
            
        result["n_fake_frames"] = sum(1 for r in frame_results if r["label"] == "FAKE")
        
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
    print(f"  Verdict    : {verdict} (Sequence Aggregated)")
    print(f"  P(fake)    : {prob:.1f}%")
    print(f"  Confidence : {conf:.1f}%")
    if "total_frames" in result:
        print(f"  Frames     : {result['total_frames']}  ({result['n_fake_frames']} flagged fake individually)")
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
                   help="Max frames to extract from disk per video")
    p.add_argument("--seq_len",        type=int, default=8,
                   help="Number of frames to feed to model LSTM per video")
    p.add_argument("--save_json",      type=str, default=None,
                   help="Optionally save full results to JSON")
    return p.parse_args()


def main():
    args = parse_args()

    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) \
             if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    model     = load_model(args.checkpoint, device, args.swin_embed_dim, args.gan_embed_dim)
    # Use VideoTransform for consistent sequence processing
    transform = VideoTransform(is_train=False, enhance=False)
    
    inp       = args.input
    results   = []

    if os.path.isdir(inp):
        results = run_on_folder(inp, model, transform, device, args.threshold)
        for r in results:
            print_result(r)
    elif Path(inp).suffix.lower() in VIDEO_EXTS:
        r = run_on_video(inp, model, transform, device, args.threshold,
                         args.max_frames, args.seq_len)
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
        clean = []
        for r in results:
            rc = {k: v for k, v in r.items() if k != "frame_results"}
            clean.append(rc)
        with open(args.save_json, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"\nResults saved → {args.save_json}")


if __name__ == "__main__":
    main()