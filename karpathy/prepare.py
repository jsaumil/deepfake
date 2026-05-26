"""
dataset.py
==========
Dataset pipeline for FaceForensics++ structured data.

FIXED: Video-level sampling. One sample = one sequence of frames from one video.
Model now receives temporal context: Input shape [B, seq_len, 3, H, W]
"""

import os
import cv2
import random
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGE_SIZE  = 224
VIDEO_EXTS  = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS  = {".jpg", ".jpeg", ".png"}


# ──────────────────────────────────────────────────────────────────────────────
#  Step 1 — Scan videos and auto-split  (no manual folders needed)
# ──────────────────────────────────────────────────────────────────────────────

def collect_videos(data_root: str) -> list[tuple[str, int]]:
    root = Path(data_root)
    samples = []

    for label, subfolder in [(0, "real"), (1, "fake")]:
        folder = root / subfolder
        if not folder.exists():
            raise FileNotFoundError(
                f"Expected folder not found: {folder}\n"
                f"Make sure your data root has 'real/' and 'fake/' subfolders."
            )
        videos = [p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS]
        if not videos:
            raise RuntimeError(f"No videos found in {folder}")
        for v in sorted(videos):
            samples.append((str(v), label))

    random.shuffle(samples)
    return samples


def split_videos(
    samples: list,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> dict:
    random.seed(seed)

    real_vids = [s for s in samples if s[1] == 0]
    fake_vids = [s for s in samples if s[1] == 1]

    def _split(lst):
        random.shuffle(lst)
        n     = len(lst)
        n_tr  = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return lst[:n_tr], lst[n_tr:n_tr + n_val], lst[n_tr + n_val:]

    real_tr, real_val, real_te = _split(real_vids)
    fake_tr, fake_val, fake_te = _split(fake_vids)

    splits = {
        "train": real_tr + fake_tr,
        "val":   real_val + fake_val,
        "test":  real_te + fake_te,
    }
    for k, v in splits.items():
        random.shuffle(v)
        n_r = sum(1 for _, l in v if l == 0)
        n_f = sum(1 for _, l in v if l == 1)
        print(f"  [{k:5s}]  {len(v):4d} videos  |  real={n_r}  fake={n_f}")

    return splits


# ──────────────────────────────────────────────────────────────────────────────
#  Step 2 — Frame extraction (with disk cache)
# ──────────────────────────────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    out_dir:    str,
    max_frames: int = 30,
    frame_step: int = 1,
) -> list[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("*.jpg"))
    if existing:
        return [str(p) for p in existing[:max_frames]]

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total > 0 and max_frames < total:
        indices = set(int(i * total / max_frames) for i in range(max_frames))
    else:
        indices = set(range(total))

    paths  = []
    fcount = 0
    saved  = 0

    while True:
        ret, frame = cap.read()
        if not ret or saved >= max_frames:
            break
        if fcount in indices:
            out_path = out_dir / f"frame_{saved:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            paths.append(str(out_path))
            saved += 1
        fcount += 1

    cap.release()
    return paths


def extract_video_frame_groups(
    video_samples: list[tuple[str, int]],
    frames_dir:    str,
    max_frames:    int = 30,
    verbose:       bool = True,
) -> list[tuple[list[str], int]]:
    """
    FIXED: Returns groups of frames belonging to the SAME video.
    Format: [ ([frame1_path, frame2_path, ...], label), ... ]
    """
    video_frame_groups = []
    total = len(video_samples)

    for idx, (video_path, label) in enumerate(video_samples):
        video_name = Path(video_path).stem
        label_name = "real" if label == 0 else "fake"
        out_dir    = Path(frames_dir) / label_name / video_name

        paths = extract_frames(video_path, str(out_dir), max_frames=max_frames)

        if not paths and verbose:
            print(f"  [warn] No frames extracted from {video_path}")
            continue

        # Append the group of frames and its label
        video_frame_groups.append((paths, label))

        if verbose and (idx + 1) % 10 == 0:
            print(f"  Extracted frames: {idx+1}/{total} videos processed...")

    return video_frame_groups


# ──────────────────────────────────────────────────────────────────────────────
#  Step 3 — Image Enhancement
# ──────────────────────────────────────────────────────────────────────────────

class ImageEnhancer:
    def __init__(self, denoise=True, sharpen_factor=1.5, contrast_factor=1.2):
        self.denoise = denoise
        self.sharpen_factor = sharpen_factor
        self.contrast_factor = contrast_factor

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.denoise:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        if self.sharpen_factor != 1.0:
            img = ImageEnhance.Sharpness(img).enhance(self.sharpen_factor)
        if self.contrast_factor != 1.0:
            img = ImageEnhance.Contrast(img).enhance(self.contrast_factor)
        return img


# ──────────────────────────────────────────────────────────────────────────────
#  Step 4 — Video Transforms (Consistent across frames!)
# ──────────────────────────────────────────────────────────────────────────────

class VideoTransform:
    """
    Applies transforms to a sequence of frames.
    Ensures spatial augmentations (flip, rotate) are identical for all frames
    in the clip so the model doesn't learn broken temporal signals.
    """
    def __init__(self, is_train: bool = True, enhance: bool = True):
        self.is_train = is_train
        self.enhance = enhance
        self.enhancer = ImageEnhancer()

    def __call__(self, clip_frames: list[Image.Image]) -> list[torch.Tensor]:
        tensors = []
        
        # Decide random spatial augmentations ONCE per clip
        do_flip = self.is_train and random.random() > 0.5
        rotation_angle = random.uniform(-10, 10) if self.is_train else 0

        for img in clip_frames:
            if self.enhance:
                img = self.enhancer(img)

            img = transforms.functional.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

            # Apply same spatial transforms to all frames in the video
            if do_flip:
                img = transforms.functional.hflip(img)
            if rotation_angle != 0:
                img = transforms.functional.rotate(img, rotation_angle)

            # Color jitter can vary slightly per frame, or be fixed. 
            # Fixed is safer for temporal consistency:
            if self.is_train:
                img = transforms.functional.adjust_brightness(img, 1.0 + random.uniform(-0.2, 0.2))
                img = transforms.functional.adjust_contrast(img, 1.0 + random.uniform(-0.2, 0.2))

            img = transforms.functional.to_tensor(img)
            img = transforms.functional.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            tensors.append(img)

        return tensors


# ──────────────────────────────────────────────────────────────────────────────
#  Step 5 — PyTorch Video Dataset
# ──────────────────────────────────────────────────────────────────────────────

class FaceForensicsVideoDataset(Dataset):
    """
    Video-level dataset. Each sample is a sequence of frames from one video.
    Returns shape: [seq_len, 3, H, W]
    """
    def __init__(
        self,
        video_frame_groups: list[tuple[list[str], int]],
        transform=None,
        seq_len: int = 8,   # How many frames to feed to the model per video
    ):
        self.groups  = video_frame_groups
        self.transform = transform
        self.seq_len = seq_len

        n_videos = len(video_frame_groups)
        n_real = sum(1 for _, l in video_frame_groups if l == 0)
        n_fake = n_videos - n_real
        print(f"  Dataset: {n_videos} videos  |  real={n_real}  fake={n_fake}")

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int):
        frame_paths, label = self.groups[idx]

        # Evenly sample `seq_len` frames from the video's total frames
        if len(frame_paths) >= self.seq_len:
            indices = np.linspace(0, len(frame_paths) - 1, self.seq_len, dtype=int)
            selected_paths = [frame_paths[i] for i in indices]
        else:
            # If video is shorter than seq_len, pad by repeating the last frame
            selected_paths = frame_paths + [frame_paths[-1]] * (self.seq_len - len(frame_paths))

        # Load PIL Images
        images = [Image.open(p).convert("RGB") for p in selected_paths]

        # Apply transforms (returns list of Tensors)
        if self.transform:
            images = self.transform(images)

        # Stack list of tensors -> shape [seq_len, 3, H, W]
        video_tensor = torch.stack(images)

        return video_tensor, torch.tensor(label, dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────────────
#  Main builder
# ──────────────────────────────────────────────────────────────────────────────

def build_ff_dataloaders(
    data_root:    str,
    frames_dir:   str  = "./frames_cache",
    batch_size:   int  = 8,      # Reduced default since video tensors are heavy
    num_workers:  int  = 4,
    max_frames:   int  = 30,     # Max frames to extract to disk per video
    seq_len:      int  = 8,      # Frames per forward pass in model
    train_ratio:  float = 0.70,
    val_ratio:    float = 0.15,
    enhance:      bool = True,
    seed:         int  = 42,
) -> dict:
    print("\n" + "="*60)
    print("  FaceForensics++ Video DataLoader Builder")
    print("="*60)

    # 1. Collect & split videos
    print("\n[1/3] Scanning videos...")
    all_videos = collect_videos(data_root)
    print(f"  Found {len(all_videos)} videos total "
          f"({sum(1 for _,l in all_videos if l==0)} real | "
          f"{sum(1 for _,l in all_videos if l==1)} fake)")

    print("\n[2/3] Splitting into train / val / test...")
    splits = split_videos(all_videos, train_ratio, val_ratio, seed)

    # 2. Extract frames for each split
    print("\n[3/3] Extracting frames (using cache if available)...")
    loaders = {}

    for split_name, video_list in splits.items():
        print(f"\n  -> {split_name} split ({len(video_list)} videos):")
        
        # Returns groups: [ ( [path1, path2...], label ), ... ]
        frame_groups = extract_video_frame_groups(
            video_list,
            frames_dir = os.path.join(frames_dir, split_name),
            max_frames = max_frames,
            verbose    = True,
        )

        is_train  = split_name == "train"
        transform = VideoTransform(is_train=is_train, enhance=enhance)
        
        dataset = FaceForensicsVideoDataset(
            frame_groups, 
            transform=transform, 
            seq_len=seq_len
        )

        loaders[split_name] = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = is_train,
            num_workers = num_workers,
            pin_memory  = torch.cuda.is_available(),
            drop_last   = is_train,
        )

    print("\n" + "="*60)
    print("  Video DataLoaders ready!")
    print("="*60 + "\n")
    return loaders


# ──────────────────────────────────────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  type=str, default="./data")
    p.add_argument("--frames_dir", type=str, default="./frames_cache")
    p.add_argument("--max_frames", type=int, default=10)
    p.add_argument("--seq_len",    type=int, default=4)
    p.add_argument("--batch_size", type=int, default=2)
    args = p.parse_args()

    loaders = build_ff_dataloaders(
        data_root  = args.data_root,
        frames_dir = args.frames_dir,
        max_frames = args.max_frames,
        seq_len    = args.seq_len,
        batch_size = args.batch_size,
        num_workers = 0,
    )

    for split, loader in loaders.items():
        videos, labels = next(iter(loader))
        print(f"{split:5s}  batch: videos={videos.shape}  labels={labels.tolist()}")
        # Expected output shape: videos=[B, seq_len, 3, H, W]

if __name__ == "__main__":
    print("Running one-time data preparation...")
    loaders = build_ff_dataloaders(
        data_root="karpathy\\FF++",
        frames_dir="./frames_cache",
        batch_size=4,
        seq_len=8,
        enhance=True
    )
    print("Data preparation complete. Caches saved to ./frames_cache")