"""
dataset.py
==========
Dataset and preprocessing utilities for DeepFake detection.

Supports:
  - Image datasets (DFDC / FaceForensics++ style folder structure)
  - Video datasets (frame extraction)
  - Image enhancement (noise reduction, sharpness, contrast)
  - Data augmentation

Expected folder structure:
    data/
      train/
        real/   *.jpg / *.png
        fake/   *.jpg / *.png
      val/
        real/
        fake/
      test/
        real/
        fake/
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ──────────────────────────────────────────────────────────────────────────────
#  Image Enhancement  (step 2 in the flow diagram)
# ──────────────────────────────────────────────────────────────────────────────

class ImageEnhancer:
    """
    Applies noise reduction, sharpness boost, and contrast adjustment
    to a PIL Image before the standard torchvision transforms.
    """

    def __init__(
        self,
        denoise: bool = True,
        sharpen_factor: float = 1.5,
        contrast_factor: float = 1.2,
    ):
        self.denoise         = denoise
        self.sharpen_factor  = sharpen_factor
        self.contrast_factor = contrast_factor

    def __call__(self, img: Image.Image) -> Image.Image:
        # 1. Noise reduction — mild Gaussian blur
        if self.denoise:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        # 2. Sharpness
        if self.sharpen_factor != 1.0:
            img = ImageEnhance.Sharpness(img).enhance(self.sharpen_factor)

        # 3. Contrast adjustment
        if self.contrast_factor != 1.0:
            img = ImageEnhance.Contrast(img).enhance(self.contrast_factor)

        return img


# ──────────────────────────────────────────────────────────────────────────────
#  Transform Builders
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_SIZE = 224

def build_train_transforms(enhance: bool = True) -> transforms.Compose:
    steps = []
    if enhance:
        steps.append(ImageEnhancer())
    steps += [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(steps)


def build_eval_transforms(enhance: bool = False) -> transforms.Compose:
    steps = []
    if enhance:
        steps.append(ImageEnhancer())
    steps += [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(steps)


# ──────────────────────────────────────────────────────────────────────────────
#  Image Dataset
# ──────────────────────────────────────────────────────────────────────────────

class DeepFakeImageDataset(Dataset):
    """
    Loads images from a folder structure:
        root/real/*.{jpg,png,jpeg}
        root/fake/*.{jpg,png,jpeg}

    Labels: 0 = real, 1 = fake
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str, transform=None):
        self.root      = Path(root)
        self.transform = transform
        self.samples   = []   # list of (path, label)

        for label, subfolder in [(0, "real"), (1, "fake")]:
            folder = self.root / subfolder
            if not folder.exists():
                raise FileNotFoundError(f"Expected folder: {folder}")
            for p in folder.iterdir():
                if p.suffix.lower() in self.EXTENSIONS:
                    self.samples.append((str(p), label))

        if not self.samples:
            raise RuntimeError(f"No images found under {root}")

        # Class balance info
        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = len(self.samples) - n_real
        print(f"[Dataset] {root}: {n_real} real | {n_fake} fake | total {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ──────────────────────────────────────────────────────────────────────────────
#  Video Dataset  (frame-level)
# ──────────────────────────────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    out_dir: str,
    max_frames: int = 30,
    frame_step: int = 1,
) -> list[str]:
    """
    Extract up to `max_frames` frames from a video at every `frame_step`-th frame.
    Saves as JPEGs and returns the list of saved paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap     = cv2.VideoCapture(video_path)
    paths   = []
    count   = 0
    saved   = 0

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_step == 0:
            out_path = out_dir / f"frame_{saved:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            paths.append(str(out_path))
            saved += 1
        count += 1

    cap.release()
    return paths


class DeepFakeVideoDataset(Dataset):
    """
    Loads video files from:
        root/real/*.{mp4,avi,mov}
        root/fake/*.{mp4,avi,mov}

    Extracts frames on the fly (cached to tmp_frames_dir).
    Each sample is one frame.
    """

    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(
        self,
        root: str,
        transform=None,
        max_frames_per_video: int = 30,
        tmp_frames_dir: str = "tmp_frames",
    ):
        self.transform = transform
        self.samples   = []   # (frame_path, label)

        root = Path(root)
        for label, subfolder in [(0, "real"), (1, "fake")]:
            folder = root / subfolder
            if not folder.exists():
                continue
            for vp in folder.iterdir():
                if vp.suffix.lower() in self.VIDEO_EXTS:
                    out_dir = Path(tmp_frames_dir) / subfolder / vp.stem
                    if not out_dir.exists() or not any(out_dir.iterdir()):
                        paths = extract_frames(str(vp), str(out_dir), max_frames_per_video)
                    else:
                        paths = sorted(str(p) for p in out_dir.iterdir())
                    self.samples.extend((p, label) for p in paths)

        print(f"[VideoDataset] {root}: {len(self.samples)} frames")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ──────────────────────────────────────────────────────────────────────────────
#  DataLoader builders
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    enhance: bool = True,
    dataset_type: str = "image",   # "image" | "video"
    max_frames_per_video: int = 30,
) -> dict:
    """
    Returns dict with keys 'train', 'val', 'test' → DataLoader objects.
    """
    DatasetClass = DeepFakeImageDataset if dataset_type == "image" else DeepFakeVideoDataset

    loaders = {}
    for split in ("train", "val", "test"):
        is_train = split == "train"
        tf = build_train_transforms(enhance) if is_train else build_eval_transforms()
        split_root = os.path.join(data_root, split)

        if dataset_type == "image":
            ds = DatasetClass(split_root, transform=tf)
        else:
            ds = DatasetClass(split_root, transform=tf, max_frames_per_video=max_frames_per_video)

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=is_train,
        )

    return loaders
