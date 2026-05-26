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


from pathlib import Path
import cv2


def extract_frames(video_path, output_dir, max_frames=30):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob("*.jpg"))
    if len(existing) >= max_frames:
        return [str(p) for p in existing[:max_frames]]

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return []

    indices = [
        int(i * total / max_frames)
        for i in range(max_frames)
    ]

    paths = []

    frame_idx = 0
    save_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx in indices:
            path = output_dir / f"frame_{save_idx:04d}.jpg"

            cv2.imwrite(str(path), frame)

            paths.append(str(path))
            save_idx += 1

        frame_idx += 1

    cap.release()

    return paths

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def build_video_index(
    split_dir,
    frame_cache_dir,
    max_frames=30,
):
    split_dir = Path(split_dir)

    samples = []

    for label_name, label in [
        ("real", 0),
        ("fake", 1),
    ]:

        video_dir = split_dir / label_name

        videos = [
            p for p in video_dir.iterdir()
            if p.suffix.lower() in VIDEO_EXTS
        ]

        for video_path in videos:

            cache_dir = (
                Path(frame_cache_dir)
                / label_name
                / video_path.stem
            )

            frame_paths = extract_frames(
                video_path,
                cache_dir,
                max_frames=max_frames,
            )

            if len(frame_paths) == 0:
                continue

            samples.append(
                (frame_paths, label)
            )

    return samples


class FaceForensicsVideoDataset(Dataset):

    def __init__(
        self,
        samples,
        transform=None,
        seq_len=8,
    ):
        self.samples = samples
        self.transform = transform
        self.seq_len = seq_len

        print(f"Videos: {len(samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        frame_paths, label = self.samples[idx]

        if len(frame_paths) >= self.seq_len:

            indices = np.linspace(
                0,
                len(frame_paths) - 1,
                self.seq_len,
                dtype=int,
            )

            frame_paths = [
                frame_paths[i]
                for i in indices
            ]

        else:

            frame_paths += [
                frame_paths[-1]
            ] * (self.seq_len - len(frame_paths))

        frames = []

        for path in frame_paths:

            img = Image.open(path).convert("RGB")

            if self.transform:
                img = self.transform(img)

            frames.append(img)

        video = torch.stack(frames)

        return video, torch.tensor(label)
    
def build_dataloaders(
    dataset_root,
    frame_cache_root="./frame_cache",
    transform=None,
    batch_size=4,
    seq_len=8,
    num_workers=4,
):

    train_samples = build_video_index(
        f"{dataset_root}/train",
        f"{frame_cache_root}/train",
    )

    val_samples = build_video_index(
        f"{dataset_root}/val",
        f"{frame_cache_root}/val",
    )

    test_samples = build_video_index(
        f"{dataset_root}/test",
        f"{frame_cache_root}/test",
    )

    train_dataset = FaceForensicsVideoDataset(
        train_samples,
        transform=transform,
        seq_len=seq_len,
    )

    val_dataset = FaceForensicsVideoDataset(
        val_samples,
        transform=transform,
        seq_len=seq_len,
    )

    test_dataset = FaceForensicsVideoDataset(
        test_samples,
        transform=transform,
        seq_len=seq_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader