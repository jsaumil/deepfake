"""
train.py
========
Training pipeline for the DeepFake Fusion Detector.

Two-phase training (matching the flow diagram):
  Phase 1 – GAN pre-training
      Train Generator + Discriminator adversarially on real face images.
      This warm-starts the GAN backbone so it learns synthesis artefacts.

  Phase 2 – Fusion model training
      Load the pre-trained GAN backbone into the fusion model.
      Train the full Swin + GAN fusion on the labelled Real/Fake dataset.

Usage:
  python train.py --data_root ./data --epochs 30 --batch_size 16 --device cuda
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score

from dataset import build_dataloaders
from fusion_model import DeepFakeFusionDetector
from gan import DeepfakeGAN


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def save_checkpoint(model: nn.Module, path: str, epoch: int, val_auc: float):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "epoch":   epoch,
        "val_auc": val_auc,
        "state":   model.state_dict(),
    }, path)
    print(f"  ✓ Checkpoint saved → {path}  (AUC={val_auc:.4f})")


# ──────────────────────────────────────────────────────────────────────────────
#  Phase 1 – GAN pre-training
# ──────────────────────────────────────────────────────────────────────────────

def pretrain_gan(
    train_loader: DataLoader,
    gan: DeepfakeGAN,
    num_epochs: int,
    save_dir: str,
):
    """Adversarially train the GAN on the REAL images only."""
    print("\n" + "="*60)
    print("  PHASE 1 — GAN Pre-training")
    print("="*60)
    gan.train(train_loader, num_epochs=num_epochs)
    gan.save(path_prefix=os.path.join(save_dir, "gan"))


# ──────────────────────────────────────────────────────────────────────────────
#  Phase 2 – Fusion model training
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: DeepFakeFusionDetector,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer=None,
    device: torch.device = torch.device("cpu"),
    is_train: bool = True,
) -> dict:
    model.train(is_train)
    total_loss = 0.0
    all_labels, all_probs = [], []

    with torch.set_grad_enabled(is_train):
        pbar = tqdm(loader, leave=False, desc="train" if is_train else "eval ")
        for imgs, labels in pbar:
            imgs   = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            logits = model(imgs)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            all_probs.extend(logits.sigmoid().squeeze(1).detach().cpu().tolist())
            all_labels.extend(labels.squeeze(1).detach().cpu().tolist())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    int_labels = [int(l) for l in all_labels]
    int_preds  = [int(p >= 0.5) for p in all_probs]

    return {
        "loss":      total_loss / len(loader),
        "auc":       roc_auc_score(int_labels, all_probs) if len(set(int_labels)) > 1 else 0.0,
        "f1":        f1_score(int_labels, int_preds, zero_division=0),
        "accuracy":  accuracy_score(int_labels, int_preds),
        "precision": precision_score(int_labels, int_preds, zero_division=0),
    }


def train_fusion(
    model: DeepFakeFusionDetector,
    loaders: dict,
    num_epochs: int,
    lr: float,
    save_dir: str,
    device: torch.device,
):
    print("\n" + "="*60)
    print("  PHASE 2 — Fusion Model Training")
    print("="*60)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    best_auc     = 0.0
    best_ckpt    = os.path.join(save_dir, "fusion_best.pt")
    history      = []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_stats = run_epoch(model, loaders["train"], criterion, optimizer, device, is_train=True)
        val_stats   = run_epoch(model, loaders["val"],   criterion, None,      device, is_train=False)
        scheduler.step()

        log = {
            "epoch":        epoch,
            "train_loss":   train_stats["loss"],
            "train_auc":    train_stats["auc"],
            "val_loss":     val_stats["loss"],
            "val_auc":      val_stats["auc"],
            "val_f1":       val_stats["f1"],
            "val_accuracy": val_stats["accuracy"],
            "val_precision":val_stats["precision"],
        }
        history.append(log)

        print(
            f"  Train  loss={train_stats['loss']:.4f}  AUC={train_stats['auc']:.4f}  "
            f"Acc={train_stats['accuracy']:.4f}"
        )
        print(
            f"  Val    loss={val_stats['loss']:.4f}  AUC={val_stats['auc']:.4f}  "
            f"F1={val_stats['f1']:.4f}  Acc={val_stats['accuracy']:.4f}  "
            f"Prec={val_stats['precision']:.4f}"
        )

        if val_stats["auc"] > best_auc:
            best_auc = val_stats["auc"]
            save_checkpoint(model, best_ckpt, epoch, best_auc)

    # Save last checkpoint
    save_checkpoint(model, os.path.join(save_dir, "fusion_last.pt"), num_epochs, val_stats["auc"])

    # Save training history
    import json
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")

    return history


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DeepFake Fusion Detector – training")
    p.add_argument("--data_root",       type=str, default="./data",
                   help="Root folder with train/val/test splits")
    p.add_argument("--dataset_type",    type=str, default="image", choices=["image", "video"])
    p.add_argument("--batch_size",      type=int, default=16)
    p.add_argument("--num_workers",     type=int, default=4)
    p.add_argument("--device",          type=str, default="auto")

    # GAN pre-training
    p.add_argument("--gan_epochs",      type=int, default=10,
                   help="Epochs for GAN pre-training (0 to skip)")
    p.add_argument("--latent_dim",      type=int, default=256)
    p.add_argument("--gan_lr",          type=float, default=1e-4)

    # Fusion model training
    p.add_argument("--epochs",          type=int, default=30)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--swin_embed_dim",  type=int, default=96)
    p.add_argument("--gan_embed_dim",   type=int, default=64)
    p.add_argument("--dropout",         type=float, default=0.4)
    p.add_argument("--freeze_swin",     action="store_true")
    p.add_argument("--freeze_gan",      action="store_true")

    # Checkpoints
    p.add_argument("--save_dir",        type=str, default="./checkpoints")
    p.add_argument("--swin_ckpt",       type=str, default=None,
                   help="Optional pre-trained Swin checkpoint")
    p.add_argument("--resume",          type=str, default=None,
                   help="Resume fusion model from checkpoint")
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    # ── Build data loaders ────────────────────────────────────────────────────
    loaders = build_dataloaders(
        data_root    = args.data_root,
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
        enhance      = True,
        dataset_type = args.dataset_type,
    )

    # ── Phase 1 – GAN pre-training ────────────────────────────────────────────
    gan = DeepfakeGAN(
        latent_dim = args.latent_dim,
        device     = str(device),
        lr_g       = args.gan_lr,
        lr_d       = args.gan_lr,
    )

    if args.gan_epochs > 0:
        pretrain_gan(
            train_loader = loaders["train"],
            gan          = gan,
            num_epochs   = args.gan_epochs,
            save_dir     = args.save_dir,
        )
    else:
        # Try loading existing GAN checkpoints
        gan_ckpt_dir = os.path.join(args.save_dir, "gan")
        if os.path.isdir(gan_ckpt_dir):
            try:
                gan.load(gan_ckpt_dir)
            except Exception as e:
                print(f"[warn] Could not load GAN checkpoint: {e}")

    # ── Phase 2 – Fusion model ────────────────────────────────────────────────
    model = DeepFakeFusionDetector(
        swin_embed_dim = args.swin_embed_dim,
        gan_embed_dim  = args.gan_embed_dim,
        freeze_swin    = args.freeze_swin,
        freeze_gan     = args.freeze_gan,
        dropout        = args.dropout,
    ).to(device)

    # Load pre-trained GAN discriminator weights into fusion model
    model.gan_disc.load_state_dict(gan.D.state_dict(), strict=False)
    print("Loaded GAN discriminator weights into fusion model.")

    # Optionally load swin checkpoint
    if args.swin_ckpt:
        model.load_pretrained_backbones(swin_ckpt=args.swin_ckpt)

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state"], strict=False)
        print(f"Resumed from {args.resume}")

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,} total | {trainable_params:,} trainable")

    train_fusion(
        model      = model,
        loaders    = loaders,
        num_epochs = args.epochs,
        lr         = args.lr,
        save_dir   = args.save_dir,
        device     = device,
    )


if __name__ == "__main__":
    main()
