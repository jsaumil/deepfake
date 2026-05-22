"""
train.py
========
Training pipeline for the DeepFake Fusion Detector on FaceForensics++.

Two-phase training:
  Phase 1 — GAN pre-training  (Discriminator learns synthesis artefacts)
  Phase 2 — Fusion model      (Swin + GAN features → Temporal LSTM → Real/Fake classifier)

UPDATED: Now trains on video sequences (B, seq_len, 3, H, W) instead of 
         independent image frames, leveraging temporal inconsistencies.

Usage:
  python train.py --data_root ./data --frames_dir ./frames_cache --seq_len 8 --epochs 30 --device cuda
"""

import argparse
import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score

from dataset import build_ff_dataloaders
from fusion_model import DeepFakeFusionDetector
from gan import DeepfakeGAN


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def save_checkpoint(model, path, epoch, val_auc):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"epoch": epoch, "val_auc": val_auc, "state": model.state_dict()}, path)
    print(f"  ✓ Checkpoint saved → {path}  (AUC={val_auc:.4f})")


# ──────────────────────────────────────────────────────────────────────────────
#  Phase 1 — GAN pre-training
# ──────────────────────────────────────────────────────────────────────────────

def pretrain_gan(train_loader, gan, num_epochs, save_dir):
    print("\n" + "="*60)
    print("  PHASE 1 — GAN Pre-training")
    print("  (Discriminator learns real vs synthesised face artefacts)")
    print("  Note: Video sequences are flattened to frames for 2D GAN.")
    print("="*60)
    gan.train(train_loader, num_epochs=num_epochs)
    gan.save(path_prefix=os.path.join(save_dir, "gan"))


# ──────────────────────────────────────────────────────────────────────────────
#  Phase 2 — Fusion model training
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, is_train):
    model.train(is_train)
    total_loss = 0.0
    all_labels, all_probs = [], []

    with torch.set_grad_enabled(is_train):
        pbar = tqdm(loader, leave=False, desc="train" if is_train else "eval ")
        for clips, labels in pbar:  # clips shape: (B, T, 3, H, W)
            clips  = clips.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            logits = model(clips)  # Model handles sequence natively via LSTM
            loss   = criterion(logits, labels)

            if is_train and optimizer:
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
        "loss":      total_loss / max(len(loader), 1),
        "auc":       roc_auc_score(int_labels, all_probs) if len(set(int_labels)) > 1 else 0.0,
        "f1":        f1_score(int_labels, int_preds, zero_division=0),
        "accuracy":  accuracy_score(int_labels, int_preds),
        "precision": precision_score(int_labels, int_preds, zero_division=0),
    }


def train_fusion(model, loaders, num_epochs, lr, save_dir, device):
    print("\n" + "="*60)
    print("  PHASE 2 — Fusion Model Training  (Swin + GAN + LSTM → Real/Fake)")
    print("="*60)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    best_auc  = 0.0
    history   = []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        tr = run_epoch(model, loaders["train"], criterion, optimizer, device, is_train=True)
        va = run_epoch(model, loaders["val"],   criterion, None,      device, is_train=False)
        scheduler.step()

        print(f"  Train  loss={tr['loss']:.4f}  AUC={tr['auc']:.4f}  Acc={tr['accuracy']:.4f}")
        print(f"  Val    loss={va['loss']:.4f}  AUC={va['auc']:.4f}  "
              f"F1={va['f1']:.4f}  Acc={va['accuracy']:.4f}  Prec={va['precision']:.4f}")

        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
                                          **{f"val_{k}": v for k, v in va.items()}})

        if va["auc"] > best_auc:
            best_auc = va["auc"]
            save_checkpoint(model, os.path.join(save_dir, "fusion_best.pt"), epoch, best_auc)

    save_checkpoint(model, os.path.join(save_dir, "fusion_last.pt"), num_epochs, va["auc"])
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ Training complete. Best Val AUC: {best_auc:.4f}")
    return history


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train DeepFake Fusion Detector on FaceForensics++")

    # Data
    p.add_argument("--data_root",        type=str,   default="./data",
                   help="Folder with real/ and fake/ video subfolders")
    p.add_argument("--frames_dir",       type=str,   default="./frames_cache",
                   help="Where to cache extracted video frames")
    p.add_argument("--max_frames",       type=int,   default=30,
                   help="Max frames extracted to disk per video")
    p.add_argument("--seq_len",          type=int,   default=8,
                   help="Number of frames per video sequence passed to model")
    p.add_argument("--batch_size",       type=int,   default=8,
                   help="Batch size (videos per batch)")
    p.add_argument("--num_workers",      type=int,   default=4)
    p.add_argument("--train_ratio",      type=float, default=0.70)
    p.add_argument("--val_ratio",        type=float, default=0.15)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--device",           type=str,   default="auto")

    # GAN pre-training
    p.add_argument("--gan_epochs",       type=int,   default=10,
                   help="Set 0 to skip GAN pre-training")
    p.add_argument("--latent_dim",       type=int,   default=256)
    p.add_argument("--gan_lr",           type=float, default=1e-4)

    # Fusion model
    p.add_argument("--epochs",           type=int,   default=30)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--swin_embed_dim",   type=int,   default=96)
    p.add_argument("--gan_embed_dim",    type=int,   default=64)
    p.add_argument("--temporal_type",    type=str,   default="lstm",
                   choices=["lstm", "avg_pool"],
                   help="Temporal aggregation method for video sequences")
    p.add_argument("--dropout",          type=float, default=0.4)
    p.add_argument("--freeze_swin",      action="store_true",
                   help="Freeze Swin backbone weights during fusion training")
    p.add_argument("--freeze_gan",       action="store_true",
                   help="Freeze GAN backbone weights during fusion training")

    # Pre-trained Checkpoints (To gather both models)
    p.add_argument("--swin_checkpoint",  type=str,   default=None,
                   help="Path to pre-trained Swin Transformer .pt file")
    p.add_argument("--gan_checkpoint",   type=str,   default=None,
                   help="Path to pre-trained GAN Discriminator .pt file (overrides Phase 1 if set)")

    # Output Checkpoints
    p.add_argument("--save_dir",         type=str,   default="./checkpoints")
    p.add_argument("--resume",           type=str,   default=None,
                   help="Resume full fusion model from checkpoint")
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    # ── Build DataLoaders ──────────────────────────────────────────────────────
    loaders = build_ff_dataloaders(
        data_root   = args.data_root,
        frames_dir  = args.frames_dir,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        max_frames  = args.max_frames,
        seq_len     = args.seq_len,
        train_ratio = args.train_ratio,
        val_ratio   = args.val_ratio,
        enhance     = True,
        seed        = args.seed,
    )

    # ── Phase 1: GAN pre-training ─────────────────────────────────────────────
    gan_ckpt_path = args.gan_checkpoint
    
    # Only run Phase 1 if we don't already have a pre-trained GAN checkpoint
    if args.gan_epochs > 0 and gan_ckpt_path is None:
        gan = DeepfakeGAN(latent_dim=args.latent_dim, device=str(device),
                          lr_g=args.gan_lr, lr_d=args.gan_lr)
        pretrain_gan(loaders["train"], gan, args.gan_epochs, args.save_dir)
        gan_ckpt_path = os.path.join(args.save_dir, "gan", "discriminator.pt")
    elif gan_ckpt_path is None:
        print("[warn] GAN pre-training skipped and no --gan_checkpoint provided.")
        print("       The GAN backbone will start from random weights.")

    # ── Phase 2: Fusion model (Gathering Both Models) ──────────────────────────
    print("\n" + "="*60)
    print("  GATHERING MODELS — Initializing Fusion Architecture")
    print("="*60)
    
    model = DeepFakeFusionDetector(
        swin_embed_dim = args.swin_embed_dim,
        gan_embed_dim  = args.gan_embed_dim,
        temporal_type  = args.temporal_type,
        freeze_swin    = args.freeze_swin,
        freeze_gan     = args.freeze_gan,
        dropout        = args.dropout,
    ).to(device)

    # 1. Gather the Swin Transformer
    if args.swin_checkpoint:
        model.load_pretrained_backbones(swin_ckpt=args.swin_checkpoint)
        print(f"✓ Swin Transformer weights loaded from: {args.swin_checkpoint}")
    else:
        print("✓ Swin Transformer initialized from scratch (no --swin_checkpoint provided)")

    # 2. Gather the GAN Discriminator
    if gan_ckpt_path and os.path.exists(gan_ckpt_path):
        model.load_pretrained_backbones(gan_ckpt=gan_ckpt_path)
        print(f"✓ GAN Discriminator weights loaded from: {gan_ckpt_path}")
    else:
        print("✓ GAN Discriminator initialized from scratch")

    # 3. Resume full fusion model if specified (overrides backbone loading)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt.get("state", ckpt), strict=False)
        print(f"✓ Full Fusion model resumed from {args.resume}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Ready: {total:,} params total | {trainable:,} trainable")
    
    if args.freeze_swin or args.freeze_gan:
        print("⚠️  Note: Some backbones are frozen. Only the unfrozen parts + LSTM + MLP will train.")

    train_fusion(model, loaders, args.epochs, args.lr, args.save_dir, device)


if __name__ == "__main__":
    main()