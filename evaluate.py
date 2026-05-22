"""
evaluate.py
===========
Evaluation script — all 6 metrics from the flow diagram:
  • AUC-ROC  (area under ROC curve)
  • F1-score  (harmonic mean P & R)
  • Accuracy  (correct predictions / total)
  • Precision  (TP / predicted positives)
  • Loss  (binary cross-entropy)
  • Confusion Matrix  (TP · TN · FP · FN)

UPDATED: Now evaluates video sequences [B, seq_len, 3, H, W] 
         instead of single images.

Usage:
  python evaluate.py \
      --checkpoint checkpoints/fusion_best.pt \
      --data_root  ./data \
      --frames_dir ./frames_cache \
      --split      test
"""

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, confusion_matrix, roc_curve,
)

from dataset import build_ff_dataloaders
from fusion_model import DeepFakeFusionDetector


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    criterion  = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for clips, labels in tqdm(loader, desc="Evaluating"):
        # clips shape: [B, seq_len, 3, H, W]
        clips     = clips.to(device)
        labels_f  = labels.float().unsqueeze(1).to(device)
        
        # Model must accept sequence input and output [B, 1] logits
        logits   = model(clips)
        
        total_loss += criterion(logits, labels_f).item()
        all_probs.extend(logits.sigmoid().squeeze(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    labels_np = np.array(all_labels, dtype=int)
    probs_np  = np.array(all_probs)
    preds_np  = (probs_np >= threshold).astype(int)

    cm = confusion_matrix(labels_np, preds_np)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics = {
        "loss":             total_loss / max(len(loader), 1),
        "auc_roc":          roc_auc_score(labels_np, probs_np) if len(set(labels_np)) > 1 else 0.0,
        "f1_score":         f1_score(labels_np, preds_np, zero_division=0),
        "accuracy":         accuracy_score(labels_np, preds_np),
        "precision":        precision_score(labels_np, preds_np, zero_division=0),
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "threshold":        threshold,
        "n_samples":        len(labels_np),
    }
    return metrics, probs_np, labels_np


def print_metrics(metrics):
    cm = metrics["confusion_matrix"]
    print("\n" + "="*52)
    print("  DeepFake Detection — Evaluation Results")
    print("="*52)
    print(f"  Videos        : {metrics['n_samples']}")
    print(f"  Threshold     : {metrics['threshold']:.2f}")
    print("-"*52)
    print(f"  Loss (BCE)    : {metrics['loss']:.4f}")
    print(f"  AUC-ROC       : {metrics['auc_roc']:.4f}")
    print(f"  F1-score      : {metrics['f1_score']:.4f}")
    print(f"  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  Precision     : {metrics['precision']:.4f}")
    print("-"*52)
    print("  Confusion Matrix:")
    print(f"    TP={cm['TP']:6d}   FP={cm['FP']:6d}")
    print(f"    FN={cm['FN']:6d}   TN={cm['TN']:6d}")
    print("="*52)


def plot_roc(labels, probs, save_path="roc_curve.png"):
    try:
        import matplotlib.pyplot as plt
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", linewidth=2)
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve — DeepFake Fusion Detector")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"ROC curve saved → {save_path}")
    except ImportError:
        print("[skip] matplotlib not installed — skipping ROC plot")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     type=str, required=True)
    p.add_argument("--data_root",      type=str, default="./data")
    p.add_argument("--frames_dir",     type=str, default="./frames_cache")
    p.add_argument("--split",          type=str, default="test", choices=["train","val","test"])
    p.add_argument("--max_frames",     type=int, default=30, help="Max frames extracted per video to disk")
    p.add_argument("--seq_len",        type=int, default=8, help="Number of frames per video sequence passed to model")
    p.add_argument("--batch_size",     type=int, default=8, help="Batch size (videos per batch)")
    p.add_argument("--num_workers",    type=int, default=4)
    p.add_argument("--device",         type=str, default="auto")
    p.add_argument("--threshold",      type=float, default=0.5)
    p.add_argument("--swin_embed_dim", type=int, default=96)
    p.add_argument("--gan_embed_dim",  type=int, default=64)
    p.add_argument("--plot_roc",       action="store_true")
    p.add_argument("--save_results",   type=str, default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    model = DeepFakeFusionDetector(
        swin_embed_dim=args.swin_embed_dim,
        gan_embed_dim=args.gan_embed_dim,
    ).to(device)
    
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("state", ckpt), strict=False)
    print(f"Loaded ← {args.checkpoint}")

    loaders = build_ff_dataloaders(
        data_root   = args.data_root,
        frames_dir  = args.frames_dir,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        max_frames  = args.max_frames,
        seq_len     = args.seq_len,
        enhance     = False,
    )

    metrics, probs, labels = evaluate(model, loaders[args.split], device, args.threshold)
    print_metrics(metrics)

    if args.plot_roc:
        plot_roc(labels, probs)
    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved → {args.save_results}")


if __name__ == "__main__":
    main()