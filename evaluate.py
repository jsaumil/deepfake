"""
evaluate.py
===========
Evaluation script — computes all metrics from the flow diagram:
  • AUC-ROC
  • F1-score
  • Accuracy
  • Precision
  • Loss (binary cross-entropy)
  • Confusion Matrix  (TP / TN / FP / FN)

Usage:
  python evaluate.py --checkpoint checkpoints/fusion_best.pt \
                     --data_root ./data --split test
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

from dataset import build_dataloaders
from fusion_model import DeepFakeFusionDetector


# ──────────────────────────────────────────────────────────────────────────────
#  Core evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: DeepFakeFusionDetector,
    loader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    criterion  = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for imgs, labels in tqdm(loader, desc="Evaluating"):
        imgs   = imgs.to(device)
        labels_f = labels.float().unsqueeze(1).to(device)

        logits = model(imgs)
        loss   = criterion(logits, labels_f)
        total_loss += loss.item()

        all_probs.extend(logits.sigmoid().squeeze(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    labels_np = np.array(all_labels, dtype=int)
    probs_np  = np.array(all_probs)
    preds_np  = (probs_np >= threshold).astype(int)

    cm = confusion_matrix(labels_np, preds_np)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics = {
        "loss":           total_loss / len(loader),
        "auc_roc":        roc_auc_score(labels_np, probs_np) if len(set(labels_np)) > 1 else 0.0,
        "f1_score":       f1_score(labels_np, preds_np, zero_division=0),
        "accuracy":       accuracy_score(labels_np, preds_np),
        "precision":      precision_score(labels_np, preds_np, zero_division=0),
        "confusion_matrix": {
            "TP": int(tp), "TN": int(tn),
            "FP": int(fp), "FN": int(fn),
        },
        "threshold": threshold,
        "n_samples": len(labels_np),
    }
    return metrics, probs_np, labels_np


# ──────────────────────────────────────────────────────────────────────────────
#  Pretty printer
# ──────────────────────────────────────────────────────────────────────────────

def print_metrics(metrics: dict):
    cm = metrics["confusion_matrix"]
    print("\n" + "="*50)
    print("  DeepFake Detection — Evaluation Results")
    print("="*50)
    print(f"  Samples       : {metrics['n_samples']}")
    print(f"  Threshold     : {metrics['threshold']:.2f}")
    print("-"*50)
    print(f"  Loss (BCE)    : {metrics['loss']:.4f}")
    print(f"  AUC-ROC       : {metrics['auc_roc']:.4f}")
    print(f"  F1-score      : {metrics['f1_score']:.4f}")
    print(f"  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  Precision     : {metrics['precision']:.4f}")
    print("-"*50)
    print("  Confusion Matrix:")
    print(f"    TP={cm['TP']:5d}   FP={cm['FP']:5d}")
    print(f"    FN={cm['FN']:5d}   TN={cm['TN']:5d}")
    print("="*50)


# ──────────────────────────────────────────────────────────────────────────────
#  Optional: plot ROC curve (saved to file)
# ──────────────────────────────────────────────────────────────────────────────

def plot_roc(labels, probs, save_path: str = "roc_curve.png"):
    try:
        import matplotlib.pyplot as plt
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", linewidth=2)
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve — DeepFake Detector")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"ROC curve saved → {save_path}")
    except ImportError:
        print("[skip] matplotlib not installed — skipping ROC plot")


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DeepFake Fusion Detector — Evaluation")
    p.add_argument("--checkpoint",    type=str, required=True,
                   help="Path to fusion model checkpoint (.pt)")
    p.add_argument("--data_root",     type=str, default="./data")
    p.add_argument("--split",         type=str, default="test", choices=["train","val","test"])
    p.add_argument("--dataset_type",  type=str, default="image", choices=["image","video"])
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--device",        type=str, default="auto")
    p.add_argument("--threshold",     type=float, default=0.5)
    p.add_argument("--swin_embed_dim",type=int, default=96)
    p.add_argument("--gan_embed_dim", type=int, default=64)
    p.add_argument("--plot_roc",      action="store_true")
    p.add_argument("--save_results",  type=str, default=None,
                   help="Save metrics JSON to this path")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    model = DeepFakeFusionDetector(
        swin_embed_dim=args.swin_embed_dim,
        gan_embed_dim=args.gan_embed_dim,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("state", ckpt)   # handle both raw and wrapped checkpoints
    model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint ← {args.checkpoint}")

    # Build loader for the chosen split only
    loaders = build_dataloaders(
        data_root    = args.data_root,
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
        enhance      = False,
        dataset_type = args.dataset_type,
    )
    loader = loaders[args.split]

    metrics, probs, labels = evaluate(model, loader, device, args.threshold)
    print_metrics(metrics)

    if args.plot_roc:
        plot_roc(labels, probs)

    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved → {args.save_results}")


if __name__ == "__main__":
    main()
