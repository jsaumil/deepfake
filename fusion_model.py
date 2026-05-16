"""
fusion_model.py
===============
Fusion Detector — combines Swin Transformer + GAN Discriminator features
to classify an image/video frame as Real or Fake.

Architecture (matches the flow diagram):
  ┌─────────────────────┐    ┌──────────────────────────┐
  │  Swin Transformer   │    │  GAN Discriminator       │
  │  (global attention) │    │  (synthesis artefacts)   │
  └────────┬────────────┘    └────────────┬─────────────┘
           │  feat_swin (768)             │  feat_gan (256)
           └─────────────┬───────────────┘
                    Concatenate  (1024)
                         │
                  Fusion MLP Head
                         │
                  Logit  → sigmoid
                         │
              Real / Fake + confidence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from swin_transformer import SwinTransformer
from gan import DeepfakeDiscriminator


# ──────────────────────────────────────────────────────────────────────────────
#  Fusion Model
# ──────────────────────────────────────────────────────────────────────────────

class DeepFakeFusionDetector(nn.Module):
    """
    Combines Swin Transformer features and GAN Discriminator features via
    concatenation, then classifies Real / Fake.

    Parameters
    ----------
    swin_embed_dim : int
        Base embedding dimension for the Swin backbone (default 96).
    gan_embed_dim  : int
        Base embedding dimension for the GAN backbone (default 64).
    freeze_swin    : bool
        If True, Swin weights are frozen (useful when loading a pre-trained
        Swin checkpoint and only training the fusion head).
    freeze_gan     : bool
        If True, GAN discriminator weights are frozen.
    dropout        : float
        Dropout rate in the fusion MLP.
    """

    def __init__(
        self,
        swin_embed_dim: int = 96,
        gan_embed_dim:  int = 64,
        freeze_swin:    bool = False,
        freeze_gan:     bool = False,
        dropout:        float = 0.4,
    ):
        super().__init__()

        # ── Swin Transformer backbone ─────────────────────────────────────────
        self.swin = SwinTransformer(embed_dim=swin_embed_dim)
        swin_feat_dim = swin_embed_dim * 8         # 768 with default

        if freeze_swin:
            for p in self.swin.parameters():
                p.requires_grad_(False)

        # ── GAN Discriminator backbone ────────────────────────────────────────
        self.gan_disc = DeepfakeDiscriminator(embed_dim=gan_embed_dim)
        gan_feat_dim  = self.gan_disc.backbone.out_dim   # 256 with default

        if freeze_gan:
            for p in self.gan_disc.parameters():
                p.requires_grad_(False)

        # ── Fusion MLP ────────────────────────────────────────────────────────
        fused_dim = swin_feat_dim + gan_feat_dim       # 768 + 256 = 1024

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),          # raw logit
        )

        self._init_head()

    def _init_head(self):
        for m in self.fusion_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_swin_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract Swin Transformer features: (B, 768)."""
        return self.swin.forward_features(x)

    def _get_gan_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract GAN Discriminator backbone features: (B, 256).
        Expects x in [0, 1]; converts to [-1, 1] internally."""
        x_norm = x * 2.0 - 1.0          # [0,1] → [-1,1]
        return self.gan_disc.backbone(x_norm)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, 224, 224) float tensor in [0, 1]

        Returns
        -------
        logits : (B, 1)  — apply sigmoid to get P(fake)
        """
        feat_swin = self._get_swin_features(x)   # (B, 768)
        feat_gan  = self._get_gan_features(x)    # (B, 256)
        fused     = torch.cat([feat_swin, feat_gan], dim=-1)   # (B, 1024)
        return self.fusion_head(fused)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> dict:
        """
        Convenience wrapper that returns human-readable results.

        Returns
        -------
        dict with keys:
            'logits'      : raw model output (B, 1)
            'probability' : P(fake) in [0, 1] (B,)
            'prediction'  : list of 'FAKE' | 'REAL' strings
            'confidence'  : confidence score in [0, 1] (B,)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            prob   = logits.sigmoid().squeeze(-1)    # (B,)
            pred   = (prob >= threshold).cpu().tolist()
            conf   = torch.where(prob >= threshold, prob, 1.0 - prob).cpu()

        return {
            "logits":      logits,
            "probability": prob.cpu(),
            "prediction":  ["FAKE" if p else "REAL" for p in pred],
            "confidence":  conf,
        }

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Saved fusion model → {path}")

    def load(self, path: str, map_location: str = "cpu") -> None:
        self.load_state_dict(
            torch.load(path, map_location=map_location), strict=False
        )
        print(f"Loaded fusion model ← {path}")

    def load_pretrained_backbones(
        self,
        swin_ckpt:  str | None = None,
        gan_ckpt:   str | None = None,
    ) -> None:
        """Load separately pre-trained backbone checkpoints."""
        if swin_ckpt:
            state = torch.load(swin_ckpt, map_location="cpu")
            self.swin.load_state_dict(state, strict=False)
            print(f"Loaded Swin weights ← {swin_ckpt}")
        if gan_ckpt:
            state = torch.load(gan_ckpt, map_location="cpu")
            self.gan_disc.load_state_dict(state, strict=False)
            print(f"Loaded GAN-Disc weights ← {gan_ckpt}")


# ──────────────────────────────────────────────────────────────────────────────
#  Smoke test
# ──────────────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = DeepFakeFusionDetector(
        swin_embed_dim=96,
        gan_embed_dim=64,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")

    # Forward pass
    x = torch.rand(2, 3, 224, 224).to(device)   # [0, 1] normalized
    logits = model(x)
    print(f"Logits shape: {logits.shape}")       # (2, 1)

    result = model.predict(x)
    print(f"Predictions : {result['prediction']}")
    print(f"Confidence  : {result['confidence'].tolist()}")


if __name__ == "__main__":
    main()
