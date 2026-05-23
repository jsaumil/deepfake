"""
gan.py
======
GAN components for DeepFake detection.

Architecture
────────────
Generator   : noise z → 224×224 RGB fake face
Discriminator: uses a lightweight Swin-based feature extractor
               to distinguish real vs GAN-generated images.

The Discriminator's feature extractor is shared with the fusion model,
so pre-training it adversarially gives the detector a strong starting
point for spotting synthesis artefacts.

UPDATED: GAN trainer now natively supports the video sequence DataLoader
         (B, T, 3, 224, 224) by flattening sequences into independent 
         2D frames for standard 2D GAN pre-training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
#  Swin helpers (self-contained, for 224×224 RGB)
# ──────────────────────────────────────────────────────────────────────────────

def window_partition(x: torch.Tensor, win: int) -> torch.Tensor:
    """(B, H, W, C) → (B*num_windows, win, win, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // win, win, W // win, win, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.reshape(-1, win, win, C)


def window_reverse(windows: torch.Tensor, win: int, H: int, W: int) -> torch.Tensor:
    """(B*num_windows, win, win, C) → (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W // win // win))
    x = windows.view(B, H // win, W // win, win, win, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.reshape(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, win: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.win       = win

        self.qkv  = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias
        coords = torch.stack(
            torch.meshgrid(torch.arange(win), torch.arange(win), indexing='ij')
        )
        coords_flat = coords.flatten(1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0)
        rel[:, :, 0] += win - 1
        rel[:, :, 1] += win - 1
        rel[:, :, 0] *= 2 * win - 1
        self.register_buffer('pos_index', rel.sum(-1))
        self.rel_bias = nn.Parameter(
            torch.zeros((2 * win - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv.unbind(0)

        attn = (Q * self.scale) @ K.transpose(-2, -1)

        # Relative position bias
        rpb = self.rel_bias[self.pos_index.reshape(-1)].reshape(N, N, self.num_heads)
        attn = attn + rpb.permute(2, 0, 1).unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(B_ // nw, nw, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        out  = (attn @ V).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)


class SwinBlock(nn.Module):
    def __init__(self, dim: int, res: tuple, win: int, shift: int, heads: int):
        super().__init__()
        self.res   = res
        self.win   = win
        self.shift = shift

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(dim, heads, win)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * dim, dim),
        )

        H, W = res
        self.mask = self._make_mask(H, W, win, shift) if shift > 0 else None

    def _make_mask(self, H: int, W: int, win: int, shift: int) -> torch.Tensor:
        img_mask = torch.zeros((1, H, W, 1))
        cnt = 0
        for h in (slice(0, -win), slice(-win, -shift), slice(-shift, None)):
            for w in (slice(0, -win), slice(-win, -shift), slice(-shift, None)):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask = window_partition(img_mask, win).view(-1, win * win)
        mask = mask.unsqueeze(1) - mask.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        H, W = self.res
        res = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))

        win_x = window_partition(x, self.win).view(-1, self.win ** 2, C)
        out   = self.attn(win_x, self.mask.to(x.device) if self.mask is not None else None)
        x     = window_reverse(out.view(-1, self.win, self.win, C), self.win, H, W)

        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))

        x = res + x.view(B, L, C)
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm      = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat(
            [x[:, 0::2, 0::2], x[:, 1::2, 0::2],
             x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1
        ).reshape(B, -1, 4 * C)
        return self.reduction(self.norm(x)), H // 2, W // 2


# ──────────────────────────────────────────────────────────────────────────────
#  Lightweight Swin backbone (for the Discriminator)
#  Works on 224×224 RGB  →  feature vector of size `out_dim`
# ──────────────────────────────────────────────────────────────────────────────

class SwinBackbone(nn.Module):
    """
    3-stage Swin Transformer feature extractor.
    Input : (B, 3, 224, 224)
    Output: (B, out_dim)   where out_dim = 4 * embed_dim
    """

    def __init__(self, embed_dim: int = 64, win: int = 7, heads: int = 4):
        super().__init__()
        C = embed_dim

        # Patch embed: 224 → 56×56
        self.patch_embed = nn.Conv2d(3, C, kernel_size=4, stride=4)
        res1 = (56, 56)

        self.stage1 = nn.Sequential(
            SwinBlock(C, res1, win=win, shift=0,       heads=heads),
            SwinBlock(C, res1, win=win, shift=win // 2, heads=heads),
        )
        self.merge1 = PatchMerging(C)           # 56 → 28, C → 2C
        res2 = (28, 28)

        self.stage2 = nn.Sequential(
            SwinBlock(C*2, res2, win=win, shift=0,       heads=heads*2),
            SwinBlock(C*2, res2, win=win, shift=win // 2, heads=heads*2),
        )
        self.merge2 = PatchMerging(C*2)         # 28 → 14, 2C → 4C
        res3 = (14, 14)

        self.stage3 = nn.Sequential(
            SwinBlock(C*4, res3, win=win, shift=0,       heads=heads*4),
            SwinBlock(C*4, res3, win=win, shift=win // 2, heads=heads*4),
        )

        self.norm    = nn.LayerNorm(C * 4)
        self.out_dim = C * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)                    # (B, C, 56, 56)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)           # (B, 56*56, C)

        x = self.stage1(x)
        x, H, W = self.merge1(x, H, W)
        x = self.stage2(x)
        x, H, W = self.merge2(x, H, W)
        x = self.stage3(x)
        x = self.norm(x)
        return x.mean(dim=1)                        # (B, out_dim)


# ──────────────────────────────────────────────────────────────────────────────
#  Generator  (noise → 224×224 RGB)
# ──────────────────────────────────────────────────────────────────────────────

class DeepfakeGenerator(nn.Module):
    """
    Latent vector z  →  224×224 RGB image.
    Maps: z (B, latent_dim) → img (B, 3, 224, 224) in [-1, 1]

    Upsampling path: 7×7 → 14 → 28 → 56 → 112 → 224
    """

    def __init__(self, latent_dim: int = 256, base_ch: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.init_size  = 7
        self.fc         = nn.Linear(latent_dim, base_ch * self.init_size ** 2)

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.net = nn.Sequential(
            nn.BatchNorm2d(base_ch),                   # 7×7
            up_block(base_ch,      base_ch // 2),      # → 14×14
            up_block(base_ch // 2, base_ch // 4),      # → 28×28
            up_block(base_ch // 4, base_ch // 8),      # → 56×56
            up_block(base_ch // 8, base_ch // 16),     # → 112×112
            up_block(base_ch // 16, base_ch // 32),    # → 224×224
            nn.Conv2d(base_ch // 32, 3, 3, padding=1),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d) and m.weight is not None:
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(z.size(0), -1, self.init_size, self.init_size)
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
#  Discriminator  (Swin backbone + head)
# ──────────────────────────────────────────────────────────────────────────────

class DeepfakeDiscriminator(nn.Module):
    """
    Discriminator for the GAN.
    Uses SwinBackbone as a feature extractor, then a small head
    to output P(real) logit.

    The `get_features()` method exposes the backbone output for the
    downstream fusion model. Expects 2D image inputs (B, 3, 224, 224).
    """

    def __init__(
        self,
        embed_dim: int = 64,
        win: int = 7,
        heads: int = 4,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = SwinBackbone(embed_dim=embed_dim, win=win, heads=heads)
        feat_dim = self.backbone.out_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            # Raw logit — use BCEWithLogitsLoss
        )
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.constant_(m.bias, 0.0)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return Swin feature vector: (B, feat_dim)."""
        x = (x + 1.0) / 2.0        # [-1,1] → [0,1]
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:  x: (B, 3, 224, 224) in [-1, 1]
        Returns: logits (B, 1)
        """
        return self.head(self.get_features(x))


# ──────────────────────────────────────────────────────────────────────────────
#  GAN trainer
# ──────────────────────────────────────────────────────────────────────────────

class DeepfakeGAN:
    """
    Wraps Generator + Discriminator with a simple train() interface.

    Pre-training the GAN on real face images forces the Discriminator
    backbone to learn synthesis artefact patterns, giving a warm-started
    feature extractor for the downstream fusion detector.
    
    UPDATED: Now handles video sequence batches (B, T, 3, 224, 224) directly
    by flattening them into independent frames for 2D GAN training.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        device: str = "cpu",
        lr_g: float = 1e-4,
        lr_d: float = 1e-4,
        beta1: float = 0.5,
        beta2: float = 0.999,
    ):
        self.device     = torch.device(device)
        self.latent_dim = latent_dim

        self.G = DeepfakeGenerator(latent_dim=latent_dim).to(self.device)
        self.D = DeepfakeDiscriminator().to(self.device)

        self.opt_G    = torch.optim.Adam(self.G.parameters(), lr=lr_g, betas=(beta1, beta2))
        self.opt_D    = torch.optim.Adam(self.D.parameters(), lr=lr_d, betas=(beta1, beta2))
        self.criterion = nn.BCEWithLogitsLoss()

    # ── label helpers ────────────────────────────────────────────────────────

    def _real_labels(self, n: int, smooth: float = 0.9) -> torch.Tensor:
        return torch.full((n, 1), smooth, device=self.device)

    def _fake_labels(self, n: int) -> torch.Tensor:
        return torch.zeros(n, 1, device=self.device)

    # ── single mini-batch step ───────────────────────────────────────────────

    def train_step(self, real_batch: torch.Tensor) -> dict:
        """
        Performs a single GAN training step.
        Accepts either (B, 3, H, W) images or (B, T, 3, H, W) video sequences.
        If video sequences are passed, they are flattened to (B*T, 3, H, W) 
        and treated as independent frames for 2D GAN training.
        """
        # Flatten video sequences into individual frames
        if real_batch.ndim == 5:
            B, T, C, H, W = real_batch.shape
            real_imgs = real_batch.view(B * T, C, H, W)
        else:
            real_imgs = real_batch
            
        real_imgs = real_imgs.to(self.device)
        N = real_imgs.size(0)  # Actual batch size (B*T or B)

        # ── Discriminator ──
        self.opt_D.zero_grad()
        d_real = self.criterion(self.D(real_imgs), self._real_labels(N))
        z      = torch.randn(N, self.latent_dim, device=self.device)
        d_fake = self.criterion(self.D(self.G(z).detach()), self._fake_labels(N))
        d_loss = (d_real + d_fake) * 0.5
        d_loss.backward()
        self.opt_D.step()

        # ── Generator ──
        self.opt_G.zero_grad()
        z      = torch.randn(N, self.latent_dim, device=self.device)
        g_loss = self.criterion(self.D(self.G(z)), self._real_labels(N, smooth=1.0))
        g_loss.backward()
        self.opt_G.step()

        return {
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
        }

    # ── epoch / full training ─────────────────────────────────────────────────

    def train_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.G.train(); self.D.train()
        totals, n = {"d_loss": 0.0, "g_loss": 0.0}, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for batch, _ in pbar:
            stats = self.train_step(batch)
            for k in totals: totals[k] += stats[k]
            n += 1
            pbar.set_postfix(D=f"{stats['d_loss']:.3f}", G=f"{stats['g_loss']:.3f}")
        return {k: v / n for k, v in totals.items()}

    def train(self, loader: DataLoader, num_epochs: int) -> None:
        print(f"GAN pre-training for {num_epochs} epochs on {self.device}")
        for ep in range(1, num_epochs + 1):
            stats = self.train_epoch(loader, ep)
            print(f"Epoch {ep:3d} | D: {stats['d_loss']:.4f} | G: {stats['g_loss']:.4f}")

    # ── inference helpers ─────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, n: int = 4) -> torch.Tensor:
        """Generate n images: (n, 3, 224, 224) in [-1, 1]."""
        self.G.eval()
        return self.G(torch.randn(n, self.latent_dim, device=self.device))

    @torch.no_grad()
    def score(self, batch: torch.Tensor) -> torch.Tensor:
        """P(real) for each image. Accepts (B, 3, H, W) or (B, T, 3, H, W)."""
        self.D.eval()
        if batch.ndim == 5:
            B, T, C, H, W = batch.shape
            batch = batch.view(B * T, C, H, W)
            scores = self.D(batch.to(self.device)).sigmoid().squeeze(-1)
            return scores.view(B, T)
        return self.D(batch.to(self.device)).sigmoid().squeeze(-1)

    def save(self, path_prefix: str = "checkpoints/gan") -> None:
        import os; os.makedirs(path_prefix, exist_ok=True)
        torch.save(self.G.state_dict(), f"{path_prefix}/generator.pt")
        torch.save(self.D.state_dict(), f"{path_prefix}/discriminator.pt")
        print(f"Saved GAN to {path_prefix}/")

    def load(self, path_prefix: str = "checkpoints/gan") -> None:
        self.G.load_state_dict(torch.load(f"{path_prefix}/generator.pt", map_location=self.device))
        self.D.load_state_dict(torch.load(f"{path_prefix}/discriminator.pt", map_location=self.device))
        print(f"Loaded GAN from {path_prefix}/")


# ──────────────────────────────────────────────────────────────────────────────
#  Smoke test
# ──────────────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    gan = DeepfakeGAN(latent_dim=256, device=device)

    g_params = sum(p.numel() for p in gan.G.parameters())
    d_params = sum(p.numel() for p in gan.D.parameters())
    print(f"Generator params   : {g_params:,}")
    print(f"Discriminator params: {d_params:,}")

    # Shape check (Image)
    z      = torch.randn(2, 256, device=device)
    fake   = gan.G(z)
    print(f"Generator output   : {fake.shape}")   # (2, 3, 224, 224)

    logits = gan.D(fake)
    print(f"Discriminator logit: {logits.shape}")  # (2, 1)

    feats  = gan.D.get_features(fake)
    print(f"Backbone features  : {feats.shape}")   # (2, 256)

    # Shape check (Video sequence from DataLoader)
    video_batch = torch.randn(2, 4, 3, 224, 224, device=device) # B=2, T=4
    stats = gan.train_step(video_batch)
    print(f"Train step (video) : D_loss={stats['d_loss']:.3f}, G_loss={stats['g_loss']:.3f}")


if __name__ == "__main__":
    main()