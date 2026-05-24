"""
swin_transformer.py
===================
Swin Transformer feature extractor for DeepFake detection.
Input : (B, 3, H, W)  — RGB face images, H=W=224 recommended
Output: (B, feature_dim)  — pooled feature vector

UPDATED: 
  - Fixed ShiftedWindowMSA forward pass (removed duplicate QKV projections)
  - Added temporal sequence processing (B, T, 3, H, W) to standalone detector
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ──────────────────────────────────────────────────────────────────────────────
#  Patch Embedding
# ──────────────────────────────────────────────────────────────────────────────

class SwinEmbedding(nn.Module):
    """
    Splits image into non-overlapping patches and linearly embeds them.
    input  : (B, 3, H, W)
    output : (B, H/4 * W/4, C)
    """

    def __init__(self, patch_size: int = 4, C: int = 96):
        super().__init__()
        self.linear_embedding = nn.Conv2d(3, C, kernel_size=patch_size, stride=patch_size)
        self.layer_norm = nn.LayerNorm(C)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_embedding(x)                      # (B, C, H/4, W/4)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return self.relu(self.layer_norm(x))


# ──────────────────────────────────────────────────────────────────────────────
#  Patch Merging  (Swin downsampling)
# ──────────────────────────────────────────────────────────────────────────────

class PatchMerging(nn.Module):
    """
    Concatenates 2×2 neighbouring patches and halves the sequence length.
    input  : (B, H*W, C)
    output : (B, H/2 * W/2, 2C)
    """

    def __init__(self, C: int):
        super().__init__()
        self.linear = nn.Linear(4 * C, 2 * C)
        self.layer_norm = nn.LayerNorm(2 * C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height = width = int(math.sqrt(x.shape[1]) / 2)
        x = rearrange(x, 'b (h s1 w s2) c -> b (h w) (s2 s1 c)',
                      s1=2, s2=2, h=height, w=width)
        return self.layer_norm(self.linear(x))


# ──────────────────────────────────────────────────────────────────────────────
#  Window / Shifted-Window Multi-head Self-Attention
# ──────────────────────────────────────────────────────────────────────────────

class ShiftedWindowMSA(nn.Module):
    """
    Window multi-head self-attention with optional cyclic shift + masking.

    input  : (B, H*W, C)
    output : (B, H*W, C)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = 7,
        mask: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mask = mask
        self.shift = window_size // 2 if mask else 0

        self.proj1 = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj2 = nn.Linear(embed_dim, embed_dim)

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))   # (2, Wh, Ww)
        coords_flat = coords.flatten(1)                                             # (2, Wh*Ww)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]       # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0)                        # (N, N, 2)
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)                          # (N, N)
        self.register_buffer('relative_position_index', relative_position_index)

    def _build_attn_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Build attention mask for shifted windows."""
        img_mask = torch.zeros((1, height, width, 1), device=device)
        slices_h = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift),
                    slice(-self.shift, None))
        slices_w = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift),
                    slice(-self.shift, None))
        cnt = 0
        for sh in slices_h:
            for sw in slices_w:
                img_mask[:, sh, sw, :] = cnt
                cnt += 1

        # window_partition
        img_mask = rearrange(
            img_mask,
            'b (h m1) (w m2) c -> (b h w) (m1 m2) c',
            m1=self.window_size, m2=self.window_size,
        ).squeeze(-1)  # (num_windows, ws*ws)

        attn_mask = img_mask.unsqueeze(1) - img_mask.unsqueeze(2)  # (nw, ws², ws²)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        return attn_mask  # (num_windows, ws², ws²)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        height = width = int(math.sqrt(L))

        # (1) Cyclic shift
        if self.shift > 0:
            x = rearrange(x, 'b (h w) c -> b h w c', h=height, w=width)
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
            x = rearrange(x, 'b h w c -> b (h w) c')

        # (2) Partition into windows
        x_2d = rearrange(x, 'b (h w) c -> b h w c', h=height, w=width)
        x_win = rearrange(
            x_2d,
            'b (nh m1) (nw m2) c -> (b nh nw) (m1 m2) c',
            m1=self.window_size, m2=self.window_size,
        )  # (B*num_windows, ws², C)

        # (3) QKV projection
        qkv_win = self.proj1(x_win)  # (B*nw, ws², 3C)
        Q, K, V = qkv_win.chunk(3, dim=-1)

        # Reshape for multi-head attention
        def split_heads(t):
            BN, N, c = t.shape
            return t.view(BN, N, self.num_heads, c // self.num_heads).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # (4) Scaled Dot-Product Attention
        att = (Q @ K.transpose(-2, -1)) / math.sqrt(self.embed_dim // self.num_heads)

        # Relative position bias
        N = self.window_size ** 2
        rpb = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, N, N)
        att = att + rpb

        # Mask for shifted windows
        if self.mask and self.shift > 0:
            attn_mask = self._build_attn_mask(height, width, x.device)  # (nw, N, N)
            nw = attn_mask.shape[0]
            att = att.view(B, nw, self.num_heads, N, N)
            att = att + attn_mask.unsqueeze(1).unsqueeze(0)
            att = att.view(-1, self.num_heads, N, N)

        att = F.softmax(att, dim=-1)
        out = att @ V  # (B*nw, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(out.shape[0], N, C)

        # (5) Reverse window partition
        num_win_h = height // self.window_size
        num_win_w = width // self.window_size
        out = rearrange(
            out,
            '(b nh nw) (m1 m2) c -> b (nh m1) (nw m2) c',
            b=B, nh=num_win_h, nw=num_win_w,
            m1=self.window_size, m2=self.window_size,
        )

        # (6) Reverse cyclic shift
        if self.shift > 0:
            out = torch.roll(out, shifts=(self.shift, self.shift), dims=(1, 2))

        out = rearrange(out, 'b h w c -> b (h w) c')
        return self.proj2(out)


# ──────────────────────────────────────────────────────────────────────────────
#  Encoder Blocks
# ──────────────────────────────────────────────────────────────────────────────

class SwinEncoderBlock(nn.Module):
    """Single Swin Transformer encoder block (W-MSA or SW-MSA + MLP)."""

    def __init__(self, embed_dim: int, num_heads: int, window_size: int, mask: bool):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.WMSA = ShiftedWindowMSA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            mask=mask,
        )
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res1 = x + self.dropout(self.WMSA(self.layer_norm1(x)))
        return res1 + self.dropout(self.MLP(self.layer_norm2(res1)))


class AlternatingEncoderBlock(nn.Module):
    """Pair of (W-MSA, SW-MSA) blocks as in the original Swin paper."""

    def __init__(self, embed_dim: int, num_heads: int, window_size: int = 7):
        super().__init__()
        self.WSA  = SwinEncoderBlock(embed_dim, num_heads, window_size, mask=False)
        self.SWSA = SwinEncoderBlock(embed_dim, num_heads, window_size, mask=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.SWSA(self.WSA(x))


# ──────────────────────────────────────────────────────────────────────────────
#  Full Swin Transformer  (Swin-T topology)
# ──────────────────────────────────────────────────────────────────────────────

class SwinTransformer(nn.Module):
    """
    Swin-T feature extractor.
    Input : (B, 3, 224, 224)
    Output: (B, 768)  — global-average-pooled features from Stage 4
    """

    def __init__(self, embed_dim: int = 96, window_size: int = 7):
        super().__init__()
        C = embed_dim

        self.Embedding   = SwinEmbedding(patch_size=4, C=C)
        self.Stage1      = AlternatingEncoderBlock(C,     num_heads=3,  window_size=window_size)
        self.PatchMerge1 = PatchMerging(C)
        self.Stage2      = AlternatingEncoderBlock(C*2,   num_heads=6,  window_size=window_size)
        self.PatchMerge2 = PatchMerging(C*2)
        self.Stage3_1    = AlternatingEncoderBlock(C*4,   num_heads=12, window_size=window_size)
        self.Stage3_2    = AlternatingEncoderBlock(C*4,   num_heads=12, window_size=window_size)
        self.Stage3_3    = AlternatingEncoderBlock(C*4,   num_heads=12, window_size=window_size)
        self.PatchMerge3 = PatchMerging(C*4)
        self.Stage4      = AlternatingEncoderBlock(C*8,   num_heads=24, window_size=window_size)
        self.norm        = nn.LayerNorm(C * 8)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled feature vector: (B, 768)."""
        x = self.Embedding(x)          # (B, 56*56, 96)
        x = self.Stage1(x)
        x = self.PatchMerge1(x)        # (B, 28*28, 192)
        x = self.Stage2(x)
        x = self.PatchMerge2(x)        # (B, 14*14, 384)
        x = self.Stage3_1(x)
        x = self.Stage3_2(x)
        x = self.Stage3_3(x)
        x = self.PatchMerge3(x)        # (B, 7*7,   768)
        x = self.Stage4(x)
        x = self.norm(x)
        return x.mean(dim=1)           # (B, 768)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


# ──────────────────────────────────────────────────────────────────────────────
#  Thin classifier wrapper (for stand-alone training / fine-tuning)
# ──────────────────────────────────────────────────────────────────────────────

class SwinTransformerDetector(nn.Module):
    """
    Swin-T + Temporal Aggregation + classification head for binary deepfake detection.
    Input : (B, T, 3, 224, 224) video sequence OR (B, 3, 224, 224) single image
    Output: logits (B, 1)  — use BCEWithLogitsLoss
    """

    def __init__(self, embed_dim: int = 96, window_size: int = 7, temporal_type: str = "lstm"):
        super().__init__()
        self.backbone = SwinTransformer(embed_dim=embed_dim, window_size=window_size)
        feat_dim = embed_dim * 8          # 768
        self.temporal_type = temporal_type.lower()

        if self.temporal_type == "lstm":
            self.temporal_lstm = nn.LSTM(
                input_size=feat_dim,
                hidden_size=feat_dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Handle 5D video sequence input (B, T, 3, H, W)
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            x_2d = x.view(B * T, C, H, W)
            feat_2d = self.backbone.forward_features(x_2d)  # (B*T, 768)
            feat_seq = feat_2d.view(B, T, -1)               # (B, T, 768)

            if self.temporal_type == "lstm":
                _, (hn, _) = self.temporal_lstm(feat_seq)
                # Concatenate final forward and backward hidden states
                hidden = torch.cat((hn[0], hn[1]), dim=-1)  # (B, 768)
            else:  # avg_pool
                hidden = feat_seq.mean(dim=1)                # (B, 768)
            return hidden
        else:
            # Fallback to 2D single image input (B, 3, H, W)
            return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

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
#  Fusion Model
# ──────────────────────────────────────────────────────────────────────────────

class DeepFakeFusionDetector(nn.Module):
    """
    Combines Swin Transformer features and GAN Discriminator features via
    concatenation, aggregates temporal information, then classifies Real / Fake.

    Parameters
    ----------
    swin_embed_dim : int
        Base embedding dimension for the Swin backbone (default 96).
    gan_embed_dim  : int
        Base embedding dimension for the GAN backbone (default 64).
    freeze_swin    : bool
        If True, Swin weights are frozen.
    freeze_gan     : bool
        If True, GAN discriminator weights are frozen.
    dropout        : float
        Dropout rate in the fusion MLP.
    temporal_type  : str
        Method to aggregate frame features into video features. 
        Options: 'lstm' (recommended) or 'avg_pool'.
    """

    def __init__(
        self,
        swin_embed_dim: int = 96,
        gan_embed_dim:  int = 64,
        freeze_swin:    bool = False,
        freeze_gan:     bool = False,
        dropout:        float = 0.4,
        temporal_type:  str = "lstm",
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

        # ── Temporal Aggregation ──────────────────────────────────────────────
        fused_dim = swin_feat_dim + gan_feat_dim       # 768 + 256 = 1024
        self.temporal_type = temporal_type.lower()

        if self.temporal_type == "lstm":
            # Bidirectional LSTM: input 1024 -> output 512*2 = 1024
            self.temporal_lstm = nn.LSTM(
                input_size=fused_dim,
                hidden_size=fused_dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
        elif self.temporal_type != "avg_pool":
            raise ValueError("temporal_type must be 'lstm' or 'avg_pool'")

        # ── Fusion MLP ────────────────────────────────────────────────────────
        # Input remains 1024 because LSTM output is 1024 (512*2) or avg_pool keeps 1024
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
        """Extract Swin Transformer features: (B*T, 768)."""
        return self.swin.forward_features(x)

    def _get_gan_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract GAN Discriminator backbone features: (B*T, 256).
        Expects x in [0, 1]; converts to [-1, 1] internally."""
        x_norm = x * 2.0 - 1.0          # [0,1] → [-1,1]
        return self.gan_disc.backbone(x_norm)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, 3, 224, 224) float tensor in [0, 1]
            B = batch size, T = sequence length (frames per video)

        Returns
        -------
        logits : (B, 1)  — apply sigmoid to get P(fake)
        """
        B, T, C, H, W = x.shape
        
        # Combine batch and time dimensions to pass through 2D backbones
        x_2d = x.view(B * T, C, H, W)
        
        # Extract spatial features per frame
        feat_swin = self._get_swin_features(x_2d)   # (B*T, 768)
        feat_gan  = self._get_gan_features(x_2d)    # (B*T, 256)
        
        fused_2d = torch.cat([feat_swin, feat_gan], dim=-1) # (B*T, 1024)
        
        # Reshape back to sequence format
        fused_seq = fused_2d.view(B, T, -1) # (B, T, 1024)
        
        # Aggregate temporal information
        if self.temporal_type == "lstm":
            _, (hn, _) = self.temporal_lstm(fused_seq)
            # hn shape: (num_layers * num_directions, B, hidden_size)
            # For bidirectional, concatenate final forward and backward hidden states
            hidden = torch.cat((hn[0], hn[1]), dim=-1) # (B, 1024)
        else: # avg_pool
            hidden = fused_seq.mean(dim=1) # (B, 1024)
            
        return self.fusion_head(hidden)

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