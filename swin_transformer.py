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


# ──────────────────────────────────────────────────────────────────────────────
#  Smoke test
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Test 2D Image Pass
    x_img = torch.randn(2, 3, 224, 224)
    model_img = SwinTransformer()
    feats = model_img(x_img)
    print(f"SwinTransformer 2D output shape: {feats.shape}")   # (2, 768)

    # Test Standalone Detector on Video Sequence
    x_vid = torch.randn(2, 4, 3, 224, 224)  # B=2, T=4
    detector = SwinTransformerDetector(temporal_type="lstm")
    logits = detector(x_vid)
    print(f"SwinTransformerDetector 5D logits shape: {logits.shape}")  # (2, 1)


if __name__ == '__main__':
    main()