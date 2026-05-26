"""
train.py
========
The single file the agent edits. Contains the FULL model architecture 
(Swin + GAN + LSTM Fusion), optimizer, and the time-budgeted training loop.

Autoresearch Rules:
- Agent may ONLY modify this file.
- Training runs for exactly 10 minutes (wall clock).
- Results are saved to results.json for the agent to read.
"""

import time
import json
import math
from cv2 import transform
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# The ONLY external import allowed. prepare.py handles data caching and loading.
from prepare import build_ff_dataloaders
from prepare1 import build_dataloaders


# ──────────────────────────────────────────────────────────────────────────────
#  1. FULL SWIN TRANSFORMER (Main Backbone)
# ──────────────────────────────────────────────────────────────────────────────

class SwinEmbedding(nn.Module):
    def __init__(self, patch_size=4, C=96):
        super().__init__()
        self.linear_embedding = nn.Conv2d(3, C, kernel_size=patch_size, stride=patch_size)
        self.layer_norm = nn.LayerNorm(C); self.relu = nn.ReLU()
    def forward(self, x):
        x = self.linear_embedding(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return self.relu(self.layer_norm(x))

class PatchMerging(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.linear = nn.Linear(4 * C, 2 * C); self.layer_norm = nn.LayerNorm(2 * C)
    def forward(self, x):
        height = width = int(math.sqrt(x.shape[1]) / 2)
        x = rearrange(x, 'b (h s1 w s2) c -> b (h w) (s2 s1 c)', s1=2, s2=2, h=height, w=width)
        return self.layer_norm(self.linear(x))

class ShiftedWindowMSA(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=7, mask=False):
        super().__init__()
        self.embed_dim, self.num_heads, self.window_size, self.mask = embed_dim, num_heads, window_size, mask
        self.shift = window_size // 2 if mask else 0
        self.proj1 = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj2 = nn.Linear(embed_dim, embed_dim)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        coords_h = torch.arange(window_size); coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flat = coords.flatten(1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0)
        relative_coords[:, :, 0] += window_size - 1; relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

    def _build_attn_mask(self, height, width, device):
        img_mask = torch.zeros((1, height, width, 1), device=device)
        slices_h = (slice(0, -self.window_size), slice(-self.window_size, -self.shift), slice(-self.shift, None))
        slices_w = (slice(0, -self.window_size), slice(-self.window_size, -self.shift), slice(-self.shift, None))
        cnt = 0
        for sh in slices_h:
            for sw in slices_w:
                img_mask[:, sh, sw, :] = cnt; cnt += 1
        img_mask = rearrange(img_mask, 'b (h m1) (w m2) c -> (b h w) (m1 m2) c', m1=self.window_size, m2=self.window_size).squeeze(-1)
        attn_mask = img_mask.unsqueeze(1) - img_mask.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0)

    def forward(self, x):
        B, L, C = x.shape; height = width = int(math.sqrt(L))
        if self.shift > 0:
            x = rearrange(x, 'b (h w) c -> b h w c', h=height, w=width)
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
            x = rearrange(x, 'b h w c -> b (h w) c')
        x_2d = rearrange(x, 'b (h w) c -> b h w c', h=height, w=width)
        x_win = rearrange(x_2d, 'b (nh m1) (nw m2) c -> (b nh nw) (m1 m2) c', m1=self.window_size, m2=self.window_size)
        qkv_win = self.proj1(x_win); Q, K, V = qkv_win.chunk(3, dim=-1)
        def split_heads(t): return t.view(t.shape[0], t.shape[1], self.num_heads, t.shape[2] // self.num_heads).transpose(1, 2)
        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
        att = (Q @ K.transpose(-2, -1)) / math.sqrt(self.embed_dim // self.num_heads)
        N = self.window_size ** 2
        rpb = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1).permute(2, 0, 1).unsqueeze(0)
        att = att + rpb
        if self.mask and self.shift > 0:
            attn_mask = self._build_attn_mask(height, width, x.device)
            nw = attn_mask.shape[0]; att = att.view(B, nw, self.num_heads, N, N)
            att = att + attn_mask.unsqueeze(1).unsqueeze(0); att = att.view(-1, self.num_heads, N, N)
        att = F.softmax(att, dim=-1); out = (att @ V).transpose(1, 2).reshape(att.shape[0], N, C)
        num_win_h = height // self.window_size; num_win_w = width // self.window_size
        out = rearrange(out, '(b nh nw) (m1 m2) c -> b (nh m1) (nw m2) c', b=B, nh=num_win_h, nw=num_win_w, m1=self.window_size, m2=self.window_size)
        if self.shift > 0: out = torch.roll(out, shifts=(self.shift, self.shift), dims=(1, 2))
        out = rearrange(out, 'b h w c -> b (h w) c')
        return self.proj2(out)

class SwinEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, mask):
        super().__init__()
        self.layer_norm1, self.layer_norm2, self.dropout = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim), nn.Dropout(0.1)
        self.WMSA = ShiftedWindowMSA(embed_dim, num_heads, window_size, mask)
        self.MLP = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(0.1), nn.Linear(embed_dim * 4, embed_dim))
    def forward(self, x):
        res1 = x + self.dropout(self.WMSA(self.layer_norm1(x)))
        return res1 + self.dropout(self.MLP(self.layer_norm2(res1)))

class AlternatingEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=7):
        super().__init__()
        self.WSA = SwinEncoderBlock(embed_dim, num_heads, window_size, mask=False)
        self.SWSA = SwinEncoderBlock(embed_dim, num_heads, window_size, mask=True)
    def forward(self, x): return self.SWSA(self.WSA(x))

class SwinTransformer(nn.Module):
    def __init__(self, embed_dim=96, window_size=7):
        super().__init__(); C = embed_dim
        self.Embedding = SwinEmbedding(C=C)
        self.Stage1 = AlternatingEncoderBlock(C, num_heads=3, window_size=window_size)
        self.PatchMerge1 = PatchMerging(C)
        self.Stage2 = AlternatingEncoderBlock(C*2, num_heads=6, window_size=window_size)
        self.PatchMerge2 = PatchMerging(C*2)
        self.Stage3_1 = AlternatingEncoderBlock(C*4, num_heads=12, window_size=window_size)
        self.Stage3_2 = AlternatingEncoderBlock(C*4, num_heads=12, window_size=window_size)
        self.Stage3_3 = AlternatingEncoderBlock(C*4, num_heads=12, window_size=window_size)
        self.PatchMerge3 = PatchMerging(C*4)
        self.Stage4 = AlternatingEncoderBlock(C*8, num_heads=24, window_size=window_size)
        self.norm = nn.LayerNorm(C * 8)
    def forward_features(self, x):
        x = self.Embedding(x); x = self.Stage1(x); x = self.PatchMerge1(x)
        x = self.Stage2(x); x = self.PatchMerge2(x)
        x = self.Stage3_1(x); x = self.Stage3_2(x); x = self.Stage3_3(x)
        x = self.PatchMerge3(x); x = self.Stage4(x); x = self.norm(x)
        return x.mean(dim=1)
    def forward(self, x): return self.forward_features(x)


# ──────────────────────────────────────────────────────────────────────────────
#  2. FULL GAN DISCRIMINATOR (Renamed PatchMerging to avoid class clash)
# ──────────────────────────────────────────────────────────────────────────────

def window_partition(x, win):
    B, H, W, C = x.shape; x = x.view(B, H // win, win, W // win, win, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, win, win, C)

def window_reverse(windows, win, H, W):
    B = int(windows.shape[0] / (H * W // win // win))
    x = windows.view(B, H // win, W // win, win, win, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H, W, -1)

class GANWindowAttention(nn.Module):
    def __init__(self, dim, num_heads, win):
        super().__init__(); self.num_heads, self.head_dim, self.scale, self.win = num_heads, dim // num_heads, (dim // num_heads) ** -0.5, win
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        coords = torch.stack(torch.meshgrid(torch.arange(win), torch.arange(win), indexing='ij')); coords_flat = coords.flatten(1)
        rel = (coords_flat[:, :, None] - coords_flat[:, None, :]).permute(1, 2, 0)
        rel[:, :, 0] += win - 1; rel[:, :, 1] += win - 1; rel[:, :, 0] *= 2 * win - 1
        self.register_buffer('pos_index', rel.sum(-1)); self.rel_bias = nn.Parameter(torch.zeros((2 * win - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)
    def forward(self, x, mask=None):
        B_, N, C = x.shape; qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv.unbind(0); attn = (Q * self.scale) @ K.transpose(-2, -1)
        rpb = self.rel_bias[self.pos_index.reshape(-1)].reshape(N, N, self.num_heads)
        attn = attn + rpb.permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]; attn = attn.view(B_ // nw, nw, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0); attn = attn.view(-1, self.num_heads, N, N)
        return self.proj((attn.softmax(dim=-1) @ V).transpose(1, 2).reshape(B_, N, C))

class GANSwinBlock(nn.Module):
    def __init__(self, dim, res, win, shift, heads):
        super().__init__(); self.res, self.win, self.shift = res, win, shift
        self.norm1, self.attn, self.norm2 = nn.LayerNorm(dim), GANWindowAttention(dim, heads, win), nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(4 * dim, dim))
        H, W = res; self.mask = self._make_mask(H, W, win, shift) if shift > 0 else None
    def _make_mask(self, H, W, win, shift):
        img_mask = torch.zeros((1, H, W, 1)); cnt = 0
        for h in (slice(0, -win), slice(-win, -shift), slice(-shift, None)):
            for w in (slice(0, -win), slice(-win, -shift), slice(-shift, None)): img_mask[:, h, w, :] = cnt; cnt += 1
        mask = window_partition(img_mask, win).view(-1, win * win); mask = mask.unsqueeze(1) - mask.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0)
    def forward(self, x):
        B, L, C = x.shape; H, W = self.res; res = x; x = self.norm1(x).view(B, H, W, C)
        if self.shift > 0: x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        win_x = window_partition(x, self.win).view(-1, self.win ** 2, C)
        out = self.attn(win_x, self.mask.to(x.device) if self.mask is not None else None)
        x = window_reverse(out.view(-1, self.win, self.win, C), self.win, H, W)
        if self.shift > 0: x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))
        x = res + x.view(B, L, C); return x + self.mlp(self.norm2(x))

class GANPatchMerging(nn.Module): # Renamed to avoid clash!
    def __init__(self, dim):
        super().__init__(); self.norm, self.reduction = nn.LayerNorm(4 * dim), nn.Linear(4 * dim, 2 * dim, bias=False)
    def forward(self, x, H, W):
        B, L, C = x.shape; x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1).reshape(B, -1, 4 * C)
        return self.reduction(self.norm(x)), H // 2, W // 2

class SwinBackbone(nn.Module):
    def __init__(self, embed_dim=64, win=7, heads=4):
        super().__init__(); C = embed_dim
        self.patch_embed = nn.Conv2d(3, C, kernel_size=4, stride=4); res1 = (56, 56)
        self.stage1 = nn.Sequential(GANSwinBlock(C, res1, win=win, shift=0, heads=heads), GANSwinBlock(C, res1, win=win, shift=win // 2, heads=heads))
        self.merge1 = GANPatchMerging(C); res2 = (28, 28)
        self.stage2 = nn.Sequential(GANSwinBlock(C*2, res2, win=win, shift=0, heads=heads*2), GANSwinBlock(C*2, res2, win=win, shift=win // 2, heads=heads*2))
        self.merge2 = GANPatchMerging(C*2); res3 = (14, 14)
        self.stage3 = nn.Sequential(GANSwinBlock(C*4, res3, win=win, shift=0, heads=heads*4), GANSwinBlock(C*4, res3, win=win, shift=win // 2, heads=heads*4))
        self.norm = nn.LayerNorm(C * 4); self.out_dim = C * 4
    def forward(self, x):
        x = self.patch_embed(x); B, C, H, W = x.shape; x = x.flatten(2).transpose(1, 2)
        x = self.stage1(x); x, H, W = self.merge1(x, H, W)
        x = self.stage2(x); x, H, W = self.merge2(x, H, W)
        x = self.stage3(x); x = self.norm(x); return x.mean(dim=1)

class DeepfakeDiscriminator(nn.Module):
    def __init__(self, embed_dim=64, win=7, heads=4):
        super().__init__()
        self.backbone = SwinBackbone(embed_dim=embed_dim, win=win, heads=heads)
        feat_dim = self.backbone.out_dim
        self.head = nn.Sequential(nn.Linear(feat_dim, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3), nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.Linear(128, 1))
    def forward(self, x): return self.head(self.backbone((x + 1.0) / 2.0))


# ──────────────────────────────────────────────────────────────────────────────
#  3. FUSION MODEL + LSTM
# ──────────────────────────────────────────────────────────────────────────────

class DeepFakeFusionDetector(nn.Module):
    def __init__(self, swin_embed_dim=96, gan_embed_dim=64, temporal_type="lstm", dropout=0.4):
        super().__init__()
        self.swin = SwinTransformer(embed_dim=swin_embed_dim)
        self.gan_disc = DeepfakeDiscriminator(embed_dim=gan_embed_dim)
        
        swin_feat_dim = swin_embed_dim * 8
        gan_feat_dim = self.gan_disc.backbone.out_dim
        fused_dim = swin_feat_dim + gan_feat_dim
        
        self.temporal_type = temporal_type
        if self.temporal_type == "lstm":
            self.temporal_lstm = nn.LSTM(fused_dim, fused_dim // 2, batch_first=True, bidirectional=True)
        
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_2d = x.view(B * T, C, H, W)
        
        feat_swin = self.swin(x_2d)
        feat_gan = self.gan_disc.backbone(x_2d * 2.0 - 1.0) # [0,1] -> [-1,1] for GAN
        
        fused_2d = torch.cat([feat_swin, feat_gan], dim=-1)
        fused_seq = fused_2d.view(B, T, -1)
        
        if self.temporal_type == "lstm":
            _, (hn, _) = self.temporal_lstm(fused_seq)
            hidden = torch.cat((hn[0], hn[1]), dim=-1)
        else:
            hidden = fused_seq.mean(dim=1)
            
        return self.fusion_head(hidden)


# ──────────────────────────────────────────────────────────────────────────────
#  4. TIME-BUDGETED TRAINING LOOP (Autoresearch Core)
# ──────────────────────────────────────────────────────────────────────────────

TIME_BUDGET_SECONDS = 10 * 60 # 10 minutes per experiment!

def run_eval(model, loader, criterion, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.float().unsqueeze(1).to(device)
            logits = model(clips)
            all_probs.extend(logits.sigmoid().cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    from sklearn.metrics import roc_auc_score
    val_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return val_auc

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    # Load Data (Uses cache from prepare.py)
    train_loader, val_loader, test_loader = build_dataloaders(
    dataset_root="./dataset_split",
    transform=transform,
    batch_size=4,
    seq_len=8,
)
    print(f"Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # ╔══════════════════════════════════════════════════════════════╗
    # ║  AGENT MODIFIES THESE PARAMETERS TO RUN EXPERIMENTS!        ║
    # ╚══════════════════════════════════════════════════════════════╝
    model = DeepFakeFusionDetector(
        swin_embed_dim=96, 
        gan_embed_dim=64, 
        temporal_type="lstm",
        dropout=0.4
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    start_time = time.time()
    step = 0

    print(f">>> Starting training. Time budget: {TIME_BUDGET_SECONDS}s")
    
    while True:
        for clips, labels in train_loader:
            if time.time() - start_time > TIME_BUDGET_SECONDS:
                print("\n[ALARM] Time budget reached! Running validation...")
                val_auc = run_eval(model, val_loader, criterion, device)
                
                # Save results for the LangGraph Agent to read!
                results = {"val_auc": val_auc, "steps_completed": step}
                with open("results.json", "w") as f:
                    json.dump(results, f, indent=2)
                
                print(f"Results saved to results.json: {results}")
                return # End experiment

            clips, labels = clips.to(device), labels.float().unsqueeze(1).to(device)
            logits = model(clips)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            if step % 5 == 0:
                elapsed = time.time() - start_time
                remaining = int(TIME_BUDGET_SECONDS - elapsed)
                print(f"Step {step} | Loss: {loss.item():.4f} | Time left: {remaining}s")

if __name__ == "__main__":
    train()