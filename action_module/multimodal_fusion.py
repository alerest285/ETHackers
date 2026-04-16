"""
Multimodal Fusion — five-modality scene encoder.

Encoders
--------
  Scene graph     → GATEncoder          (3-layer Graph Attention Network)
  Segmented image → ViTSegEncoder       (ViT-S/16, MAE-pretrained via timm)
  BBox image      → BBoxRoIEncoder      (shared ResNet-50 trunk + RoI Align)
  Depth map       → DilatedResNetEncoder (ResNet-34, dilated layer3/4)
  Point cloud     → ColorPointNetPlusPlus (PointNet++ with color/class features)

Fusion
------
  Five (embed_dim,) tokens → learnable modality embeddings → 2-layer
  TransformerEncoder (multi-head self-attention) → mean pool → (embed_dim,).

Dependencies
------------
  Core   : torch, torchvision
  Optional: timm  (for MAE-pretrained ViT; falls back to a built-in ViT-lite)

Public API
----------
  model = MultimodalFusion(embed_dim=256)
  emb   = model(graph_data, seg_image, bbox_image, depth_map, point_cloud)
  # emb : FloatTensor (embed_dim,)

  graph_data = detections_to_graph_data(detections, graph)
  seg_t      = pil_to_tensor(pil_image)          # → (3, H, W)
  depth_t    = depth_array_to_tensor(depth_arr)  # → (1, H, W)
  pc_t       = point_cloud_to_tensor(xyz_arr)    # → (N, 3) or (N, 3+C)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    import timm
    _TIMM = True
except ImportError:
    _TIMM = False

try:
    from torchvision.models import resnet50, resnet34, ResNet50_Weights, ResNet34_Weights
    from torchvision.ops import roi_align
    _TV = True
except ImportError:
    _TV = False

# ── feature constants ─────────────────────────────────────────────────────────

_RISK_GROUPS      = ["HUMAN", "VEHICLE", "OBSTACLE", "SAFETY_MARKER", "BACKGROUND"]
_PROXIMITY_LABELS = ["CLOSE", "MEDIUM", "FAR", "NAVIGABLE"]
_PATH_ZONES       = ["CENTER", "PERIPHERAL"]
_REL_POSITIONS    = ["in front of", "behind", "to the left of", "to the right of"]

NODE_FEAT_DIM = len(_RISK_GROUPS) + len(_PROXIMITY_LABELS) + len(_PATH_ZONES) + 4  # 15
EDGE_FEAT_DIM = len(_REL_POSITIONS) + 2                                             # 6


# ══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════════════

def _scatter_softmax(
    values:  torch.Tensor,  # (E, H)
    dst:     torch.Tensor,  # (E,) — destination node index for each edge
    n_nodes: int,
) -> torch.Tensor:          # (E, H)
    """Per-destination softmax over incoming edge scores (one head at a time)."""
    E, H = values.shape
    device = values.device

    # running max per destination node for numerical stability
    max_v = torch.full((n_nodes, H), -1e9, device=device)
    for e in range(E):
        d = dst[e].item()
        max_v[d] = torch.maximum(max_v[d], values[e])

    exp_v = torch.exp(values - max_v[dst])            # (E, H)

    denom = torch.zeros(n_nodes, H, device=device)
    denom.scatter_add_(0, dst.unsqueeze(1).expand_as(exp_v), exp_v)

    return exp_v / (denom[dst] + 1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# 1. GAT Encoder  (scene graph → embed_dim)
# ══════════════════════════════════════════════════════════════════════════════

class _GATLayer(nn.Module):
    """
    One multi-head Graph Attention layer with edge-feature-augmented attention.

    Attention coefficient for edge (i→j):
        e_ij = LeakyReLU( a^T [Wh_i ‖ Wh_j] ) + W_e · edge_feat_ij
        α_ij = softmax_{k∈N(i)} e_ik
        h_i' = σ( Σ_j α_ij · W · h_j )

    H heads are concatenated (intermediate layers) or averaged (last layer).
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int,
                 n_heads: int = 4, concat: bool = True, dropout: float = 0.1):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = out_dim // n_heads
        self.concat   = concat

        self.W   = nn.Linear(in_dim, out_dim, bias=False)
        self.W_e = nn.Linear(edge_dim, n_heads, bias=False)
        self.a   = nn.Parameter(torch.empty(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        self.leaky = nn.LeakyReLU(0.2)
        self.drop  = nn.Dropout(dropout)
        self.norm  = nn.LayerNorm(out_dim if concat else out_dim // n_heads * n_heads)
        self.act   = nn.GELU()

    def forward(
        self,
        h:          torch.Tensor,   # (N, in_dim)
        edge_index: torch.Tensor,   # (2, E)
        edge_feat:  torch.Tensor,   # (E, edge_dim)
    ) -> torch.Tensor:              # (N, out_dim)
        N   = h.shape[0]
        H   = self.n_heads
        D   = self.head_dim
        dev = h.device

        Wh = self.W(h).reshape(N, H, D)          # (N, H, D)

        if edge_index.shape[1] == 0:
            out = Wh.reshape(N, H * D) if self.concat else Wh.mean(1).reshape(N, -1)
            return self.norm(self.act(out))

        src, dst = edge_index[0], edge_index[1]  # (E,)

        # attention logits: (E, H)
        attn = (torch.cat([Wh[src], Wh[dst]], dim=-1) * self.a.unsqueeze(0)).sum(-1)
        attn = self.leaky(attn)                  # (E, H)
        if edge_feat.shape[0] > 0:
            attn = attn + self.W_e(edge_feat)    # (E, H) bias from edge features

        alpha = _scatter_softmax(attn, dst, N)   # (E, H)
        alpha = self.drop(alpha)                 # (E, H)

        # aggregate: h_out_i = Σ_j α_ij * Wh_j  (message: source contributes to dest)
        agg = torch.zeros(N, H, D, device=dev)
        for k in range(H):
            contrib = alpha[:, k:k+1] * Wh[src, k, :]          # (E, D)
            agg[:, k].scatter_add_(0, dst.unsqueeze(1).expand(-1, D), contrib)

        if self.concat:
            out = agg.reshape(N, H * D)
        else:
            out = agg.mean(dim=1)                # average heads (last layer)

        return self.norm(self.act(out))


class _GlobalAttentionPool(nn.Module):
    """Attention-weighted global pooling: z = Σ_i softmax(gate_i) * h_i."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.gate = nn.Linear(in_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (N, D) → (D,)
        weights = F.softmax(self.gate(h), dim=0)         # (N, 1)
        return (weights * h).sum(0)                       # (D,)


class GATEncoder(nn.Module):
    """
    3-layer GAT with 4 attention heads per layer.
    Layers 1–2 concatenate heads; layer 3 averages them.
    Global attention pooling produces the graph-level embedding.
    """

    def __init__(self, node_dim: int, edge_dim: int,
                 hidden_dim: int, embed_dim: int, n_heads: int = 4):
        super().__init__()
        self.layer1 = _GATLayer(node_dim,   hidden_dim,    edge_dim, n_heads, concat=True)
        self.layer2 = _GATLayer(hidden_dim, hidden_dim,    edge_dim, n_heads, concat=True)
        self.layer3 = _GATLayer(hidden_dim, embed_dim,     edge_dim, n_heads, concat=False)
        self.pool   = _GlobalAttentionPool(embed_dim)

    def forward(
        self,
        node_feats: torch.Tensor,   # (N, node_dim)
        edge_index: torch.Tensor,   # (2, E)
        edge_feats: torch.Tensor,   # (E, edge_dim)
    ) -> torch.Tensor:              # (embed_dim,)
        h = self.layer1(node_feats, edge_index, edge_feats)
        h = self.layer2(h,          edge_index, edge_feats)
        h = self.layer3(h,          edge_index, edge_feats)
        return self.pool(h)


# ══════════════════════════════════════════════════════════════════════════════
# 2. ViT Segmented Image Encoder  (seg_image → embed_dim)
# ══════════════════════════════════════════════════════════════════════════════

class _ViTLite(nn.Module):
    """
    Minimal ViT-Small/16 fallback (used when timm is not installed).
    6 transformer layers, 6 heads, embed_dim=384 → projected to out_dim.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 depth: int = 6, n_heads: int = 6, vit_dim: int = 384,
                 out_dim: int = 256):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, vit_dim, patch_size, stride=patch_size)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, vit_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches + 1, vit_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=vit_dim, nhead=n_heads, dim_feedforward=vit_dim * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm        = nn.LayerNorm(vit_dim)
        self.proj        = nn.Linear(vit_dim, out_dim)
        self.n_patches   = n_patches
        self.vit_dim     = vit_dim

    def forward(self, x: torch.Tensor,
                return_patches: bool = False) -> torch.Tensor | tuple:
        # x: (3, H, W)  →  (1, 3, H, W)
        x   = x.unsqueeze(0)
        tok = self.patch_embed(x).flatten(2).transpose(1, 2)  # (1, N, D)
        tok = torch.cat([self.cls_token, tok], dim=1)
        tok = tok + self.pos_embed
        tok = self.transformer(tok)
        tok = self.norm(tok)                       # (1, N+1, D)
        cls = self.proj(tok[:, 0]).squeeze(0)      # (out_dim,)
        if return_patches:
            return cls, tok[:, 1:].squeeze(0)      # (out_dim,), (N, D)
        return cls


class ViTSegEncoder(nn.Module):
    """
    ViT-S/16 initialized from MAE pretraining (timm) when available.
    Falls back to _ViTLite otherwise.

    forward(x, return_patches=False) → (embed_dim,)
                                     or ((embed_dim,), (N_patches, vit_dim))
    """

    def __init__(self, embed_dim: int = 256, img_size: int = 224,
                 pretrained: bool = True):
        super().__init__()

        if _TIMM:
            self.backbone = timm.create_model(
                "vit_small_patch16_224",
                pretrained=pretrained,
                num_classes=0,           # remove classification head
                global_pool="",          # return all tokens
            )
            vit_dim = self.backbone.embed_dim   # 384 for ViT-S
        else:
            self.backbone = _ViTLite(img_size=img_size, out_dim=embed_dim)
            vit_dim = self.backbone.vit_dim

        self.proj = nn.Linear(vit_dim, embed_dim)
        self._uses_timm = _TIMM

    def forward(self, x: torch.Tensor,
                return_patches: bool = False) -> torch.Tensor | tuple:
        """x: (3, H, W)"""
        if not self._uses_timm:
            return self.backbone(x, return_patches=return_patches)

        # timm path
        out = self.backbone(x.unsqueeze(0))      # (1, N+1, D)
        cls     = self.proj(out[:, 0]).squeeze(0)  # (embed_dim,)
        patches = self.proj(out[:, 1:]).squeeze(0) # (N, embed_dim)
        if return_patches:
            return cls, patches
        return cls


# ══════════════════════════════════════════════════════════════════════════════
# 3. BBox RoI Encoder  (bbox_image + boxes → embed_dim)
# ══════════════════════════════════════════════════════════════════════════════

class _MiniTransformer(nn.Module):
    """2-layer self-attention transformer over a set of object tokens."""
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.tf = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (1, N, D) → (N, D)
        return self.tf(x).squeeze(0)


class BBoxRoIEncoder(nn.Module):
    """
    ResNet-50 trunk (stem + layer1 + layer2 = shared CNN features) +
    RoI Align per bounding box (7×7) + 2-layer object transformer → mean pool.

    The shared_trunk parameter allows you to pass in a SharedCNNTrunk instance
    so the early ResNet layers are literally the same weights as used elsewhere.
    If None, a private trunk is created.

    boxes (list of [x1,y1,x2,y2] in pixel coords) are passed to roi_align
    alongside the feature map.
    """

    ROI_SIZE = 7

    def __init__(self, embed_dim: int = 256, shared_trunk: nn.Module | None = None):
        super().__init__()

        if shared_trunk is not None:
            self.trunk = shared_trunk
            trunk_out  = 512   # ResNet-50 layer2 output channels
        elif _TV:
            base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.trunk = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2,
            )
            trunk_out = 512
        else:
            # lightweight fallback trunk
            self.trunk = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                nn.GroupNorm(8, 64), nn.GELU(), nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(64, 256, 3, padding=1, bias=False),
                nn.GroupNorm(16, 256), nn.GELU(),
                nn.Conv2d(256, 512, 3, padding=1, bias=False),
                nn.GroupNorm(32, 512), nn.GELU(),
            )
            trunk_out = 512

        roi_flat   = trunk_out * self.ROI_SIZE * self.ROI_SIZE
        self.roi_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(roi_flat, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.obj_transformer = _MiniTransformer(embed_dim)
        self.uses_tv         = _TV

    def forward(
        self,
        x:     torch.Tensor,        # (3, H, W)
        boxes: list[list[float]],   # [[x1,y1,x2,y2], ...] in pixel space
    ) -> torch.Tensor:              # (embed_dim,)
        feat_map = self.trunk(x.unsqueeze(0))    # (1, C, H', W')
        _, _, H_f, W_f = feat_map.shape
        H, W = x.shape[1], x.shape[2]

        if not boxes or not self.uses_tv:
            # no boxes or no RoI Align: global avg pool
            return self.roi_proj(
                nn.functional.adaptive_avg_pool2d(feat_map, self.ROI_SIZE).flatten(1)
            ).squeeze(0)

        # scale boxes from image pixels to feature-map pixels
        sx, sy = W_f / W, H_f / H
        rois_t = torch.tensor(
            [[0, b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy] for b in boxes],
            dtype=torch.float32, device=x.device,
        )  # (N_boxes, 5)

        # RoI Align: (N_boxes, C, 7, 7)
        roi_feats = roi_align(feat_map, rois_t, self.ROI_SIZE, aligned=True)
        obj_tok   = self.roi_proj(roi_feats)     # (N_boxes, embed_dim)

        # self-attention over objects, then mean pool
        fused = self.obj_transformer(obj_tok.unsqueeze(0))   # (N_boxes, embed_dim)
        return fused.mean(0)                                  # (embed_dim,)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Dilated ResNet Depth Encoder  (depth_map → embed_dim)
# ══════════════════════════════════════════════════════════════════════════════

class DilatedResNetEncoder(nn.Module):
    """
    ResNet-34 with dilated convolutions in layer3 (dilation=2) and layer4
    (dilation=4) to preserve spatial resolution.
    Accepts single-channel input (depth map).
    Global average pool → linear projection → embed_dim.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()

        if _TV:
            base = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

            # Replace first conv: 3-channel → 1-channel
            base.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)

            # Apply dilation to layer3 and layer4
            for block in base.layer3:
                block.conv2.dilation  = (2, 2)
                block.conv2.padding   = (2, 2)
                block.conv2.stride    = (1, 1)
                if block.downsample is not None:
                    block.downsample[0].stride = (1, 1)
            for block in base.layer4:
                block.conv2.dilation  = (4, 4)
                block.conv2.padding   = (4, 4)
                block.conv2.stride    = (1, 1)
                if block.downsample is not None:
                    block.downsample[0].stride = (1, 1)

            self.backbone = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4,
            )
            backbone_out = 512
        else:
            # lightweight fallback
            self.backbone = nn.Sequential(
                nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False),
                nn.GroupNorm(8, 64), nn.GELU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1, bias=False),
                nn.GroupNorm(16, 128), nn.GELU(),
                nn.Conv2d(128, 256, 3, dilation=2, padding=2, bias=False),
                nn.GroupNorm(32, 256), nn.GELU(),
                nn.Conv2d(256, 512, 3, dilation=4, padding=4, bias=False),
                nn.GroupNorm(32, 512), nn.GELU(),
            )
            backbone_out = 512

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (1, H, W) → (embed_dim,)
        feat = self.backbone(x.unsqueeze(0))               # (1, C, H', W')
        return self.proj(self.pool(feat)).squeeze(0)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pre-pool feature map for the depth reconstruction decoder."""
        return self.backbone(x.unsqueeze(0))


# ══════════════════════════════════════════════════════════════════════════════
# 5. Colored PointNet++ Encoder  (point_cloud → embed_dim)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _fps(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    N       = xyz.shape[0]
    npoint  = min(npoint, N)
    sel     = torch.zeros(npoint, dtype=torch.long, device=xyz.device)
    dist    = torch.full((N,), float("inf"), device=xyz.device)
    farthest = int(torch.randint(0, N, (1,)).item())
    for i in range(npoint):
        sel[i]   = farthest
        d        = ((xyz - xyz[farthest]) ** 2).sum(-1)
        dist     = torch.minimum(dist, d)
        farthest = int(dist.argmax().item())
    return sel


def _ball_query(radius: float, nsample: int,
                xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    diff  = new_xyz.unsqueeze(1) - xyz.unsqueeze(0)     # (M, N, 3)
    dist2 = (diff ** 2).sum(-1)
    dist2 = dist2.masked_fill(dist2 > radius ** 2, 1e9)
    nsample = min(nsample, xyz.shape[0])
    return dist2.topk(nsample, dim=-1, largest=False)[1]  # (M, nsample)


def _mlp(dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
    return nn.Sequential(*layers)


class _SA(nn.Module):
    """Set Abstraction layer with optional per-point color/class features."""
    def __init__(self, npoint: int, radius: float, nsample: int,
                 in_dim: int, mlp_dims: list[int]):
        super().__init__()
        self.npoint  = npoint
        self.radius  = radius
        self.nsample = nsample
        self.mlp     = _mlp([3 + in_dim] + mlp_dims)
        self.out_dim = mlp_dims[-1]

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor | None) \
            -> tuple[torch.Tensor, torch.Tensor]:
        npoint    = min(self.npoint, xyz.shape[0])
        idx_cent  = _fps(xyz, npoint)
        new_xyz   = xyz[idx_cent]                          # (M, 3)
        group_idx = _ball_query(self.radius, self.nsample, xyz, new_xyz)
        M, K      = group_idx.shape

        g_xyz = xyz[group_idx] - new_xyz.unsqueeze(1)     # (M, K, 3) relative
        g_in  = torch.cat([g_xyz, feat[group_idx]], -1) if feat is not None else g_xyz

        out = self.mlp(g_in.reshape(M * K, -1)).reshape(M, K, self.out_dim)
        return new_xyz, out.max(1)[0]                      # (M, out_dim)


class ColorPointNetPlusPlus(nn.Module):
    """
    PointNet++ that accepts per-point color or class-label features.

    point_cloud shape: (N, 3)       — xyz only
                    or (N, 3+C)     — xyz + C color/class channels
    """

    def __init__(self, embed_dim: int = 256, color_dim: int = 3):
        super().__init__()
        self.color_dim = color_dim
        self.sa1 = _SA(512,  0.2, 16, color_dim, [64, 64, 128])
        self.sa2 = _SA(128,  0.4, 32, 128,        [128, 128, 256])
        self.sa3 = _SA(32,   0.8, 64, 256,        [256, 512, 1024])
        self.global_mlp = nn.Sequential(
            nn.Linear(1024, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:   # (N, 3+C) → (embed_dim,)
        if pts.shape[1] > 3:
            xyz, color = pts[:, :3], pts[:, 3:]
        else:
            xyz, color = pts, None

        xyz1, f1 = self.sa1(xyz,  color)
        xyz2, f2 = self.sa2(xyz1, f1)
        xyz3, f3 = self.sa3(xyz2, f2)

        return self.global_mlp(f3.mean(0))                  # (embed_dim,)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Shared CNN Trunk  (optional; passed to BBoxRoIEncoder)
# ══════════════════════════════════════════════════════════════════════════════

class SharedCNNTrunk(nn.Module):
    """
    ResNet-50 stem + layer1 + layer2.
    Create one instance and pass it to both BBoxRoIEncoder and any other
    encoder that wants to share early CNN weights.
    """
    def __init__(self):
        super().__init__()
        if _TV:
            base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.net = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2,
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                nn.GroupNorm(8, 64), nn.GELU(), nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(64, 512, 3, padding=1, bias=False),
                nn.GroupNorm(32, 512), nn.GELU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Cross-Attention Fusion
# ══════════════════════════════════════════════════════════════════════════════

class CrossAttentionFusion(nn.Module):
    """
    Stack 5 modality tokens + learnable modality embeddings →
    TransformerEncoder (n_layers layers, n_heads heads) → mean pool.
    """
    N_MODALITIES = 5

    def __init__(self, embed_dim: int, n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.modality_emb = nn.Embedding(self.N_MODALITIES, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, tokens: list[torch.Tensor]) -> torch.Tensor:
        device = tokens[0].device
        x   = torch.stack(tokens, 0).unsqueeze(0)          # (1, 5, D)
        ids = torch.arange(self.N_MODALITIES, device=device)
        x   = x + self.modality_emb(ids).unsqueeze(0)
        x   = self.transformer(x)                           # (1, 5, D)
        return x.squeeze(0).mean(0)                         # (D,)


# ══════════════════════════════════════════════════════════════════════════════
# 8. MultimodalFusion — top-level scene encoder
# ══════════════════════════════════════════════════════════════════════════════

class MultimodalFusion(nn.Module):
    """
    Five-modality scene encoder.

    Parameters
    ----------
    embed_dim    : shared embedding size for all encoders (default 256)
    gnn_hidden   : GAT hidden dim (default 128)
    n_heads      : attention heads in cross-attention fusion (default 8)
    n_layers     : transformer layers in cross-attention (default 2)
    color_dim    : per-point color channels for PointNet++ (default 3 for RGB)
    vit_pretrained : load MAE-pretrained ViT weights via timm (default True)
    share_trunk  : if True, creates a SharedCNNTrunk and passes it to BBoxRoIEncoder
    """

    def __init__(
        self,
        embed_dim:     int  = 256,
        gnn_hidden:    int  = 128,
        n_heads:       int  = 8,
        n_layers:      int  = 2,
        color_dim:     int  = 3,
        vit_pretrained: bool = True,
        share_trunk:   bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        trunk = SharedCNNTrunk() if share_trunk else None

        self.gat_encoder   = GATEncoder(NODE_FEAT_DIM, EDGE_FEAT_DIM,
                                        gnn_hidden, embed_dim)
        self.seg_encoder   = ViTSegEncoder(embed_dim, pretrained=vit_pretrained)
        self.bbox_encoder  = BBoxRoIEncoder(embed_dim, shared_trunk=trunk)
        self.depth_encoder = DilatedResNetEncoder(embed_dim)
        self.pc_encoder    = ColorPointNetPlusPlus(embed_dim, color_dim)
        self.fusion        = CrossAttentionFusion(embed_dim, n_heads, n_layers)
        self.null_token    = nn.Parameter(torch.zeros(embed_dim))

    def forward(
        self,
        graph_data:  dict | None,
        seg_image:   torch.Tensor | None,   # (3, H, W)
        bbox_image:  torch.Tensor | None,   # (3, H, W)
        bbox_boxes:  list[list[float]] | None,  # [[x1,y1,x2,y2], ...]
        depth_map:   torch.Tensor | None,   # (1, H, W)
        point_cloud: torch.Tensor | None,   # (N, 3) or (N, 3+C)
    ) -> torch.Tensor:                      # (embed_dim,)
        dev = self.null_token.device
        tokens: list[torch.Tensor] = []

        # Graph
        if graph_data is not None and graph_data["node_features"].shape[0] > 0:
            tokens.append(self.gat_encoder(
                graph_data["node_features"].to(dev),
                graph_data["edge_index"].to(dev),
                graph_data["edge_features"].to(dev),
            ))
        else:
            tokens.append(self.null_token.clone())

        # Segmented image
        tokens.append(
            self.seg_encoder(seg_image.to(dev))
            if seg_image is not None else self.null_token.clone()
        )

        # BBox image
        if bbox_image is not None:
            boxes = bbox_boxes or []
            tokens.append(self.bbox_encoder(bbox_image.to(dev), boxes))
        else:
            tokens.append(self.null_token.clone())

        # Depth map
        tokens.append(
            self.depth_encoder(depth_map.to(dev))
            if depth_map is not None else self.null_token.clone()
        )

        # Point cloud
        if point_cloud is not None and point_cloud.shape[0] >= 8:
            tokens.append(self.pc_encoder(point_cloud.to(dev)))
        else:
            tokens.append(self.null_token.clone())

        return self.fusion(tokens)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline integration helpers
# ══════════════════════════════════════════════════════════════════════════════

def _onehot(val: str, vocab: list[str]) -> list[float]:
    return [1.0 if val == v else 0.0 for v in vocab]


def detections_to_graph_data(detections: list[dict], graph=None) -> dict:
    if not detections:
        return {
            "node_features": torch.zeros(0, NODE_FEAT_DIM),
            "edge_index":    torch.zeros(2, 0, dtype=torch.long),
            "edge_features": torch.zeros(0, EDGE_FEAT_DIM),
        }

    node_rows = []
    for d in detections:
        feat = (
            _onehot(d.get("risk_group",      "BACKGROUND"), _RISK_GROUPS)
            + _onehot(d.get("proximity_label","FAR"),        _PROXIMITY_LABELS)
            + _onehot(d.get("path_zone",      "PERIPHERAL"), _PATH_ZONES)
            + [
                float(d.get("depth_score",   0.5) or 0.5),
                float(d.get("score",         0.0) or 0.0),
                float(d.get("mask_score",    0.0) or 0.0),
                float(d.get("relative_area", 0.0) or 0.0),
            ]
        )
        node_rows.append(feat)

    node_features = torch.tensor(node_rows, dtype=torch.float32)

    if graph is None:
        edge_index    = torch.zeros(2, 0, dtype=torch.long)
        edge_features = torch.zeros(0, EDGE_FEAT_DIM, dtype=torch.float32)
    else:
        src_l, dst_l, edge_rows = [], [], []
        for e in getattr(graph, "edges", []):
            src_l.append(getattr(e, "from_id",   0))
            dst_l.append(getattr(e, "to_id",     0))
            rel  = getattr(e, "relative_position", "in front of")
            dist = float(getattr(e, "distance_3d", 0.0))
            blk  = float(getattr(e, "blocking",    False))
            edge_rows.append(_onehot(rel, _REL_POSITIONS) + [dist, blk])

        if edge_rows:
            edge_index    = torch.tensor([src_l, dst_l], dtype=torch.long)
            edge_features = torch.tensor(edge_rows,       dtype=torch.float32)
        else:
            edge_index    = torch.zeros(2, 0, dtype=torch.long)
            edge_features = torch.zeros(0, EDGE_FEAT_DIM, dtype=torch.float32)

    return {"node_features": node_features,
            "edge_index":    edge_index,
            "edge_features": edge_features}


def pil_to_tensor(image: Image.Image, channels: int = 3) -> torch.Tensor:
    """PIL Image → float32 (C, H, W) in [0, 1]."""
    if channels == 1:
        arr = np.array(image.convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def depth_array_to_tensor(depth: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(depth).unsqueeze(0)


def point_cloud_to_tensor(pts: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(pts.astype(np.float32))
