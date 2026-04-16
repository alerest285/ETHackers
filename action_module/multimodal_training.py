"""
Multimodal SSL Pretraining — self-supervised training for MultimodalFusion.

Objectives (one per encoder, plus cross-modal consistency):
  1. MAE on segmented image         (75% masked patch reconstruction)
  2. MAE on bbox image              (same ViT-style decoder, shared weights)
  3. Scale-invariant depth loss     (Eigen et al., λ=0.5; decoder: UNet-style)
  4. Point-MAE                      (mask 75% of tokens; decoder: transformer + Chamfer)
  5. Masked graph node/edge         (BERT-style masking; decoder: linear heads)
  6. InfoNCE grounding loss         (align text-grounded Grounding DINO regions with image patches)
  7. Geometric consistency          (L2 between depth-encoder and pc-encoder embeddings)

Training
  AdamW, cosine LR decay (warmup 5% of steps), gradient clip 1.0, 200 epochs.
  All SSL losses are combined with fixed weights (tunable via loss_weights).

Usage
  from action_module.multimodal_training import MultimodalPretrainer, pretrain
  model = pretrain(train_dir="data/pipeline_output", epochs=200, embed_dim=256)
  # model is a MultimodalFusion instance with trained encoder weights.
  # Decoder heads are discarded after training.

Dataset directory layout expected (matches pipeline_output):
  <root>/detections/<stem>.json      — enriched detections (node/edge features)
  <root>/overlays/<stem>_overlay.png — segmented image (RGB)
  <root>/scene_graphs/<stem>.json    — scene graph (for graph encoder)
  <root>/depth_maps/<stem>.npy       — (H,W) float32 normalised depth
  <root>/point_clouds/<stem>.npy     — (N,3+C) float32 point cloud
  <root>/bbox_images/<stem>.png      — bbox crop image (RGB)
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from action_module.multimodal_fusion import (
    MultimodalFusion,
    GATEncoder,
    ViTSegEncoder,
    BBoxRoIEncoder,
    DilatedResNetEncoder,
    ColorPointNetPlusPlus,
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
    detections_to_graph_data,
    pil_to_tensor,
    depth_array_to_tensor,
    point_cloud_to_tensor,
)

# ── constants ─────────────────────────────────────────────────────────────────
_MASK_RATIO   = 0.75   # fraction of patches / nodes / points to mask
_TAU          = 0.07   # InfoNCE temperature
_DEPTH_LAMBDA = 0.5    # scale-invariant depth loss λ (Eigen et al.)
_IMG_SIZE     = 224

# ── default loss weights ──────────────────────────────────────────────────────
DEFAULT_LOSS_WEIGHTS = {
    "mae_seg":       1.0,
    "mae_bbox":      1.0,
    "depth":         1.0,
    "point_mae":     1.0,
    "graph_mask":    1.0,
    "grounding":     0.5,   # λ1
    "geo_consist":   0.5,   # λ2
}


# ══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ══════════════════════════════════════════════════════════════════════════════

def mae_loss(pred: torch.Tensor, target: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-squared reconstruction loss over masked tokens only.
    pred, target : (N_patches, patch_dim)
    mask         : (N_patches,) bool — True = masked (must reconstruct)
    """
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    diff = (pred[mask] - target[mask]) ** 2
    return diff.mean()


def scale_invariant_depth_loss(pred_log: torch.Tensor,
                                gt_log:   torch.Tensor,
                                lam:      float = _DEPTH_LAMBDA) -> torch.Tensor:
    """
    Eigen et al. scale-invariant log-depth loss.
    pred_log, gt_log : (H*W,) log-depth tensors
    """
    d = pred_log - gt_log                     # (n,)
    n = d.numel()
    if n == 0:
        return pred_log.new_tensor(0.0)
    return (d ** 2).mean() - lam * (d.sum() ** 2) / (n ** 2)


def chamfer_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Symmetric Chamfer Distance between two point sets.
    p : (M, 3),  q : (N, 3)
    """
    diff  = p.unsqueeze(1) - q.unsqueeze(0)   # (M, N, 3)
    dist2 = (diff ** 2).sum(-1)               # (M, N)
    cd    = dist2.min(1)[0].mean() + dist2.min(0)[0].mean()
    return cd


def info_nce_loss(anchors: torch.Tensor, positives: torch.Tensor,
                  tau: float = _TAU) -> torch.Tensor:
    """
    InfoNCE / NT-Xent contrastive loss.
    anchors   : (B, D) — image patch embeddings
    positives : (B, D) — text-grounded region embeddings (one per anchor)
    In-batch negatives.
    """
    anchors   = F.normalize(anchors,   dim=-1)
    positives = F.normalize(positives, dim=-1)
    logits    = anchors @ positives.T / tau       # (B, B)
    labels    = torch.arange(len(anchors), device=anchors.device)
    return F.cross_entropy(logits, labels)


def geo_consistency_loss(depth_emb: torch.Tensor,
                          pc_emb:    torch.Tensor) -> torch.Tensor:
    """L2 distance between depth and point-cloud embeddings (push to align)."""
    return F.mse_loss(F.normalize(depth_emb, dim=-1),
                      F.normalize(pc_emb,    dim=-1))


# ══════════════════════════════════════════════════════════════════════════════
# Decoder heads  (training-time only; discarded after pretraining)
# ══════════════════════════════════════════════════════════════════════════════

class MAEDecoder(nn.Module):
    """
    Transformer-based MAE decoder for image patch reconstruction.
    Produces (N_masked, patch_dim) pixel predictions.
    """
    def __init__(self, embed_dim: int, patch_dim: int, depth: int = 4,
                 n_heads: int = 8, decoder_dim: int = 128):
        super().__init__()
        self.proj    = nn.Linear(embed_dim, decoder_dim)
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_tok, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim, nhead=n_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=depth)
        self.head = nn.Linear(decoder_dim, patch_dim)

    def forward(self, visible_tokens: torch.Tensor,     # (N_vis, embed_dim)
                n_total:         int) -> torch.Tensor:   # → (N_total, patch_dim)
        """
        visible_tokens : encoded visible patches
        n_total        : total patch count (visible + masked)
        Returns predictions for ALL positions (masked ones are meaningful).
        """
        v = self.proj(visible_tokens).unsqueeze(0)      # (1, N_vis, decoder_dim)
        # build full sequence: visible + mask tokens (order doesn't matter for loss)
        n_mask  = n_total - visible_tokens.shape[0]
        masks_t = self.mask_tok.expand(1, n_mask, -1)   # (1, N_mask, decoder_dim)
        tgt     = torch.cat([v, masks_t], dim=1)        # (1, N_total, decoder_dim)
        out     = self.transformer(tgt, v)              # (1, N_total, decoder_dim)
        return self.head(out.squeeze(0))                # (N_total, patch_dim)


class GraphMaskedDecoder(nn.Module):
    """
    Node classification + attribute prediction decoder for masked graph nodes.
    Two linear heads: node-class prediction and raw-attribute reconstruction.
    """
    def __init__(self, embed_dim: int, node_dim: int = NODE_FEAT_DIM,
                 n_classes: int = 5):
        super().__init__()
        self.class_head = nn.Linear(embed_dim, n_classes)
        self.attr_head  = nn.Linear(embed_dim, node_dim)

    def forward(self, node_embs: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:
        # node_embs: (N, embed_dim)  (only masked nodes passed during training)
        return self.class_head(node_embs), self.attr_head(node_embs)


class EdgeDecoder(nn.Module):
    """Edge relation-type decoder from concatenated src+dst node embeddings."""
    def __init__(self, embed_dim: int, n_edge_types: int = 4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_edge_types),
        )

    def forward(self, src_emb: torch.Tensor,
                dst_emb: torch.Tensor) -> torch.Tensor:
        # src_emb, dst_emb : (E, embed_dim)
        return self.head(torch.cat([src_emb, dst_emb], dim=-1))


class DepthReconDecoder(nn.Module):
    """
    UNet-style depth reconstruction decoder.
    Takes the (1, C, H', W') feature map from DilatedResNetEncoder.feature_map()
    and upsamples back to (1, H, W) log-depth predictions.
    """
    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.up1 = self._up_block(in_channels,  256)
        self.up2 = self._up_block(256,          128)
        self.up3 = self._up_block(128,           64)
        self.up4 = self._up_block(64,            32)
        self.out = nn.Conv2d(32, 1, 1)

    @staticmethod
    def _up_block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, out_c // 8), out_c),
            nn.GELU(),
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        # feat_map: (1, C, H', W')
        x = self.up1(feat_map)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.out(x)                      # (1, 1, H_out, W_out)


class PointCloudDecoder(nn.Module):
    """
    Transformer-based point cloud decoder for Point-MAE.
    Reconstructs masked point tokens → (N_masked_groups, n_pts, 3).
    """
    def __init__(self, embed_dim: int, n_pts: int = 32,
                 depth: int = 4, n_heads: int = 8):
        super().__init__()
        self.n_pts    = n_pts
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_tok, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_pts * 3),
        )

    def forward(self, vis_tokens: torch.Tensor,         # (N_vis, D)
                n_total:     int) -> torch.Tensor:       # → (N_total, n_pts, 3)
        n_mask  = n_total - vis_tokens.shape[0]
        v       = vis_tokens.unsqueeze(0)               # (1, N_vis, D)
        masks_t = self.mask_tok.expand(1, n_mask, -1)   # (1, N_mask, D)
        tgt     = torch.cat([v, masks_t], dim=1)        # (1, N_total, D)
        out     = self.transformer(tgt, v).squeeze(0)   # (N_total, D)
        return self.head(out).reshape(-1, self.n_pts, 3)  # (N_total, n_pts, 3)


# ══════════════════════════════════════════════════════════════════════════════
# Pretrainer
# ══════════════════════════════════════════════════════════════════════════════

class MultimodalPretrainer(nn.Module):
    """
    Wraps MultimodalFusion and attaches all SSL decoder heads.
    After training, call .encoders_only() to get a plain MultimodalFusion
    instance with pretrained weights and no decoder overhead.
    """

    def __init__(self, fusion: MultimodalFusion,
                 loss_weights: dict[str, float] | None = None):
        super().__init__()
        self.fusion = fusion
        D = fusion.embed_dim

        # ── decoder heads ─────────────────────────────────────────────────────
        # ViT patch dim = patch_size**2 * 3 = 16*16*3 = 768
        patch_dim = 16 * 16 * 3
        self.mae_decoder_seg  = MAEDecoder(D, patch_dim)
        self.mae_decoder_bbox = MAEDecoder(D, patch_dim)

        self.depth_decoder   = DepthReconDecoder(in_channels=512)
        self.pc_decoder      = PointCloudDecoder(D)

        # graph decoder: final GAT layer node-level embeddings needed
        # we expose intermediate node embeddings via a hook
        self.graph_node_dec  = GraphMaskedDecoder(D)
        self.edge_dec        = EdgeDecoder(D)

        self.lw = {**DEFAULT_LOSS_WEIGHTS, **(loss_weights or {})}

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _random_mask(n: int, ratio: float,
                     device: torch.device) -> torch.Tensor:
        """Returns a bool mask of shape (n,); True = masked."""
        n_mask = max(1, int(n * ratio))
        idx    = torch.randperm(n, device=device)[:n_mask]
        mask   = torch.zeros(n, dtype=torch.bool, device=device)
        mask[idx] = True
        return mask

    # ── individual SSL losses ─────────────────────────────────────────────────

    def _mae_image_loss(
        self,
        encoder:   ViTSegEncoder,
        decoder:   MAEDecoder,
        img:       torch.Tensor,           # (3, H, W)
    ) -> torch.Tensor:
        """MAE loss for one image encoder (seg or bbox)."""
        # get all patch tokens
        _, patches = encoder(img, return_patches=True)   # (N, vit_dim)
        N = patches.shape[0]
        mask = self._random_mask(N, _MASK_RATIO, img.device)

        # encode only visible patches (simulate MAE encoder)
        vis_patches = patches[~mask]                     # (N_vis, vit_dim)
        preds = decoder(vis_patches, N)                  # (N, patch_dim)

        # targets: raw pixel values reshaped to patches
        # resize img to _IMG_SIZE to match ViT
        img_rs = F.interpolate(img.unsqueeze(0), _IMG_SIZE, mode="bilinear",
                               align_corners=False).squeeze(0)  # (3, 224, 224)
        P = 16
        H_p = W_p = _IMG_SIZE // P
        # (3, H_p, P, W_p, P) → (N, P*P*3)
        target_patches = (
            img_rs
            .reshape(3, H_p, P, W_p, P)
            .permute(1, 3, 2, 4, 0)
            .reshape(H_p * W_p, P * P * 3)
        )
        return mae_loss(preds, target_patches.detach(), mask)

    def _depth_loss(
        self,
        depth_map: torch.Tensor,           # (1, H, W)
    ) -> torch.Tensor:
        """Scale-invariant depth reconstruction via UNet decoder."""
        feat_map = self.fusion.depth_encoder.feature_map(depth_map)  # (1,C,H',W')
        pred     = self.depth_decoder(feat_map)                       # (1,1,H_o,W_o)
        # resize GT to decoder output size
        H_o, W_o = pred.shape[-2], pred.shape[-1]
        gt = F.interpolate(depth_map.unsqueeze(0), (H_o, W_o),
                           mode="bilinear", align_corners=False).squeeze(0)  # (1,H_o,W_o)
        # clamp to avoid log(0)
        pred_log = torch.log(pred.squeeze().clamp(min=1e-4).flatten())
        gt_log   = torch.log(gt.squeeze().clamp(min=1e-4).flatten())
        return scale_invariant_depth_loss(pred_log, gt_log)

    def _point_mae_loss(
        self,
        pc: torch.Tensor,                  # (N, 3+C)
    ) -> torch.Tensor:
        """Point-MAE: mask 75% of SA-level tokens, reconstruct with Chamfer."""
        # We treat each SA centroid group as a "token".
        # For simplicity: subsample N points into G groups of K pts each.
        G, K = 32, 32
        N = pc.shape[0]
        if N < G * K:
            return pc.new_tensor(0.0)

        # sample G centroids via FPS (reuse _fps from fusion)
        from action_module.multimodal_fusion import _fps
        idx_c = _fps(pc[:, :3], G)
        centroids = pc[idx_c, :3]               # (G, 3)

        # nearest K points per centroid
        diff  = pc[:, :3].unsqueeze(0) - centroids.unsqueeze(1)  # (G, N, 3)
        dists = (diff ** 2).sum(-1)
        knn   = dists.topk(K, dim=-1, largest=False)[1]           # (G, K)
        groups = pc[knn]                                           # (G, K, 3+C)

        # simple token embedding: mean + linear
        if not hasattr(self, "_pt_tok_proj"):
            feat_dim = pc.shape[1]
            self._pt_tok_proj = nn.Linear(feat_dim, self.fusion.embed_dim).to(pc.device)
        tokens = self._pt_tok_proj(groups.reshape(G, K, -1).mean(1))  # (G, D)

        mask = self._random_mask(G, _MASK_RATIO, pc.device)
        vis_tokens = tokens[~mask]                                 # (G_vis, D)
        preds = self.pc_decoder(vis_tokens, G)                    # (G, n_pts, 3)

        # target: local centred point sets for masked groups
        # shape: (G, K, 3) relative to centroid
        target_pts = (groups[:, :, :3] - centroids.unsqueeze(1))  # (G, K, 3)
        masked_pred   = preds[mask]                               # (G_m, n_pts, 3)
        masked_target = target_pts[mask]                          # (G_m, K, 3)

        if masked_pred.shape[0] == 0:
            return pc.new_tensor(0.0)
        # Chamfer per masked group, mean
        total = pc.new_tensor(0.0)
        for i in range(masked_pred.shape[0]):
            total = total + chamfer_distance(masked_pred[i], masked_target[i])
        return total / masked_pred.shape[0]

    def _graph_mask_loss(
        self,
        graph_data: dict,
    ) -> torch.Tensor:
        """Masked node/edge reconstruction."""
        dev  = self.fusion.null_token.device
        nf   = graph_data["node_features"].to(dev)       # (N, NODE_FEAT_DIM)
        ei   = graph_data["edge_index"].to(dev)          # (2, E)
        ef   = graph_data["edge_features"].to(dev)       # (E, EDGE_FEAT_DIM)
        N    = nf.shape[0]

        if N == 0:
            return nf.new_tensor(0.0)

        node_mask = self._random_mask(N, _MASK_RATIO, dev)

        # zero-out masked node features
        nf_masked = nf.clone()
        nf_masked[node_mask] = 0.0

        # obtain node-level embeddings via GAT layers (bypass global pool)
        enc = self.fusion.gat_encoder
        h   = enc.layer1(nf_masked, ei, ef)
        h   = enc.layer2(h,          ei, ef)
        h   = enc.layer3(h,          ei, ef)             # (N, embed_dim)

        # node attribute + class reconstruction
        node_embs_masked = h[node_mask]                  # (N_m, D)
        cls_pred, attr_pred = self.graph_node_dec(node_embs_masked)

        # node class: argmax of risk-group one-hot (first 5 dims)
        node_class_gt = nf[node_mask, :5].argmax(-1)     # (N_m,)
        loss_node_cls = F.cross_entropy(cls_pred, node_class_gt)
        loss_node_attr = F.mse_loss(attr_pred, nf[node_mask].detach())

        # edge relation reconstruction (if edges exist)
        loss_edge = nf.new_tensor(0.0)
        E = ei.shape[1]
        if E > 0:
            edge_mask = self._random_mask(E, _MASK_RATIO, dev)
            src_emb = h[ei[0][edge_mask]]
            dst_emb = h[ei[1][edge_mask]]
            edge_pred = self.edge_dec(src_emb, dst_emb)
            # relation type: argmax of first 4 dims of edge feature
            edge_gt = ef[edge_mask, :4].argmax(-1)
            loss_edge = F.cross_entropy(edge_pred, edge_gt)

        return loss_node_cls + loss_node_attr + loss_edge

    def _grounding_loss(
        self,
        seg_img:    torch.Tensor,           # (3, H, W)
        boxes:      list[list[float]],      # [[x1,y1,x2,y2], ...]
    ) -> torch.Tensor:
        """
        InfoNCE grounding loss: patch embeddings vs. RoI-pooled region embeddings.
        Aligns each detected box with the most-overlapping image patch.
        """
        if not boxes:
            return seg_img.new_tensor(0.0)

        dev = seg_img.device
        _, patches = self.fusion.seg_encoder(seg_img, return_patches=True)
        # patches: (N_patches, D)

        # Region embeddings via BBoxRoI encoder (per-box)
        # We run the BBox encoder in "per-box" mode using full seg_img
        # Re-using the obj_transformer path — extract per-box tokens
        enc   = self.fusion.bbox_encoder
        feat  = enc.trunk(seg_img.unsqueeze(0))
        _, _, H_f, W_f = feat.shape
        H, W  = seg_img.shape[1], seg_img.shape[2]
        sx, sy = W_f / W, H_f / H

        try:
            from torchvision.ops import roi_align
            rois = torch.tensor(
                [[0, b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy] for b in boxes],
                dtype=torch.float32, device=dev,
            )
            roi_feats = roi_align(feat, rois, enc.ROI_SIZE, aligned=True)
            region_embs = enc.roi_proj(roi_feats)               # (B, D)
        except Exception:
            return seg_img.new_tensor(0.0)

        # For each box, find the patch with highest IoU overlap as anchor
        P     = 16
        H_p   = W_p = _IMG_SIZE // P
        patch_cx = (torch.arange(W_p, device=dev) * P + P // 2) * (W / _IMG_SIZE)
        patch_cy = (torch.arange(H_p, device=dev) * P + P // 2) * (H / _IMG_SIZE)

        anchors: list[torch.Tensor] = []
        for b in boxes:
            cx_b = (b[0] + b[2]) / 2
            cy_b = (b[1] + b[3]) / 2
            # find nearest patch centroid
            diff_x = (patch_cx - cx_b).abs()
            diff_y = (patch_cy - cy_b).abs()
            ix = diff_x.argmin().item()
            iy = diff_y.argmin().item()
            patch_idx = iy * W_p + ix
            patch_idx = min(patch_idx, patches.shape[0] - 1)
            anchors.append(patches[patch_idx])

        anchors_t = torch.stack(anchors, 0)  # (B, D)
        return info_nce_loss(anchors_t, region_embs)

    # ── forward (one training step) ───────────────────────────────────────────

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        """
        Compute all SSL losses for a single sample batch.
        Returns a dict of named losses + 'total'.
        """
        dev      = self.fusion.null_token.device
        losses: dict[str, torch.Tensor] = {}

        seg_img  = batch.get("seg_image")         # (3, H, W) | None
        bbox_img = batch.get("bbox_image")        # (3, H, W) | None
        boxes    = batch.get("boxes", [])
        depth    = batch.get("depth_map")         # (1, H, W) | None
        pc       = batch.get("point_cloud")       # (N, 3+C) | None
        graph    = batch.get("graph_data")        # dict | None

        # 1. MAE — segmented image
        if seg_img is not None:
            losses["mae_seg"] = self._mae_image_loss(
                self.fusion.seg_encoder, self.mae_decoder_seg, seg_img.to(dev))
        else:
            losses["mae_seg"] = seg_img.new_tensor(0.0) if seg_img else \
                                self.fusion.null_token.new_tensor(0.0)

        # 2. MAE — bbox image
        if bbox_img is not None:
            losses["mae_bbox"] = self._mae_image_loss(
                self.fusion.seg_encoder, self.mae_decoder_bbox, bbox_img.to(dev))
        else:
            losses["mae_bbox"] = self.fusion.null_token.new_tensor(0.0)

        # 3. Depth
        if depth is not None:
            losses["depth"] = self._depth_loss(depth.to(dev))
        else:
            losses["depth"] = self.fusion.null_token.new_tensor(0.0)

        # 4. Point-MAE
        if pc is not None:
            losses["point_mae"] = self._point_mae_loss(pc.to(dev))
        else:
            losses["point_mae"] = self.fusion.null_token.new_tensor(0.0)

        # 5. Graph masking
        if graph is not None and graph["node_features"].shape[0] > 0:
            losses["graph_mask"] = self._graph_mask_loss(graph)
        else:
            losses["graph_mask"] = self.fusion.null_token.new_tensor(0.0)

        # 6. InfoNCE grounding loss
        if seg_img is not None and boxes:
            losses["grounding"] = self._grounding_loss(seg_img.to(dev), boxes)
        else:
            losses["grounding"] = self.fusion.null_token.new_tensor(0.0)

        # 7. Geometric consistency
        if depth is not None and pc is not None:
            d_emb = self.fusion.depth_encoder(depth.to(dev))
            p_emb = self.fusion.pc_encoder(pc.to(dev))
            losses["geo_consist"] = geo_consistency_loss(d_emb, p_emb)
        else:
            losses["geo_consist"] = self.fusion.null_token.new_tensor(0.0)

        # Weighted sum
        total = sum(self.lw.get(k, 1.0) * v for k, v in losses.items())
        losses["total"] = total
        return losses

    def encoders_only(self) -> MultimodalFusion:
        """Return the fusion model with trained encoder weights (no decoders)."""
        return self.fusion


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class PipelineOutputDataset(Dataset):
    """
    Loads pipeline_output artefacts for SSL pretraining.
    Missing files are silently replaced with None (losses skip gracefully).
    """

    def __init__(self, root: str | Path, img_size: int = _IMG_SIZE):
        self.root     = Path(root)
        self.img_size = img_size

        # collect stems from detections dir (primary anchor)
        det_dir = self.root / "detections"
        if not det_dir.exists():
            raise FileNotFoundError(f"detections/ not found under {root}")
        self.stems = sorted(p.stem for p in det_dir.glob("*.json"))

    def __len__(self) -> int:
        return len(self.stems)

    def _load_img(self, path: Path) -> Optional[torch.Tensor]:
        if not path.exists():
            return None
        try:
            img = Image.open(path).convert("RGB").resize(
                (self.img_size, self.img_size), Image.BILINEAR)
            return pil_to_tensor(img)   # (3, H, W) float32 [0,1]
        except Exception:
            return None

    def _load_depth(self, path: Path) -> Optional[torch.Tensor]:
        if not path.exists():
            return None
        try:
            arr = np.load(path).astype(np.float32)
            return depth_array_to_tensor(arr)   # (1, H, W)
        except Exception:
            return None

    def _load_pc(self, path: Path) -> Optional[torch.Tensor]:
        if not path.exists():
            return None
        try:
            arr = np.load(path).astype(np.float32)
            return point_cloud_to_tensor(arr)   # (N, 3+C)
        except Exception:
            return None

    def _load_graph(self, stem: str) -> Optional[dict]:
        p = self.root / "scene_graphs" / f"{stem}.json"
        d = self.root / "detections"   / f"{stem}.json"
        if not p.exists() or not d.exists():
            return None
        try:
            with open(d) as f:
                dets = json.load(f)
            # scene graph JSON produced by SceneGraphBuilder
            import sys
            sys.path.insert(0, str(Path(__file__).parents[1] / "3d-module"))
            from lift_3d import SceneGraphBuilder
            # We already have the graph JSON; parse into graph_data dict
            with open(p) as f:
                sg = json.load(f)
            # Fallback: reconstruct graph_data from detection JSON
            return detections_to_graph_data(dets, sg)
        except Exception:
            return None

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]
        return {
            "stem":        stem,
            "seg_image":   self._load_img(
                self.root / "overlays" / f"{stem}_overlay.png"),
            "bbox_image":  self._load_img(
                self.root / "bbox_images" / f"{stem}.png"),
            "depth_map":   self._load_depth(
                self.root / "depth_maps" / f"{stem}.npy"),
            "point_cloud": self._load_pc(
                self.root / "point_clouds" / f"{stem}.npy"),
            "graph_data":  self._load_graph(stem),
            "boxes":       self._load_boxes(stem),
        }

    def _load_boxes(self, stem: str) -> list[list[float]]:
        p = self.root / "detections" / f"{stem}.json"
        if not p.exists():
            return []
        try:
            with open(p) as f:
                dets = json.load(f)
            return [d["box"] for d in dets if "box" in d]
        except Exception:
            return []


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def pretrain(
    train_dir:    str | Path,
    epochs:       int   = 200,
    embed_dim:    int   = 256,
    lr:           float = 1e-4,
    weight_decay: float = 0.05,
    warmup_frac:  float = 0.05,
    clip_grad:    float = 1.0,
    save_every:   int   = 20,
    save_dir:     str | Path = "data/multimodal_checkpoints",
    loss_weights: dict | None = None,
    device:       str | None  = None,
    num_workers:  int   = 4,
) -> MultimodalFusion:
    """
    Full SSL pretraining loop.

    Returns the MultimodalFusion model with pretrained encoder weights.
    Decoder heads are not included in the returned model.
    """
    dev = torch.device(
        device if device else
        ("cuda" if torch.cuda.is_available() else
         ("mps"  if torch.backends.mps.is_available() else "cpu"))
    )
    print(f"Pretraining on device: {dev}")

    # ── dataset ───────────────────────────────────────────────────────────────
    dataset = PipelineOutputDataset(train_dir)
    # simple collate: each item is independent (no batching across samples)
    loader  = DataLoader(
        dataset, batch_size=1, shuffle=True,
        num_workers=num_workers, collate_fn=lambda x: x[0],
        pin_memory=(dev.type == "cuda"),
    )
    print(f"Dataset: {len(dataset)} samples")

    # ── model ─────────────────────────────────────────────────────────────────
    fusion    = MultimodalFusion(embed_dim=embed_dim).to(dev)
    pretrainer = MultimodalPretrainer(fusion, loss_weights=loss_weights).to(dev)

    n_steps = epochs * len(dataset)
    n_warmup = max(1, int(n_steps * warmup_frac))

    optimizer = torch.optim.AdamW(
        pretrainer.parameters(), lr=lr, weight_decay=weight_decay,
    )

    def lr_lambda(step: int) -> float:
        if step < n_warmup:
            return step / n_warmup
        progress = (step - n_warmup) / max(1, n_steps - n_warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # ── loop ──────────────────────────────────────────────────────────────────
    global_step = 0
    for epoch in range(1, epochs + 1):
        epoch_losses: dict[str, float] = {}
        epoch_count = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            optimizer.zero_grad()

            try:
                loss_dict = pretrainer(batch)
            except Exception as e:
                tqdm.write(f"  [skip] {batch.get('stem','?')}: {e}")
                global_step += 1
                scheduler.step()
                continue

            total = loss_dict["total"]
            if not torch.isfinite(total):
                tqdm.write(f"  [skip] non-finite loss at step {global_step}")
                global_step += 1
                scheduler.step()
                continue

            total.backward()
            nn.utils.clip_grad_norm_(pretrainer.parameters(), clip_grad)
            optimizer.step()
            scheduler.step()

            # accumulate for logging
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
            epoch_count += 1
            global_step += 1

            pbar.set_postfix(loss=f"{total.item():.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        # epoch summary
        if epoch_count > 0:
            avg = {k: v / epoch_count for k, v in epoch_losses.items()}
            parts = "  ".join(f"{k}={v:.4f}" for k, v in avg.items())
            print(f"[Epoch {epoch:03d}]  {parts}")

        if epoch % save_every == 0 or epoch == epochs:
            ckpt = save_path / f"pretrain_ep{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "fusion_state": fusion.state_dict(),
                "pretrainer_state": pretrainer.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, ckpt)
            print(f"  Checkpoint saved → {ckpt}")

    print("\nPretraining complete.")
    return pretrainer.encoders_only()


def load_pretrained(
    checkpoint: str | Path,
    embed_dim:  int = 256,
    device:     str | None = None,
) -> MultimodalFusion:
    """Load a MultimodalFusion model from a pretraining checkpoint."""
    dev = torch.device(device or "cpu")
    ckpt = torch.load(checkpoint, map_location=dev)
    model = MultimodalFusion(embed_dim=embed_dim).to(dev)
    model.load_state_dict(ckpt["fusion_state"])
    model.eval()
    return model


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multimodal SSL pretraining")
    parser.add_argument("--train-dir",   type=Path, default=Path("data/pipeline_output"),
                        help="Root of pipeline_output directory")
    parser.add_argument("--epochs",      type=int,  default=200)
    parser.add_argument("--embed-dim",   type=int,  default=256)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--save-dir",    type=Path,  default=Path("data/multimodal_checkpoints"))
    parser.add_argument("--save-every",  type=int,   default=20)
    parser.add_argument("--workers",     type=int,   default=4)
    parser.add_argument("--device",      type=str,   default=None)
    args = parser.parse_args()

    pretrain(
        train_dir    = args.train_dir,
        epochs       = args.epochs,
        embed_dim    = args.embed_dim,
        lr           = args.lr,
        save_dir     = args.save_dir,
        save_every   = args.save_every,
        num_workers  = args.workers,
        device       = args.device,
    )
