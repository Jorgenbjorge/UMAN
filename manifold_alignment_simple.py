# manifold_alignment_simple.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftGlobalLocalManifoldAlignment(nn.Module):
    """
    loss_align = (beta)*L_global + (1-beta)*L_local + reg_terms
    - L_global: batch-level InfoNCE，保证主损失不易快速塌陷到极小
    - L_local : patch<->token 双向 soft matching，体现“局部流形对齐”
    - reg_terms:
        * entropy-to-topk: 让patch/token权重有效支撑接近 topk_ratio，而不是均匀/崩到一两处
        * variance floor: similarity/weights 方差太小 -> 惩罚，直接抑制 mode collapse
        * topology KL: patch邻接分布 与 matched-text邻接分布 KL 对齐
    """

class SoftGlobalLocalManifoldAlignment(nn.Module):
    def __init__(
        self,
        dim: int = None,
        feature_dim: int = None,   # ✅ 兼容旧参数名
        init_temp: float = 0.07,
        topk_ratio: float = 0.30,
        topo_weight: float = 0.20,
        entropy_weight: float = 0.05,
        var_weight: float = 0.10,
        gate_entropy_weight: float = 0.01,
        topo_temp: float = 0.10,
        pool_temp: float = 0.07,
        **kwargs,                  # ✅ 防止你那边还有其它旧参数名
    ):
        super().__init__()

        # ✅ 统一维度参数（优先 feature_dim，其次 dim）
        if feature_dim is not None:
            self.dim = int(feature_dim)
        elif dim is not None:
            self.dim = int(dim)
        else:
            self.dim = 512

        self.topk_ratio = float(topk_ratio)
        self.pool_temp = float(pool_temp)

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / init_temp))

        self.topo_weight = float(topo_weight)
        self.topo_temp = float(topo_temp)

        self.entropy_weight = float(entropy_weight)
        self.var_weight = float(var_weight)
        self.gate_entropy_weight = float(gate_entropy_weight)

        # ✅ gate 输入维度用 self.dim
        self.gate = nn.Sequential(
            nn.Linear(4 * self.dim, 2 * self.dim),
            nn.GELU(),
            nn.Linear(2 * self.dim, 1),
        )


    @staticmethod
    def _safe_entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # p: [..., K] probability simplex
        return -(p * (p + eps).log()).sum(dim=-1)

    def _info_nce(self, img_feat: torch.Tensor, txt_feat: torch.Tensor) -> torch.Tensor:
        """
        img_feat: [B, D], txt_feat: [B, D]
        """
        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)

        scale = self.logit_scale.exp().clamp(max=100.0)
        logits_i2t = scale * (img_feat @ txt_feat.t())
        logits_t2i = scale * (txt_feat @ img_feat.t())
        labels = torch.arange(img_feat.size(0), device=img_feat.device)

        loss_i = F.cross_entropy(logits_i2t, labels)
        loss_t = F.cross_entropy(logits_t2i, labels)
        return 0.5 * (loss_i + loss_t)

    def _topology_kl(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        A,B: [B, N, N] row-stochastic (soft adjacency)
        """
        # KL(A||B) row-wise
        return F.kl_div((B + 1e-8).log(), A, reduction="batchmean")

    def forward(
        self,
        img_global: torch.Tensor,      # [B, D]
        img_patches: torch.Tensor,     # [B, N, D]
        txt_global: torch.Tensor,      # [B, D]
        txt_tokens: torch.Tensor,      # [B, L, D]
        return_info: bool = False,
    ):
        """
        Returns:
            img_patches_aligned: [B, N, D]  (对齐后的patch表征，可用于可视化/下游)
            alignment_info: dict (包含 similarity_map, weights 等)
        """
        B, N, D = img_patches.shape
        _, L, _ = txt_tokens.shape

        img_global = F.normalize(img_global, dim=-1)
        txt_global = F.normalize(txt_global, dim=-1)
        img_patches = F.normalize(img_patches, dim=-1)
        txt_tokens = F.normalize(txt_tokens, dim=-1)

        # =========================
        # 1) Patch-Token Similarity
        # =========================
        scale = self.logit_scale.exp().clamp(max=100.0)
        sim = scale * torch.bmm(img_patches, txt_tokens.transpose(1, 2))  # [B, N, L]

        # patch importance: based on max token similarity
        patch_score = sim.max(dim=-1).values  # [B, N]
        patch_weights = F.softmax(patch_score / max(self.pool_temp, 1e-6), dim=-1)  # [B, N]

        # token importance: based on max patch similarity
        token_score = sim.max(dim=1).values  # [B, L]
        token_weights = F.softmax(token_score / max(self.pool_temp, 1e-6), dim=-1)  # [B, L]

        # =====================================
        # 2) Local soft matching (bi-direction)
        # =====================================
        # patch -> token assignment
        A_p2t = F.softmax(sim / max(self.pool_temp, 1e-6), dim=-1)          # [B, N, L]
        matched_txt_for_patch = torch.bmm(A_p2t, txt_tokens)               # [B, N, D]
        matched_txt_for_patch = F.normalize(matched_txt_for_patch, dim=-1)

        # token -> patch assignment
        A_t2p = F.softmax(sim.transpose(1, 2) / max(self.pool_temp, 1e-6), dim=-1)  # [B, L, N]
        matched_img_for_token = torch.bmm(A_t2p, img_patches)               # [B, L, D]
        matched_img_for_token = F.normalize(matched_img_for_token, dim=-1)

        # local cosine alignment (weighted)
        cos_patch = (img_patches * matched_txt_for_patch).sum(dim=-1)       # [B, N]
        cos_token = (txt_tokens * matched_img_for_token).sum(dim=-1)        # [B, L]
        L_local_patch = 1.0 - (patch_weights * cos_patch).sum(dim=-1)       # [B]
        L_local_token = 1.0 - (token_weights * cos_token).sum(dim=-1)       # [B]
        L_local = 0.5 * (L_local_patch + L_local_token)                    # [B]

        # ==========================================================
        # 3) Global representation from soft top-k-like pooling
        #    (keep it inside loss_align; no external CLIP loss)
        # ==========================================================
        img_local_pool = torch.einsum("bn,bnd->bd", patch_weights, img_patches)  # [B,D]
        txt_local_pool = torch.einsum("bl,bld->bd", token_weights, txt_tokens)  # [B,D]
        img_local_pool = F.normalize(img_local_pool, dim=-1)
        txt_local_pool = F.normalize(txt_local_pool, dim=-1)

        # gate beta: soft combine global and local inside loss_align
        gate_in = torch.cat([img_global, txt_global, img_local_pool, txt_local_pool], dim=-1)  # [B,4D]
        beta = torch.sigmoid(self.gate(gate_in)).squeeze(-1)  # [B] in (0,1)

        img_fused = F.normalize(beta.unsqueeze(-1) * img_global + (1.0 - beta).unsqueeze(-1) * img_local_pool, dim=-1)
        txt_fused = F.normalize(beta.unsqueeze(-1) * txt_global + (1.0 - beta).unsqueeze(-1) * txt_local_pool, dim=-1)

        # global InfoNCE (scalar)
        L_global = self._info_nce(img_fused, txt_fused)  # scalar

        # Make L_global per-sample compatible by broadcasting
        L_global_vec = L_global * torch.ones_like(L_local)

        # =========================
        # 4) Anti-collapse regularizers
        # =========================
        # (a) entropy-to-topk: target effective support size ~ topk_ratio * N
        # entropy H ~ log(K_eff). We set target H* = log(topkN)
        topkN = max(1, int(self.topk_ratio * N))
        topkL = max(1, int(self.topk_ratio * L))
        target_H_patch = math.log(topkN + 1e-8)
        target_H_token = math.log(topkL + 1e-8)

        H_patch = self._safe_entropy(patch_weights)  # [B]
        H_token = self._safe_entropy(token_weights)  # [B]
        reg_entropy = (H_patch - target_H_patch).pow(2).mean() + (H_token - target_H_token).pow(2).mean()

        # (b) variance floor (very important for your现象：loss_align秒变极小)
        # Encourage std of similarity and weights to stay above a floor.
        sim_std = patch_score.std(dim=-1)           # [B]
        w_std = patch_weights.std(dim=-1)           # [B]
        # floors can be tuned; these are conservative to avoid collapse
        sim_floor = 0.15
        w_floor = 0.08
        reg_var = F.relu(sim_floor - sim_std).mean() + F.relu(w_floor - w_std).mean()

        # (c) gate entropy regularization: avoid beta stuck at all-0 or all-1 early
        # Encourage some uncertainty (not too sharp), but weight small
        H_beta = -(beta * (beta + 1e-8).log() + (1 - beta) * (1 - beta + 1e-8).log())
        reg_gate = -H_beta.mean()  # negative entropy -> penalize low entropy => keep moderate

        # (d) topology (optional but recommended)
        # Build row-stochastic adjacency on patches and matched text (patch-level)
        if self.topo_weight > 0:
            # patch affinity
            aff_v = torch.bmm(img_patches, img_patches.transpose(1, 2)) / max(self.topo_temp, 1e-6)  # [B,N,N]
            A_v = F.softmax(aff_v, dim=-1)

            # matched text affinity (patch-level, using matched_txt_for_patch)
            aff_t = torch.bmm(matched_txt_for_patch, matched_txt_for_patch.transpose(1, 2)) / max(self.topo_temp, 1e-6)
            A_t = F.softmax(aff_t, dim=-1)

            reg_topo = self._topology_kl(A_v, A_t)
        else:
            reg_topo = torch.zeros([], device=img_patches.device)

        # =========================
        # 5) Final loss_align
        # =========================
        # per-sample combine for global/local
        loss_align_vec = beta * L_global_vec + (1.0 - beta) * L_local  # [B]
        loss_align = loss_align_vec.mean()

        loss_align = (
            loss_align
            + self.entropy_weight * reg_entropy
            + self.var_weight * reg_var
            + self.gate_entropy_weight * reg_gate
            + self.topo_weight * reg_topo
        )

        # For visualization: similarity_map = normalized patch_score to [0,1] by sigmoid on scaled cos
        # NOTE: patch_score already scaled; we map it for heatmap stability.
        sim_map = torch.sigmoid(patch_score / (scale + 1e-8))  # [B,N] in (0,1)

        alignment_info = {
            "loss_align": loss_align,
            "similarity_map": sim_map.detach(),     # [B,N]
            "weights": patch_weights.detach(),      # [B,N]
            "beta": beta.detach(),                  # [B]
            "reg_entropy": reg_entropy.detach(),
            "reg_var": reg_var.detach(),
            "reg_gate": reg_gate.detach(),
            "reg_topo": reg_topo.detach(),
        }

        # aligned patches output (for potential downstream / local visualization)
        img_patches_aligned = matched_txt_for_patch  # align to text manifold (one reasonable choice)

        if return_info:
            return img_patches_aligned, alignment_info
        else:
            return img_patches_aligned, {}