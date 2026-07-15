from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# -----------------------------
# Plot style helpers
# -----------------------------

def _set_pub_style() -> None:
    """A clean, journal-friendly matplotlib style (no seaborn dependency)."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "axes.grid": False,
        "savefig.dpi": 300,
    })


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def _safe_sqrt_grid(n: int) -> Tuple[int, int]:
    """Try to reshape N patches into a near-square grid."""
    s = int(np.sqrt(n))
    if s * s == n:
        return s, s
    # Fallback to a rectangle close to square
    for h in range(s, 0, -1):
        if n % h == 0:
            return h, n // h
    return 1, n


def _denormalize_image(img_chw: np.ndarray) -> np.ndarray:
    """Best-effort denormalization for visualization.

    - If 3-channel, assume ImageNet normalization.
    - If 1-channel, min-max normalize.
    """
    if img_chw.ndim != 3:
        raise ValueError(f"Expected CHW image, got shape {img_chw.shape}")

    c, h, w = img_chw.shape
    if c == 3:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        x = img_chw * std + mean
        x = np.clip(x, 0.0, 1.0)
        return np.transpose(x, (1, 2, 0))
    # Single-channel or other: normalize to [0,1]
    x = img_chw[0]
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x


def _try_load_tokenizer(bert_path: str):
    """Load tokenizer without internet; returns None if unavailable."""
    try:
        from transformers import AutoTokenizer  # type: ignore
        return AutoTokenizer.from_pretrained(bert_path, local_files_only=True)
    except Exception:
        return None


def _decode_tokens(tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[str]:
    """Decode token strings (without special tokens) for a single sample."""
    ids = input_ids.detach().cpu().tolist()
    am = attention_mask.detach().cpu().tolist()
    out: List[str] = []
    for tid, m in zip(ids, am):
        if m == 0:
            break
        tok = tokenizer.convert_ids_to_tokens(tid)
        if tok in {"[CLS]", "[SEP]", "[PAD]"}:
            continue
        out.append(tok)
    return out


# -----------------------------
# Core visualization outputs
# -----------------------------


@dataclass
class AlignmentMaps:
    """A compact container for alignment fields."""
    similarity_map: np.ndarray  # [N]
    weights_map: np.ndarray     # [N]
    manifold_distance: Optional[np.ndarray] = None  # [N]
    token_relevance: Optional[np.ndarray] = None    # [L] optional


class ALTAVisualizer:
    """ALTA model visualization tool.

    This class is safe to import from training. It does NOT assume the existence
    of CLIP/global/local losses; it only needs your model to provide:
      - target_model.image_encoder(img, mask_ratio=0.0) -> (img_global, img_local, ...)
      - target_model.bert_encoder.bert(...) -> last_hidden_state
      - target_model.text_local_proj(...) (projection to D)
      - target_model.manifold_align(img_local, txt_tokens) -> (aligned, alignment_info)

    If `adaptive_fusion` exists, we can still visualize its scalar gates.
    Otherwise, the same API produces an alignment-statistics overview.
    """

    def __init__(self, model, device, save_dir: str = "./visualizations"):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        _mkdir(save_dir)
        _set_pub_style()

    # -----------------------------
    # Public, training-compatible API
    # -----------------------------

    @torch.no_grad()
    def visualize_fusion_weights(
        self,
        dataloader,
        max_samples: int = 200,
        organ_labels=None,
        epoch: int = 0,
    ) -> str:
        """Epoch-level summary figure.

        Compatibility note:
        - Your engine_pretrain.py calls this every 10 epochs.
        - If your model still has `adaptive_fusion`, we plot α distributions.
        - If not, we fall back to a manifold-alignment statistics overview.
        """
        target_model = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(target_model, "adaptive_fusion"):
            return self._visualize_adaptive_fusion_gates(dataloader, max_samples, organ_labels, epoch)
        return self.visualize_alignment_overview(dataloader, max_samples=max_samples, epoch=epoch)

    @torch.no_grad()
    def visualize_alignment_overview(self, dataloader, max_samples: int = 200, epoch: int = 0) -> str:
        """A robust, explainable summary of the manifold alignment behavior.

        Figure panels (TMI-friendly):
        (a) similarity distribution across patches (mean per sample)
        (b) weight entropy distribution (how selective the alignment is)
        (c) correlation: mean similarity vs. weight entropy
        (d) histogram of per-patch weights (aggregated)
        """
        self.model.eval()
        target_model = self.model.module if hasattr(self.model, "module") else self.model

        sim_means: List[float] = []
        sim_stds: List[float] = []
        entropies: List[float] = []
        all_weights: List[np.ndarray] = []

        n_seen = 0
        for batch_idx, batch in enumerate(dataloader):
            if n_seen >= max_samples:
                break
            if not isinstance(batch, dict) or "img" not in batch:
                continue

            img = batch["img"].to(self.device)
            txt_inputs = batch["ids"].to(self.device)
            txt_attention_mask = batch["attention_mask"].to(self.device)

            if img.ndim == 5:
                img = img.squeeze(1)
            if txt_inputs.ndim == 3:
                txt_inputs = txt_inputs.squeeze(1)
                txt_attention_mask = txt_attention_mask.squeeze(1)

            # Use mask_ratio=0 to ensure all patches are visible for interpretation
            _, img_local, _, _, _ = target_model.image_encoder(img, mask_ratio=0.0)
            txt_output = target_model.bert_encoder.bert(
                input_ids=txt_inputs,
                attention_mask=txt_attention_mask,
                return_dict=True,
            )
            txt_hidden = txt_output.last_hidden_state
            txt_tokens = target_model.text_local_proj(txt_hidden[:, 1:, :])
            txt_tokens = F.normalize(txt_tokens, dim=-1)

            _, alignment_info = target_model.manifold_align(img_local, txt_tokens)
            sim_map = alignment_info.get("similarity_map", None)
            weights = alignment_info.get("weights", None)
            if sim_map is None or weights is None:
                continue

            sim_map = sim_map.detach()
            weights = weights.detach()

            # per-sample statistics
            sim_means.extend(sim_map.mean(dim=1).cpu().tolist())
            sim_stds.extend(sim_map.std(dim=1).cpu().tolist())

            # entropy of normalized weights
            w = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            ent = -(w * torch.log(w + 1e-8)).sum(dim=1)
            entropies.extend(ent.cpu().tolist())
            all_weights.append(_to_numpy(weights).reshape(-1))

            n_seen += int(img.shape[0])

        if len(sim_means) == 0:
            # Fail gracefully (still return a path)
            save_path = os.path.join(self.save_dir, f"fusion_weights_epoch{epoch}.png")
            fig = plt.figure(figsize=(8, 3))
            plt.text(0.5, 0.5, "No alignment_info collected for visualization.", ha="center", va="center")
            plt.axis("off")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
            self.model.train()
            return save_path

        sim_means_np = np.asarray(sim_means)
        sim_stds_np = np.asarray(sim_stds)
        ent_np = np.asarray(entropies)
        weights_np = np.concatenate(all_weights, axis=0)

        fig = plt.figure(figsize=(14, 3.8))

        # (a) similarity mean distribution
        ax1 = plt.subplot(1, 4, 1)
        ax1.hist(sim_means_np, bins=30, edgecolor="black", linewidth=0.6)
        ax1.set_title("(a) Patch Similarity (mean/sample)")
        ax1.set_xlabel("mean(similarity_map)")
        ax1.set_ylabel("Count")
        ax1.grid(True, alpha=0.25)

        # (b) weight entropy distribution
        ax2 = plt.subplot(1, 4, 2)
        ax2.hist(ent_np, bins=30, edgecolor="black", linewidth=0.6)
        ax2.set_title("(b) Alignment Selectivity (entropy)")
        ax2.set_xlabel("H(weights)")
        ax2.set_ylabel("Count")
        ax2.grid(True, alpha=0.25)

        # (c) correlation scatter
        ax3 = plt.subplot(1, 4, 3)
        ax3.scatter(sim_means_np, ent_np, s=12, alpha=0.55, edgecolors="none")
        # Trend line
        if len(sim_means_np) >= 2:
            z = np.polyfit(sim_means_np, ent_np, 1)
            p = np.poly1d(z)
            xs = np.linspace(sim_means_np.min(), sim_means_np.max(), 50)
            ax3.plot(xs, p(xs), linestyle="--", linewidth=2)
        ax3.set_title("(c) Similarity vs. Selectivity")
        ax3.set_xlabel("mean(similarity_map)")
        ax3.set_ylabel("H(weights)")
        ax3.grid(True, alpha=0.25)

        # (d) per-patch weight distribution (aggregated)
        ax4 = plt.subplot(1, 4, 4)
        ax4.hist(weights_np, bins=40, edgecolor="black", linewidth=0.6)
        ax4.set_title("(d) Per-patch Alignment Weight")
        ax4.set_xlabel("weights")
        ax4.set_ylabel("Count")
        ax4.grid(True, alpha=0.25)

        fig.suptitle(f"Manifold Alignment Overview (Epoch {epoch})", y=1.02)
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, f"fusion_weights_epoch{epoch}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        self.model.train()
        return save_path

    @torch.no_grad()
    def visualize_single_alignment_sample(
        self,
        img: torch.Tensor,
        txt_inputs: torch.Tensor,
        txt_attention_mask: torch.Tensor,
        epoch: int,
        iteration: int,
        save_prefix: str = "alignment",
        topk_patches: float = 0.30,
        decode_tokens: bool = True,
    ) -> Dict[str, str]:
        """Save an interpretable, publication-ready alignment explanation for ONE sample.

        Outputs (2 files by default):
          1) {save_prefix}_maps_epoch{epoch:03d}_iter{iteration:04d}.png
             - Original image
             - Patch-word similarity heatmap
             - Alignment weights heatmap
             - Weight overlay
             - Top-K patch boxes (salient patches)

          2) {save_prefix}_tokens_epoch{epoch:03d}_iter{iteration:04d}.png
             - Top tokens by relevance (weight-aware coverage)
        """
        self.model.eval()
        target_model = self.model.module if hasattr(self.model, "module") else self.model

        # Normalize shapes
        if img.ndim == 5:
            img = img.squeeze(1)
        if img.ndim == 3:
            img = img.unsqueeze(0)
        if txt_inputs.ndim == 3:
            txt_inputs = txt_inputs.squeeze(1)
            txt_attention_mask = txt_attention_mask.squeeze(1)

        img = img.to(self.device)
        txt_inputs = txt_inputs.to(self.device)
        txt_attention_mask = txt_attention_mask.to(self.device)

        # Forward
        _, img_local, _, _, _ = target_model.image_encoder(img, mask_ratio=0.0)
        txt_output = target_model.bert_encoder.bert(
            input_ids=txt_inputs,
            attention_mask=txt_attention_mask,
            return_dict=True,
        )
        txt_hidden = txt_output.last_hidden_state
        txt_tokens = target_model.text_local_proj(txt_hidden[:, 1:, :])
        txt_tokens = F.normalize(txt_tokens, dim=-1)

        _, alignment_info = target_model.manifold_align(img_local, txt_tokens)
        sim_map_t = alignment_info.get("similarity_map")
        weights_t = alignment_info.get("weights")
        dist_t = alignment_info.get("manifold_distance", None)
        if sim_map_t is None or weights_t is None:
            raise RuntimeError("alignment_info missing similarity_map/weights")

        # [1, N] -> [N]
        sim_map = _to_numpy(sim_map_t[0])
        weights = _to_numpy(weights_t[0])
        dist = _to_numpy(dist_t[0]) if dist_t is not None else None

        n_patches = sim_map.shape[0]
        gh, gw = _safe_sqrt_grid(n_patches)
        sim_2d = sim_map.reshape(gh, gw)
        w_2d = weights.reshape(gh, gw)

        # Prepare image for display
        img_chw = _to_numpy(img[0])
        img_disp = _denormalize_image(img_chw)
        is_gray = (isinstance(img_disp, np.ndarray) and img_disp.ndim == 2)

        # Top-K patches by weight
        k = max(1, int(n_patches * float(topk_patches)))
        top_idx = np.argsort(-weights)[:k]
        top_mask = np.zeros(n_patches, dtype=np.float32)
        top_mask[top_idx] = 1.0
        top_mask_2d = top_mask.reshape(gh, gw)

        out_paths: Dict[str, str] = {}

        # ----------------
        # Figure 1: Maps
        # ----------------
        fig = plt.figure(figsize=(16, 4.5))

        ax1 = plt.subplot(1, 5, 1)
        ax1.imshow(img_disp, cmap="gray" if is_gray else None)
        ax1.set_title("(a) Image")
        ax1.axis("off")

        ax2 = plt.subplot(1, 5, 2)
        im2 = ax2.imshow(sim_2d, cmap="magma", interpolation="bilinear")
        ax2.set_title(f"(b) Patch-Token Similarity\nmean={sim_map.mean():.3f}, std={sim_map.std():.3f}")
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = plt.subplot(1, 5, 3)
        im3 = ax3.imshow(w_2d, cmap="viridis", interpolation="bilinear", vmin=0.0, vmax=1.0)
        ax3.set_title(f"(c) Alignment Weights\nmean={weights.mean():.3f}, std={weights.std():.3f}")
        ax3.axis("off")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        ax4 = plt.subplot(1, 5, 4)
        overlay = self._overlay_heatmap_on_image(img_disp, w_2d)
        ax4.imshow(overlay, cmap="gray" if overlay.ndim == 2 else None)
        ax4.set_title("(d) Weight Overlay")
        ax4.axis("off")

        ax5 = plt.subplot(1, 5, 5)
        ax5.imshow(img_disp, cmap="gray" if is_gray else None)
        ax5.set_title(f"(e) Top-{int(topk_patches*100)}% Patches")
        ax5.axis("off")
        self._draw_patch_grid_boxes(ax5, img_disp, top_mask_2d, edge_alpha=0.65)

        fig.suptitle(f"Manifold Alignment Explanation (Epoch {epoch}, Iter {iteration})", y=1.02)
        plt.tight_layout()
        save_maps = os.path.join(self.save_dir, f"{save_prefix}_maps_epoch{epoch:03d}_iter{iteration:04d}.png")
        plt.savefig(save_maps, dpi=300, bbox_inches="tight")
        plt.close(fig)
        out_paths["maps"] = save_maps

        # ----------------
        # Figure 2: Token relevance (weight-aware)
        # ----------------
        token_fig_path = self._save_token_relevance_figure(
            target_model=target_model,
            txt_inputs=txt_inputs[0],
            txt_attention_mask=txt_attention_mask[0],
            img_patches=img_local[0],
            txt_tokens=txt_tokens[0],
            patch_weights=weights_t[0],
            epoch=epoch,
            iteration=iteration,
            save_prefix=save_prefix,
            decode_tokens=decode_tokens,
        )
        out_paths["tokens"] = token_fig_path

        # Optional: manifold distance map figure (if present)
        if dist is not None:
            dist_2d = dist.reshape(gh, gw)
            figd = plt.figure(figsize=(6.2, 5.2))
            axd = plt.gca()
            imd = axd.imshow(dist_2d, cmap="cividis", interpolation="bilinear")
            axd.set_title(f"Manifold Distance (Epoch {epoch}, Iter {iteration})\nmean={dist.mean():.3f}")
            axd.axis("off")
            plt.colorbar(imd, ax=axd, fraction=0.046, pad=0.04)
            save_dist = os.path.join(self.save_dir, f"{save_prefix}_dist_epoch{epoch:03d}_iter{iteration:04d}.png")
            plt.tight_layout()
            plt.savefig(save_dist, dpi=300, bbox_inches="tight")
            plt.close(figd)
            out_paths["distance"] = save_dist

        self.model.train()
        return out_paths

    # -----------------------------
    # Internal helpers
    # -----------------------------

    def _overlay_heatmap_on_image(self, img: np.ndarray, heat_2d: np.ndarray) -> np.ndarray:
        """Overlay a low-res heatmap (patch grid) onto the image."""
        h = heat_2d.astype(np.float32)
        h = (h - h.min()) / (h.max() - h.min() + 1e-8)

        if img.ndim == 2:
            H, W = img.shape
        else:
            H, W = img.shape[:2]
        gh, gw = heat_2d.shape
        scale_h = max(1, H // gh)
        scale_w = max(1, W // gw)
        up = np.kron(h, np.ones((scale_h, scale_w), dtype=np.float32))
        up = up[:H, :W]

        cmap = plt.get_cmap("jet")
        colored = cmap(up)[..., :3]

        if img.ndim == 2:
            base = np.stack([img, img, img], axis=-1)
        else:
            base = img

        alpha = 0.50
        out = np.clip((1 - alpha) * base + alpha * colored, 0.0, 1.0)
        return out

    def _draw_patch_grid_boxes(
        self,
        ax,
        img: np.ndarray,
        mask_2d: np.ndarray,
        edge_alpha: float = 0.7,
    ) -> None:
        """Draw rectangles for selected patches on an axis."""
        gh, gw = mask_2d.shape
        if img.ndim == 2:
            H, W = img.shape
        else:
            H, W = img.shape[:2]
        ph = H / gh
        pw = W / gw

        for r in range(gh):
            for c in range(gw):
                if mask_2d[r, c] <= 0:
                    continue
                rect = Rectangle(
                    (c * pw, r * ph),
                    pw,
                    ph,
                    fill=False,
                    linewidth=1.2,
                    edgecolor=(1.0, 0.2, 0.2, edge_alpha),
                )
                ax.add_patch(rect)

    def _save_token_relevance_figure(
        self,
        target_model,
        txt_inputs: torch.Tensor,
        txt_attention_mask: torch.Tensor,
        img_patches: torch.Tensor,
        txt_tokens: torch.Tensor,
        patch_weights: torch.Tensor,
        epoch: int,
        iteration: int,
        save_prefix: str,
        decode_tokens: bool,
        topk_tokens: int = 12,
    ) -> str:
        """Token-level interpretability: which words drive patch alignment.

        We compute a genuine patch-token similarity matrix:
            S[p, t] = cosine(patch_p, token_t)

        Then define token relevance as a *weight-aware* coverage score:
            r(t) = sum_p w(p) * softplus( S[p,t] )
        where w(p) are the alignment weights from the manifold module.

        This makes the plot interpretable in your setting:
        - patches selected by the manifold alignment contribute more;
        - tokens that explain those selected patches rise to the top.
        """

        token_strings: Optional[List[str]] = None
        if decode_tokens:
            args = getattr(target_model, "args", None)
            bert_path = getattr(args, "bert_path", None)
            if bert_path is not None:
                tok = _try_load_tokenizer(bert_path)
                if tok is not None:
                    token_strings = _decode_tokens(tok, txt_inputs, txt_attention_mask)

        L_in = int((txt_attention_mask > 0).sum().item())
        L_eff = max(1, L_in - 2)

        patches = F.normalize(img_patches.detach(), dim=-1)           # [N, D]
        tokens = F.normalize(txt_tokens[:L_eff].detach(), dim=-1)     # [L, D]
        w = patch_weights.detach().float()
        w = w / (w.sum() + 1e-8)
        sim = patches @ tokens.transpose(0, 1)                        # [N, L]
        rel = (w.unsqueeze(-1) * F.softplus(sim)).sum(dim=0)          # [L]
        rel = rel / (rel.sum() + 1e-8)
        relevance = rel.cpu().numpy()

        k = min(topk_tokens, len(relevance))
        idx = np.argsort(-relevance)[:k]
        vals = relevance[idx]

        if token_strings is not None and len(token_strings) > 0:
            token_strings = token_strings[:L_eff]
            labels = [token_strings[i] if i < len(token_strings) else f"t{i}" for i in idx]
        else:
            labels = [f"t{i}" for i in idx]

        fig = plt.figure(figsize=(8.2, 3.8))
        ax = plt.gca()
        y = np.arange(k)
        ax.barh(y, vals)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Token relevance (weight-aware)")
        ax.set_title("Token-level Guidance for Alignment")
        ax.grid(True, axis="x", alpha=0.25)

        fig.suptitle(f"Epoch {epoch}, Iter {iteration}", y=0.98)
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"{save_prefix}_tokens_epoch{epoch:03d}_iter{iteration:04d}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return save_path

    @torch.no_grad()
    def _visualize_adaptive_fusion_gates(self, dataloader, max_samples: int, organ_labels, epoch: int) -> str:
        """Original α-gate visualization (kept for backward compatibility)."""
        self.model.eval()
        target_model = self.model.module if hasattr(self.model, "module") else self.model

        alphas_img: List[float] = []
        alphas_txt: List[float] = []

        n_seen = 0
        for batch in dataloader:
            if n_seen >= max_samples:
                break
            if not isinstance(batch, dict) or "img" not in batch:
                continue

            img = batch["img"].to(self.device)
            txt_inputs = batch["ids"].to(self.device)
            txt_attention_mask = batch["attention_mask"].to(self.device)

            if img.ndim == 5:
                img = img.squeeze(1)
            if txt_inputs.ndim == 3:
                txt_inputs = txt_inputs.squeeze(1)
                txt_attention_mask = txt_attention_mask.squeeze(1)

            img_global, img_local, _, _, _ = target_model.image_encoder(img, mask_ratio=0.0)

            txt_embed = target_model.bert_encoder(
                txt_inputs,
                txt_attention_mask,
                output_cls_projected_embedding=True,
                return_dict=True,
            ).cls_projected_embedding
            txt_embed = target_model.text_proj_adapter(txt_embed) if hasattr(target_model, "text_proj_adapter") else txt_embed
            txt_global = F.normalize(txt_embed, dim=-1)

            txt_output = target_model.bert_encoder.bert(
                input_ids=txt_inputs,
                attention_mask=txt_attention_mask,
                return_dict=True,
            )
            txt_tokens = target_model.text_local_proj(txt_output.last_hidden_state[:, 1:, :])
            txt_tokens = F.normalize(txt_tokens, dim=-1)

            img_local_pooled = img_local.mean(dim=1)
            _, alpha_i = target_model.adaptive_fusion(img_global, img_local_pooled)

            txt_local_pooled = txt_tokens.mean(dim=1)
            _, alpha_t = target_model.adaptive_fusion(txt_global, txt_local_pooled)

            alphas_img.extend(alpha_i.squeeze(-1).detach().cpu().tolist())
            alphas_txt.extend(alpha_t.squeeze(-1).detach().cpu().tolist())
            n_seen += int(img.shape[0])

        fig = plt.figure(figsize=(10.5, 3.8))
        ax1 = plt.subplot(1, 2, 1)
        ax1.hist(alphas_img, bins=30, alpha=0.7, edgecolor="black", linewidth=0.6)
        ax1.set_title("(a) Image Fusion Gate α")
        ax1.set_xlabel("α")
        ax1.set_ylabel("Count")
        ax1.grid(True, alpha=0.25)

        ax2 = plt.subplot(1, 2, 2)
        ax2.hist(alphas_txt, bins=30, alpha=0.7, edgecolor="black", linewidth=0.6)
        ax2.set_title("(b) Text Fusion Gate α")
        ax2.set_xlabel("α")
        ax2.set_ylabel("Count")
        ax2.grid(True, alpha=0.25)

        fig.suptitle(f"Adaptive Global-Local Fusion Gates (Epoch {epoch})", y=1.02)
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"fusion_weights_epoch{epoch}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        self.model.train()
        return save_path


# -----------------------------
# Optional quick test
# -----------------------------

def _quick_smoke_test() -> None:
    """A tiny smoke test that only checks plotting utilities (no model)."""
    _set_pub_style()
    fig = plt.figure(figsize=(4, 2))
    ax = plt.gca()
    x = np.random.randn(200)
    ax.hist(x, bins=30, edgecolor="black", linewidth=0.6)
    ax.set_title("Smoke Test")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _mkdir("./_vis_test")
    p = "./_vis_test/smoke.png"
    plt.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[visualization_utils] Smoke test saved: {p}")


if __name__ == "__main__":
    _quick_smoke_test()
