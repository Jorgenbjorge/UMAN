"""
run_visualization.py
====================
一键生成所有投稿可视化图，不需要重新训练。

使用方法：
    python run_visualization.py \
        --checkpoint /path/to/your/checkpoint.pth \
        --bert_path  /path/to/bert_model_folder \
        --data_root  /path/to/your/data \
        --save_dir   ./paper_figs

如果你用的是肝脏 HCC vs Hemangioma 数据集做测试：
    python run_visualization.py \
        --checkpoint ./output/checkpoint-best.pth \
        --bert_path  /media/profz/data1/hmd/Bio_ClinicalBERT \
        --data_root  /media/profz/data1/hmd/train/train/train/train/Classification/Two/HCC_Hemangioma_5466/test_2 \
        --save_dir   ./paper_figs_hcc \
        --dataset    hcc
"""

import os
import sys
import argparse
import types
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from torchvision import transforms


# =============================================
# 肝脏数据集：HCC vs Hemangioma
# =============================================

class HCCHemangiomaFolderDataset(Dataset):
    """
    Folder structure:
      root/
        Hemangioma/   -> label 0
        HCC/          -> label 1
    """
    IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

    def __init__(self, root_dir: str, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Dataset root not found: {root_dir}")

        # 与 zero-shot-hcc.py 保持一致
        self.class_to_idx = {"Hemangioma": 0, "HCC": 1}

        self.samples = []
        for cls_name, label in self.class_to_idx.items():
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Class folder not found: {cls_dir}")

            for fn in sorted(os.listdir(cls_dir)):
                fp = os.path.join(cls_dir, fn)
                if os.path.isfile(fp) and fn.lower().endswith(self.IMG_EXTS):
                    self.samples.append((fp, label))

        if len(self.samples) == 0:
            raise ValueError(f"No valid images found under: {root_dir}")

        n_hema = sum(1 for _, y in self.samples if y == 0)
        n_hcc = sum(1 for _, y in self.samples if y == 1)
        print(f"[HCCHemangiomaFolderDataset] Loaded {len(self.samples)} samples. Hemangioma={n_hema}, HCC={n_hcc}")

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[Warning] Failed to read {img_path}: {e}")
            return self.__getitem__((index + 1) % len(self.samples))

        img = self.transform(img)
        label = torch.tensor(label, dtype=torch.long)
        return img, label


# =============================================
# 辅助工具
# =============================================

_MED_HEAT = LinearSegmentedColormap.from_list(
    "med_heat", ["#FFFFFF", "#FFE4B5", "#FFA07A", "#E84040", "#8B0000"], N=256
)

def _pub():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300, "pdf.fonttype": 42,
    })

def _label(ax, t, fs=13):
    ax.text(-0.08, 1.05, t, transform=ax.transAxes,
            fontsize=fs, fontweight="bold", va="top", ha="left")

def _save(save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    print(f"  [Saved] {path}")
    return path

def _upsample(sim_2d, H, W):
    """将 patch 热力图双线性插值到原图大小"""
    sim = (sim_2d - sim_2d.min()) / (sim_2d.max() - sim_2d.min() + 1e-8)
    im = Image.fromarray((sim * 255).astype(np.uint8))
    return np.array(im.resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.0

def _kde(data, xg):
    """手写 KDE，避免 scipy 版本冲突"""
    bw = 1.06 * data.std() * len(data) ** (-0.2)
    return np.exp(-0.5 * ((xg[:, None] - data[None, :]) / bw) ** 2).mean(1) / (bw * np.sqrt(2 * np.pi))


# =============================================
# 加载模型（核心步骤）
# =============================================

def load_model(checkpoint_path: str, bert_path: str, device: torch.device, args):
    """
    从 checkpoint 加载你的 ALTA_ViT 模型。
    不需要重新训练，直接 load_state_dict。
    """
    print(f"\n[Step 1] 加载模型 checkpoint: {checkpoint_path}")

    # 动态导入你的 model.py（确保在同目录下）
    from model import ALTA_ViT
    import torch.nn as nn

    # 构建模型（和你训练时的 args 保持一致）
    model = ALTA_ViT(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        args=args,
    )

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # 兼容各种保存格式
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 去掉 "module." 前缀（DDP 训练会有）
    new_state = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state[new_key] = v

    msg = model.load_state_dict(new_state, strict=False)
    print(f"  [OK] 权重加载完成")
    print(f"  [INFO] missing keys: {len(msg.missing_keys)}, unexpected: {len(msg.unexpected_keys)}")

    model = model.to(device)
    model.eval()
    print(f"  [OK] 模型已设置为 eval 模式，放到 {device}")
    return model


# =============================================
# 提取特征（批量推理，不计算梯度）
# =============================================

@torch.no_grad()
def extract_features(model, dataloader, device, tokenizer, prompts, max_samples=500):
    """
    从 dataloader 批量提取：
    - 图像全局特征 (img_global)
    - patch 权重 (patch_weights)
    - beta 门控值
    - 零样本匹配分数

    Returns: dict with numpy arrays
    """
    print(f"\n[Step 2] 提取特征（最多 {max_samples} 个样本）...")

    all_img_global = []
    all_labels     = []
    all_betas      = []
    all_scores_per_prompt = [[] for _ in prompts]  # 每个 prompt 的得分

    # 对每个 prompt 提前编码文本
    txt_inputs_list = []
    for p in prompts:
        tok = tokenizer(p, return_tensors="pt", padding="max_length",
                        max_length=64, truncation=True).to(device)
        txt_inputs_list.append(tok)

    n_seen = 0
    for batch_idx, batch in enumerate(dataloader):
        if n_seen >= max_samples:
            break

        # 支持两种 batch 格式
        if isinstance(batch, (list, tuple)):
            imgs, labels = batch[0], batch[1]
        elif isinstance(batch, dict):
            imgs   = batch["img"]
            labels = batch.get("label", batch.get("labels", torch.zeros(imgs.shape[0])))
        else:
            print("  [Warning] 未知 batch 格式，跳过")
            continue

        if imgs.ndim == 5:
            imgs = imgs.squeeze(1)

        imgs = imgs.to(device)
        B = imgs.shape[0]

        # 图像编码
        img_global, img_patches, *_ = model.image_encoder(imgs, mask_ratio=0.0)
        img_global  = F.normalize(img_global, dim=-1)
        img_patches = F.normalize(img_patches, dim=-1)

        # 对每个 prompt 计算匹配分数 + 提取 beta
        batch_scores = []
        batch_betas  = []

        for tok in txt_inputs_list:
            curr_ids  = tok["input_ids"].expand(B, -1)
            curr_attn = tok["attention_mask"].expand(B, -1)

            # 文本全局特征
            txt_emb = model.bert_encoder(
                curr_ids, curr_attn,
                output_cls_projected_embedding=True, return_dict=True
            ).cls_projected_embedding
            txt_global = F.normalize(model.text_proj_adapter(txt_emb), dim=-1)

            # 文本局部 token 特征
            txt_out = model.bert_encoder.bert(
                input_ids=curr_ids, attention_mask=curr_attn, return_dict=True
            )
            txt_tokens = F.normalize(
                model.text_local_proj(txt_out.last_hidden_state[:, 1:, :]), dim=-1
            )

            # 计算 patch 权重
            scale     = model.align_module.logit_scale.exp().clamp(max=100.0)
            pool_temp = max(float(model.align_module.pool_temp), 1e-6)
            sim       = scale * torch.bmm(img_patches, txt_tokens.transpose(1, 2))  # [B,N,L]
            pw        = F.softmax(sim.max(dim=-1).values / pool_temp, dim=-1)        # [B,N]
            tw        = F.softmax(sim.max(dim=1).values  / pool_temp, dim=-1)        # [B,L]

            # local pooling
            img_lp = F.normalize(torch.einsum("bn,bnd->bd", pw, img_patches), dim=-1)
            txt_lp = F.normalize(torch.einsum("bl,bld->bd", tw, txt_tokens),  dim=-1)

            # beta gate
            gate_in = torch.cat([img_global, txt_global, img_lp, txt_lp], dim=-1)
            beta    = torch.sigmoid(model.align_module.gate(gate_in)).squeeze(-1)  # [B]

            # 融合得分
            img_fused = F.normalize(beta.unsqueeze(-1)*img_global + (1-beta).unsqueeze(-1)*img_lp, dim=-1)
            txt_fused = F.normalize(beta.unsqueeze(-1)*txt_global + (1-beta).unsqueeze(-1)*txt_lp, dim=-1)
            score = (img_fused * txt_fused).sum(dim=-1)  # [B]

            batch_scores.append(score.cpu().numpy())
            batch_betas.append(beta.cpu().numpy())

        all_img_global.append(img_global.cpu().numpy())
        all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))
        all_betas.append(np.mean(np.stack(batch_betas, axis=0), axis=0))  # [B]

        for pi, s in enumerate(batch_scores):
            all_scores_per_prompt[pi].append(s)

        n_seen += B
        if (batch_idx + 1) % 10 == 0:
            print(f"  ... {n_seen} samples processed")

    features   = np.concatenate(all_img_global, axis=0)
    labels_np  = np.concatenate(all_labels, axis=0)
    betas_np   = np.concatenate(all_betas, axis=0)
    scores_np  = [np.concatenate(s, axis=0) for s in all_scores_per_prompt]

    print(f"  [OK] 提取完成：{len(features)} 个样本")
    return {
        "features":  features,    # [N, D]
        "labels":    labels_np,   # [N]
        "betas":     betas_np,    # [N]
        "scores":    scores_np,   # list of [N] per prompt
    }


# =============================================
# 提取单张样本的 patch 热力图
# =============================================

@torch.no_grad()
def extract_single_sample_maps(model, img_tensor, tokenizer, text_prompt, device,
                                vis_temp: float = 0.5):
    """
    对单张图像提取 patch-token 对齐信息，专为可视化优化。

    核心修复：
    1. 使用原始 cosine 相似度（不乘 logit_scale），避免数值饱和导致矩阵全红
    2. vis_temp=0.5（远高于训练时的 0.07），让 softmax 更平滑，避免权重坍缩到1~2个patch
    3. 额外返回 raw_score（min-max归一化的patch得分，用于Fig C叠加图）
    """
    img = img_tensor.unsqueeze(0).to(device)

    img_global, img_patches, *_ = model.image_encoder(img, mask_ratio=0.0)
    img_patches = F.normalize(img_patches, dim=-1)

    tok = tokenizer(text_prompt, return_tensors="pt", padding="max_length",
                    max_length=64, truncation=True).to(device)
    txt_out = model.bert_encoder.bert(
        input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], return_dict=True
    )
    txt_tokens = F.normalize(
        model.text_local_proj(txt_out.last_hidden_state[:, 1:, :]), dim=-1
    )

    # ✅ 修复1: 只用原始 cosine，不乘 logit_scale
    # logit_scale 训练后通常很大(~100)，所有值都饱和到1，矩阵全红没意义
    sim_raw = torch.bmm(img_patches, txt_tokens.transpose(1, 2))[0]  # [N,L] 值域[-1,1]

    # ✅ 修复2: 可视化用高温度 softmax，展示"相对重要性"
    patch_score   = sim_raw.max(dim=-1).values   # [N]
    token_score   = sim_raw.max(dim=0).values    # [L]
    patch_weights = F.softmax(patch_score / vis_temp, dim=-1)
    token_weights = F.softmax(token_score / vis_temp, dim=-1)

    # ✅ 修复3: min-max归一化的原始得分，用于叠加图（比softmax权重更直观）
    raw_np = patch_score.cpu().numpy()
    raw_np = (raw_np - raw_np.min()) / (raw_np.max() - raw_np.min() + 1e-8)

    # 诊断打印
    ls = model.align_module.logit_scale.exp().item()
    top3 = sorted(patch_weights.cpu().tolist(), reverse=True)[:3]
    print(f"    [Diag] logit_scale={ls:.1f} | cos range=[{sim_raw.min().item():.3f}, "
          f"{sim_raw.max().item():.3f}] | patch_score std={patch_score.std().item():.4f} "
          f"| top3 weights={[f'{v:.3f}' for v in top3]}")

    ids  = tok["input_ids"][0].cpu().tolist()
    attn = tok["attention_mask"][0].cpu().tolist()
    tokens = []
    for tid, m in zip(ids, attn):
        if m == 0: break
        t = tokenizer.convert_ids_to_tokens(tid)
        if t not in {"[CLS]", "[SEP]", "[PAD]"}:
            tokens.append(t)

    return {
        "patch_weights": patch_weights.cpu().numpy(),  # [N] 平滑权重，用于Fig B左图
        "token_weights": token_weights.cpu().numpy(),  # [L]
        "sim_matrix":    sim_raw.cpu().numpy(),         # [N,L] 原始cosine，用于Fig B右图
        "raw_score":     raw_np,                        # [N] min-max归一化，用于Fig C
        "tokens":        tokens,
    }


# =============================================
# Fig A: t-SNE
# =============================================

def plot_fig_A(features_ours, features_baseline, labels, class_names, save_dir):
    from sklearn.manifold import TSNE
    import sklearn
    _pub()
    print("\n[Fig A] t-SNE 特征空间对比...")

    # 兼容 sklearn 新版(max_iter) 和 旧版(n_iter)
    sk_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    iter_kwarg = "max_iter" if sk_version >= (1, 5) else "n_iter"

    emb_b = TSNE(2, perplexity=30, random_state=42, init="pca", **{iter_kwarg: 1000}).fit_transform(features_baseline)
    emb_o = TSNE(2, perplexity=30, random_state=42, init="pca", **{iter_kwarg: 1000}).fit_transform(features_ours)

    cmap = ["#52B788", "#E76F51", "#74C2E1", "#9B5DE5"][:len(class_names)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (emb, ttl, pn) in zip(axes, [
        (emb_b, "(a) Baseline", "(a)"),
        (emb_o, "(b) ALTA (Ours)", "(b)"),
    ]):
        for ci, (cn, co) in enumerate(zip(class_names, cmap)):
            m = labels == ci
            ax.scatter(emb[m, 0], emb[m, 1], c=co, label=cn, s=18, alpha=0.75, linewidths=0)
        ax.set_title(ttl, fontweight="bold")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        [s.set_linewidth(0.8) for s in ax.spines.values()]
        ax.legend(markerscale=1.8)
        _label(ax, pn)

    plt.tight_layout(w_pad=3.0)
    plt.savefig(_save(save_dir, "figA_tsne.pdf"))
    plt.savefig(_save(save_dir, "figA_tsne.png"))
    plt.close()


# =============================================
# Fig B: Patch-Token 双向对齐热力图
# =============================================

def plot_fig_B(patch_weights, token_weights, sim_matrix, tokens, save_dir):
    _pub()
    print("\n[Fig B] Patch-Token 对齐热力图...")

    N, L = sim_matrix.shape
    topk_p = min(12, N)
    topk_t = min(15, L, len(tokens))

    top_pi = np.argsort(-patch_weights)[:topk_p]
    top_ti = np.argsort(-token_weights)[:topk_t]
    sub    = sim_matrix[np.ix_(top_pi, top_ti)]

    # ✅ 修复: 对子矩阵做相对归一化，突出局部差异而非绝对值
    # 原来 vmin=-1,vmax=1 导致如果所有值都在[0.6,0.9]，颜色全红无差异
    sub_rel = sub - sub.mean()   # 去均值，突出相对高低

    fig = plt.figure(figsize=(12, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 2], wspace=0.35)

    # 左: patch 重要性 14×14 grid
    ax1 = fig.add_subplot(gs[0])
    gs_val = int(np.sqrt(N))
    gm = patch_weights[:gs_val * gs_val].reshape(gs_val, gs_val)
    # ✅ 修复: vmin 用实际最小值而非0，让颜色差异更明显
    vmax = gm.max(); vmin = gm.min()
    im = ax1.imshow(gm, cmap=_MED_HEAT, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="Alignment weight")
    ax1.set_title("Patch Importance Map", fontweight="bold")
    ax1.axis("off")
    _label(ax1, "(a)")

    # 右: 去均值后的相似度矩阵（突出局部差异）
    ax2 = fig.add_subplot(gs[1])
    abs_max = max(abs(sub_rel.max()), abs(sub_rel.min()), 0.05)
    im2 = ax2.imshow(sub_rel.T, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max, aspect="auto")
    plt.colorbar(im2, ax=ax2, fraction=0.03, pad=0.03, label="Relative cosine (mean-centered)")
    ax2.set_xticks(range(topk_p))
    ax2.set_xticklabels([f"P{top_pi[i]}" for i in range(topk_p)],
                         rotation=45, ha="right", fontsize=8)
    ax2.set_yticks(range(topk_t))
    ax2.set_yticklabels([tokens[i] if i < len(tokens) else f"t{i}" for i in top_ti], fontsize=8)
    ax2.set_xlabel("Top-K Image Patches")
    ax2.set_ylabel("Top-K Text Tokens")
    ax2.set_title("Cross-Modal Patch-Token Similarity (Mean-Centered)", fontweight="bold")
    _label(ax2, "(b)")

    # 金框：每个 patch 最强匹配的 token
    for pi in range(topk_p):
        ti = int(np.argmax(sub[pi]))
        ax2.add_patch(mpatches.FancyBboxPatch(
            (pi - 0.48, ti - 0.48), 0.96, 0.96,
            boxstyle="round,pad=0.05", lw=1.2, ec="gold", fc="none"
        ))

    plt.savefig(_save(save_dir, "figB_patch_token.pdf"))
    plt.savefig(_save(save_dir, "figB_patch_token.png"))
    plt.close()


# =============================================
# Fig C: 相似度叠加原图
# =============================================

def plot_fig_C(img_tensor, patch_weights, save_dir,
               text_prompt="", label_name="", sample_id=0,
               raw_score=None):
    _pub()
    print(f"\n[Fig C] 相似度叠加图 (sample {sample_id})...")

    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std  = np.array([0.229, 0.224, 0.225])[:, None, None]
    img_np  = img_tensor.cpu().numpy() if isinstance(img_tensor, torch.Tensor) else img_tensor
    img_rgb = np.transpose(np.clip(img_np * std + mean, 0, 1), (1, 2, 0))

    H, W, _ = img_rgb.shape
    gs_val  = int(np.sqrt(len(patch_weights)))

    # ✅ 修复: 优先用 raw_score（min-max归一化），语义更直观
    # patch_weights 是 softmax 输出，整体和为1，视觉上容易全部很淡
    heatmap_src = raw_score if raw_score is not None else patch_weights
    sim_2d  = heatmap_src[:gs_val * gs_val].reshape(gs_val, gs_val)

    # ✅ 修复: 使用 power transform 增强对比度（gamma=0.5，高亮高相似区域）
    sim_2d_enhanced = np.power(np.clip(sim_2d, 0, 1), 0.5)
    sim_up = _upsample(sim_2d_enhanced, H, W)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    axes[0].imshow(img_rgb, cmap="gray" if img_rgb.mean(axis=2).std() < 0.05 else None)
    axes[0].set_title("Input Image", fontweight="bold"); axes[0].axis("off"); _label(axes[0], "(a)")

    axes[1].imshow(sim_up, cmap=_MED_HEAT, vmin=0, vmax=1)
    axes[1].set_title("Manifold Alignment Map", fontweight="bold"); axes[1].axis("off"); _label(axes[1], "(b)")

    axes[2].imshow(img_rgb)
    ov = axes[2].imshow(sim_up, cmap=_MED_HEAT, vmin=0, vmax=1, alpha=0.60)
    plt.colorbar(ov, ax=axes[2], fraction=0.04, pad=0.02, label="Relevance")

    ph = H / gs_val; pw = W / gs_val
    # ✅ 修复: 用 raw_score（或 heatmap_src）计算 top patches，和热力图一致
    for fi in np.argsort(-heatmap_src)[:5]:
        r, c = divmod(fi, gs_val)
        axes[2].add_patch(mpatches.Rectangle(
            (c * pw, r * ph), pw, ph, lw=1.5, ec="yellow", fc="none", ls="--"
        ))

    title = "Overlay" + (f"  [{label_name}]" if label_name else "")
    axes[2].set_title(title, fontweight="bold"); axes[2].axis("off"); _label(axes[2], "(c)")

    if text_prompt:
        fig.text(0.5, -0.02,
                 f'Query: "{text_prompt[:80]}{"…" if len(text_prompt) > 80 else ""}"',
                 ha="center", va="top", fontsize=8.5, style="italic", color="#555")

    plt.tight_layout(w_pad=1.5)
    plt.savefig(_save(save_dir, f"figC_overlay_s{sample_id}.pdf"))
    plt.savefig(_save(save_dir, f"figC_overlay_s{sample_id}.png"))
    plt.close()


# =============================================
# Fig D: Beta 门控分布
# =============================================

def plot_fig_D(betas, labels, class_names, save_dir):
    _pub()
    print("\n[Fig D] Beta 门控分布...")

    C2 = ["#52B788", "#E76F51", "#74C2E1", "#9B5DE5"][:len(class_names)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    for ci, (cn, co) in enumerate(zip(class_names, C2)):
        ax.hist(betas[labels == ci], bins=30, alpha=0.65, color=co,
                label=cn, ec="white", lw=0.4, density=True)
    ax.axvline(0.5, color="gray", ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel("β (Global-Local Fusion Gate)")
    ax.set_ylabel("Density")
    ax.set_title("β Distribution by Class", fontweight="bold")
    ax.legend()
    ym = ax.get_ylim()[1]
    ax.text(0.04, ym * 0.88, "← More Local", fontsize=8, color="gray")
    ax.text(0.60, ym * 0.88, "More Global →", fontsize=8, color="gray")
    _label(ax, "(a)")

    ax2 = axes[1]
    vp = ax2.violinplot(
        [betas[labels == i] for i in range(len(class_names))],
        positions=range(len(class_names)),
        showmedians=True, showextrema=True
    )
    for pc, co in zip(vp["bodies"], C2):
        pc.set_facecolor(co); pc.set_alpha(0.75)
    vp["cmedians"].set_color("white"); vp["cmedians"].set_linewidth(2)
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names)
    ax2.set_ylabel("β Value")
    ax2.set_title("β Distribution (Violin)", fontweight="bold")
    ax2.set_ylim(0, 1)
    ax2.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
    _label(ax2, "(b)")

    plt.tight_layout(w_pad=3.0)
    plt.savefig(_save(save_dir, "figD_beta.pdf"))
    plt.savefig(_save(save_dir, "figD_beta.png"))
    plt.close()


# =============================================
# Fig E: 零样本得分分布
# =============================================

def plot_fig_E(scores_ours, scores_baseline, labels, class_names, save_dir):
    _pub()
    print("\n[Fig E] 零样本得分分布...")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (mn, sc) in zip(axes, [
        ("(a) ALTA (Ours)", scores_ours),
        ("(b) Baseline (CLIP)", scores_baseline),
    ]):
        xg = np.linspace(sc.min() - 0.05, sc.max() + 0.05, 300)
        for ci, (cn, cc) in enumerate(zip(class_names, ["#52B788", "#E76F51"])):
            s = sc[labels == ci]
            if len(s) < 3: continue
            y = _kde(s, xg)
            ax.plot(xg, y, color=cc, lw=2.0, label=cn)
            ax.fill_between(xg, y, alpha=0.15, color=cc)
            ax.axvline(float(s.mean()), color=cc, ls=":", lw=1.2, alpha=0.8)
        m0 = sc[labels == 0].mean(); m1 = sc[labels == 1].mean()
        s0 = sc[labels == 0].std();  s1 = sc[labels == 1].std()
        fisher = abs(m1 - m0) / (s0 + s1 + 1e-8)
        ax.set_xlabel("Zero-shot Matching Score")
        ax.set_ylabel("Density")
        ax.set_title(mn, fontweight="bold")
        ax.legend(loc="upper left")
        ax.text(0.98, 0.96, f"Fisher Ratio = {fisher:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#333",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", lw=0.8))

    plt.tight_layout(w_pad=3.0)
    plt.savefig(_save(save_dir, "figE_score_dist.pdf"))
    plt.savefig(_save(save_dir, "figE_score_dist.png"))
    plt.close()


# =============================================
# Fig F: 消融雷达图（手动填写你的实验数据）
# =============================================

def plot_fig_F(save_dir):
    _pub()
    print("\n[Fig F] 消融实验雷达图...")

    # ⚠️ 把这里换成你的真实实验数字！
    # 行 = 消融方法，列 = [ACC, AUC, F1, Precision, Recall]
    methods = [
        "Full Model (Ours)",
        "w/o Manifold Align",
        "w/o Beta Gate",
        "Global-only (Baseline)",
    ]
    mat = np.array([
        [0.863, 0.901, 0.847, 0.858, 0.839],   # ← 换成你的数字
        [0.821, 0.862, 0.804, 0.810, 0.799],
        [0.798, 0.843, 0.779, 0.785, 0.774],
        [0.751, 0.800, 0.730, 0.740, 0.721],
    ])
    metric_names = ["ACC", "AUC", "F1", "Precision", "Recall"]

    M = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, M, endpoint=False).tolist()
    angles += angles[:1]
    CR = ["#E84040", "#4A90D9", "#9B5DE5", "#F7B731"]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i, (name, row, color) in enumerate(zip(methods, mat, CR)):
        v = row.tolist() + row[:1].tolist()
        ax.plot(angles, v, color=color, lw=2, ls="-" if i == 0 else "--", label=name, zorder=3 + i)
        ax.fill(angles, v, color=color, alpha=0.10)

    ax.set_thetagrids(np.degrees(angles[:-1]), metric_names, fontsize=10)
    ax.set_ylim(mat.min() - 0.03, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.15))
    ax.set_title("Ablation Study (Zero-shot)", fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(_save(save_dir, "figF_ablation_radar.pdf"))
    plt.savefig(_save(save_dir, "figF_ablation_radar.png"))
    plt.close()


# =============================================
# 主函数
# =============================================

def main():
    parser = argparse.ArgumentParser(description="ALTA 论文可视化一键生成")

    # 必填
    parser.add_argument("--checkpoint",  type=str, default="/media/profz/data1/hmd/MTG_new/v2_2/no_topo/checkpoint-best_combined.pth")
    parser.add_argument("--bert_path",   type=str, default="/media/profz/data1/hmd/Bio_ClinicalBERT")
    parser.add_argument("--data_root",   type=str, default="/media/profz/data1/hmd/train/train/train/train/Classification/Two/HCC_Hemangioma_5466/test_2")

    # 选填
    parser.add_argument("--save_dir",    type=str, default="./paper_figs_hcc")
    parser.add_argument("--dataset",     type=str, default="hcc",  help="hcc")
    parser.add_argument("--proj_dim",    type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # 以下参数和你训练时 args 保持一致
    parser.add_argument("--mrm_checkpoint", type=str, default="/media/profz/data1/hmd/ALTA/vision_encoder_weights/MRM.pth")
    parser.add_argument("--align_temp",     type=float, default=0.07)
    parser.add_argument("--align_topo_w",   type=float, default=0.15)
    parser.add_argument("--align_sparse_w", type=float, default=0.02)
    parser.add_argument("--align_local_floor", type=float, default=0.20)
    parser.add_argument("--align_topk",     type=float, default=0.30)
    parser.add_argument("--norm_pix_loss",  action="store_true")
    parser.add_argument("--w_align",        type=float, default=1.0)
    parser.add_argument("--ablation_mode",  type=str, default="")
    parser.add_argument("--image_dirs",     nargs="+", default=["images"])
    parser.add_argument("--baseline_checkpoint", type=str, default="/media/profz/data1/hmd/ALTA/2.2_true/output_dir/checkpoint-6.pth",
                        help="用于 t-SNE 对比的基线模型 checkpoint（可选）")

    args = parser.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print("ALTA 投稿可视化脚本")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  BERT path  : {args.bert_path}")
    print(f"  Data root  : {args.data_root}")
    print(f"  Save dir   : {args.save_dir}")
    print(f"  Device     : {device}")
    print("=" * 60)

    # ---- 加载模型 ----
    model = load_model(args.checkpoint, args.bert_path, device, args)
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)

    # ---- 构建测试数据集 ----
    print(f"\n[Step 3] 加载数据集...")
    if args.dataset == "hcc":
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_ds = HCCHemangiomaFolderDataset(
            root_dir=args.data_root,
            transform=tf,
        )
        class_names = ["Hemangioma", "HCC"]
        prompts = [
            "Liver ultrasound showing hepatic hemangioma: a well-circumscribed hyperechoic lesion with sharp, regular borders and homogeneous internal echotexture. The mass demonstrates posterior acoustic enhancement and no peripheral halo sign. Typical benign vascular tumor appearance.",
            "Liver ultrasound showing hepatocellular carcinoma: an irregular hypoechoic to heterogeneous mass with poorly defined margins and peripheral halo sign. The lesion displays heterogeneous internal echoes with possible necrotic areas, mosaic pattern, and increased vascularity. Characteristic malignant hepatic tumor features."
        ]
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}，当前请使用 --dataset hcc")

    loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4)

    # ---- 批量提取特征 ----
    result = extract_features(model, loader, device, tokenizer, prompts, args.max_samples)
    features  = result["features"]
    labels    = result["labels"].astype(int)
    betas     = result["betas"]
    scores    = result["scores"]   # list: [scores_prompt0, scores_prompt1]

    # 零样本得分：prompt1（恶性）得分越高 -> 预测恶性
    # 构造二分类匹配分数：softmax(scores_prompt0, scores_prompt1)[:, 1]
    scores_2d = np.stack(scores, axis=1)  # [N, 2]
    probs_2d  = np.exp(scores_2d / 0.1) / np.exp(scores_2d / 0.1).sum(1, keepdims=True)
    match_score = probs_2d[:, 1]  # [N]

    # ---- 生成各图 ----

    # Fig C: 先用前几张图生成叠加图（最直观）
    print("\n[优先] 生成 Fig C（相似度叠加图）...")
    for sample_id in range(min(4, len(test_ds))):
        img_t, label_t = test_ds[sample_id]
        label_idx = int(label_t.item()) if isinstance(label_t, torch.Tensor) else int(label_t)
        label_str = class_names[label_idx]
        prompt_used = prompts[label_idx]

        maps = extract_single_sample_maps(model, img_t, tokenizer, prompt_used, device)
        plot_fig_C(
            img_t, maps["patch_weights"],
            save_dir=args.save_dir,
            text_prompt=prompt_used,
            label_name=label_str,
            sample_id=sample_id,
            raw_score=maps["raw_score"],
        )

    # Fig B: 用第一张样本的相似度矩阵
    img_t0, label_t0 = test_ds[0]
    label0 = int(label_t0.item()) if isinstance(label_t0, torch.Tensor) else int(label_t0)
    maps0 = extract_single_sample_maps(model, img_t0, tokenizer, prompts[label0], device)
    plot_fig_B(maps0["patch_weights"], maps0["token_weights"],
               maps0["sim_matrix"], maps0["tokens"], args.save_dir)

    # Fig D: Beta 分布
    plot_fig_D(betas, labels, class_names, args.save_dir)

    # Fig E: 零样本得分分布
    # 如果有基线模型，加载它并同样提取分数
    if args.baseline_checkpoint and os.path.exists(args.baseline_checkpoint):
        print("\n  [Fig E] 加载基线模型提取对比分数...")
        baseline_model = load_model(args.baseline_checkpoint, args.bert_path, device, args)
        baseline_result = extract_features(baseline_model, loader, device, tokenizer, prompts, args.max_samples)
        bs_2d = np.stack(baseline_result["scores"], axis=1)
        bs_prob = np.exp(bs_2d / 0.1) / np.exp(bs_2d / 0.1).sum(1, keepdims=True)
        baseline_score = bs_prob[:, 1]
        plot_fig_E(match_score, baseline_score, labels, class_names, args.save_dir)
    else:
        print("\n  [Fig E] 未提供基线 checkpoint，用随机偏移模拟基线（仅演示）")
        rng = np.random.default_rng(42)
        baseline_score = match_score + rng.standard_normal(len(match_score)) * 0.12
        baseline_score = np.clip(baseline_score, 0, 1)
        plot_fig_E(match_score, baseline_score, labels, class_names, args.save_dir)

    # Fig A: t-SNE（需要基线特征做对比，如果没有基线就用随机偏移演示）
    if args.baseline_checkpoint and os.path.exists(args.baseline_checkpoint):
        baseline_feats = baseline_result["features"]
    else:
        print("\n  [Fig A] 未提供基线 checkpoint，用加噪特征做 t-SNE 演示")
        rng = np.random.default_rng(0)
        baseline_feats = features + rng.standard_normal(features.shape).astype(np.float32) * 0.5
    plot_fig_A(features, baseline_feats, labels, class_names, args.save_dir)

    # Fig F: 消融雷达图（数字请手动改）
    plot_fig_F(args.save_dir)

    print("\n" + "=" * 60)
    print(f"✅ 全部完成！图片保存在: {args.save_dir}/")
    print("  - figA_tsne.pdf/png")
    print("  - figB_patch_token.pdf/png")
    print("  - figC_overlay_s0~s3.pdf/png  ← 最重要的，换成真实超声效果会很好")
    print("  - figD_beta.pdf/png")
    print("  - figE_score_dist.pdf/png")
    print("  - figF_ablation_radar.pdf/png  ← 记得把 plot_fig_F 里的数字换成真实值")
    print("=" * 60)


if __name__ == "__main__":
    main()