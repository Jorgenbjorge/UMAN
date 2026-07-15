import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed # type: ignore
from util.pos_embed import get_2d_sincos_pos_embed
from adapter_block import Block
from masking_utils import random_masking, patchify
from cxrbert import CXRBertModel
from loss_functions import MLMLoss
from vision_decoder import VisionDecoder
from manifold_alignment_simple import SoftGlobalLocalManifoldAlignment


class ImageProjectionHead(nn.Module):
    def __init__(self, input_dim, proj_dim) -> None:
        super().__init__()
        self.dense_to_hidden = nn.Linear(input_dim, proj_dim)
        self.transform_act_fn = nn.functional.gelu
        self.LayerNorm = nn.LayerNorm(proj_dim, eps=1e-12)
        self.dense_to_output = nn.Linear(proj_dim, proj_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_to_hidden(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dense_to_output(hidden_states)
        return hidden_states


class ImageEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., norm_layer=nn.LayerNorm, args=None):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=norm_layer
            )
            for _ in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.projection_head = ImageProjectionHead(embed_dim, args.proj_dim)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # 域适应 BN（你原来就有）
        self.domain_norm = nn.BatchNorm1d(embed_dim)

    def forward_feature(self, x, mask_ratio=0.0):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        if mask_ratio > 0:
            x, mask, ids_restore = random_masking(x, mask_ratio)
        else:
            mask = None
            ids_restore = None

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # local features 做域归一
        x[:, 1:, :] = self.domain_norm(x[:, 1:, :].transpose(1, 2)).transpose(1, 2)

        return x, mask, ids_restore

    def forward(self, img_stack, mask_ratio=0.0):
        img = img_stack
        if len(img.shape) == 5:
            B, N_views, C, H, W = img.shape
            img = img.reshape(B * N_views, C, H, W)
        else:
            B, C, H, W = img.shape

        x, mask, ids_restore = self.forward_feature(img, mask_ratio)

        # global
        global_feature = x[:, 0, :]
        global_feature = self.projection_head(global_feature)
        global_feature = F.normalize(global_feature, dim=-1)

        # local
        local_features = x[:, 1:, :]
        B_local, N_local, D_local = local_features.shape
        local_features_flat = local_features.reshape(B_local * N_local, D_local)
        local_features_proj = self.projection_head(local_features_flat)
        local_features = local_features_proj.reshape(B_local, N_local, -1)
        local_features = F.normalize(local_features, dim=-1)

        return global_feature, local_features, x, mask, ids_restore

class _LocalOnlyManifoldAlignWrapper(torch.nn.Module):
    """
    兼容旧 zero-shot 评估接口：model.manifold_align(img_local, txt_tokens)
    - 若没有提供 global，就用 mean pooling 自动构造 global
    - 内部仍调用新的 align_module（全局/局部软结合 + 流形对齐）
    """
    def __init__(self, align_module):
        super().__init__()
        self.align_module = align_module

    def forward(self, img_patches, txt_tokens, img_global=None, txt_global=None, return_info=True):
        # img_patches: [B, N, D], txt_tokens: [B, L, D]
        if img_global is None:
            img_global = F.normalize(img_patches.mean(dim=1), dim=-1)   # [B, D]
        if txt_global is None:
            txt_global = F.normalize(txt_tokens.mean(dim=1), dim=-1)    # [B, D]

        # 走你新的对齐模块
        aligned_patches, alignment_info = self.align_module(
            img_global=img_global,
            img_patches=img_patches,
            txt_global=txt_global,
            txt_tokens=txt_tokens,
            return_info=return_info
        )
        return aligned_patches, alignment_info
    @torch.no_grad()
    def score_pairs(self, img_global, img_patches, txt_global, txt_tokens):
        """
        给 zero-shot 用：返回每个样本与该 prompt 的匹配分数 [B]
        逻辑与 align_module 内部一致：patch/token soft pooling -> gate(beta) -> fused cosine
        """
        # normalize
        img_global = F.normalize(img_global, dim=-1)
        txt_global = F.normalize(txt_global, dim=-1)
        img_patches = F.normalize(img_patches, dim=-1)
        txt_tokens = F.normalize(txt_tokens, dim=-1)

        # scale & similarity
        scale = self.align_module.logit_scale.exp().clamp(max=100.0)
        sim = scale * torch.bmm(img_patches, txt_tokens.transpose(1, 2))  # [B, N, L]

        pool_temp = getattr(self.align_module, "pool_temp", 0.07)
        pool_temp = max(float(pool_temp), 1e-6)

        # patch weights (soft top-k prior)
        patch_score = sim.max(dim=-1).values                # [B, N]
        patch_weights = F.softmax(patch_score / pool_temp, dim=-1)

        # token weights
        token_score = sim.max(dim=1).values                 # [B, L]
        token_weights = F.softmax(token_score / pool_temp, dim=-1)

        # local pooling
        img_local_pool = torch.einsum("bn,bnd->bd", patch_weights, img_patches)  # [B, D]
        txt_local_pool = torch.einsum("bl,bld->bd", token_weights, txt_tokens)  # [B, D]
        img_local_pool = F.normalize(img_local_pool, dim=-1)
        txt_local_pool = F.normalize(txt_local_pool, dim=-1)

        # gate beta (global-local soft fusion)
        gate_in = torch.cat([img_global, txt_global, img_local_pool, txt_local_pool], dim=-1)  # [B, 4D]
        beta = torch.sigmoid(self.align_module.gate(gate_in)).squeeze(-1)  # [B]

        img_fused = F.normalize(beta.unsqueeze(-1) * img_global + (1.0 - beta).unsqueeze(-1) * img_local_pool, dim=-1)
        txt_fused = F.normalize(beta.unsqueeze(-1) * txt_global + (1.0 - beta).unsqueeze(-1) * txt_local_pool, dim=-1)

        # cosine score
        score = (img_fused * txt_fused).sum(dim=-1)  # [B]
        return score

class ALTA_ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, decoder_embed_dim=512, decoder_depth=8,
                 decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, args=None):
        super().__init__()

        self.args = args
        self.image_encoder = ImageEncoder(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, norm_layer=norm_layer, args=args
        )

        # ====== 视觉预训练权重加载（保留你原逻辑）======
        mrm_checkpoint_path = getattr(args, 'mae_path', '/media/profz/data1/hmd/ALTA/vision_encoder_weights/MRM.pth')
        if os.path.exists(mrm_checkpoint_path):
            print(f"[Model] Loading MRM pretrained weights from: {mrm_checkpoint_path}")
            try:
                checkpoint = torch.load(mrm_checkpoint_path, map_location='cpu', weights_only=False)
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

                encoder_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.image_encoder.'):
                        new_k = k.replace('module.image_encoder.', '')
                        encoder_state_dict[new_k] = v
                    elif k.startswith('image_encoder.'):
                        new_k = k.replace('image_encoder.', '')
                        encoder_state_dict[new_k] = v
                    elif not k.startswith('bert_encoder') and not k.startswith('decoder') and not k.startswith('vision_decoder'):
                        encoder_state_dict[k] = v

                msg = self.image_encoder.load_state_dict(encoder_state_dict, strict=False)
                print(f"[Model] Load state dict msg: {msg}")

                print("[Model] Configuring parameter freezing...")
                for name, param in self.image_encoder.named_parameters():
                    param.requires_grad = False

                    if 'adapter' in name.lower() or 'projection_head' in name.lower():
                        param.requires_grad = True

                    # 解冻后 3 层（你原来也是这么写的）
                    if ('blocks.9.' in name or 'blocks.10.' in name or 'blocks.11.' in name):
                        param.requires_grad = True

                    if 'norm' in name.lower():
                        param.requires_grad = True

                    if 'domain_norm' in name.lower():
                        param.requires_grad = True
                    if 'ls1' in name or 'ls2' in name:
                        param.requires_grad = True
                        
                trainable_img = sum(p.numel() for p in self.image_encoder.parameters() if p.requires_grad)
                total_img = sum(p.numel() for p in self.image_encoder.parameters())
                print(f"[Model] Image Encoder: {trainable_img:,}/{total_img:,} trainable ({trainable_img/total_img*100:.1f}%)")

            except Exception as e:
                print(f"[ERROR] Failed to load MRM checkpoint: {e}")
        else:
            print(f"[WARNING] MRM checkpoint not found: {mrm_checkpoint_path}")

        # ====== MIM decoder ======
        self.vision_decoder = VisionDecoder(
            encoder_embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
            patch_size=patch_size, in_chans=in_chans
        )

        # ====== Text encoder ======
        self.bert_encoder = CXRBertModel.from_pretrained(args.bert_path)

        # 冻结 BERT 主体，只解冻 proj_head + 最后两层（保留你原逻辑）
        for param in self.bert_encoder.parameters():
            param.requires_grad = False
        for param in self.bert_encoder.proj_head.parameters():
            param.requires_grad = True
        for layer in self.bert_encoder.bert.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        # BERT 输出维度适配
        if isinstance(self.bert_encoder.proj_head, nn.Sequential):
            last_layer = list(self.bert_encoder.proj_head.children())[-1]
            bert_out_dim = last_layer.out_features
        else:
            bert_out_dim = 128

        if bert_out_dim != args.proj_dim:
            self.text_proj_adapter = nn.Linear(bert_out_dim, args.proj_dim)
        else:
            self.text_proj_adapter = nn.Identity()

        self.text_local_proj = nn.Linear(768, args.proj_dim)

        # ====== 仅保留 MLM ======
        self.mlm_loss = MLMLoss(self.bert_encoder.config.vocab_size)

        self.patch_size = patch_size

        # ====== 核心：soft global-local manifold alignment -> loss_align ======
        self.align_module = SoftGlobalLocalManifoldAlignment(
            feature_dim=args.proj_dim,
            temperature=getattr(args, "align_temp", 0.07),
            topo_weight=getattr(args, "align_topo_w", 0),
            sparse_weight=getattr(args, "align_sparse_w", 0.02),
            local_weight_floor=getattr(args, "align_local_floor", 0.20),
            topk_ratio=getattr(args, "align_topk", 0.30),
        )
        self.manifold_align = _LocalOnlyManifoldAlignWrapper(self.align_module)

        trainable_total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[Model] TOTAL: {trainable_total:,}/{total_params:,} trainable ({trainable_total/total_params*100:.1f}%)")

    def forward_image_feature(self, img_stack, mask_ratio=0.0):
        return self.image_encoder(img_stack, mask_ratio)

    def forward(self, img_stack, a_img_stack, l_img_stack, la_img_stack,
                txt_inputs, txt_attention_mask, txt_labels,
                mask_ratio=0.75):

        img = img_stack
        if len(img.shape) == 5:
            B, N_views, C, H, W = img.shape
            img = img.reshape(B * N_views, C, H, W)
        else:
            B, C, H, W = img.shape

        # ====== image encode (masking for MIM) ======
        img_global, img_local, encoder_output, mask, ids_restore = self.image_encoder(
            img, mask_ratio=mask_ratio
        )

        # ====== text flatten ======
        if len(txt_inputs.shape) == 3:
            txt_inputs_flat = txt_inputs.squeeze(1)
            txt_attention_mask_flat = txt_attention_mask.squeeze(1)
            txt_labels_flat = txt_labels.squeeze(1)
        else:
            txt_inputs_flat = txt_inputs
            txt_attention_mask_flat = txt_attention_mask
            txt_labels_flat = txt_labels

        # ====== text global ======
        txt_embed = self.bert_encoder(
            txt_inputs_flat,
            txt_attention_mask_flat,
            output_cls_projected_embedding=True,
            return_dict=True
        ).cls_projected_embedding
        txt_embed = self.text_proj_adapter(txt_embed)
        txt_global = F.normalize(txt_embed, dim=-1)

        # ====== text local tokens ======
        txt_output = self.bert_encoder.bert(
            input_ids=txt_inputs_flat,
            attention_mask=txt_attention_mask_flat,
            return_dict=True
        )
        txt_hidden = txt_output.last_hidden_state  # [B, L, 768]
        txt_tokens = self.text_local_proj(txt_hidden[:, 1:, :])  # [B, L-1, D]
        txt_tokens = F.normalize(txt_tokens, dim=-1)

        # ====== loss_align (模块内部 soft global-local) ======
        aligned_patches, alignment_info = self.align_module(
            img_global=img_global,
            img_patches=img_local,
            txt_global=txt_global,
            txt_tokens=txt_tokens,
            return_info=True
        )

        # loss_align 在 alignment_info 里
        loss_align = alignment_info["loss_align"] if isinstance(alignment_info, dict) and "loss_align" in alignment_info else None
        if loss_align is None:
            raise RuntimeError("alignment_info missing key 'loss_align'. Please check align_module outputs.")


        # ====== MLM ======
        loss_mlm = self.mlm_loss(
            txt_inputs_flat, txt_attention_mask_flat, txt_labels_flat, self.bert_encoder
        )

        # ====== MIM ======
        # 注意：VisionDecoder 期望 encoder_output 含 CLS + visible patches
        pred_patches = self.vision_decoder(encoder_output, ids_restore)
        target_patches = patchify(img, self.patch_size)

        if getattr(self.args, 'norm_pix_loss', False):
            mean = target_patches.mean(dim=-1, keepdim=True)
            var = target_patches.var(dim=-1, keepdim=True)
            target_patches = (target_patches - mean) / (var + 1e-6) ** .5

        loss_mim = (pred_patches - target_patches) ** 2
        loss_mim = loss_mim.mean(dim=-1)  # [B, N]
        loss_mim = (loss_mim * mask).sum() / (mask.sum() + 1e-8)

        return loss_align, loss_mlm, loss_mim, alignment_info