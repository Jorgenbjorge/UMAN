# improved_loss_functions.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class ImprovedClipLoss(nn.Module):
    """
    改进的CLIP损失（参考BLIP）：
    1. 双向对比损失
    2. Hard negative mining
    3. Temperature自适应
    """
    def __init__(self, proj_dim=512, init_temp=0.07):
        super().__init__()
        # 可学习的temperature（BLIP做法）
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / init_temp)))
        self.proj_dim = proj_dim
        
    def forward(self, image_features, text_features, hard_negatives=False, domain_negatives=False):
        """
        Args:
            image_features: [B, D]
            text_features: [B, D]
            hard_negatives: 是否使用hard negative mining
        """
        # 归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 温度缩放
        logit_scale = self.logit_scale.exp()
        logit_scale = torch.clamp(logit_scale, max=100)  # 防止过大
        
        # 相似度矩阵 [B, B]
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        
        # 标签
        labels = torch.arange(logits_per_image.shape[0], device=image_features.device)
        
        # === Hard Negative Mining（可选）===
        if hard_negatives and logits_per_image.shape[0] > 2:
            # 找到最难的负样本（相似度最高的错误配对）
            with torch.no_grad():
                # 对于每个样本，找到最相似的负样本
                mask = torch.eye(logits_per_image.shape[0], device=image_features.device).bool()
                logits_per_image_masked = logits_per_image.masked_fill(mask, -1e4)
                hard_neg_idx = logits_per_image_masked.argmax(dim=1)
            
            # 增强hard negative的权重（加大惩罚）
            hard_neg_mask = F.one_hot(hard_neg_idx, num_classes=logits_per_image.shape[0])
            logits_per_image = logits_per_image + 0.5 * hard_neg_mask  # 提升hard negative logit
        
        # InfoNCE损失
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        if domain_negatives:  # 🔥 新增：域负样本（batch内shuffle模拟跨域）
            shuffled_idx = torch.randperm(image_features.shape[0])
            domain_neg_sim = logit_scale * image_features @ text_features[shuffled_idx].T
            domain_mask = torch.eye(domain_neg_sim.shape[0], device=image_features.device) * -1e4
            domain_neg_sim += domain_mask
            hard_domain_idx = domain_neg_sim.argmax(dim=1)
            hard_domain_mask = F.one_hot(hard_domain_idx, num_classes=domain_neg_sim.shape[0])
            logits_per_image += 0.3 * hard_domain_mask  # 轻微增强域负惩罚
        
        return loss, logits_per_image


class ManifoldAlignedLocalLoss(nn.Module):
    """
    改进版：集成对比学习和稀疏性正则
    """
    def __init__(self, proj_dim=512, temperature=0.07):
        super().__init__()
        self.proj_dim = proj_dim
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / temperature)))
        
    def forward(self, img_aligned, txt_tokens, alignment_info=None):
        """
        Args:
            img_aligned: [B, N_vis, D] - 流形对齐后的图像特征
            txt_tokens: [B, L, D] - 文本token特征（用于global对比）
            alignment_info: dict - 包含辅助损失
        """
        B, N_img, D = img_aligned.shape
        _, N_txt, _ = txt_tokens.shape
        
        # 归一化
        img_aligned = F.normalize(img_aligned, dim=-1)
        txt_tokens = F.normalize(txt_tokens, dim=-1)
        
        # ========== 1. Token-level对比损失 ========== 
        similarity = torch.bmm(img_aligned, txt_tokens.transpose(1, 2))  # [B, N_img, N_txt]
        max_sim_per_patch, matched_token_idx = similarity.max(dim=-1)  # [B, N_img]
        
        logit_scale = self.logit_scale.exp().clamp(max=100)
        
        # 展平计算
        img_flat = img_aligned.reshape(B * N_img, D)
        txt_flat = txt_tokens.reshape(B * N_txt, D)
        
        global_sim = logit_scale * img_flat @ txt_flat.T
        
        matched_token_global_idx = matched_token_idx + torch.arange(B, device=img_aligned.device).unsqueeze(1) * N_txt
        matched_token_global_idx = matched_token_global_idx.reshape(B * N_img)
        
        loss_token = F.cross_entropy(global_sim, matched_token_global_idx)
        
        # ========== 2. 总损失 ========== 
        loss = loss_token
        
        return loss


class ImprovedITMLoss(nn.Module):
    """
    改进版：集成全局/局部匹配（参考BLIP）
    """
    def __init__(self, proj_dim=512):
        super().__init__()
        self.proj_dim = proj_dim
        self.itm_head = nn.Linear(2 * proj_dim, 2)  # 🔥 修改：输入维度为2 * proj_dim，因为cat后维度翻倍
        
    def forward(self, img_feat, txt_feat):
        # 拼接全局特征
        itm_input = torch.cat([img_feat, txt_feat], dim=-1)
        
        logits = self.itm_head(itm_input)
        
        # 标签：假设0为不匹配，1为匹配（需外部提供或负采样）
        # 这里简化假设所有为正样本（实际需负采样）
        labels = torch.ones(img_feat.shape[0], dtype=torch.long, device=img_feat.device)
        
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        
        return loss


class MLMLoss(nn.Module):
    """保持原有实现（已经足够好）"""
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, txt_inputs, txt_attention_mask, txt_labels, bert_encoder):
        if txt_inputs.dim() == 3:
            B, V, N = txt_inputs.shape
            txt_inputs = txt_inputs.reshape(B * V, N)
            txt_attention_mask = txt_attention_mask.reshape(B * V, N)
            txt_labels = txt_labels.reshape(B * V, N)
        
        mlm_output = bert_encoder(
            input_ids=txt_inputs,
            attention_mask=txt_attention_mask,
            output_mlm_logits=True,
            return_dict=True
        )
        prediction_scores = mlm_output.logits
        
        loss_fct = CrossEntropyLoss()
        mlm_loss = loss_fct(
            prediction_scores.view(-1, self.vocab_size),
            txt_labels.view(-1)
        )
        
        return mlm_loss


class CrossModalConsistencyLoss(nn.Module):
    """
    【新增】跨模态一致性损失
    
    确保：
    1. 全局特征和局部聚合特征一致
    2. 不同增强视角的特征一致（如果有多视角）
    """
    def __init__(self, proj_dim=512):
        super().__init__()
        self.proj_dim = proj_dim
        
    def forward(self, img_global, img_local, txt_global, txt_local):
        """
        Args:
            img_global: [B, D] - 全局图像特征
            img_local: [B, N, D] - 局部图像特征
            txt_global: [B, D] - 全局文本特征
            txt_local: [B, L, D] - 局部文本特征
        """
        # 从局部特征聚合全局表示
        img_global_from_local = F.normalize(img_local.mean(dim=1), dim=-1)
        txt_global_from_local = F.normalize(txt_local.mean(dim=1), dim=-1)
        
        img_global = F.normalize(img_global, dim=-1)
        txt_global = F.normalize(txt_global, dim=-1)
        
        # 一致性约束：全局特征应与局部聚合一致
        loss_img_consistency = 1.0 - (img_global * img_global_from_local).sum(dim=-1).mean()
        loss_txt_consistency = 1.0 - (txt_global * txt_global_from_local).sum(dim=-1).mean()
        
        # 跨模态一致性：图像全局应与文本全局对齐
        loss_cross_modal = 1.0 - (img_global * txt_global).sum(dim=-1).mean()
        
        total_loss = loss_img_consistency + loss_txt_consistency + loss_cross_modal
        
        return total_loss


# 🔥 新增：模态内拓扑保持损失（用于迁移）
class TopologyPreservationLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, feats1, feats2):
        dist1 = torch.cdist(feats1, feats1)
        dist2 = torch.cdist(feats2, feats2)
        prob1 = F.softmax(-dist1 / self.temperature, dim=-1)
        prob2 = F.softmax(-dist2 / self.temperature, dim=-1)
        return F.kl_div(prob1.log(), prob2, reduction='batchmean')


# ============== 测试代码 ==============
if __name__ == "__main__":
    print("=== 测试改进的损失函数 ===\n")
    
    B, N, L, D = 4, 196, 50, 512
    
    # 1. 测试ImprovedClipLoss
    print("1. ImprovedClipLoss:")
    clip_loss = ImprovedClipLoss(D)
    img_feat = torch.randn(B, D)
    txt_feat = torch.randn(B, D)
    loss, logits = clip_loss(img_feat, txt_feat, hard_negatives=True)
    print(f"   Loss: {loss.item():.4f}, Logits: {logits.shape}")
    print(f"   Temperature: {clip_loss.logit_scale.exp().item():.4f}")
    
    # 2. 测试ManifoldAlignedLocalLoss
    print("\n2. ManifoldAlignedLocalLoss:")
    local_loss = ManifoldAlignedLocalLoss(D)
    img_aligned = torch.randn(B, N, D)
    txt_tokens = torch.randn(B, L, D)
    alignment_info = {
        'similarity_map': torch.rand(B, N),
        'weights': torch.rand(B, N)
    }
    loss = local_loss(img_aligned, txt_tokens, alignment_info)
    print(f"   Loss: {loss.item():.4f}")
    
    # 3. 测试ImprovedITMLoss
    print("\n3. ImprovedITMLoss:")
    itm_loss = ImprovedITMLoss(D)
    loss = itm_loss(img_feat, txt_feat)
    print(f"   Loss: {loss.item():.4f}")
    
    # 4. 测试CrossModalConsistencyLoss
    print("\n4. CrossModalConsistencyLoss:")
    consistency_loss = CrossModalConsistencyLoss(D)
    img_global = torch.randn(B, D)
    img_local = torch.randn(B, N, D)
    txt_global = torch.randn(B, D)
    txt_local = torch.randn(B, L, D)
    loss = consistency_loss(img_global, img_local, txt_global, txt_local)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n=== 所有测试通过 ===")