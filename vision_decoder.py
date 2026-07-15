import torch
import torch.nn as nn
from timm.models.vision_transformer import Block as TransformerBlock


class VisionDecoder(nn.Module):
    """
    轻量级Vision Decoder用于MIM任务
    参考MAE和论文设计
    """
    def __init__(
        self,
        encoder_embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        patch_size=16,
        in_chans=3,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_size = patch_size
        self.in_chans = in_chans
        
        # Encoder → Decoder 投影
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        
        # Mask token: 用于填充被遮盖的位置
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # === 新增：CLS token ===
        self.cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # 位置编码（可学习或固定）
        # 224/16 = 14, 所以有14*14=196个patch, +1 CLS token = 197
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, 197, decoder_embed_dim),
            requires_grad=False  # 使用固定的正弦位置编码
        )
        
        # Transformer Decoder Blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=norm_layer
            )
            for _ in range(decoder_depth)
        ])
        
        # 输出层
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # 预测头: 从decoder输出 → 像素值
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size**2 * in_chans,
            bias=True
        )
        
        # 初始化
        torch.nn.init.normal_(self.mask_token, std=0.02)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.initialize_weights()
    
    def initialize_weights(self):
        """初始化位置编码（正弦编码）"""
        from util.pos_embed import get_2d_sincos_pos_embed
        
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(196**0.5),
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, ids_restore):
        """
        Args:
            x: [B, N_visible+1, encoder_embed_dim] - 含CLS token的encoder输出
            ids_restore: [B, N] - 用于恢复原始顺序的索引
        
        Returns:
            pred: [B, N, patch_size^2 * in_chans] - 重建的patch像素
        """
        # 拆分CLS与patch
        cls_token, x = x[:, :1, :], x[:, 1:, :]  # [B, 1, D], [B, N_vis, D]
        
        # 投影到decoder维度
        cls_token = self.decoder_embed(cls_token)
        x = self.decoder_embed(x)
        
        B, N_vis, D = x.shape
        N = ids_restore.shape[1]  # patch总数 (196)
        
        # 创建mask tokens以填补被mask掉的patch
        mask_tokens = self.mask_token.repeat(B, N - N_vis, 1)
        
        # 拼接非mask patches和mask tokens
        x_ = torch.cat([x, mask_tokens], dim=1)
        
        # 根据ids_restore恢复原始顺序
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, D)
        )
        
        # 加回CLS token
        x_full = torch.cat([cls_token, x_], dim=1)  # [B, 197, D]
        
        # 添加位置编码（197对齐 ✅）
        x = x_full + self.decoder_pos_embed
        
        # Transformer解码器
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # 预测像素
        x = self.decoder_pred(x)
        
        # 移除CLS token（仅重建patch部分）
        x = x[:, 1:, :]  # [B, 196, patch_size^2 * in_chans]
        return x


# ============== 测试代码 ==============
if __name__ == "__main__":
    B = 4
    encoder_dim = 768
    N_visible = 49  # 25%可见patches
    ids_restore = torch.argsort(torch.rand(B, 196), dim=1)
    encoder_out = torch.randn(B, N_visible + 1, encoder_dim)  # 含CLS

    decoder = VisionDecoder()
    pred = decoder(encoder_out, ids_restore)

    print("输入:", encoder_out.shape)
    print("输出:", pred.shape)  # [4, 196, 768]
