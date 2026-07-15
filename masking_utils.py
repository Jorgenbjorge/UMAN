import torch
import numpy as np


def random_masking(x, mask_ratio):
    """
    执行随机遮盖（参考MAE论文）
    
    Args:
        x: [B, N, D] - Batch, Num_patches, Dimension
        mask_ratio: 遮盖比例 (e.g., 0.75)
    
    Returns:
        x_masked: [B, N*(1-mask_ratio), D] - 非遮盖的patches
        mask: [B, N] - 0表示保留，1表示遮盖
        ids_restore: [B, N] - 用于恢复原始顺序的索引
    """
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))  # 保留的patch数量
    
    # 生成随机噪声用于shuffle
    noise = torch.rand(B, N, device=x.device)  # [B, N]
    
    # 排序得到shuffle后的索引
    ids_shuffle = torch.argsort(noise, dim=1)  # [B, N]
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # [B, N] - 用于恢复
    
    # 保留前len_keep个patch（非遮盖）
    ids_keep = ids_shuffle[:, :len_keep]  # [B, len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    
    # 生成mask: 0表示保留，1表示遮盖
    mask = torch.ones([B, N], device=x.device)  # [B, N]
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)  # 恢复到原始顺序
    
    return x_masked, mask, ids_restore


def unpatchify(x, patch_size=16):
    """
    将patch序列还原为图像
    
    Args:
        x: [B, N, patch_size^2 * 3] - patches
        patch_size: patch大小
    
    Returns:
        imgs: [B, 3, H, W]
    """
    p = patch_size
    h = w = int(x.shape[1]**.5)  # 假设是正方形
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs


def patchify(imgs, patch_size=16):
    """
    将图像转换为patch序列（用于计算MIM loss的target）
    
    Args:
        imgs: [B, 3, H, W]
        patch_size: patch大小
    
    Returns:
        x: [B, N, patch_size^2 * 3]
    """
    p = patch_size
    assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0
    
    h = imgs.shape[2] // p
    w = imgs.shape[3] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    return x


# ============== 测试代码 ==============
if __name__ == "__main__":
    # 测试masking
    B, N, D = 4, 196, 768
    x = torch.randn(B, N, D)
    
    x_masked, mask, ids_restore = random_masking(x, mask_ratio=0.75)
    
    print(f"原始输入: {x.shape}")
    print(f"遮盖后: {x_masked.shape}")  # 应该是 [4, 49, 768]
    print(f"保留率: {x_masked.shape[1] / N:.2%}")  # 应该是 25%
    print(f"Mask形状: {mask.shape}")  # [4, 196]
    print(f"遮盖比例: {mask.sum() / mask.numel():.2%}")  # 应该是 75%
    
    # 测试patchify/unpatchify
    imgs = torch.randn(2, 3, 224, 224)
    patches = patchify(imgs, patch_size=16)
    print(f"\n图像: {imgs.shape} → Patches: {patches.shape}")
    
    imgs_recon = unpatchify(patches, patch_size=16)
    print(f"重建: {imgs_recon.shape}")
    print(f"重建误差: {(imgs - imgs_recon).abs().max().item():.6f}")
