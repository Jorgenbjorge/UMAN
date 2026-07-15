import torch
import torch.nn as nn

# 原始导入：
# from timm.models.vision_transformer import DropPath, Mlp, LayerScale, Attention, Block

# 修复后的导入：
# LayerScale 在不同版本的 timm 中位置不同，尝试多种导入方式
from timm.models.vision_transformer import DropPath, Mlp, Attention, Block

# 尝试从不同位置导入 LayerScale
try:
    from timm.models.layers import LayerScale  # 新版本 timm
except ImportError:
    try:
        from timm.models.vision_transformer import LayerScale  # 旧版本 timm
    except ImportError:
        # 如果都失败，创建一个简单的 LayerScale 实现
        class LayerScale(nn.Module):
            def __init__(self, dim, init_values=1e-5, inplace=False):
                super().__init__()
                self.inplace = inplace
                self.gamma = nn.Parameter(init_values * torch.ones(dim)) if init_values else None

            def forward(self, x):
                if self.gamma is not None:
                    return x.mul_(self.gamma) if self.inplace else x * self.gamma
                return x

# --- 修复 'to_2tuple' 导入错误 (兼容新旧版 timm) ---
try:
    # 尝试从新位置导入 (timm 0.9+)
    from timm.layers import to_2tuple
except ImportError:
    try:
        # 尝试从旧位置导入 (timm < 0.9)
        from timm.models.layers import to_2tuple
    except ImportError:
        try:
            # 尝试从更旧的位置导入
            from timm.models.layers.helpers import to_2tuple
        except ImportError:
            # 如果都失败了，手动定义这个函数
            import collections.abc
            from itertools import repeat

            def to_2tuple(x):
                if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
                    return tuple(x)
                return tuple(repeat(x, 2))
# ---------------------------------------------------------------


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attention_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            # attention_mask shape: (B, N) -> (B, 1, 1, N)
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,  # LayerScale 初始化值
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 只保留 Attention Adapter
        self.attn_adapter = Adapter(dim, skip_connect=True)

    def forward(self, x, attention_mask=None):
        if attention_mask is not None:
            # Attention + Adapter
            x = x + self.drop_path1(
                self.attn_adapter(self.ls1(self.attn(self.norm1(x), attention_mask)))
            )

            # 标准 FFN，不再加 mlp_adapter
            x = x + self.drop_path2(
                self.ls2(self.mlp(self.norm2(x)))
            )

        else:
            # Attention + Adapter
            x = x + self.drop_path1(
                self.attn_adapter(self.ls1(self.attn(self.norm1(x))))
            )

            # 标准 FFN，不再加 mlp_adapter
            x = x + self.drop_path2(
                self.ls2(self.mlp(self.norm2(x)))
            )

        return x