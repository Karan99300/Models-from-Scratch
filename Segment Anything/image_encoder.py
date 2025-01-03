import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import LayerNorm2d, MLPBlock

class PatchEmbed(nn.Module):
    def __init__(self, kernel_size=(16,16), stride=(16,16), padding=(0,0), in_channels=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size, stride, padding)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x

def get_rel_pos(q_size, k_size, rel_pos):
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size = max_rel_dist,
            mode = "linear"
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
        
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    
    return rel_pos_resized[relative_coords.long()]

def add_decomposed_rel_pos(attention, q, rel_pos_h, rel_pos_w, q_size, k_size):
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_size, k_size, rel_pos_h)
    Rw = get_rel_pos(q_size, k_size, rel_pos_w)
    
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, rel_pos_h)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    
    attention = (
        attention.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attention

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, use_rel_pos=False, input_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim//num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rol_pos = use_rel_pos
        if self.use_rol_pos:
            assert (
                input_size is not None
            ), "Input Size must be provided using relative positional encoding."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            
    def forward(self, x):
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H*W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x
    
def window_partition(x, window_size):
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

def window_unpartition(windows, window_size, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm, act_layer=nn.GELU, use_rel_pos=False, rel_pos_zero_init=True, window_size=0, input_size=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size
        
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

class ImageEncoderViT(nn.Module):
    def __init__(self, img_size=1024, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4.0, out_chans=256, qkv_bias=True, norm_layer=nn.LayerNorm, act_layer=nn.GELU,
        use_abs_pos=True, use_rel_pos=False, rel_pos_zero_init=True, window_size=0, global_attn_indexes=(),):
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed((patch_size, patch_size), (patch_size, patch_size), in_chans, embed_dim)
        self.pos_embed = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, img_size//patch_size, img_size//patch_size, embed_dim))
            
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(embed_dim, num_heads, mlp_ratio, qkv_bias, norm_layer, act_layer, use_rel_pos, rel_pos_zero_init, 
                          window_size if i not in global_attn_indexes else 0, (img_size//patch_size, img_size//patch_size))
            self.blocks.append(block)
            
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False,),
            LayerNorm2d(out_chans)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x