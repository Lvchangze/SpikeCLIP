import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from transformers import CLIPProcessor, CLIPModel
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models import create_model

__all__ = ['vision_encoder', 'language_encoder', 'CLIPv_encoder']
backend = 'torch'
# backend = 'cupy'


def _make_mask(attention_mask, T=4):
    B, L = attention_mask.shape
    mask = []
    for i in range(B):
        k = attention_mask[i].sum()
        per_mask = torch.full((L, L), 1, device=attention_mask.device)
        per_mask[k:, :]=0
        per_mask[:, k:]=0
        mask.append(per_mask)
    mask = torch.stack(mask, dim=0)
    mask = mask.unsqueeze(0).repeat(T, 1, 1, 1)
    return mask





class Vision_Tokenizer(nn.Module):
    def __init__(self, in_channels=2, embed_dims=256):
        super().__init__()

        self.block0_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.block0_bn = nn.BatchNorm2d(embed_dims // 8)
        self.block1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        
        self.block1_conv = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn = nn.BatchNorm2d(embed_dims // 4)
        self.block2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        
        self.block2_conv = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn = nn.BatchNorm2d(embed_dims // 2)
        self.block3_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        
        self.block3_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block3_conv = nn.Conv2d(embed_dims // 2, embed_dims // 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn = nn.BatchNorm2d(embed_dims // 1)
        self.block4_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        
        self.block4_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.block4_bn = nn.BatchNorm2d(embed_dims)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.block0_conv(x.flatten(0, 1))
        x = self.block0_bn(x).reshape(T, B, -1, H, W)
        x = self.block1_lif(x).flatten(0, 1)
        
        x = self.block1_conv(x)
        x = self.block1_bn(x).reshape(T, B, -1, H, W)
        x = self.block2_lif(x).flatten(0, 1)
        
        x = self.block2_conv(x)
        x = self.block2_bn(x).reshape(T, B, -1, H, W)
        x = self.block3_mp(x.flatten(0, 1)).reshape(T, B, -1, int(H / 2), int(W / 2))
        x = self.block3_lif(x).flatten(0, 1)
        
        x = self.block3_conv(x)
        x = self.block3_bn(x).reshape(T, B, -1, int(H / 2), int(W / 2))
        x = self.block4_mp(x.flatten(0, 1)).reshape(T, B, -1, int(H / 4), int(W / 4))
        x = self.block4_lif(x).flatten(0, 1)
        
        x = self.block4_conv(x)
        x = self.block4_bn(x).reshape(T, B, -1, int(H / 4), int(W / 4))
        
        return x
   

class Vision_MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)

        self.mlp2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.mlp1_lif(x)
        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x).reshape(T, B, self.c_hidden, H, W)

        x = self.mlp2_lif(x)
        x = self.mlp2_conv(x.flatten(0, 1))
        x = self.mlp2_bn(x).reshape(T, B, C, H, W)
        return x


class Vision_SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend=backend)
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_lif(x)

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N)
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * 0.125

        x = x.transpose(3, 4).reshape(T, B, C, N)
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_bn(self.proj_conv(x)).reshape(T, B, C, H, W)
        return x


class Vision_Transformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop_path=None):
        super().__init__()
        self.attn = Vision_SelfAttention(dim, num_heads=num_heads)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Vision_MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class Vision_Encoder(nn.Module):
    def __init__(self, in_channels=3, target_dim=512, embed_dims=384, num_heads=12, mlp_ratios=4, depths=4, T=4):
        super().__init__()
        
        weights = [-2.3026, -1.6094, -0.9163, -0.2231]
        self.weights_tensor = nn.Parameter(torch.tensor(weights).unsqueeze(0).unsqueeze(2))
        self.depths = depths
        self.T = T

        patch_embed = Vision_Tokenizer(in_channels=in_channels, embed_dims=embed_dims)

        block = nn.ModuleList([Vision_Transformer(dim=embed_dims, num_heads=num_heads, 
                                                  mlp_ratio=mlp_ratios) for j in range(depths)])


        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        self.proj = nn.Linear(embed_dims, target_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x = patch_embed(x)
        for blk in block:
            x = blk(x)
        return x.flatten(3).mean(3)

    def forward(self, pixel_values):
        x = pixel_values.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)    # [T, B, D]
        x = x.transpose(0, 1)
        x = torch.sum(x * self.weights_tensor.to(x.device), dim=1)
        x = self.proj(x)
        return x



class Language_Tokenizer(nn.Module):
    def __init__(self, clip_embedding, clip_emb_dim=512, dim=384):
        super().__init__()
        self.emb = clip_embedding
        self.redim = nn.Linear(clip_emb_dim, dim)
    
    def forward(self, x):
        x = self.emb(x)
        x = self.redim(x)
        return x


class Language_SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        
        self.q_conv = nn.Linear(dim, dim, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        
        self.k_conv = nn.Linear(dim, dim, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        
        self.v_conv = nn.Linear(dim, dim, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend=backend)
        self.proj_conv = nn.Linear(dim, dim, bias=False)
        self.proj_bn = nn.BatchNorm1d(dim)

    def forward(self, x, attention_mask):
        T, B, L, D = x.shape
        T, B, L, L = attention_mask.shape
        attention_mask = attention_mask.unsqueeze(2)
        
        x = self.proj_lif(x)
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)     # [T*B, L, D]
        q_conv_out = self.q_bn(q_conv_out.transpose(-2, -1)).reshape(T, B, L, D)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.reshape(T, B, L, self.num_heads, D // self.num_heads).permute(0, 1, 3, 2, 4)

        k_conv_out = self.k_conv(x_for_qkv) 
        k_conv_out = self.k_bn(k_conv_out.transpose(-2, -1)).reshape(T, B, L, D)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.reshape(T, B, L, self.num_heads, D // self.num_heads).permute(0, 1, 3, 2, 4)

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out.transpose(-2, -1)).reshape(T, B, L, D)
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.reshape(T, B, L, self.num_heads, D // self.num_heads).permute(0, 1, 3, 2, 4)     # [T, B, h, L, d]

        attn = (q @ k.transpose(-2, -1))    # [T, B, h, L, L]
        attn = attn.masked_fill(attention_mask==0, 0)
        x = (attn @ v) * 0.125              # [T, B, h, L, d]

        x = x.transpose(2, 3).reshape(T, B, L, D)
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_bn(self.proj_conv(x).transpose(-2, -1)).reshape(T, B, L, D)
        return x


class Language_MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        
        self.mlp1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.mlp1_conv = nn.Linear(in_features, hidden_features, bias=False)
        self.mlp1_bn = nn.BatchNorm1d(hidden_features)

        self.mlp2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.mlp2_conv = nn.Linear(hidden_features, out_features, bias=False)
        self.mlp2_bn = nn.BatchNorm1d(out_features)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, L, D = x.shape

        x = self.mlp1_lif(x)
        x = self.mlp1_conv(x)
        x = self.mlp1_bn(x.flatten(0, 1).transpose(-2, -1)).reshape(T, B, L, self.hidden_features)

        x = self.mlp2_lif(x)
        x = self.mlp2_conv(x)
        x = self.mlp2_bn(x.flatten(0, 1).transpose(-2, -1)).reshape(T, B, L, D)
        return x


class Language_Transformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.attn = Language_SelfAttention(dim, num_heads=num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Language_MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x, attention_mask):  
        T, B, L, D = x.shape    
        x = x + self.attn(x, attention_mask)
        x = x + self.mlp(x)
        return x


class Language_Encoder(nn.Module):
    def __init__(self, clip_embedding, clip_dim=512, mlp_ratios=4, depths=4, T=4):
        super(Language_Encoder, self).__init__()

        # ###############################################################################
        self.mlp_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)   #
        self.fc1 = nn.Linear(20*clip_dim, mlp_ratios*clip_dim)                          #
        # self.ln1 = nn.LayerNorm( mlp_ratios*clip_dim)
                                                                                        #
        self.mlp_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)   #
        self.fc2 = nn.Linear(mlp_ratios*clip_dim, clip_dim)                             #
        # ###############################################################################

        self.clip_embedding = clip_embedding
        
        weights = [-2.3026, -1.6094, -0.9163, -0.2231]
        self.T = T
        self.weights_tensor = nn.Parameter(torch.tensor(weights).unsqueeze(0).unsqueeze(2))
        

    def forward(self, x, attention_mask):
        x = self.clip_embedding(x)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1)     
        x = x.view(x.size(0),x.size(1), -1)  
  
        x = self.mlp_lif1(x)
        x = self.fc1(x)    
        # x = self.ln1(x)

        x = self.mlp_lif2(x)
        x = self.fc2(x)      

        x = x.transpose(0, 1)           # [B, T, D]
        x = torch.sum(x * self.weights_tensor.to(x.device), dim=1)
        return x


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input):
        return input * torch.sigmoid(1.702 * input)


class CLIP_Tokenizer(nn.Module):
    def __init__(self, in_channels=2, embed_dims=256):
        super().__init__()
        
        self.activation_fn = QuickGELUActivation()

        self.block0_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.block0_bn = nn.BatchNorm2d(embed_dims // 8)

        self.block1_conv = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn = nn.BatchNorm2d(embed_dims // 4)

        self.block2_conv = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn = nn.BatchNorm2d(embed_dims // 2)
        
        self.block3_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block3_conv = nn.Conv2d(embed_dims // 2, embed_dims // 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn = nn.BatchNorm2d(embed_dims // 1)

        self.block4_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.block4_bn = nn.BatchNorm2d(embed_dims)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.block0_conv(x)
        x = self.block0_bn(x)
        
        x = self.block1_conv(x)
        x = self.block1_bn(x)

        x = self.block2_conv(x)
        x = self.block2_bn(x)
        
        x = self.block3_mp(x).reshape(B, -1, int(H / 2), int(W / 2))
        x = self.block3_conv(x)
        x = self.block3_bn(x).reshape(B, -1, int(H / 2), int(W / 2))

        x = self.block4_mp(x).reshape(B, -1, int(H / 4), int(W / 4))
        x = self.block4_conv(x)
        x = self.block4_bn(x).reshape(B, -1, int(H / 4), int(W / 4))
        
        return x.flatten(2).transpose(1, 2)


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, embed_dims, image_size, patch_size, num_channels):
        super().__init__()
        self.embed_dims = embed_dims
        self.image_size = image_size
        self.patch_size = patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dims))

        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=self.embed_dims,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dims)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class CLIPAttention(nn.Module):

    def __init__(self, embed_dims=384, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dims
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states):
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_probs = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output


class CLIPMLP(nn.Module):
    def __init__(self, embed_dims=384, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or embed_dims
        self.activation_fn = QuickGELUActivation()
        self.fc1 = nn.Linear(embed_dims, hidden_features)
        self.fc2 = nn.Linear(hidden_features, embed_dims)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    def __init__(self, embed_dims=384, num_heads=8, mlp_ratios=4):
        super().__init__()
        self.embed_dim = embed_dims
        hidden_features = int(embed_dims * mlp_ratios)
        
        self.self_attn = CLIPAttention(embed_dims=embed_dims, num_heads=num_heads)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(embed_dims=embed_dims, hidden_features=hidden_features)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        
    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(nn.Module):
    def __init__(self, embed_dims=384, num_heads=8, mlp_ratios=4, depths=4):
        super().__init__()
        self.layers = nn.ModuleList([CLIPEncoderLayer(embed_dims, num_heads, mlp_ratios) for _ in range(depths)])
        
    def forward(self, inputs_embeds):
        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states)
        return hidden_states


class CLIPVisionTransformer(nn.Module):
    def __init__(self, in_channels=3, target_dim=512, embed_dims=384, num_heads=8, mlp_ratios=4, depths=4, image_size=32, patch_size=16):
        super().__init__()
        embed_dim = embed_dims
        
        self.embeddings = CLIP_Tokenizer(in_channels=in_channels, embed_dims=embed_dims)
        self.pre_layrnorm = nn.LayerNorm(embed_dim)
        self.encoder = CLIPEncoder(embed_dims=embed_dims, num_heads=num_heads, mlp_ratios=mlp_ratios, depths=depths)
        self.post_layernorm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dims, target_dim)

    
    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)

        hidden_states = self.pre_layrnorm(hidden_states)
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        pooled_output = last_hidden_state.mean(1)
        pooled_output = self.post_layernorm(pooled_output)
        pooled_output = self.proj(pooled_output)
        
        return pooled_output 




def vision_encoder(pretrained=False, **kwargs):
    model = Vision_Encoder(
        **kwargs
    )
    return model


def language_encoder(pretrained=False, **kwargs):
    model = Language_Encoder(
        **kwargs
    )
    return model


def CLIPv_encoder(pretrained=False, **kwargs):
    model = CLIPVisionTransformer(
        **kwargs
    )
    return model
