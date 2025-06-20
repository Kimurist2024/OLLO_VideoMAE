import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.io as io
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
import numpy as np
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
import math
import random
from PIL import Image

# ===== 高性能VideoTransforms =====
class VideoRandomResizedCrop:
    """VideoMAE専用ランダムリサイズクロップ"""
    def __init__(self, size=(224, 224), scale=(0.5, 1.0), ratio=(3./4., 4./3.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self, video):
        # video: (T, C, H, W)
        T, C, H, W = video.shape
        
        # パラメータ計算
        area = H * W
        target_area = random.uniform(*self.scale) * area
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        aspect_ratio = math.exp(random.uniform(*log_ratio))
        
        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))
        
        if w <= W and h <= H:
            i = random.randint(0, H - h)
            j = random.randint(0, W - w)
        else:
            # Fallback to center crop
            in_ratio = W / H
            if in_ratio < min(self.ratio):
                w = W
                h = int(round(w / min(self.ratio)))
            elif in_ratio > max(self.ratio):
                h = H
                w = int(round(h * max(self.ratio)))
            else:
                w, h = W, H
            i = (H - h) // 2
            j = (W - w) // 2
        
        # 全フレームに同じクロップを適用
        cropped = video[:, :, i:i+h, j:j+w]
        
        # リサイズ
        return F.interpolate(cropped, size=self.size, mode='bilinear', align_corners=False)

class VideoRandomHorizontalFlip:
    """時間一貫性のある水平反転"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, video):
        if random.random() < self.p:
            return torch.flip(video, dims=[-1])  # 幅方向に反転
        return video

class VideoColorJitter:
    """時間一貫性のあるカラージッター"""
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, video):
        # 全フレームに同じパラメータを適用
        brightness_factor = random.uniform(max(0, 1-self.brightness), 1+self.brightness) if self.brightness else None
        contrast_factor = random.uniform(max(0, 1-self.contrast), 1+self.contrast) if self.contrast else None
        saturation_factor = random.uniform(max(0, 1-self.saturation), 1+self.saturation) if self.saturation else None
        hue_factor = random.uniform(-self.hue, self.hue) if self.hue else None
        
        # フレームごとに変換適用
        transformed_frames = []
        for frame in video:
            if brightness_factor is not None:
                frame = TF.adjust_brightness(frame, brightness_factor)
            if contrast_factor is not None:
                frame = TF.adjust_contrast(frame, contrast_factor)
            if saturation_factor is not None:
                frame = TF.adjust_saturation(frame, saturation_factor)
            if hue_factor is not None:
                frame = TF.adjust_hue(frame, hue_factor)
            transformed_frames.append(frame)
        
        return torch.stack(transformed_frames)

class VideoRandomRotation:
    """時間一貫性のある回転"""
    def __init__(self, degrees=15):
        self.degrees = degrees
    
    def __call__(self, video):
        angle = random.uniform(-self.degrees, self.degrees)
        transformed_frames = []
        for frame in video:
            rotated = TF.rotate(frame, angle, interpolation=TF.InterpolationMode.BILINEAR)
            transformed_frames.append(rotated)
        return torch.stack(transformed_frames)

class VideoRandomGrayscale:
    """ランダムグレースケール変換"""
    def __init__(self, p=0.1):
        self.p = p
    
    def __call__(self, video):
        if random.random() < self.p:
            # RGB係数でグレースケール
            gray = 0.299 * video[:, 0:1] + 0.587 * video[:, 1:2] + 0.114 * video[:, 2:3]
            return gray.repeat(1, 3, 1, 1)
        return video

class VideoRandomGaussianBlur:
    """ランダムガウシアンブラー"""
    def __init__(self, kernel_size=3, sigma=(0.1, 2.0), p=0.2):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p
    
    def __call__(self, video):
        if random.random() < self.p:
            sigma = random.uniform(*self.sigma)
            transformed_frames = []
            for frame in video:
                blurred = TF.gaussian_blur(frame, kernel_size=self.kernel_size, sigma=sigma)
                transformed_frames.append(blurred)
            return torch.stack(transformed_frames)
        return video

class VideoTemporalShift:
    """時間軸シフト（フレーム順序の変更）"""
    def __init__(self, max_shift=2):
        self.max_shift = max_shift
    
    def __call__(self, video):
        T = video.shape[0]
        if T > self.max_shift * 2:
            shift = random.randint(-self.max_shift, self.max_shift)
            if shift != 0:
                video = torch.roll(video, shifts=shift, dims=0)
        return video

class VideoMixUp:
    """Video MixUp（2つの動画の線形結合）"""
    def __init__(self, alpha=0.4, p=0.3):
        self.alpha = alpha
        self.p = p
    
    def __call__(self, video, other_video=None):
        if other_video is not None and random.random() < self.p:
            lam = np.random.beta(self.alpha, self.alpha)
            return lam * video + (1 - lam) * other_video
        return video

class VideoCutMix:
    """Video CutMix（空間的マスキング）"""
    def __init__(self, alpha=1.0, p=0.2):
        self.alpha = alpha
        self.p = p
    
    def __call__(self, video):
        if random.random() < self.p:
            lam = np.random.beta(self.alpha, self.alpha)
            T, C, H, W = video.shape
            
            # カット領域計算
            cut_ratio = np.sqrt(1. - lam)
            cut_w = int(W * cut_ratio)
            cut_h = int(H * cut_ratio)
            
            cx = np.random.randint(W)
            cy = np.random.randint(H)
            
            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)
            
            # マスク領域をゼロで埋める
            video[:, :, bby1:bby2, bbx1:bbx2] = 0
        
        return video

class VideoNormalize:
    """VideoMAE用正規化"""
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
    
    def __call__(self, video):
        return (video - self.mean.to(video.device)) / self.std.to(video.device)

class VideoCompose:
    """動画変換の合成"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, video):
        for transform in self.transforms:
            video = transform(video)
        return video

def get_video_transforms(mode='train', img_size=224):
    """VideoMAE専用の最適化されたデータ拡張"""
    
    if mode == 'train':
        return VideoCompose([
            VideoRandomResizedCrop(size=(img_size, img_size), scale=(0.5, 1.0)),
            VideoRandomHorizontalFlip(p=0.5),
            VideoColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            VideoRandomRotation(degrees=10),
            VideoRandomGrayscale(p=0.1),
            VideoRandomGaussianBlur(p=0.2),
            VideoTemporalShift(max_shift=2),
            VideoCutMix(p=0.3),
            VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # validation
        return VideoCompose([
            VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.io as io
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
import numpy as np
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
import math
import random
from PIL import Image
from functools import partial
import torch.utils.checkpoint as checkpoint

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                             attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]), 
                            stride=(self.tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)

class VisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, fc_drop_rate=0., drop_rate=0., 
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0.,
                 use_learnable_pos_emb=False, init_scale=0., all_frames=16, tubelet_size=2, 
                 use_checkpoint=False, use_mean_pooling=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, 
            num_frames=all_frames, tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                  init_values=init_values)
            for i in range(depth)])
        
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, _, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:   
            for blk in self.blocks:
                x = blk(x)

        x = self.norm(x)
        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))
        else:
            return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(self.fc_dropout(x))
        return x

def get_model():
    """VideoMAE Base model with enhanced head"""
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=58,
        all_frames=16,
        tubelet_size=2,
        use_mean_pooling=True,
        init_values=0.1,
        use_checkpoint=False
    )
    return model

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None, use_flash_attn=False):
        super().__init__()
        self.num_heads = num_heads
        self.use_flash_attn = use_flash_attn
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        self.head_dim = head_dim
        self.all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, self.all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(self.all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(self.all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_flash_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            x = x.transpose(1, 2).reshape(B, N, self.all_head_dim)
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, self.all_head_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, use_flash_attn=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, attn_head_dim, use_flash_attn)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1 = self.gamma_2 = None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = tubelet_size
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (num_frames // tubelet_size)
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
                              stride=(tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

def get_sinusoid_encoding_table(n_position, d_hid):
    def get_angle(pos, i):
        return pos / np.power(10000, 2 * (i // 2) / d_hid)
    table = np.array([[get_angle(pos, i) for i in range(d_hid)] for pos in range(n_position)])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.tensor(table, dtype=torch.float).unsqueeze(0)

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=58, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, fc_drop_rate=0., drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0., use_learnable_pos_emb=False,
                 init_scale=0., all_frames=16, tubelet_size=2, use_checkpoint=False, use_mean_pooling=True,
                 use_flash_attn=True):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, all_frames, tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) if use_learnable_pos_emb else get_sinusoid_encoding_table(num_patches, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[i], init_values,
                  norm_layer=norm_layer, act_layer=nn.GELU, use_flash_attn=use_flash_attn)
            for i in range(depth)
        ])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.fc_dropout = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        
        # 改良されたヘッド層（ドロップアウト強化）
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # 重み初期化
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed.to(x.device)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.fc_norm(x.mean(1)) if self.fc_norm else x[:, 0]

    def forward(self, x):
        return self.head(self.fc_dropout(self.forward_features(x)))

def get_model():
    return VisionTransformer()

class AdvancedFocalLoss(nn.Module):
    """高性能Focal Loss with Label Smoothing"""
    def __init__(self, alpha=0.5, gamma=1.0, pos_weight=None, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        # Label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing * 0.5
        
        # BCE loss with pos_weight
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none'
        )
        
        # Focal weight
        pt = torch.exp(-bce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        return (focal_weight * bce_loss).mean()

class EnhancedEgo4DDataset(Dataset):
    def __init__(self, annotation_file, video_root, transform=None, num_frames=16, num_classes=58, mode='train'):
        with open(annotation_file, "r") as f:
            data = json.load(f)
        self.annotations = data["annotations"]
        self.video_root = video_root
        self.transform = transform
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.mode = mode
        
        # MixUp用の事前読み込み（訓練時のみ）
        self.preloaded_videos = {}
        if mode == 'train':
            self._preload_sample_videos()

    def _preload_sample_videos(self, max_preload=100):
        """MixUp用にいくつかの動画を事前読み込み"""
        print("Preloading sample videos for MixUp...")
        indices = random.sample(range(len(self.annotations)), min(max_preload, len(self.annotations)))
        
        for idx in indices:
            try:
                ann = self.annotations[idx]
                video_path = os.path.join(self.video_root, ann["video_url"])
                video, _, _ = io.read_video(video_path, pts_unit='sec')
                self.preloaded_videos[idx] = self._process_video(video)
            except:
                continue

    def _process_video(self, video):
        """動画の基本的な前処理"""
        T = video.shape[0]
        if T < self.num_frames:
            repeat_factor = (self.num_frames + T - 1) // T
            video = video.repeat(repeat_factor, 1, 1, 1)
        
        # フレームサンプリング
        if self.mode == 'train' and T > self.num_frames:
            start_idx = random.randint(0, T - self.num_frames)
            indices = torch.arange(start_idx, start_idx + self.num_frames)
        else:
            indices = torch.linspace(0, T - 1, self.num_frames).long()
        
        video = video[indices].permute(0, 3, 1, 2).float() / 255.0
        return video

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        video_path = os.path.join(self.video_root, ann["video_url"])
        
        # 動画読み込み
        try:
            if idx in self.preloaded_videos:
                video = self.preloaded_videos[idx].clone()
            else:
                video, _, _ = io.read_video(video_path, pts_unit='sec')
                video = self._process_video(video)
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # フォールバック: ランダムノイズ
            video = torch.rand(self.num_frames, 3, 224, 224)
        
        # MixUp用の追加動画（訓練時のみ）
        mixup_video = None
        if self.mode == 'train' and self.preloaded_videos and random.random() < 0.3:
            mixup_idx = random.choice(list(self.preloaded_videos.keys()))
            if mixup_idx != idx:
                mixup_video = self.preloaded_videos[mixup_idx].clone()
        
        # データ拡張適用
        if self.transform:
            video = self.transform(video)
            # MixUpの場合
            if mixup_video is not None and hasattr(self.transform, 'transforms'):
                # MixUp変換を探す
                for t in self.transform.transforms:
                    if isinstance(t, VideoMixUp):
                        video = t(video, mixup_video)
                        break
        
        # ラベル処理
        label = ann["label"]
        if not isinstance(label, list):
            label = [label]
        target = torch.zeros(self.num_classes)
        for l in label:
            if 0 <= l < self.num_classes:
                target[l] = 1.0
        
        return video.permute(1, 0, 2, 3), target  # (C, T, H, W)

def get_enhanced_transforms(mode='train'):
    """強化されたVideoMAE専用データ拡張"""
    return get_video_transforms(mode=mode, img_size=224)

def compute_sample_weights_fast(annotation_file, num_classes=58):
    """高速なサンプル重み計算（動画読み込み不要）"""
    print("Computing sample weights for balanced sampling...")
    
    with open(annotation_file, "r") as f:
        data = json.load(f)
    
    # 各クラスの出現回数を計算
    class_counts = torch.zeros(num_classes)
    sample_labels = []
    
    for ann in data["annotations"]:
        labels = ann["label"]
        if not isinstance(labels, list):
            labels = [labels]
        
        target = torch.zeros(num_classes)
        for label in labels:
            if 0 <= label < num_classes:
                target[label] = 1.0
                class_counts[label] += 1
        
        sample_labels.append(target)
    
    # クラス重みを計算（逆頻度）
    total_samples = len(data["annotations"])
    class_weights = total_samples / (class_counts + 1e-8)
    
    # 各サンプルの重みを計算
    sample_weights = []
    for target in sample_labels:
        # サンプルが持つクラスの重みの平均
        active_classes = target > 0
        if active_classes.sum() > 0:
            weight = (class_weights * active_classes).sum() / active_classes.sum()
        else:
            weight = 1.0
        sample_weights.append(weight.item())
    
    sample_weights = torch.tensor(sample_weights)
    
    print(f"Sample weights - Min: {sample_weights.min():.2f}, Max: {sample_weights.max():.2f}, Mean: {sample_weights.mean():.2f}")
    return sample_weights

def compute_pos_weights_smart(annotation_file, num_classes=58):
    """スマートなpos_weight計算"""
    with open(annotation_file, "r") as f:
        data = json.load(f)
    
    class_counts = torch.zeros(num_classes)
    total_samples = len(data["annotations"])
    
    for ann in data["annotations"]:
        labels = ann["label"]
        if not isinstance(labels, list):
            labels = [labels]
        
        for label in labels:
            if 0 <= label < num_classes:
                class_counts[label] += 1
    
    # 改良されたpos_weight計算
    pos_weights = (total_samples - class_counts) / (class_counts + 1e-8)
    
    # より賢いクリッピング
    median_weight = pos_weights.median()
    pos_weights = torch.clamp(pos_weights, min=0.1, max=median_weight * 5)
    
    print(f"Class distribution: {class_counts[:10]}...")
    print(f"Pos weights range: [{pos_weights.min():.2f}, {pos_weights.max():.2f}]")
    print(f"Mean pos weight: {pos_weights.mean():.2f}")
    
    return pos_weights

def ultra_test_time_augmentation(model, video, device):
    """安全なTTA - エラー処理強化版"""
    model.eval()
    predictions = []
    
    # 基本変換のみ（安全な変換）
    safe_transforms = [
        lambda x: x,  # Original
        lambda x: torch.flip(x, dims=[4]),  # H-flip  
        lambda x: torch.flip(x, dims=[3]),  # V-flip
        lambda x: torch.flip(x, dims=[2]),  # T-flip
        lambda x: torch.clamp(x * 1.05, 0, 1),  # Brightness + (軽微)
        lambda x: torch.clamp(x * 0.95, 0, 1),  # Brightness - (軽微)
    ]
    
    # 重み設定
    weights = torch.tensor([0.35, 0.25, 0.15, 0.15, 0.05, 0.05]).to(device)
    
    # 予測実行
    for i, (transform, weight) in enumerate(zip(safe_transforms, weights)):
        try:
            # 変換適用
            transformed_video = transform(video)
            
            # サイズチェック
            if transformed_video.shape != video.shape:
                print(f"Size mismatch in transform {i}, using original")
                transformed_video = video
            
            # 予測
            with torch.no_grad():
                with autocast():
                    pred = torch.sigmoid(model(transformed_video))
                    predictions.append(pred * weight)
                    
        except Exception as e:
            print(f"TTA transform {i} failed: {e}, using original")
            # フォールバック: オリジナル予測
            try:
                with torch.no_grad():
                    with autocast():
                        pred = torch.sigmoid(model(video))
                        predictions.append(pred * weight)
            except Exception as e2:
                print(f"Fallback also failed: {e2}, skipping")
                continue
    
    # アンサンブル
    if predictions:
        ensemble_pred = sum(predictions)
        # 実際に使用された重みで正規化
        actual_weight = sum(weights[:len(predictions)])
        return ensemble_pred / actual_weight
    else:
        # 最後の手段: 通常予測
        print("All TTA failed, using simple prediction")
        with torch.no_grad():
            with autocast():
                return torch.sigmoid(model(video))

def find_optimal_threshold_advanced(model, val_loader, device, num_thresholds=100):
    """高精度閾値最適化"""
    print("Advanced threshold optimization...")
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for videos, targets in tqdm(val_loader, desc="Collecting predictions", leave=False):
            videos, targets = videos.to(device), targets.to(device)
            ensemble_preds = ultra_test_time_augmentation(model, videos, device)
            all_probs.append(ensemble_preds.cpu())
            all_targets.append(targets.cpu())
    
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # より細かい閾値探索
    best_threshold = 0.5
    best_f1 = 0.0
    
    # 粗い探索
    coarse_thresholds = np.linspace(0.05, 0.95, 20)
    coarse_f1s = []
    
    for threshold in coarse_thresholds:
        preds = (all_probs > threshold).astype(float)
        _, _, f1 = compute_precision_recall_f1(preds, all_targets)
        coarse_f1s.append(f1)
    
    # 最良の範囲を特定
    best_idx = np.argmax(coarse_f1s)
    center = coarse_thresholds[best_idx]
    
    # 細かい探索
    fine_range = 0.1
    fine_thresholds = np.linspace(max(0.05, center - fine_range), 
                                 min(0.95, center + fine_range), num_thresholds)
    
    for threshold in fine_thresholds:
        preds = (all_probs > threshold).astype(float)
        _, _, f1 = compute_precision_recall_f1(preds, all_targets)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")
    return best_threshold, best_f1

def compute_precision_recall_f1(y_pred, y_target):
    """
    y_pred: 0 or 1 binary prediction.
    target: binary prediction
    """
    tp = np.sum(y_pred * y_target)
    fp = np.sum(y_pred * (1 - y_target))
    fn = np.sum((1 - y_pred) * y_target)
    
    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    
    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    
    if precision + recall <= 1e-4:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1

def evaluate_with_threshold(model, val_loader, device, threshold=0.5):
    """TTA付き評価"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for videos, targets in val_loader:
            videos, targets = videos.to(device), targets.to(device)
            ensemble_preds = ultra_test_time_augmentation(model, videos, device)
            preds = (ensemble_preds > threshold).float()
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return compute_precision_recall_f1(all_preds, all_targets)

def train_model_ultra_performance():
    print("🚀 Ultra Performance Training Mode")
    print("FlashAttention Enabled:", torch.backends.cuda.flash_sdp_enabled())

    # パス設定
    annotation_dir = "/home/ollo/VideoMAE/videomae-clean"
    video_root = "/srv/shared/data/ego4d/short_clips/verb_annotation_simple"
    checkpoint_path = "/home/ollo/VideoMAE/checkpoints/vit_b_hybrid_pt_800e_k710_ft.pth"
    train_json = os.path.join(annotation_dir, "20250512_annotations_train.json")
    val_json = os.path.join(annotation_dir, "20250512_annotations_val.json")

    # 強化されたデータセット
    train_dataset = EnhancedEgo4DDataset(train_json, video_root, 
                                       get_enhanced_transforms('train'), mode='train')
    val_dataset = EnhancedEgo4DDataset(val_json, video_root, 
                                     get_enhanced_transforms('val'), mode='val')

    # WeightedRandomSamplerで不均衡データのバランス調整（高速版）
    sample_weights = compute_sample_weights_fast(train_json, num_classes=58)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # 重複サンプリング許可
    )

    # より大きなバッチサイズ
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            sampler=sampler,  # samplerを使う場合はshuffle=False
                            num_workers=6, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=6, pin_memory=True)

    print(f"✅ WeightedRandomSampler configured for balanced training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)

    # 事前学習済みモデル読み込み
    if os.path.exists(checkpoint_path):
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint_data, strict=False)
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"⚠️ Checkpoint not found at {checkpoint_path}")

    # スマートなpos_weight計算
    print("Computing smart positive weights...")
    pos_weights = compute_pos_weights_smart(train_json, num_classes=58).to(device)

    # 高性能Focal Loss
    criterion = AdvancedFocalLoss(alpha=0.5, gamma=1.0, pos_weight=pos_weights, label_smoothing=0.05)
    print("✅ Advanced Focal Loss configured")

    # 最適化器（異なる学習率）
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'head' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 5e-5},  # Backbone: 低い学習率
        {'params': head_params, 'lr': 5e-4}       # Head: 高い学習率
    ], weight_decay=0.05)

    # Warm-up + Cosine Annealing
    num_epochs = 25
    warmup_epochs = 3
    total_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader) * warmup_epochs
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        else:
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    # 学習ループ
    best_f1 = 0.0
    best_threshold = 0.5
    patience = 8
    no_improvement = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for videos, targets in pbar:
            videos, targets = videos.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(videos)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / num_batches

        # Validation (every 3 epochs for threshold optimization)
        if (epoch + 1) % 3 == 0:
            optimal_threshold, _ = find_optimal_threshold_advanced(model, val_loader, device)
            best_threshold = optimal_threshold

        # Standard validation
        precision, recall, f1 = evaluate_with_threshold(model, val_loader, device, best_threshold)
        
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\n[Epoch {epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Threshold: {best_threshold:.4f}")
        print(f"  🎯 Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Early stopping and best model saving
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "videomae_ultra_best.pth")
            print(f"  🏆 New best F1: {best_f1:.4f}! Model saved.")
            no_improvement = 0
        else:
            no_improvement += 1
        
        if no_improvement >= patience and epoch > 10:
            print(f"  🛑 Early stopping triggered after {patience} epochs without improvement")
            break
        
        print("-" * 80)

    # Final results
    print(f"\n🎉 Training Complete!")
    print(f"🏆 Best F1 Score: {best_f1:.4f}")
    print(f"🎯 Optimal Threshold: {best_threshold:.4f}")
    
    # Save final artifacts
    torch.save(model.state_dict(), "videomae_ultra_final.pth")
    
    results = {
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'final_epoch': epoch + 1
    }
    
    import pickle
    with open('ultra_training_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to: ultra_training_results.pkl")
    print(f"Best model saved to: videomae_ultra_best.pth")
    print(f"Final model saved to: videomae_ultra_final.pth")

if __name__ == "__main__":
    train_model_ultra_performance()