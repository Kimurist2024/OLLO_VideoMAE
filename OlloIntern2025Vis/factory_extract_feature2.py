
import collections
import sys
import time
import argparse
import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torchvision import transforms as torch_transforms
from timm.models.layers import drop_path, to_2tuple, trunc_normal_


from models import test_net
from inference_utils import input_maker, detector, data_writer


import decord
from decord import VideoReader
import sys


sys.path.append('/home/ollo/VideoMAE')
sys.path.append('/home/ollo/VideoMAE/OlloIntern2025Vis')


try:
    import video_transforms as video_transforms
    import volume_transforms as volume_transforms
except ImportError:
    print("Warning: VideoMAE transforms not found, using fallback transforms")
    import torchvision.transforms as transforms
    

    class VideoTransforms:
        @staticmethod
        def Compose(transforms_list):
            return transforms.Compose(transforms_list)
        
        @staticmethod
        def Resize(size, interpolation='bilinear'):
            return transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        
        @staticmethod
        def CenterCrop(size):
            return transforms.CenterCrop(size)
        
        @staticmethod
        def Normalize(mean, std):
            return transforms.Normalize(mean=mean, std=std)
    #何これ　　embededを参照してってchatGPTに打ったら出てきたけど
    class VolumeTransforms:
        @staticmethod
        def ToFloatTensorInZeroOne():
            return transforms.ToTensor()
    
    video_transforms = VideoTransforms()
    volume_transforms = VolumeTransforms()



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
        if init_values and init_values > 0:
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

def fix_position_embedding(model, img_size=224, patch_size=16, all_frames=16, tubelet_size=2, embed_dim=768):
    """Fix missing position embedding"""
    
    # Calculate correct number of patches
    patches_per_frame = (img_size // patch_size) ** 2  # 14*14 = 196 for 224x224
    temporal_patches = all_frames // tubelet_size        # 16//2 = 8
    total_patches = patches_per_frame * temporal_patches # 196*8 = 1568
    
    print(f"🔧 FIXING POSITION EMBEDDING:")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"  Patches per frame: {patches_per_frame}")
    print(f"  Frames: {all_frames}, Tubelet size: {tubelet_size}")
    print(f"  Temporal patches: {temporal_patches}")
    print(f"  Total patches needed: {total_patches}")
    print(f"  Embedding dimension: {embed_dim}")
    
    # Check current pos_embed
    if hasattr(model, 'pos_embed'):
        current_shape = model.pos_embed.shape
        print(f"  Current pos_embed shape: {current_shape}")
        
        if current_shape[1] == total_patches:
            print(f"  ✅ Position embedding is correct!")
            return model
        else:
            print(f"  ⚠️ Position embedding shape mismatch!")
    else:
        print(f"  ❌ No position embedding found!")
    
    # Create correct sinusoidal position embedding
    pos_embed = get_sinusoid_encoding_table(total_patches, embed_dim)
    
    # Register as buffer (non-trainable parameter)
    if hasattr(model, 'pos_embed'):
        # Replace existing
        del model.pos_embed
    
    model.register_buffer('pos_embed', pos_embed)
    print(f"  ✅ New position embedding created: {pos_embed.shape}")
    print(f"  ✅ Position embedding fixed successfully!")
    
    return model

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=58, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, fc_drop_rate=0., drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0., use_learnable_pos_emb=False,
                 init_scale=0., all_frames=16, tubelet_size=2, use_checkpoint=False, use_mean_pooling=True,
                 use_flash_attn=True):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, all_frames, tubelet_size)
        num_patches = self.patch_embed.num_patches
        
        # Always create position embedding (learnable or sinusoidal)
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)
            print(f"✅ Created learnable position embedding: {self.pos_embed.shape}")
        else:
            pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)
            self.register_buffer('pos_embed', pos_embed)
            print(f"✅ Created sinusoidal position embedding: {pos_embed.shape}")
            
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
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        trunc_normal_(self.head.weight, std=.02)

    def forward_features(self, x):
        x = self.patch_embed(x)
        
        # Ensure pos_embed matches input
        if x.shape[1] != self.pos_embed.shape[1]:
            print(f"⚠️ Shape mismatch: input patches {x.shape[1]}, pos_embed {self.pos_embed.shape[1]}")
            # Try to interpolate or truncate pos_embed if needed
            if x.shape[1] < self.pos_embed.shape[1]:
                pos_embed = self.pos_embed[:, :x.shape[1], :]
            else:
                # Need to extend pos_embed (this shouldn't happen with correct setup)
                pos_embed = self.pos_embed
        else:
            pos_embed = self.pos_embed
            
        x = x + pos_embed.to(x.device)
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.fc_norm(x.mean(1)) if self.fc_norm else x[:, 0]

    def forward(self, x):
        return self.head(self.fc_dropout(self.forward_features(x)))

def get_optimized_model(num_classes=58, img_size=224, all_frames=16, tubelet_size=2, use_flash_attn=True):
    """Create optimized model matching the training configuration"""
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        fc_drop_rate=0.,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        init_values=0.,
        use_learnable_pos_emb=False,
        init_scale=0.,
        all_frames=all_frames,
        tubelet_size=tubelet_size,
        use_checkpoint=False,
        use_mean_pooling=True,
        use_flash_attn=use_flash_attn
    )

# ========================================
# FACTORY-OPTIMIZED PREPROCESSING
# ========================================

# Actual category mappings (Japanese labels)
ACTUAL_CATEGORIES = {
    0: "r_写っている", 1: "r_tc_はい", 2: "r_切る", 3: "r_削る", 4: "r_掘る",
    5: "r_描く", 6: "r_注ぐ", 7: "r_引く", 8: "r_押す", 9: "r_置く",
    10: "r_取る", 11: "r_押さえる", 12: "r_回す", 13: "r_ひっくり返す", 14: "r_落とす",
    15: "r_叩く", 16: "r_縛る・折る", 17: "r_(ネジなどを)締める", 18: "r_(ネジなどを)緩める",
    19: "r_開ける", 20: "r_閉める", 21: "r_洗う", 22: "r_拭く", 23: "r_捨てる",
    24: "r_投げる", 25: "r_混ぜる", 26: "l_写っている", 27: "l_tc_はい", 28: "l_切る",
    29: "l_削る", 30: "l_掘る", 31: "l_描く", 32: "l_注ぐ", 33: "l_引く",
    34: "l_押す", 35: "l_置く", 36: "l_取る", 37: "l_押さえる", 38: "l_回す",
    39: "l_ひっくり返す", 40: "l_落とす", 41: "l_叩く", 42: "l_縛る・折る",
    43: "l_(ネジなどを)締める", 44: "l_(ネジなどを)緩める", 45: "l_開ける",
    46: "l_閉める", 47: "l_洗う", 48: "l_拭く", 49: "l_捨てる", 50: "l_投げる",
    51: "l_混ぜる", 52: "c_右手から左手に持ち替える", 53: "c_左手から右手に持ち替える",
    54: "s_物を触って調べる", 55: "s_メモ、本などを読む・調べる", 56: "m_はい", 57: "t_はい"
}

def apply_factory_preprocessing(img, crop_hands=True, enhance_contrast=True):
    """Apply factory-specific preprocessing optimizations"""
    if img is None:
        return img
        
    if crop_hands:
        h, w = img.shape[:2]
        # Focus on center-bottom area where most hand work happens
        h_start = max(0, int(h * 0.15))  # Start from 15% down
        h_end = min(h, int(h * 0.95))    # End at 95% down  
        w_start = max(0, int(w * 0.05))  # Start from 5% right
        w_end = min(w, int(w * 0.95))    # End at 95% right
        img = img[h_start:h_end, w_start:w_end]
    
    if enhance_contrast:
        # Enhance contrast for better hand detection in factory lighting
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    
    return img

class FactoryOptimizedPreprocessor(input_maker.Preprocessor):
    """Factory-optimized version of the Preprocessor class"""
    
    def __init__(self, *args, crop_hands=True, enhance_contrast=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.crop_hands = crop_hands
        self.enhance_contrast = enhance_contrast
    
    def update_image(self, orig_img, crop_area_bbox: list, frame: int, timestamp: int, is_last: bool):
        time_info = {}
        
        if crop_area_bbox is None:
            pass
        else:
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_area_bbox
            orig_img = orig_img[crop_y1:crop_y2, crop_x1:crop_x2]
            
        if self.centercrop and orig_img is not None:
            orig_img_h, orig_img_w = orig_img.shape[:2]
            crop_x1 = (orig_img_w - orig_img_h) // 2
            crop_x2 = crop_x1 + orig_img_h
            orig_img = orig_img[:, crop_x1:crop_x2]
            
        if self.img_w is None and self.img_h is None:
            if is_last:
                raise NotImplementedError("error")
            self.img_h, self.img_w, *_ = orig_img.shape
            
        if is_last:
            transformed_img = self.latest_image_list[-1]
            frame = self.latest_frame_list[-1]
            timestamp = self.latest_timestamp_list[-1]
        else:
            # Apply factory-specific preprocessing
            orig_img = apply_factory_preprocessing(orig_img, self.crop_hands, self.enhance_contrast)
            
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            orig_img = cv2.resize(orig_img, self.resize_img_wh)
            
            transform_st = time.perf_counter()
            
            transpose_st = time.perf_counter()
            orig_img = orig_img.transpose(2, 0, 1)
            transpose_time = time.perf_counter() - transpose_st
            time_info["transpose_time"] = transpose_time
            
            div_st = time.perf_counter()
            transformed_img = orig_img.astype(np.float32) / 255.0
            div_time = time.perf_counter() - div_st
            time_info["div_time"] = div_time
            
            sub_div_st = time.perf_counter()
            for i in range(transformed_img.shape[0]):
                transformed_img[i] -= self.mean_array[i]
                transformed_img[i] *= self.std_multiple_array[i]
            sub_div_time = time.perf_counter() - sub_div_st
            time_info["sub_div_time"] = sub_div_time
            
            transform_time = time.perf_counter() - transform_st
            time_info["transform_time"] = transform_time
            
        self.latest_image_list.append(transformed_img)
        self.latest_frame_list.append(frame)
        self.latest_timestamp_list.append(timestamp)
        
        model_name2x_tensor_list = {}
        pid = -100
        batch_info_list = []
        start_preprocess = time.perf_counter()
        
        if len(self.latest_image_list) >= self.window_size:
            insert_st = time.perf_counter()
            for i, img in enumerate(self.latest_image_list[-self.window_size :]):
                self.preallocated_buffer[i] = img
            insert_time = time.perf_counter() - insert_st
            time_info["insert_time"] = insert_time
            
            transpose_window_st = time.perf_counter()
            input_images = self.preallocated_buffer.copy().transpose(1, 0, 2, 3)
            transpose_window_time = time.perf_counter() - transpose_window_st
            time_info["transpose_window_time"] = transpose_window_time
            
            model_name2x_tensor_list["main"] = [input_images]
            insert_frame_list = self.latest_frame_list[-self.window_size :]
            insert_timestamp_list = self.latest_timestamp_list[-self.window_size :]
            
            batch_info_list.append({
                "pid": pid,
                "frame_list": insert_frame_list,
                "timestamp_list": insert_timestamp_list,
            })
            
            self.latest_image_list = self.latest_image_list[self.slide_size :]
            self.latest_frame_list = self.latest_frame_list[self.slide_size :]
            self.latest_timestamp_list = self.latest_timestamp_list[self.slide_size :]
            
        preprocess_time = time.perf_counter() - start_preprocess
        time_info["prepare_preprocess_time"] = preprocess_time
        
        return (model_name2x_tensor_list, batch_info_list, [], time_info)

# ========================================
# FACTORY ANALYSIS FUNCTIONS
# ========================================

def analyze_and_save_factory_results(all_output_results, output_dir, video_name, args):
    """Analyze results and save factory-specific analysis WITHOUT confidence thresholds"""
    analysis = {
        'all_detections': [],  # すべての検出結果を保存
        'dominant_actions': [],
        'workflow_stages': [],
        'activity_summary': {}
    }
    
    print(f"Analyzing {len(all_output_results)} output results...")
    
    # Process output results - NO CONFIDENCE THRESHOLD
    for result in all_output_results:
        features = result['features']
        frame_list = result['frame_list']
        timestamp_list = result.get('timestamp_list', [])
        
        # Apply sigmoid to get probabilities if needed
        if isinstance(features, np.ndarray):
            if features.max() > 1.0:  # Assume logits if values > 1
                features = 1 / (1 + np.exp(-features))  # Sigmoid
        
        # Process each frame
        if len(features.shape) == 2:  # (time, num_classes)
            for i, feature in enumerate(features):
                if i < len(frame_list):
                    # すべてのクラスの情報を保存（信頼度制限なし）
                    frame_detections = []
                    for class_id, confidence in enumerate(feature):
                        if confidence > 0.01:  # 非常に低い閾値（ほぼノイズを除外するだけ）
                            action_name = ACTUAL_CATEGORIES.get(class_id, f"unknown_{class_id}")
                            frame_detections.append({
                                'class_id': class_id,
                                'action': action_name,
                                'confidence': float(confidence)
                            })
                    
                    # 信頼度でソート（降順）
                    frame_detections.sort(key=lambda x: x['confidence'], reverse=True)
                    
                    # このフレームのすべての検出を保存
                    analysis['all_detections'].append({
                        'frame': frame_list[i],
                        'timestamp': timestamp_list[i] if i < len(timestamp_list) else 0.0,
                        'detections': frame_detections
                    })
                    
                    # 最も信頼度の高いアクションをdominant_actionsに追加
                    if frame_detections:
                        top_detection = frame_detections[0]
                        analysis['dominant_actions'].append({
                            'frame': frame_list[i],
                            'timestamp': timestamp_list[i] if i < len(timestamp_list) else 0.0,
                            'action': top_detection['action'],
                            'class_id': top_detection['class_id'],
                            'confidence': top_detection['confidence']
                        })
                        
        elif len(features.shape) == 1:  # Single prediction for entire window
            # すべてのクラスの情報を保存
            window_detections = []
            for class_id, confidence in enumerate(features):
                if confidence > 0.01:
                    action_name = ACTUAL_CATEGORIES.get(class_id, f"unknown_{class_id}")
                    window_detections.append({
                        'class_id': class_id,
                        'action': action_name,
                        'confidence': float(confidence)
                    })
            
            window_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            analysis['all_detections'].append({
                'frame_range': frame_list,
                'timestamp_range': timestamp_list,
                'detections': window_detections
            })
            
            if window_detections:
                top_detection = window_detections[0]
                analysis['dominant_actions'].append({
                    'frame_range': frame_list,
                    'timestamp_range': timestamp_list,
                    'action': top_detection['action'],
                    'class_id': top_detection['class_id'],
                    'confidence': top_detection['confidence']
                })
    
    # Create workflow stages based on dominant actions
    if analysis['dominant_actions']:
        analysis['dominant_actions'].sort(key=lambda x: x.get('timestamp', x.get('frame', 0)))
        
        current_stage = {
            'action': analysis['dominant_actions'][0]['action'],
            'start_time': analysis['dominant_actions'][0].get('timestamp', 0),
            'frames': [],
            'confidences': []
        }
        
        for action_info in analysis['dominant_actions']:
            if action_info['action'] == current_stage['action']:
                if 'frame' in action_info:
                    current_stage['frames'].append(action_info['frame'])
                current_stage['confidences'].append(action_info['confidence'])
            else:
                # Finalize current stage
                current_stage['duration_frames'] = len(current_stage['frames'])
                current_stage['avg_confidence'] = np.mean(current_stage['confidences'])
                analysis['workflow_stages'].append(current_stage.copy())
                
                # Start new stage
                current_stage = {
                    'action': action_info['action'],
                    'start_time': action_info.get('timestamp', 0),
                    'frames': [action_info.get('frame', 0)] if 'frame' in action_info else [],
                    'confidences': [action_info['confidence']]
                }
        
        # Add final stage
        if current_stage['frames']:
            current_stage['duration_frames'] = len(current_stage['frames'])
            current_stage['avg_confidence'] = np.mean(current_stage['confidences'])
            analysis['workflow_stages'].append(current_stage)
    
    # Activity summary
    if analysis['dominant_actions']:
        actions = [action['action'] for action in analysis['dominant_actions']]
        unique_actions = list(set(actions))
        
        # クラスごとの統計情報も計算
        class_statistics = {}
        for detection_frame in analysis['all_detections']:
            for detection in detection_frame['detections']:
                class_id = detection['class_id']
                if class_id not in class_statistics:
                    class_statistics[class_id] = {
                        'action_name': detection['action'],
                        'count': 0,
                        'confidences': [],
                        'max_confidence': 0,
                        'mean_confidence': 0
                    }
                class_statistics[class_id]['count'] += 1
                class_statistics[class_id]['confidences'].append(detection['confidence'])
        
        # 統計値を計算
        for class_id, stats in class_statistics.items():
            stats['max_confidence'] = max(stats['confidences'])
            stats['mean_confidence'] = np.mean(stats['confidences'])
            del stats['confidences']  # メモリ節約のため
        
        analysis['activity_summary'] = {
            'total_detected_actions': len(analysis['dominant_actions']),
            'unique_actions_count': len(unique_actions),
            'unique_actions': unique_actions,
            'action_frequencies': {action: actions.count(action) for action in unique_actions},
            'most_frequent_action': max(unique_actions, key=lambda x: actions.count(x)) if unique_actions else 'none',
            'avg_confidence': np.mean([action['confidence'] for action in analysis['dominant_actions']]),
            'high_confidence_actions': len([a for a in analysis['dominant_actions'] if a['confidence'] > 0.7]),
            'class_statistics': class_statistics
        }
    
    # Save factory analysis results
    factory_results = {
        'video_name': video_name,
        'analysis_config': {
            'window_size': args.window_size,
            'input_size': args.input_size,
            'num_classes': args.num_classes,
            'analysis_fps': args.analysis_fps,
            'checkpoint_path': args.checkpoint_path,
            'model_architecture': 'CustomViT_from_paste_txt',
            'confidence_threshold': 'none (all detections saved)',
            'flash_attention': args.use_flash_attn if hasattr(args, 'use_flash_attn') else True
        },
        'factory_analysis': analysis,
        'class_mapping': ACTUAL_CATEGORIES,
        'workflow_summary': {
            'num_workflow_stages': len(analysis['workflow_stages']),
            'total_actions_detected': analysis['activity_summary'].get('total_detected_actions', 0),
            'unique_actions_detected': analysis['activity_summary'].get('unique_actions_count', 0),
            'most_frequent_action': analysis['activity_summary'].get('most_frequent_action', 'none'),
            'avg_confidence': analysis['activity_summary'].get('avg_confidence', 0.0),
            'high_confidence_actions': analysis['activity_summary'].get('high_confidence_actions', 0)
        }
    }
    
    output_file = os.path.join(output_dir, f'{video_name}_factory_analysis.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(factory_results, f, indent=2, ensure_ascii=False, 
                  default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else str(x))
    
    print(f"Factory analysis saved to: {output_file}")
    return factory_results



class OptimizedModelDetector:
    """Optimized detector for custom ViT model with proper tensor handling"""
    
    def __init__(self, model, device_id=0, inference_batch_size=8, logger=None):
        self.model = model
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.inference_batch_size = inference_batch_size
        self.logger = logger
        self.output_checked = False
        
        print(f"Model loaded on device: {self.device}")
        if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
            print(f"FlashAttention enabled: {torch.backends.cuda.flash_sdp_enabled()}")
    
    def _debug_model_output(self, outputs):
        """Debug model output to understand format"""
        print(f"🔍 DEBUG MODEL OUTPUT:")
        print(f"  Shape: {outputs.shape}")
        print(f"  Range: [{outputs.min():.4f}, {outputs.max():.4f}]")
        print(f"  Mean: {outputs.mean():.4f}")
        print(f"  Std: {outputs.std():.4f}")
        
        
        if len(outputs.shape) == 2:
            top_values, top_indices = torch.topk(outputs[0], 5)
            print(f"  Top 5 values: {top_values.cpu().numpy()}")
            print(f"  Top 5 indices: {top_indices.cpu().numpy()}")
        
       
        if outputs.max() > 5.0 or outputs.min() < -5.0:
            print(f"  → Detected: LOGITS (applying sigmoid)")
            return torch.sigmoid(outputs)
        elif outputs.max() <= 1.0 and outputs.min() >= 0.0 and outputs.sum(dim=-1).mean() < 10.0:
            print(f"  → Detected: PROBABILITIES (no sigmoid needed)")
            return outputs
        else:
            print(f"  → Uncertain format, applying sigmoid for safety")
            return torch.sigmoid(outputs)
    
    def update(self, detected_model_name2x_tensor_list, detected_batch_info_list, 
               is_last=False, now_video_frame=None, now_timestamp=None, model_name="main"):
        
        if not detected_model_name2x_tensor_list or model_name not in detected_model_name2x_tensor_list:
            return []
        
        results = []
        
        with torch.no_grad():
            for batch_idx, x_tensor in enumerate(detected_model_name2x_tensor_list[model_name]):
                
                if isinstance(x_tensor, np.ndarray):
                    x_tensor = torch.from_numpy(x_tensor).float()
                
                
                if len(x_tensor.shape) == 4: 
                    x_tensor = x_tensor.unsqueeze(0) 
                elif len(x_tensor.shape) == 5:  
                    pass
                else:
                    if self.logger:
                        self.logger.warning(f"Unexpected tensor shape: {x_tensor.shape}")
                    continue
                
                x_tensor = x_tensor.to(self.device)
                
                try:
                    
                    outputs = self.model(x_tensor)
                    
                   
                    if not self.output_checked:
                        features = self._debug_model_output(outputs).cpu().numpy()
                        self.output_checked = True
                    else:
                        
                        if outputs.max() > 5.0 or outputs.min() < -5.0:
                            features = torch.sigmoid(outputs).cpu().numpy()
                        elif outputs.max() <= 1.0 and outputs.min() >= 0.0:
                            features = outputs.cpu().numpy()
                        else:
                            features = torch.sigmoid(outputs).cpu().numpy()
                    
                  
                    batch_info = detected_batch_info_list[batch_idx] if batch_idx < len(detected_batch_info_list) else {}
                    
                    result = {
                        'features': features,
                        'frame_list': batch_info.get('frame_list', []),
                        'timestamp_list': batch_info.get('timestamp_list', []),
                        'pid': batch_info.get('pid', -1),
                        'video_frame': now_video_frame,
                        'timestamp': now_timestamp
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error in model forward pass: {e}")
                    else:
                        print(f"Error in model forward pass: {e}")
                    continue
        
        return results



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="可視化したい動画ファイル")
    parser.add_argument("--output_dir", type=str, required=True, help="model の予測結果を保存するディレクトリ")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="読み込むモデルチェックポイントのパス")
    parser.add_argument("--analysis_fps", type=int, required=True, help="動画の分析のfps (例: 10)")
    parser.add_argument("--inference_batch_size", type=int, required=True, help="推論のバッチサイズ")
    parser.add_argument("--window_size", type=int, required=True, help="model 学習時の model の windowsize")
    parser.add_argument("--input_size", type=int, default=224, help="model 学習時の model の input size")
    parser.add_argument("--num_classes", type=int, default=58, help="model 学習時の model の num_classes")
    parser.add_argument("--tubelet_size", type=int, default=2, help="model 学習時の model の tubelet_size")
    
   
    parser.add_argument("--crop_hands", action='store_true', default=True, help="Focus on hand workspace region")
    parser.add_argument("--enhance_contrast", action='store_true', default=True, help="Enhance contrast for factory lighting")
    parser.add_argument("--use_flash_attn", action='store_true', default=True, help="Use FlashAttention for better performance")
    
    return parser.parse_args()

def main(args, logger=None):
    start = time.time()
    if logger is None:
        print_ = print
    else:
        print_ = logger.info
        
    msg = "####### OPTIMIZED FACTORY EXTRACT FEATURE #######"
    print_("#" * len(msg))
    print_(msg)
    print_("#" * len(msg))
    
    
    start_frame = 0
    end_frame = 10000000
    
    
    decord.bridge.set_bridge('torch')
    read_cap = VideoReader(args.video_path)
    
    fps = int(round(read_cap.get_avg_fps()))
    total_frame_num = len(read_cap)
    end_frame = min(end_frame, total_frame_num)
    if end_frame == -1:
        end_frame = total_frame_num
    
    cudnn.benchmark = True
  
    print_(f"Loading optimized model with:")
    print_(f"  - num_classes: {args.num_classes}")
    print_(f"  - input_size: {args.input_size}")
    print_(f"  - window_size: {args.window_size}")
    print_(f"  - tubelet_size: {args.tubelet_size}")
    print_(f"  - flash_attention: {args.use_flash_attn}")
    
    
    model = get_optimized_model(
        num_classes=args.num_classes,
        img_size=args.input_size,
        all_frames=args.window_size,
        tubelet_size=args.tubelet_size,
        use_flash_attn=args.use_flash_attn
    )
    
   
    checkpoint_path = args.checkpoint_path
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            
            
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            print_(f" Model weights loaded successfully from {checkpoint_path}")
            if missing_keys:
                print_(f" Missing keys: {missing_keys}")
            if unexpected_keys:
                print_(f"  Unexpected keys: {unexpected_keys}")
           
            if 'pos_embed' in missing_keys:
                print_(f"🔧 Fixing missing position embedding...")
                model = fix_position_embedding(
                    model, 
                    img_size=args.input_size,
                    patch_size=16,
                    all_frames=args.window_size,
                    tubelet_size=args.tubelet_size,
                    embed_dim=768
                )
                print_(f" Position embedding fixed!")
            else:
                print_(f" Position embedding already present")
                
        except Exception as e:
            print_(f"Error loading checkpoint: {e}")
            print_(f"Please ensure the checkpoint matches the model architecture")
            return
    else:
        print_(f" Checkpoint not found at {checkpoint_path}")
        return
    
  
    if total_frame_num > 1000:
        skip_length = 1
        slide_size = 2
    else:
        skip_length = 1
        slide_size = 1
    
    
    transform = video_transforms.Compose([
        video_transforms.Resize(256, interpolation='bilinear'),
        video_transforms.CenterCrop(size=(args.input_size, args.input_size)),
        volume_transforms.ToFloatTensorInZeroOne(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    
    model_name2data_transform = {"main": transform}
    
    
    input_preprocessor = FactoryOptimizedPreprocessor(
        logger=logger,
        model_name2data_transform=model_name2data_transform,
        analysis_img_wh=(args.input_size, args.input_size),
        window_size=args.window_size,
        skipper=skip_length,
        slide_size=slide_size,
        is_keep_aspratio=False,
        resize_img_wh=(args.input_size, args.input_size),
        centercrop=False,
        crop_hands=args.crop_hands,
        enhance_contrast=args.enhance_contrast,
    )
    
   
    model_detector = OptimizedModelDetector(
        model=model,
        device_id=0,
        inference_batch_size=args.inference_batch_size,
        logger=logger
    )
    
   
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    analysis_fps = args.analysis_fps
    video_fps = fps
    
    
    analysis_frame_set = set()
    all_output_results = []  
    
    print_(f"Processing video:")
    print_(f"  - start frame = {start_frame}, end frame = {end_frame}")
    print_(f"  - video fps = {video_fps}, analysis fps = {analysis_fps}")
    print_(f"  - factory optimizations: crop_hands={args.crop_hands}, enhance_contrast={args.enhance_contrast}")
    
    time_info_history = collections.defaultdict(list)
    progress_bar = tqdm(total=end_frame - start_frame + 1)
    prev_video_index = 0
    
    iter_st = time.time()
    video_analysis_skipper = int(round(video_fps / analysis_fps))
    first_analysis_frame = 0
    now_timestamp = 0.0
    
    for fi in range(start_frame, end_frame + 1):
        try:
            img = read_cap[fi].numpy() 
        except (IndexError, decord.DECORDError):
            break
            
        video_frame = fi
        diff_frame = fi - prev_video_index
        progress_bar.update(diff_frame)
        prev_video_index = fi
        now_timestamp += 1 / fps
        
        is_infer = fi == end_frame
        analysis_frame = video_frame // video_analysis_skipper
        
        if not is_infer:
            if (analysis_frame - first_analysis_frame) % skip_length != 0:
                continue
                
        if analysis_frame in analysis_frame_set:
            continue
            
        analysis_frame_set.add(analysis_frame)
        
      
        preprocess_st = time.time()
        (
            model_name2x_tensor_list,
            batch_info_list,
            pop_pid_list,
            time_info,
        ) = input_preprocessor.update_image(
            orig_img=img,
            crop_area_bbox=None,
            frame=analysis_frame,
            timestamp=now_timestamp,
            is_last=is_infer,
        )
        preprocess_time = time.time() - preprocess_st
        time_info["preprocess_time"] = preprocess_time
        
    
        model_forward_st = time.time()
        output_result = model_detector.update(
            detected_model_name2x_tensor_list=model_name2x_tensor_list,
            detected_batch_info_list=batch_info_list,
            is_last=False,
            now_video_frame=fi,
            now_timestamp=now_timestamp,
        )
        
      
        all_output_results.extend(output_result)
        
        model_forward_time = time.time() - model_forward_st
        time_info["model_forward_time"] = model_forward_time
        
        duration = time.time() - iter_st
        time_info["iteration_duration"] = duration
        iter_st = time.time()
        
        for key, value in time_info.items():
            time_info_history[key].append(value)
    
    progress_bar.close()
    

    print_(f"Processing final frames...")
    
 
    last_update_generator = input_preprocessor.last_update(
        inference_length=args.window_size
    )
    
    for last_output in last_update_generator:
        (
            model_name2x_tensor_list,
            batch_info_list,
            pop_pid_list,
            time_info,
            fi,
        ) = last_output
        
        video_frame = end_frame + fi * skip_length + 1
        
        if len(model_name2x_tensor_list):
            output_result = model_detector.update(
                detected_model_name2x_tensor_list=model_name2x_tensor_list,
                detected_batch_info_list=batch_info_list,
                now_video_frame=video_frame,
                is_last=False,
            )
            all_output_results.extend(output_result)
        
        now_timestamp = video_frame / fps
    
   
    output_result = model_detector.update(
        detected_model_name2x_tensor_list={},
        detected_batch_info_list=[],
        is_last=True,
        now_video_frame=video_frame,
        now_timestamp=now_timestamp
    )
    all_output_results.extend(output_result)
    
    
    time_info_summary = {}
    for key, value in time_info_history.items():
        mean = np.mean(value)
        sum_val = np.sum(value)
        time_info_summary[key] = {"mean": mean, "sum": sum_val}
    
    print_(f"Performance summary: {time_info_summary}")
    

    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    factory_results = analyze_and_save_factory_results(all_output_results, output_dir, video_name, args)
    
    total_time = time.time() - start
    print_(f"Total processing time: {total_time:.2f} sec")
    
   
    summary = factory_results['workflow_summary']
    print_(f"\n" + "="*60)
    print_(f"OPTIMIZED FACTORY ANALYSIS SUMMARY")
    print_(f"="*60)
    print_(f"Video: {video_name}")
    print_(f"Model: Custom ViT (from paste.txt)")
    print_(f"FlashAttention: {args.use_flash_attn}")
    print_(f"Total actions detected: {summary['total_actions_detected']}")
    print_(f"Unique actions: {summary['unique_actions_detected']}")
    print_(f"Workflow stages: {summary['num_workflow_stages']}")
    print_(f"High confidence actions: {summary['high_confidence_actions']}")
    print_(f"Average confidence: {summary['avg_confidence']:.3f}")
    print_(f"Most frequent action: {summary['most_frequent_action']}")
    print_(f"="*60)
    
  
    if 'class_statistics' in factory_results['factory_analysis']['activity_summary']:
        print_(f"\n" + "="*60)
        print_(f"TOP DETECTED CLASSES")
        print_(f"="*60)
        class_stats = factory_results['factory_analysis']['activity_summary']['class_statistics']
        sorted_classes = sorted(class_stats.items(), key=lambda x: x[1]['max_confidence'], reverse=True)
        
        for class_id, stats in sorted_classes[:10]:  
            print_(f"Class {class_id}: {stats['action_name']}")
            print_(f"  - Count: {stats['count']}")
            print_(f"  - Max confidence: {stats['max_confidence']:.4f}")
            print_(f"  - Mean confidence: {stats['mean_confidence']:.4f}")
        print_(f"="*60)
    
    print_(f"Analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    args = get_args()
   
    logger = None
    main(args, logger=logger)