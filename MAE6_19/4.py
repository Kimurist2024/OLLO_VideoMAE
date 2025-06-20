#!/usr/bin/env python3
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings('ignore')

# Model definitions (same as training code)
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
        self.head = nn.Linear(embed_dim, num_classes)
        trunc_normal_(self.head.weight, std=.02)

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

# Load class labels from annotation file
def load_class_labels(annotation_file):
    """Load class labels from annotation JSON file"""
    # Actual class labels based on provided categories
    label_names = {
        0: "r_写っている",
        1: "r_tc_はい",
        2: "r_切る",
        3: "r_削る",
        4: "r_掘る",
        5: "r_描く",
        6: "r_注ぐ",
        7: "r_引く",
        8: "r_押す",
        9: "r_置く",
        10: "r_取る",
        11: "r_押さえる",
        12: "r_回す",
        13: "r_ひっくり返す",
        14: "r_落とす",
        15: "r_叩く",
        16: "r_縛る・折る",
        17: "r_(ネジなどを)締める",
        18: "r_(ネジなどを)緩める",
        19: "r_開ける",
        20: "r_閉める",
        21: "r_洗う",
        22: "r_拭く",
        23: "r_捨てる",
        24: "r_投げる",
        25: "r_混ぜる",
        26: "l_写っている",
        27: "l_tc_はい",
        28: "l_切る",
        29: "l_削る",
        30: "l_掘る",
        31: "l_描く",
        32: "l_注ぐ",
        33: "l_引く",
        34: "l_押す",
        35: "l_置く",
        36: "l_取る",
        37: "l_押さえる",
        38: "l_回す",
        39: "l_ひっくり返す",
        40: "l_落とす",
        41: "l_叩く",
        42: "l_縛る・折る",
        43: "l_(ネジなどを)締める",
        44: "l_(ネジなどを)緩める",
        45: "l_開ける",
        46: "l_閉める",
        47: "l_洗う",
        48: "l_拭く",
        49: "l_捨てる",
        50: "l_投げる",
        51: "l_混ぜる",
        52: "c_右手から左手に持ち替える",
        53: "c_左手から右手に持ち替える",
        54: "s_物を触って調べる",
        55: "s_メモ、本などを読む・調べる",
        56: "m_はい",
        57: "t_はい"
    }
    
    return label_names

class VideoInference:
    def __init__(self, model_path, annotation_file, device='cuda'):
        # Ensure we're using the correct CUDA device
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using CUDA device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
            
        self.model = get_model().to(self.device)
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}")
            print("Trying alternate path...")
            # Try without full path
            alt_path = 'videomae_finetuned_interactive.pth'
            if os.path.exists(alt_path):
                model_path = alt_path
                print(f"Found model at {model_path}")
            else:
                raise FileNotFoundError(f"Model file not found at {model_path} or {alt_path}")
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.num_frames = 16
        self.label_names = load_class_labels(annotation_file)
        
        # Try to load Japanese font
        self.font_path = None
        self.font = None
        font_candidates = [
            '/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf',
            '/usr/share/fonts/truetype/takao-mincho/TakaoMincho.ttf',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf',
            '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',  # macOS
            'C:/Windows/Fonts/msgothic.ttc',  # Windows
            'C:/Windows/Fonts/YuGothM.ttc',   # Windows
        ]
        
        for font_path in font_candidates:
            if os.path.exists(font_path):
                self.font_path = font_path
                try:
                    self.font = ImageFont.truetype(self.font_path, 24)
                    print(f"Using Japanese font: {font_path}")
                    break
                except:
                    continue
        
        if self.font is None:
            print("Warning: Japanese font not found. Text may not display correctly.")
            # Use default font as fallback
            try:
                self.font = ImageFont.load_default()
            except:
                self.font = None
    
    def preprocess_frames(self, frames):
        """Preprocess frames for model input"""
        processed_frames = []
        for frame in frames:
            frame_tensor = self.transform(frame)
            processed_frames.append(frame_tensor)
        
        # Stack frames and add batch dimension
        video_tensor = torch.stack(processed_frames)  # [T, C, H, W]
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # [C, T, H, W]
        video_tensor = video_tensor.unsqueeze(0)  # [1, C, T, H, W]
        
        return video_tensor
    
    def predict(self, frames):
        """Make prediction on frames"""
        with torch.no_grad():
            video_tensor = self.preprocess_frames(frames).to(self.device)
            outputs = self.model(video_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # Get top-k predictions
        top_k = 5
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_probs = probs[top_indices]
        
        predictions = []
        for idx, prob in zip(top_indices, top_probs):
            if prob > 0.1:  # Threshold for showing predictions
                predictions.append((self.label_names[idx], prob))
        
        return predictions
    
    def process_video(self, video_path, output_path, fps=10):
        """Process video and add predictions overlay"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        orig_fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame skip for desired fps
        frame_skip = max(1, orig_fps // fps)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Buffer for frames
        frame_buffer = []
        frame_count = 0
        
        print(f"Processing video: {video_path}")
        print(f"Original FPS: {orig_fps}, Output FPS: {fps}")
        print(f"Total frames: {total_frames}")
        
        pbar = tqdm(total=total_frames)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames to achieve desired fps
            if frame_count % frame_skip == 0:
                frame_buffer.append(frame)
                
                # Keep only the last num_frames
                if len(frame_buffer) > self.num_frames:
                    frame_buffer.pop(0)
                
                # Make prediction when we have enough frames
                if len(frame_buffer) == self.num_frames:
                    predictions = self.predict(frame_buffer)
                    
                    # Draw predictions on frame
                    frame_with_pred = self.draw_predictions(frame.copy(), predictions)
                    out.write(frame_with_pred)
            
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Output saved to: {output_path}")
    
    def draw_predictions(self, frame, predictions):
        """Draw predictions on frame using PIL for Japanese text support"""
        # Convert CV2 image to PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # Text properties
        padding = 15
        line_height = 35
        
        # Calculate maximum text width for background
        max_width = 0
        display_predictions = predictions[:8]  # Show top 8
        
        # If font is available, calculate text sizes
        if self.font:
            for label, prob in display_predictions:
                text = f"{label}: {prob:.2f}"
                bbox = draw.textbbox((0, 0), text, font=self.font)
                text_width = bbox[2] - bbox[0]
                max_width = max(max_width, text_width)
        else:
            # Fallback estimation
            max_width = 300
        
        # Draw semi-transparent background
        bg_height = len(display_predictions) * line_height + 20
        overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(
            [(padding - 5, 10), (padding + max_width + 25, bg_height + 10)],
            fill=(0, 0, 0, 180)
        )
        pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(pil_image)
        
        # Draw predictions
        y_offset = 40
        for i, (label, prob) in enumerate(display_predictions):
            text = f"{label}: {prob:.2f}"
            
            # Determine text color based on label prefix
            if label.startswith('r_'):
                text_color = (0, 255, 0)  # Bright green for right hand
            elif label.startswith('l_'):
                text_color = (0, 191, 255)  # Deep sky blue for left hand
            elif label.startswith('c_'):
                text_color = (255, 255, 0)  # Yellow for change
            elif label.startswith('s_'):
                text_color = (255, 0, 255)  # Magenta for search
            elif label.startswith('m_'):
                text_color = (255, 165, 0)  # Orange for moving
            elif label.startswith('t_'):
                text_color = (0, 255, 255)  # Cyan for turning
            else:
                text_color = (255, 255, 255)  # White for others
            
            # Draw text with shadow for better visibility
            # Shadow
            draw.text((padding + 2, y_offset + 2), text, font=self.font, fill=(0, 0, 0))
            # Main text
            draw.text((padding, y_offset), text, font=self.font, fill=text_color)
            
            y_offset += line_height
        
        # Convert back to CV2 format
        frame_result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return frame_result

def main():
    # Check if CUDA device is properly set
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print(f"Using CUDA device: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    parser = argparse.ArgumentParser(description='Video inference with visualization')
    parser.add_argument('--model_path', type=str, default='/home/ollo/VideoMAE/videomae-clean/videomae_finetuned_interactive.pth',
                       help='Path to trained model')
    parser.add_argument('--video_dir', type=str, default='/home/ollo/Ollo_video',
                       help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, default='/home/ollo/VideoMAE/videomae-clean/MAE6_19/inference_results',
                       help='Output directory for processed videos')
    parser.add_argument('--annotation_file', type=str, 
                       default='/home/ollo/VideoMAE/videomae-clean/20250512_annotations_train.json',
                       help='Annotation file for class labels')
    parser.add_argument('--fps', type=int, default=10,
                       help='Output video FPS')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize inference engine
    inference = VideoInference(args.model_path, args.annotation_file, args.device)
    
    # Process all videos in directory
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(args.video_dir).glob(f'*{ext}'))
    
    print(f"Found {len(video_files)} videos to process")
    
    for video_path in video_files:
        output_path = os.path.join(args.output_dir, f'inference_{video_path.name}')
        try:
            inference.process_video(str(video_path), output_path, args.fps)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue

if __name__ == "__main__":
    main()