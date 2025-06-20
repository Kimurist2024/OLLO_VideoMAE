#!/usr/bin/env python3
"""
Standalone VideoMAE Fine-tuning Script for Ego4D
Usage: python 7.py
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
import math
import sys
import time
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

# ================== Configuration ==================
class Config:
    # Data paths
    annotation_dir = "/home/ollo/VideoMAE/videomae-clean"
    video_root = "/srv/shared/data/ego4d/short_clips/verb_annotation_simple"
    checkpoint_path = "/home/ollo/VideoMAE/checkpoints/vit_b_hybrid_pt_800e_k710_ft.pth"
    train_json = os.path.join(annotation_dir, "20250512_annotations_train.json")
    val_json = os.path.join(annotation_dir, "20250512_annotations_val.json")
    
    # Output
    output_dir = "./output_ego4d_finetune"
    
    # Model
    model_type = "vit_base_patch16_224"
    num_frames = 16
    tubelet_size = 2
    input_size = 224
    patch_size = 16
    
    # Training
    batch_size = 8
    epochs = 30
    lr = 1e-3
    weight_decay = 0.05
    warmup_epochs = 5
    save_freq = 5
    
    # Video
    sampling_rate = 4
    
    # System
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

config = Config()

# ================== Video Transforms ==================
class VideoNormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
    
    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

class RandomResizedCropVideo:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(0.75, 1.3333)):
        self.size = size
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self, video):
        # video shape: [T, C, H, W]
        _, _, height, width = video.shape
        
        area = height * width
        for _ in range(10):
            target_area = np.random.uniform(*self.scale) * area
            aspect_ratio = np.random.uniform(*self.ratio)
            
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if 0 < w <= width and 0 < h <= height:
                i = np.random.randint(0, height - h + 1)
                j = np.random.randint(0, width - w + 1)
                video = video[:, :, i:i+h, j:j+w]
                # Resize to target size
                video = F.interpolate(video, size=(self.size, self.size), mode='bilinear', align_corners=False)
                return video
        
        # Fallback
        video = F.interpolate(video, size=(self.size, self.size), mode='bilinear', align_corners=False)
        return video

class CenterCropVideo:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, video):
        _, _, h, w = video.shape
        th, tw = self.size, self.size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return video[:, :, i:i+th, j:j+tw]

# ================== Dataset ==================
class Ego4DDataset(Dataset):
    def __init__(self, annotation_json, video_root, mode='train', config=None):
        self.video_root = video_root
        self.mode = mode
        self.config = config or Config()
        
        # Load annotations
        with open(annotation_json, 'r') as f:
            data = json.load(f)
        
        # Handle the JSON structure with categories and videos
        if isinstance(data, dict):
            # Extract categories mapping
            if 'categories' in data:
                self.categories = data['categories']
                self.idx_to_label = {idx: label for label, idx in self.categories.items()}
                self.label_to_idx = self.categories
                self.num_classes = len(self.categories)
                self.labels = list(self.categories.keys())
                
                # Now extract video annotations
                self.annotations = []
                for key, value in data.items():
                    if key != 'categories':
                        if isinstance(value, dict) and 'videos' in value:
                            # Format: {"split_name": {"videos": {...}}}
                            for video_name, label_idx in value['videos'].items():
                                self.annotations.append({
                                    'file_name': video_name,
                                    'label_idx': label_idx,
                                    'label': self.idx_to_label[label_idx]
                                })
                        elif isinstance(value, list):
                            # Format: {"videos": [...]}
                            for item in value:
                                self.annotations.append(item)
                
                # If no annotations found in nested structure, look for 'videos' key
                if not self.annotations and 'videos' in data:
                    videos_data = data['videos']
                    if isinstance(videos_data, dict):
                        for video_name, label_idx in videos_data.items():
                            self.annotations.append({
                                'file_name': video_name,
                                'label_idx': label_idx,
                                'label': self.idx_to_label.get(label_idx, f'unknown_{label_idx}')
                            })
                    elif isinstance(videos_data, list):
                        self.annotations = videos_data
                        
            else:
                # Old format handling - direct video to label mapping
                self.annotations = []
                self.labels = set()
                for video_name, label in data.items():
                    if isinstance(label, int):
                        label_str = f"action_{label}"
                    else:
                        label_str = str(label)
                    self.annotations.append({
                        'file_name': video_name,
                        'label': label_str
                    })
                    self.labels.add(label_str)
                
                self.labels = sorted(list(self.labels))
                self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
                self.num_classes = len(self.labels)
                
        elif isinstance(data, list):
            self.annotations = data
            # Extract labels from list
            self.labels = sorted(list(set([ann.get('label', 'unknown') for ann in self.annotations])))
            self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
            self.num_classes = len(self.labels)
        else:
            raise ValueError(f"Unexpected JSON structure. Type: {type(data)}")
        
        print(f"[{mode}] Found {self.num_classes} classes: {list(self.labels)[:5]}...")
        print(f"[{mode}] Loaded {len(self.annotations)} samples")
        
        if self.annotations and len(self.annotations) > 0:
            print(f"[{mode}] Sample annotation: {self.annotations[0]}")
        
        # Transforms
        self.normalize = VideoNormalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        if mode == 'train':
            self.spatial_transform = RandomResizedCropVideo(config.input_size)
        else:
            self.spatial_transform = CenterCropVideo(config.input_size)
    
    def _load_video(self, video_path):
        """Load video using available backend"""
        try:
            # Try decord first (faster)
            import decord
            decord.bridge.set_bridge('torch')
            
            vr = decord.VideoReader(video_path, num_threads=1)
            total_frames = len(vr)
            
            if total_frames == 0:
                return torch.zeros(self.config.num_frames, 3, self.config.input_size, self.config.input_size)
            
            # Sample frame indices
            indices = self._sample_indices(total_frames)
            frames = vr.get_batch(indices).float() / 255.0  # [T, H, W, C]
            frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
            
        except ImportError:
            # Fallback to OpenCV
            try:
                import cv2
            except ImportError:
                print("ERROR: Neither decord nor opencv-python is installed!")
                print("Please install one of them:")
                print("  pip install opencv-python")
                print("  or")
                print("  pip install decord")
                raise
            
            cap = cv2.VideoCapture(video_path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                cap.release()
                return torch.zeros(self.config.num_frames, 3, self.config.input_size, self.config.input_size)
            
            # Sample frame indices
            indices = self._sample_indices(total_frames)
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = torch.from_numpy(frame).float() / 255.0
                    frames.append(frame)
                else:
                    # Use last valid frame or zeros
                    if frames:
                        frames.append(frames[-1])
                    else:
                        frames.append(torch.zeros(frame.shape if 'frame' in locals() else (224, 224, 3)))
            
            cap.release()
            
            # Convert to [T, C, H, W]
            frames = torch.stack(frames).permute(0, 3, 1, 2)
        
        return frames
    
    def _sample_indices(self, total_frames):
        """Sample frame indices"""
        if self.mode == 'train':
            # Random start
            start = np.random.randint(0, max(1, total_frames - self.config.num_frames * self.config.sampling_rate))
        else:
            # Center start
            start = max(0, (total_frames - self.config.num_frames * self.config.sampling_rate) // 2)
        
        indices = []
        for i in range(self.config.num_frames):
            idx = start + i * self.config.sampling_rate
            idx = min(idx, total_frames - 1)
            indices.append(idx)
        
        return indices
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # Get video file name and label
        video_name = ann['file_name']
        
        # Handle label - could be string or index
        if 'label' in ann:
            label_str = ann['label']
            label = self.label_to_idx.get(label_str, 0)
        elif 'label_idx' in ann:
            label = ann['label_idx']
        else:
            print(f"Warning: No label found for {video_name}")
            label = 0
        
        video_path = os.path.join(self.video_root, video_name)
        
        # Check if file exists, try with .mp4 extension if not
        if not os.path.exists(video_path) and not video_name.endswith('.mp4'):
            video_path = os.path.join(self.video_root, video_name + '.mp4')
        
        # Load and transform video
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
                
            frames = self._load_video(video_path)
            frames = self.spatial_transform(frames)
            frames = self.normalize(frames)
        except Exception as e:
            if idx < 5:  # Only print first few errors
                print(f"Error loading {video_path}: {e}")
            frames = torch.zeros(self.config.num_frames, 3, self.config.input_size, self.config.input_size)
        
        return frames, label

# ================== Vision Transformer Model ==================
class PatchEmbed(nn.Module):
    """Video to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, 
                 num_frames=16, tubelet_size=2):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        
        self.num_patches = (img_size // patch_size) ** 2 * (num_frames // tubelet_size)
        self.proj = nn.Conv3d(in_chans, embed_dim, 
                             kernel_size=(tubelet_size, patch_size, patch_size),
                             stride=(tubelet_size, patch_size, patch_size))

    def forward(self, x):
        # x: [B, T, C, H, W] -> [B, C, T, H, W]
        x = x.transpose(1, 2)
        x = self.proj(x)
        # [B, D, T', H', W'] -> [B, T'*H'*W', D]
        x = x.flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                            attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer for Video"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., num_frames=16, tubelet_size=2):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, 
                                    num_frames, tubelet_size)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate)
            for i in range(depth)])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def create_model(model_type, num_classes, num_frames=16, tubelet_size=2):
    """Create ViT model"""
    if model_type == "vit_base_patch16_224":
        model = VisionTransformer(
            img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            num_classes=num_classes, num_frames=num_frames, tubelet_size=tubelet_size
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

# ================== Training Functions ==================
def train_epoch(model, dataloader, criterion, optimizer, epoch, device):
    model.train()
    losses = []
    correct = 0
    total = 0
    
    for batch_idx, (videos, labels) in enumerate(dataloader):
        videos, labels = videos.to(device), labels.to(device)
        
        # Forward
        outputs = model(videos)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        losses.append(loss.item())
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch} [{batch_idx}/{len(dataloader)}] '
                  f'Loss: {loss.item():.4f} '
                  f'Acc: {100.*correct/total:.2f}%')
    
    return np.mean(losses), 100. * correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    losses = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for videos, labels in dataloader:
            videos, labels = videos.to(device), labels.to(device)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            losses.append(loss.item())
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return np.mean(losses), 100. * correct / total

def load_checkpoint(model, checkpoint_path):
    """Load pre-trained checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Find state dict
    state_dict = None
    for key in ['model', 'model_state', 'state_dict']:
        if key in checkpoint:
            state_dict = checkpoint[key]
            break
    if state_dict is None:
        state_dict = checkpoint
    
    # Clean state dict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k.startswith('encoder.'):
            k = k[8:]
        # Skip head weights
        if 'head' in k:
            continue
        new_state_dict[k] = v
    
    # Load weights
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded pre-trained weights: {msg}")
    
    return model

# ================== Main Function ==================
def main():
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*50)
    print("VideoMAE Fine-tuning for Ego4D")
    print("="*50)
    
    # Load datasets
    print("\nLoading datasets...")
    
    # Quick JSON structure check
    try:
        with open(config.train_json, 'r') as f:
            train_data = json.load(f)
            print(f"Train JSON keys: {list(train_data.keys())}")
    except Exception as e:
        print(f"Error reading train JSON: {e}")
    
    train_dataset = Ego4DDataset(config.train_json, config.video_root, 'train', config)
    val_dataset = Ego4DDataset(config.val_json, config.video_root, 'val', config)
    
    num_classes = train_dataset.num_classes
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True
    )
    
    # Create model
    print(f"\nCreating model: {config.model_type}")
    model = create_model(config.model_type, num_classes, config.num_frames, config.tubelet_size)
    
    # Load pre-trained weights
    if os.path.exists(config.checkpoint_path):
        model = load_checkpoint(model, config.checkpoint_path)
    else:
        print(f"Warning: Checkpoint not found at {config.checkpoint_path}")
    
    model = model.to(config.device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # Cosine scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )
    
    # Training loop
    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.lr}")
    print("="*50)
    
    best_acc = 0.0
    
    for epoch in range(config.epochs):
        print(f"\nEpoch: {epoch+1}/{config.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch, config.device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, config.device)
        
        # Update scheduler
        scheduler.step()
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config.__dict__
            }
            torch.save(checkpoint, os.path.join(config.output_dir, 'best_model.pth'))
            print(f'Saved best model with accuracy: {best_acc:.2f}%')
        
        if (epoch + 1) % config.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'acc': val_acc,
                'config': config.__dict__
            }
            torch.save(checkpoint, os.path.join(config.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print("\n" + "="*50)
    print(f"Training completed! Best accuracy: {best_acc:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()