import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.io as io
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from tqdm.auto import tqdm
import numpy as np
import warnings
import logging
from pathlib import Path
import argparse
import time
from collections import OrderedDict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 警告を無視
warnings.filterwarnings("ignore", category=FutureWarning)

# AVION paths - adjusted for MAE6_14 directory
sys.path.append('/home/ollo/VideoMAE/videomae-clean')
sys.path.append('/home/ollo/VideoMAE')
sys.path.append('/home/ollo/VideoMAE/videomae-clean/MAE6_14')
sys.path.append('/home/ollo/VideoMAE/videomae-clean/MAE6_7')
sys.path.append('/home/ollo/VideoMAE/AVION')

try:
    from AVION.avion.models.model_videomae import VisionTransformer
    from mixup import Mixup
    import volume_transforms as volume_transforms
except ImportError as e:
    logger.warning(f"Could not import AVION modules: {e}")
    logger.info("Falling back to basic implementations")

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip

# Ensure volume_transforms has Compose
try:
    volume_transforms.Compose = Compose
except:
    pass

class ImprovedEgo4DFlashDataset(Dataset):
    def __init__(self, annotation_file, video_root, transform=None, num_frames=16, 
                 num_classes=58, is_training=True, clip_duration=2.0):
        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)["annotations"]
        
        self.video_root = video_root
        self.transform = transform
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.is_training = is_training
        self.clip_duration = clip_duration
        
        # Filter out invalid annotations
        valid_annotations = []
        for ann in self.annotations:
            video_path = os.path.join(self.video_root, ann["video_url"])
            if os.path.exists(video_path):
                valid_annotations.append(ann)
            else:
                logger.warning(f"Video file not found: {video_path}")
        
        self.annotations = valid_annotations
        logger.info(f"Loaded {len(self.annotations)} valid annotations")
    
    def temporal_sampling(self, video_length, start_time=None, end_time=None):
        """Improved temporal sampling strategy"""
        if start_time is not None and end_time is not None:
            # Use provided temporal bounds
            start_frame = int(start_time * 30)  # Assuming 30fps
            end_frame = int(end_time * 30)
            target_frames = min(end_frame - start_frame, self.num_frames)
        else:
            # Use full video
            start_frame = 0
            target_frames = min(video_length, self.num_frames)
        
        if video_length <= self.num_frames:
            # Pad or repeat frames
            indices = torch.linspace(0, video_length - 1, self.num_frames).long()
        else:
            if self.is_training:
                # Random temporal cropping for training
                max_start = video_length - self.num_frames
                start_idx = torch.randint(0, max_start + 1, (1,)).item()
                indices = torch.arange(start_idx, start_idx + self.num_frames)
            else:
                # Uniform sampling for validation
                indices = torch.linspace(0, video_length - 1, self.num_frames).long()
        
        return indices
    
    def spatial_augmentation(self, video):
        """Enhanced spatial augmentation"""
        if not self.is_training:
            return video
        
        C, T, H, W = video.shape
        
        # Random resized crop
        scale = torch.uniform(0.8, 1.0).item()
        ratio = torch.uniform(0.75, 1.33).item()
        
        area = H * W
        target_area = area * scale
        
        w = int(round(torch.sqrt(torch.tensor(target_area * ratio)).item()))
        h = int(round(torch.sqrt(torch.tensor(target_area / ratio)).item()))
        
        if torch.rand(1) < 0.5 and w <= W and h <= H:
            i = torch.randint(0, H - h + 1, (1,)).item()
            j = torch.randint(0, W - w + 1, (1,)).item()
            video = video[:, :, i:i+h, j:j+w]
            
            # Resize back to target size
            video = torch.nn.functional.interpolate(
                video, size=(224, 224), mode='bilinear', align_corners=False
            )
        
        return video
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        video_path = os.path.join(self.video_root, ann["video_url"])
        
        try:
            # Load video with specific start/end times if available
            start_time = ann.get("start_time", 0)
            end_time = ann.get("end_time", None)
            
            if end_time:
                video, _, _ = io.read_video(
                    video_path, 
                    start_pts=start_time, 
                    end_pts=end_time,
                    pts_unit='sec'
                )
            else:
                video, _, _ = io.read_video(video_path, pts_unit='sec')
            
            if video.numel() == 0:
                raise ValueError("Empty video tensor")
                
        except Exception as e:
            logger.warning(f"Video loading error: {video_path}, {e}")
            # Create dummy video
            video = torch.zeros(self.num_frames, 224, 224, 3)
        
        # Temporal sampling
        T = video.shape[0]
        if T > 0:
            indices = self.temporal_sampling(T, ann.get("start_time"), ann.get("end_time"))
            video = video[indices]
        
        # Convert to (C, T, H, W) format and normalize
        video = video.permute(3, 0, 1, 2).float() / 255.0
        
        # Spatial augmentation
        video = self.spatial_augmentation(video)
        
        # Apply transforms
        if self.transform:
            # Convert back to (T, C, H, W) for volume_transforms
            video = video.permute(1, 0, 2, 3)
            video = self.transform(video)
            # Convert back to (C, T, H, W)
            video = video.permute(1, 0, 2, 3)
        
        # Handle multi-label targets
        label = ann["label"]
        if not isinstance(label, list):
            label = [label]
        
        target = torch.zeros(self.num_classes)
        for l in label:
            if 0 <= l < self.num_classes:
                target[l] = 1.0
        
        return video, target

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()

class AdaptiveLoss(nn.Module):
    """Adaptive loss that combines BCE and Focal Loss"""
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0):
        super(AdaptiveLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / targets.size(1)
        
        # BCE loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Focal loss component
        pt = torch.sigmoid(inputs)
        focal_weight = self.alpha * torch.pow(1 - pt, self.gamma)
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()

def convert_checkpoint_keys(checkpoint_dict):
    """チェックポイントのキーを変換してモデルと一致させる"""
    new_dict = {}
    
    for key, value in checkpoint_dict.items():
        new_key = key
        
        # module.visual.* から visual.* への変換
        if key.startswith('module.visual.'):
            new_key = key.replace('module.visual.', '')
        # module.* から * への変換
        elif key.startswith('module.'):
            new_key = key.replace('module.', '')
        
        # さらなるキーマッピング
        # transformer -> blocks への変換
        if 'transformer.resblocks.' in new_key:
            new_key = new_key.replace('transformer.resblocks.', 'blocks.')
        
        # Wqkv -> qkv への変換
        if '.attn.Wqkv.' in new_key:
            new_key = new_key.replace('.attn.Wqkv.', '.attn.qkv.')
        
        # その他の一般的な変換
        if '.ln_1.' in new_key:
            new_key = new_key.replace('.ln_1.', '.norm1.')
        elif '.ln_2.' in new_key:
            new_key = new_key.replace('.ln_2.', '.norm2.')
        
        new_dict[new_key] = value
    
    return new_dict

def load_checkpoint_with_key_matching(model, checkpoint_path, device):
    """チェックポイントをロードし、キーの不一致を処理する"""
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        ckpt_raw = torch.load(checkpoint_path, map_location=device)
        
        # チェックポイントの構造を確認
        if 'state_dict' in ckpt_raw:
            ckpt = ckpt_raw['state_dict']
            logger.info("Loading parameters from 'state_dict' key")
        elif 'model' in ckpt_raw:
            ckpt = ckpt_raw['model']
            logger.info("Loading parameters from 'model' key")
        else:
            ckpt = ckpt_raw
            logger.info("Loading parameters from root level")
        
        # キー変換
        ckpt_converted = convert_checkpoint_keys(ckpt)
        
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(ckpt_converted.keys())
        
        # 共通するキーを見つける
        common_keys = model_keys.intersection(ckpt_keys)
        missing_keys = model_keys - ckpt_keys
        unexpected_keys = ckpt_keys - model_keys
        
        logger.info(f"Model keys: {len(model_keys)}, Checkpoint keys: {len(ckpt_keys)}")
        logger.info(f"Common keys: {len(common_keys)}, Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
        
        # 共通するキーのみを使用してロード
        if len(common_keys) > 0:
            filtered_ckpt = {k: v for k, v in ckpt_converted.items() if k in common_keys}
            
            # strict=Falseでロード（不足するキーは無視）
            missing_keys_load, unexpected_keys_load = model.load_state_dict(filtered_ckpt, strict=False)
            
            logger.info(f"Successfully loaded {len(filtered_ckpt)} parameters")
            if missing_keys_load:
                logger.info(f"Initialized {len(missing_keys_load)} parameters randomly")
            
            # ロード成功率を計算
            success_rate = len(filtered_ckpt) / len(model_keys) * 100
            logger.info(f"Load success rate: {success_rate:.1f}%")
            
            return True
        else:
            logger.warning("No compatible keys found, continuing with random initialization")
            return False
            
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return False

def compute_metrics(predictions, targets, threshold=0.5):
    """Compute comprehensive metrics"""
    pred_binary = (predictions > threshold).astype(np.float32)
    
    # Per-class metrics
    tp = np.sum(pred_binary * targets, axis=0)
    fp = np.sum(pred_binary * (1 - targets), axis=0)
    fn = np.sum((1 - pred_binary) * targets, axis=0)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    return {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }

def validate_model(model, val_loader, criterion, device):
    """Comprehensive validation function"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for videos, targets in tqdm(val_loader, desc="Validation"):
            videos, targets = videos.to(device), targets.to(device)
            
            with autocast():
                outputs = model(videos)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            # Store predictions and targets
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Find optimal threshold
    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.1, 0.9, 0.1):
        metrics = compute_metrics(all_preds, all_targets, threshold)
        if metrics['macro_f1'] > best_f1:
            best_f1 = metrics['macro_f1']
            best_threshold = threshold
    
    # Compute final metrics with best threshold
    final_metrics = compute_metrics(all_preds, all_targets, best_threshold)
    final_metrics['best_threshold'] = best_threshold
    final_metrics['val_loss'] = total_loss / len(val_loader)
    
    return final_metrics

def get_args_parser():
    parser = argparse.ArgumentParser(description='AVION finetune ego4d cls', add_help=False)
    
    # Dataset - with default paths
    parser.add_argument('--dataset', default='ego4d_cls', type=str)
    parser.add_argument('--root', default='/srv/shared/data/ego4d/short_clips/verb_annotation_simple', 
                       type=str, help='path to dataset root')
    parser.add_argument('--train-metadata', type=str,
                       default='/home/ollo/VideoMAE/videomae-clean/20250512_annotations_train.json')
    parser.add_argument('--val-metadata', type=str,
                       default='/home/ollo/VideoMAE/videomae-clean/20250512_annotations_val.json')
    parser.add_argument('--output-dir', default='./output', type=str, help='output dir')
    parser.add_argument('--pretrain-model', default='/home/ollo/VideoMAE/checkpoints/avion_finetune_cls_lavila_vitb_best_converted.pt', 
                       type=str, help='path of pretrained model')
    
    # Video processing
    parser.add_argument('--num-crops', default=1, type=int, help='number of crops for testing')
    parser.add_argument('--num-clips', default=1, type=int, help='number of clips for testing')
    parser.add_argument('--clip-length', default=16, type=int, help='clip length')
    parser.add_argument('--clip-stride', default=2, type=int, help='clip stride')
    parser.add_argument('--norm-style', default='openai', type=str, choices=['openai', 'timm'])
    
    # Model
    parser.add_argument('--grad-checkpointing', action='store_true', dest='use_grad_checkpointing')
    parser.add_argument('--use-flash-attn', action='store_true', dest='use_flash_attn')
    parser.add_argument('--patch-dropout', default=0., type=float)
    parser.add_argument('--drop-path-rate', default=0.2, type=float)
    parser.add_argument('--dropout-rate', default=0.5, type=float)
    parser.add_argument('--num-classes', default=58, type=int)
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    
    # Mixup
    parser.add_argument('--mixup', type=float, default=0.2)
    parser.add_argument('--cutmix', type=float, default=0.5)
    parser.add_argument('--mixup-prob', type=float, default=0.5)
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5)
    parser.add_argument('--mixup-mode', type=str, default='batch')
    parser.add_argument('--smoothing', type=float, default=0.1)
    
    # Training - with optimized defaults
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--warmup-epochs', default=5, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--optimizer', default='adamw', choices=['adamw', 'sgd'], type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float)
    parser.add_argument('--lr-end', default=1e-6, type=float)
    parser.add_argument('--update-freq', default=1, type=int)
    parser.add_argument('--wd', default=0.05, type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval-freq', default=5, type=int)
    parser.add_argument('--disable-amp', action='store_true')
    parser.add_argument('--grad-clip-norm', default=1.0, type=float)
    
    # System
    parser.add_argument('--print-freq', default=10, type=int)
    parser.add_argument('-j', '--workers', default=8, type=int)
    parser.add_argument('--evaluate', action='store_true', help='eval only')
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str)
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int)
    
    return parser

def main(args):
    """Main training function"""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    try:
        model = VisionTransformer(
            num_classes=args.num_classes,
            use_flash_attn=args.use_flash_attn,
            drop_path_rate=args.drop_path_rate
        ).to(device)
        logger.info("Model created with Flash Attention")
    except Exception as e:
        logger.warning(f"Flash Attention failed: {e}")
        try:
            model = VisionTransformer(
                num_classes=args.num_classes,
                use_flash_attn=False,
                drop_path_rate=args.drop_path_rate
            ).to(device)
            logger.info("Model created without Flash Attention")
        except Exception as e2:
            logger.error(f"Failed to create VisionTransformer: {e2}")
            # Fallback to basic model
            model = nn.Sequential(
                nn.AdaptiveAvgPool3d((args.clip_length, 224, 224)),
                nn.Flatten(),
                nn.Linear(args.clip_length * 224 * 224 * 3, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, args.num_classes)
            ).to(device)
            logger.info("Using fallback basic model")
    
    # Load pretrained weights if available
    load_checkpoint_with_key_matching(model, args.pretrain_model, device)
    
    # Setup transforms
    try:
        train_transform = volume_transforms.Compose([
            volume_transforms.Resize((256, 256)),
            volume_transforms.RandomHorizontalFlip(p=0.5),
            volume_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = volume_transforms.Compose([
            volume_transforms.Resize((224, 224)),
            volume_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    except:
        logger.warning("volume_transforms not available, using basic transforms")
        train_transform = None
        val_transform = None
    
    # Create datasets
    train_dataset = ImprovedEgo4DFlashDataset(
        args.train_metadata, args.root, train_transform,
        args.clip_length, args.num_classes, is_training=True
    )
    val_dataset = ImprovedEgo4DFlashDataset(
        args.val_metadata, args.root, val_transform,
        args.clip_length, args.num_classes, is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers, 
        pin_memory=True, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers, 
        pin_memory=True
    )
    
    logger.info(f"Train loader length: {len(train_loader)}")
    logger.info(f"Val loader length: {len(val_loader)}")
    
    # Setup loss function and optimizer
    criterion = AdaptiveLoss(
        alpha=0.25, 
        gamma=2.0, 
        label_smoothing=args.smoothing
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        betas=args.betas
    )
    
    # Learning rate scheduler with warmup
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=args.lr_end
    )
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[args.warmup_epochs]
    )
    
    # Mixup
    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0:
        try:
            mixup_fn = Mixup(
                mixup_alpha=args.mixup,
                cutmix_alpha=args.cutmix,
                prob=args.mixup_prob,
                switch_prob=args.mixup_switch_prob,
                mode=args.mixup_mode,
                label_smoothing=args.smoothing,
                num_classes=args.num_classes
            )
            logger.info("Mixup activated")
        except Exception as e:
            logger.warning(f"Mixup setup failed: {e}")
    
    # Training setup
    scaler = GradScaler(enabled=not args.disable_amp)
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    best_f1 = 0.0
    
    logger.info(f"Starting training for {args.epochs} epochs")
    
    for epoch in range(args.start_epoch, args.epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for videos, targets in progress_bar:
            videos, targets = videos.to(device), targets.to(device)
            
            # Apply mixup
            if mixup_fn is not None and epoch >= args.warmup_epochs:
                try:
                    videos, targets = mixup_fn(videos, targets)
                except:
                    pass  # Continue without mixup if it fails
            
            optimizer.zero_grad()
            
            with autocast(enabled=not args.disable_amp):
                outputs = model(videos)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        scheduler.step()
        avg_train_loss = total_loss / max(num_batches, 1)
        
        # Validation phase (every eval_freq epochs)
        if (epoch + 1) % args.eval_freq == 0:
            val_metrics = validate_model(model, val_loader, criterion, device)
            
            logger.info(
                f"Epoch {epoch+1}: "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val F1: {val_metrics['macro_f1']:.4f}, "
                f"Best Threshold: {val_metrics['best_threshold']:.2f}"
            )
            
            # Save best model
            if val_metrics['macro_f1'] > best_f1:
                best_f1 = val_metrics['macro_f1']
                save_path = os.path.join(args.output_dir, f"avion_best_f1_{best_f1:.4f}_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_f1,
                    'args': args
                }, save_path)
                logger.info(f"New best model saved: {save_path}")
            
            # Early stopping
            if early_stopping(val_metrics['macro_f1'], model):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        else:
            logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}")
    
    logger.info(f"Training completed. Best F1: {best_f1:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('AVION Ego4D training - 5.py', parents=[get_args_parser()])
    args = parser.parse_args()
    
    print("🎯 AVION Ego4D Enhanced Training System - 5.py")
    print("📂 Location: /home/ollo/VideoMAE/videomae-clean/MAE6_14/5.py")
    print("=" * 60)
    
    # Set all default paths if not provided
    if not args.pretrain_model:
        args.pretrain_model = '/home/ollo/VideoMAE/checkpoints/avion_finetune_cls_lavila_vitb_best_converted.pt'
    if not args.train_metadata:
        args.train_metadata = '/home/ollo/VideoMAE/videomae-clean/20250512_annotations_train.json'
    if not args.val_metadata:
        args.val_metadata = '/home/ollo/VideoMAE/videomae-clean/20250512_annotations_val.json'
    if not args.root:
        args.root = '/srv/shared/data/ego4d/short_clips/verb_annotation_simple'
    if not args.output_dir or args.output_dir == './':
        args.output_dir = '/home/ollo/VideoMAE/videomae-clean/MAE6_14/output'
    
    # Ensure all paths exist
    if not os.path.exists(args.pretrain_model):
        logger.warning(f"⚠️  Pretrained model not found: {args.pretrain_model}")
        logger.info("Continuing with random initialization...")
    if not os.path.exists(args.train_metadata):
        logger.error(f"❌ Train metadata not found: {args.train_metadata}")
        sys.exit(1)
    if not os.path.exists(args.val_metadata):
        logger.error(f"❌ Val metadata not found: {args.val_metadata}")
        sys.exit(1)
    if not os.path.exists(args.root):
        logger.error(f"❌ Video root directory not found: {args.root}")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("🚀 Starting AVION Ego4D training - 5.py")
    logger.info("📂 Running from: /home/ollo/VideoMAE/videomae-clean/MAE6_14/")
    logger.info("🔧 Enhanced training with advanced features:")
    logger.info("   • AdaptiveLoss (BCE + Focal Loss)")
    logger.info("   • Early Stopping")
    logger.info("   • Advanced Data Augmentation") 
    logger.info("   • Automatic Threshold Optimization")
    logger.info("   • Comprehensive Metrics Tracking")
    logger.info("")
    logger.info("⚙️  Configuration:")
    logger.info(f"  📁 Pretrained model: {args.pretrain_model}")
    logger.info(f"  📊 Train metadata: {args.train_metadata}")
    logger.info(f"  🔍 Val metadata: {args.val_metadata}")
    logger.info(f"  🎥 Video root: {args.root}")
    logger.info(f"  💾 Output dir: {args.output_dir}")
    logger.info(f"  🔢 Batch size: {args.batch_size}")
    logger.info(f"  🔄 Epochs: {args.epochs}")
    logger.info(f"  📈 Learning rate: {args.lr}")
    logger.info(f"  🏷️  Num classes: {args.num_classes}")
    logger.info(f"  🎬 Clip length: {args.clip_length}")
    logger.info(f"  ⚡ Flash attention: {args.use_flash_attn}")
    logger.info(f"  🎭 Mixup alpha: {args.mixup}")
    logger.info(f"  ✂️  Cutmix alpha: {args.cutmix}")
    logger.info(f"  🎯 Drop path rate: {args.drop_path_rate}")
    logger.info(f"  ⚖️  Weight decay: {args.wd}")
    logger.info("=" * 60)
    
    try:
        main(args)
        logger.info("🎉 Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("⏹️  Training interrupted by user")
    except Exception as e:
        logger.error(f"💥 Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("=" * 60)
    print("📝 5.py execution completed")
    print("📂 Check output directory for saved models and logs")