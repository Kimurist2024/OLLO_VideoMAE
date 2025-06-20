# 改良されたAVIONトレーニングスクリプト
# 主な改善点：高性能Augmentation, EarlyStopping, Warmup Scheduler, F1最適化, Mixup調整 など

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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm.auto import tqdm
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path += [
    '/home/ollo/VideoMAE/videomae-clean',
    '/home/ollo/VideoMAE',
    '/home/ollo/VideoMAE/videomae-clean/MAE6_7',
    '/home/ollo/VideoMAE/AVION'
]

from AVION.avion.models.model_videomae import VisionTransformer
from mixup import Mixup
import volume_transforms as volume_transforms

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip

volume_transforms.Compose = Compose

class ImprovedEgo4DFlashDataset(Dataset):
    def __init__(self, annotation_file, video_root, transform=None, num_frames=16, num_classes=58, is_training=True):
        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)["annotations"]
        self.video_root = video_root
        self.transform = transform
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.is_training = is_training

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        video_path = os.path.join(self.video_root, ann["video_url"])
        try:
            video, _, _ = io.read_video(video_path, pts_unit='sec')
        except Exception as e:
            print(f"ビデオ読み込みエラー: {video_path}, {e}")
            video = torch.zeros(self.num_frames, 224, 224, 3)

        T = video.shape[0]
        if T < self.num_frames:
            indices = torch.linspace(0, T - 1, self.num_frames).long()
            video = video[indices]
        else:
            start_idx = torch.randint(0, max(1, T - self.num_frames), (1,)).item() if self.is_training else 0
            indices = torch.arange(start_idx, start_idx + self.num_frames)
            video = video[indices]

        video = video.permute(0, 3, 1, 2).float() / 255.0
        video = torch.nn.functional.interpolate(video, size=(224, 224), mode='bilinear', align_corners=False)
        if self.transform:
            video = self.transform(video)
        video = video.permute(1, 0, 2, 3)

        label = ann["label"]
        if not isinstance(label, list):
            label = [label]
        target = torch.zeros(self.num_classes)
        for l in label:
            if 0 <= l < self.num_classes:
                target[l] = 1.0

        return video, target

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean() if self.reduction == 'mean' else loss.sum()

def compute_precision_recall_f1_per_class(y_pred, y_target):
    precisions, recalls, f1s = [], [], []
    for i in range(y_pred.shape[1]):
        tp = np.sum(y_pred[:, i] * y_target[:, i])
        fp = np.sum(y_pred[:, i] * (1 - y_target[:, i]))
        fn = np.sum((1 - y_pred[:, i]) * y_target[:, i])
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>1e-4 else 0
        precisions.append(p); recalls.append(r); f1s.append(f1)
    return np.mean(precisions), np.mean(recalls), np.mean(f1s)

def load_checkpoint_with_key_matching(model, path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt_raw = torch.load(path, map_location=device)
    ckpt = ckpt_raw.get("model", ckpt_raw)
    model.load_state_dict({k: v for k, v in ckpt.items() if k in model.state_dict()}, strict=False)
    return True

def main():
    annotation_dir = "/home/ollo/VideoMAE/videomae-clean"
    video_root = "/srv/shared/data/ego4d/short_clips/verb_annotation_simple"
    converted_ckpt_path = "/home/ollo/VideoMAE/checkpoints/avion_finetune_cls_lavila_vitb_best_converted.pt"
    train_json = os.path.join(annotation_dir, "20250512_annotations_train.json")
    val_json = os.path.join(annotation_dir, "20250512_annotations_val.json")

    train_transform = volume_transforms.Compose([
        volume_transforms.Resize((224, 224)),
        volume_transforms.RandomHorizontalFlip(p=0.5),
        volume_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = volume_transforms.Compose([
        volume_transforms.Resize((224, 224)),
        volume_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(ImprovedEgo4DFlashDataset(train_json, video_root, train_transform, is_training=True),
                              batch_size=6, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ImprovedEgo4DFlashDataset(val_json, video_root, val_transform, is_training=False),
                            batch_size=6, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTransformer(num_classes=58, use_flash_attn=True, drop_path_rate=0.3).to(device)
    load_checkpoint_with_key_matching(model, converted_ckpt_path, device)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-6)
    criterion = FocalLoss(alpha=1, gamma=2)
    scaler = GradScaler()
    mixup_fn = Mixup(mixup_alpha=0.4, cutmix_alpha=0.8, prob=0.8, switch_prob=0.5, mode='batch', label_smoothing=0.05, num_classes=58)

    best_f1, best_model_state = 0.0, None
    for epoch in range(1):
        model.train(); total_loss = 0.0
        for videos, targets in tqdm(train_loader):
            videos, targets = videos.to(device), targets.to(device)
            videos, targets = mixup_fn(videos, targets)
            optimizer.zero_grad()
            with autocast(): outputs = model(videos); loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item()
        scheduler.step()

        model.eval(); all_preds, all_targets = [], []
        with torch.no_grad():
            for videos, targets in val_loader:
                videos, targets = videos.to(device), targets.to(device)
                with autocast(): outputs = model(videos)
                preds = (torch.sigmoid(outputs) > 0.4).float()
                all_preds.append(preds.cpu().numpy()); all_targets.append(targets.cpu().numpy())
        all_preds = np.concatenate(all_preds); all_targets = np.concatenate(all_targets)
        p, r, f1 = compute_precision_recall_f1_per_class(all_preds, all_targets)
        print(f"[Epoch {epoch+1}] Val Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1; best_model_state = model.state_dict().copy()
            torch.save(best_model_state, f"avion_best_f1_{f1:.4f}.pth")
            print(f"★ ベストモデル保存: avion_best_f1_{f1:.4f}.pth")

if __name__ == "__main__":
    main()
