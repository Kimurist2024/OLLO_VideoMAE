import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import random
import torchvision  # ← 追加

# --- パス設定 ---
ROOT_DIR = "/home/ollo/VideoMAE"
AVION_DIR = os.path.join(ROOT_DIR, "AVION")
WEIGHTS_PATH = os.path.join(ROOT_DIR, "checkpoints", "avion_finetune_cls_lavila_vitb_best_converted.pt")
ANNOTATION_FILE = "/home/ollo/datasets/Ego4D/annotations/ego4d_finetune.json"
VIDEO_ROOT = "/home/ollo/datasets/Ego4D/videos"

# --- sys.path 設定 ---
import sys
sys.path.append(AVION_DIR)
sys.path.append('/home/ollo/VideoMAE/')
# --- AVION関連モジュール ---
from AVION.avion.models.model_videomae import VisionTransformer
from AVION.avion.utils.meters import AverageMeter
from AVION.avion.optim.schedulers import cosine_scheduler
import AVION.avion.data.transforms as volume_transforms

# --- Transform Compose定義（修正済） ---
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip
volume_transforms.Compose = Compose

# --- Ego4D Dataset ---
class Ego4DFlashDataset(Dataset):
    def __init__(self, annotation_file, video_root, transform=None, num_frames=16, num_classes=58):
        with open(annotation_file, "r") as f:
            data = json.load(f)
        self.annotations = data["annotations"]
        self.video_root = video_root
        self.transform = transform
        self.num_frames = num_frames
        self.num_classes = num_classes

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations[idx]
        video_path = os.path.join(self.video_root, sample["video"])
        frames, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
        frames = frames.permute(0, 3, 1, 2)[:self.num_frames]
        if self.transform:
            frames = self.transform(frames)
        label = torch.tensor(sample["label"], dtype=torch.float32)
        return frames, label

# --- Focal Loss 定義 ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

# --- モデル初期化と重み読み込み ---
def build_model():
    model = VisionTransformer(img_size=224, num_classes=58, tubelet_size=2)
    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    return model

# --- 学習ループ ---
def train_one_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    loss_meter = AverageMeter()
    for videos, labels in tqdm(loader):
        videos, labels = videos.to(device), labels.to(device)
        with autocast():
            outputs = model(videos)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        loss_meter.update(loss.item(), videos.size(0))
    return loss_meter.avg

# --- 評価ループ ---
def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for videos, labels in tqdm(loader):
            videos = videos.to(device)
            outputs = model(videos).sigmoid().cpu()
            preds.append((outputs > 0.5).int().numpy())
            gts.append(labels.int().numpy())
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    return f1_score(gts, preds, average="macro")

# --- メイン処理 ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = volume_transforms.Compose([
        volume_transforms.Permute(),
        volume_transforms.AdaptiveTemporalCrop(16),
        volume_transforms.GroupMultiScaleCrop((224, 224), [1, .875, .75]),
        volume_transforms.RandomHorizontalFlip(),
        volume_transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    ])

    dataset = Ego4DFlashDataset(ANNOTATION_FILE, VIDEO_ROOT, transform)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    model = build_model().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    lr_scheduler = cosine_scheduler(1e-4, 1e-6, 10, len(train_loader))
    scaler = GradScaler()
    criterion = FocalLoss()

    for epoch in range(10):
        lr = lr_scheduler[epoch * len(train_loader)]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device)
        f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val F1 Score: {f1:.4f}")

if __name__ == '__main__':
    main()