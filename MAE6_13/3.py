import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.io as io
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
import numpy as np
import warnings
from torchvision import transforms as tv_transforms

# ワーニング抑制
warnings.filterwarnings("ignore", category=FutureWarning)

# パス設定
sys.path += [
    '/home/ollo/VideoMAE/AVION'
]
sys.path.append('/home/ollo/VideoMAE/')

# AVION関連インポート
from AVION.avion.models.model_videomae import VisionTransformer
from AVION.avion.utils.meters import AverageMeter
from AVION.avion.optim.schedulers import cosine_scheduler
import AVION.avion.data.transforms as volume_transforms

# Compose定義
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip

volume_transforms.Compose = Compose

# Datasetクラス
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

# FocalLoss定義
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

# 評価指標

def compute_precision_recall_f1(y_pred, y_target):
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




# チェックポイント読込
def load_checkpoint_with_key_matching(model, path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt_raw = torch.load(path, map_location=device)
    ckpt = ckpt_raw.get("state_dict", ckpt_raw)
    model.load_state_dict({k: v for k, v in ckpt.items() if k in model.state_dict()}, strict=False)
    return True

def main():
    annotation_dir = "/home/ollo/VideoMAE/videomae-clean"
    video_root = "/srv/shared/data/ego4d/short_clips/verb_annotation_simple"
    converted_ckpt_path = "/home/ollo/VideoMAE/checkpoints/avion_finetune_cls_lavila_vitb_best_converted.pt"
    train_json = os.path.join(annotation_dir, "20250512_annotations_train.json")
    val_json = os.path.join(annotation_dir, "20250512_annotations_val.json")

    train_transform = Compose([
        tv_transforms.Resize((224, 224)),
        tv_transforms.RandomHorizontalFlip(p=0.5),
        tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = Compose([
        tv_transforms.Resize((224, 224)),
        tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ImprovedEgo4DFlashDataset(train_json, video_root, train_transform, is_training=True)
    val_dataset = ImprovedEgo4DFlashDataset(val_json, video_root, val_transform, is_training=False)
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTransformer(num_classes=58, use_flash_attn=True, drop_path_rate=0.3).to(device)

    try:
        ckpt = torch.load(converted_ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
        print("[\u2713] Checkpoint successfully loaded from AVION")
    except Exception as e:
        print(f"[!] Error loading checkpoint: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    num_epochs = 10
    lr_schedule = cosine_scheduler(
        base_value=5e-4,
        final_value=1e-6,
        epochs=num_epochs,
        niter_per_ep=len(train_loader),
        warmup_epochs=2,
        start_warmup_value=1e-7
    )

    criterion = FocalLoss(alpha=1, gamma=2)
    scaler = GradScaler()

    best_f1 = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        loss_meter = AverageMeter(name="loss")

        for it, (videos, targets) in enumerate(tqdm(train_loader)):
            global_step = epoch * len(train_loader) + it
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[global_step]

            videos, targets = videos.to(device), targets.to(device)
            # Mixupなし（そのまま）

            optimizer.zero_grad()
            with autocast():
                outputs = model(videos)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), videos.size(0))

        print(f"[Epoch {epoch+1}] Train Loss: {loss_meter.avg:.4f}")

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for videos, targets in val_loader:
                videos, targets = videos.to(device), targets.to(device)
                with autocast():
                    outputs = model(videos)
                preds = (torch.sigmoid(outputs) > 0.4).float()
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        precision, recall, f1 = compute_precision_recall_f1(all_preds, all_targets)

        print(f"[Epoch {epoch+1}] Val Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, f"avion_best_f1_{f1:.4f}.pth")
            print(f"\u2605 ベストモデル保存: avion_best_f1_{f1:.4f}.pth")

if __name__ == "__main__":
    main()