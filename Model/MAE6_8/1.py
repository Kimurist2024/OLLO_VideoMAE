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
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
import numpy as np

# パス設定
sys.path.append('/home/ollo/VideoMAE/videomae-clean')
sys.path.append('/home/ollo/VideoMAE')
sys.path.append('/home/ollo/VideoMAE/videomae-clean/MAE6_7')

from AVION.avion.models.model_videomae import VisionTransformer
from mixup import Mixup
import volume_transforms as volume_transforms

# Compose定義
class Compose:
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, clip):
        for t in self.transforms: clip = t(clip)
        return clip
volume_transforms.Compose = Compose

# データセット定義
class Ego4DFlashDataset(Dataset):
    def __init__(self, annotation_file, video_root, transform=None, num_frames=16, num_classes=58):
        with open(annotation_file, "r") as f: self.annotations = json.load(f)["annotations"]
        self.video_root, self.transform, self.num_frames, self.num_classes = video_root, transform, num_frames, num_classes

    def __len__(self): return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        video_path = os.path.join(self.video_root, ann["video_url"])
        video, _, _ = io.read_video(video_path, pts_unit='sec')
        T = video.shape[0]
        if T < self.num_frames:
            video = video.repeat((self.num_frames + T - 1) // T, 1, 1, 1)
        indices = torch.linspace(0, T - 1, self.num_frames).long()
        video = video[indices].permute(0, 3, 1, 2).float() / 255.0
        if self.transform: video = self.transform(video)
        video = video.permute(1, 0, 2, 3)
        label = ann["label"]
        if not isinstance(label, list): label = [label]
        target = torch.zeros(self.num_classes)
        for l in label:
            if 0 <= l < self.num_classes: target[l] = 1.0
        return video, target

# F1計算
def compute_precision_recall_f1(y_pred, y_target):
    tp = np.sum(y_pred * y_target)
    fp = np.sum(y_pred * (1 - y_target))
    fn = np.sum((1 - y_pred) * y_target)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 1e-4 else 0
    return precision, recall, f1

# メイン
def main():
    annotation_dir = "/home/ollo/VideoMAE/videomae-clean"
    video_root = "/srv/shared/data/ego4d/short_clips/verb_annotation_simple"
    converted_ckpt_path = "/home/ollo/VideoMAE/checkpoints/avion_finetune_cls_lavila_vitb_best_converted.pt"
    train_json = os.path.join(annotation_dir, "20250512_annotations_train.json")
    val_json = os.path.join(annotation_dir, "20250512_annotations_val.json")

    transform = volume_transforms.Compose([
        volume_transforms.Resize((224, 224)),
        volume_transforms.RandomHorizontalFlip(p=0.5),
        volume_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = Ego4DFlashDataset(train_json, video_root, transform)
    val_dataset = Ego4DFlashDataset(val_json, video_root, transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTransformer(num_classes=58, use_flash_attn=True, drop_path_rate=0.2).to(device)

    # 重みロードと一致検証
    if os.path.exists(converted_ckpt_path):
        ckpt = torch.load(converted_ckpt_path, map_location=device)
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(ckpt.keys())
        assert model_keys == ckpt_keys, "Checkpoint and model keys do not match"
        diff_sum = sum(torch.sum((ckpt[k] - model.state_dict()[k]).abs()).item() for k in model_keys if torch.is_tensor(ckpt[k]))
        print(f"Loaded checkpoint. Weight difference sum: {diff_sum:.6f}")
        model.load_state_dict(ckpt)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {converted_ckpt_path}")

    mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, mode='batch', label_smoothing=0.1, num_classes=58)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / 5 if epoch < 5 else 0.5 * (1 + np.cos(np.pi * (epoch - 5) / 5)))
    criterion, scaler = nn.BCEWithLogitsLoss(), GradScaler()

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for videos, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            videos, targets = videos.to(device), targets.to(device)
            videos, targets = mixup_fn(videos, targets)
            optimizer.zero_grad()
            with autocast():
                outputs = model(videos)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        scheduler.step()
        print(f"[Epoch {epoch+1}] Training Loss: {total_loss / len(train_loader):.4f}")

        # 評価
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for videos, targets in val_loader:
                videos, targets = videos.to(device), targets.to(device)
                with autocast(): outputs = model(videos)
                preds = (torch.sigmoid(outputs) > 0.3).float()
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        precision, recall, f1 = compute_precision_recall_f1(all_preds, all_targets)
        print(f"[Epoch {epoch+1}] Val Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    torch.save(model.state_dict(), "avion_finetuned_from_converted_ckpt.pth")
    print("Saved model to avion_finetuned_from_converted_ckpt.pth")

if __name__ == "__main__":
    main()
