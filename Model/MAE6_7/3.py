import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.io as io
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
import numpy as np

sys.path.append('/home/ollo/videomae-clean')  # mixup.py
sys.path.append('/home/ollo/AVION')           # avion モジュール

from mixup import Mixup
from avion.models.model_videomae import VisionTransformer

# --- Dataset ---
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
        ann = self.annotations[idx]
        video_path = os.path.join(self.video_root, ann["video_url"])
        video, _, _ = io.read_video(video_path, pts_unit='sec')
        T = video.shape[0]
        if T < self.num_frames:
            repeat_factor = (self.num_frames + T - 1) // T
            video = video.repeat(repeat_factor, 1, 1, 1)
        indices = torch.linspace(0, T - 1, self.num_frames).long()
        video = video[indices].permute(0, 3, 1, 2).float() / 255.0
        if self.transform:
            video = torch.stack([self.transform(frame) for frame in video])
        video = video.permute(1, 0, 2, 3)
        label = ann["label"]
        if not isinstance(label, list):
            label = [label]
        target = torch.zeros(self.num_classes)
        for l in label:
            if 0 <= l < self.num_classes:
                target[l] = 1.0
        return video, target

# --- Metric ---
def compute_precision_recall_f1(y_pred, y_target):
    tp = np.sum(y_pred * y_target)
    fp = np.sum(y_pred * (1 - y_target))
    fn = np.sum((1 - y_pred) * y_target)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 1e-4 else 0
    return precision, recall, f1

# --- Main ---
def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])

    train_dataset = Ego4DFlashDataset(args.train_json, args.video_root, transform)
    val_dataset = Ego4DFlashDataset(args.val_json, args.video_root, transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = VisionTransformer(num_classes=58, use_flash_attn=True).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {args.checkpoint}")

    mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0,
        prob=1.0, switch_prob=0.5,
        mode='batch', label_smoothing=0.1,
        num_classes=58
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    model.train()
    for epoch in range(args.epochs):
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

        print(f"[Epoch {epoch+1}] Train Loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for videos, targets in val_loader:
                videos, targets = videos.to(device), targets.to(device)
                with autocast():
                    outputs = model(videos)
                preds = (torch.sigmoid(outputs) > 0.3).float()
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        precision, recall, f1 = compute_precision_recall_f1(all_preds, all_targets)
        print(f"[Epoch {epoch+1}] Val Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "avion_finetuned.pth"))
    print(f"Saved model to {args.output_dir}/avion_finetuned.pth")

# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune AVION on Ego4D 58-class task")
    parser.add_argument("--train-json", type=str, default="/home/ollo/videomae-clean/20250512_annotations_train.json")
    parser.add_argument("--val-json", type=str, default="/home/ollo/videomae-clean/20250512_annotations_val.json")
    parser.add_argument("--video-root", type=str, default="/srv/shared/data/ego4d/short_clips/verb_annotation_simple")
    parser.add_argument("--checkpoint", type=str, default="/home/ollo/VideoMAE/checkpoints/avion_finetune_mir_lavila_vitb_best.pt")
    parser.add_argument("--output-dir", type=str, default="./output_finetune")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args)
