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
from tqdm.auto import tqdm
import numpy as np

# --- パス設定 ---
sys.path.append('/home/ollo/videomae-clean')  # mixup.py の場所
sys.path.append('/home/ollo/AVION')           # avion モジュールのルート

# --- インポート ---
from mixup import Mixup
from avion.models.model_videomae import VisionTransformer

# --- データセット定義 ---
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

# --- 評価指標計算 ---
def compute_precision_recall_f1(y_pred, y_target):
    tp = np.sum(y_pred * y_target)
    fp = np.sum(y_pred * (1 - y_target))
    fn = np.sum((1 - y_pred) * y_target)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 1e-4 else 0
    return precision, recall, f1

# --- メイントレーニング関数 ---
def train_model_interactive():
    print("FlashAttention Enabled:", torch.backends.cuda.flash_sdp_enabled())

    annotation_dir = "/home/ollo/videomae-clean"
    video_root = "/srv/shared/data/ego4d/short_clips/verb_annotation_simple"
    checkpoint_path = "/home/ollo/VideoMAE/checkpoints/vit_b_hybrid_pt_800e_k710_ft.pth"
    train_json = os.path.join(annotation_dir, "20250512_annotations_train.json")
    val_json = os.path.join(annotation_dir, "20250512_annotations_val.json")

    transform = transforms.Compose([transforms.Resize((224, 224))])
    train_dataset = Ego4DFlashDataset(train_json, video_root, transform)
    val_dataset = Ego4DFlashDataset(val_json, video_root, transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- モデル構築 (FlashAttention有効) ---
    model = VisionTransformer(num_classes=58, use_flash_attn=True).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint_data, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("No checkpoint found. Training from scratch.")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0,
        prob=1.0, switch_prob=0.5,
        mode='batch', label_smoothing=0.1,
        num_classes=58
    )

    for epoch in range(20):
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

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Training Loss: {avg_loss:.4f}")

        # --- 検証 ---
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for videos, targets in val_loader:
                videos, targets = videos.to(device), targets.to(device)
                with autocast():
                    outputs = model(videos)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        precision, recall, f1 = compute_precision_recall_f1(all_preds, all_targets)
        print(f"[Epoch {epoch+1}] Validation - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    torch.save(model.state_dict(), "avion_videomae_finetuned_flash_mixup.pth")
    print("Training complete. Model saved to 'avion_videomae_finetuned_flash_mixup.pth'.")

# --- 実行エントリポイント ---
if __name__ == "__main__":
    train_model_interactive()
