import os
import sys
import json
import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from torchvision import transforms
from timm.scheduler import CosineLRScheduler

# パス設定
sys.path.append("/home/ollo/VideoMAE/")
from engine_for_finetuning import train_one_epoch
from engine_for_finetuning import get_model  # FlashAttention付きVideoMAE

# ==== F1スコア計算関数 ====
def compute_precision_recall_f1(y_pred, y_target):
    """
    y_pred: 0 or 1 binary prediction (numpy array)
    y_target: 0 or 1 binary ground truth (numpy array)
    """
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

# ==== パラメータ設定 ====
annotation_dir = "/home/ollo/VideoMAE/videomae-clean"
video_root = "/srv/shared/data/ego4d/short_clips/verb_annotation_simple"
train_json = os.path.join(annotation_dir, "20250512_annotations_train.json")
val_json = os.path.join(annotation_dir, "20250512_annotations_val.json")
pretrained_ckpt = "/home/ollo/VideoMAE/checkpoints/vit_b_hybrid_pt_800e_k710_ft.pth"

num_classes = 57
num_epochs = 20
batch_size = 8
lr = 1e-4
num_frames = 16
img_size = 224
update_freq = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Ego4D Dataset ====
class Ego4DDataset(Dataset):
    def __init__(self, annotation_path, video_root, transform=None):
        with open(annotation_path, 'r') as f:
            self.annotations = json.load(f)
        self.video_root = video_root
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        video_path = os.path.join(self.video_root, ann["video_id"])  # .pt expected
        video = torch.load(video_path)  # Tensor: [3, T, H, W]
        if self.transform:
            video = self.transform(video)
        label = ann["label"]
        return video, label

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.Normalize([0.5], [0.5]),
])

train_dataset = Ego4DDataset(train_json, video_root, transform=transform)
val_dataset = Ego4DDataset(val_json, video_root, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ==== モデル構築 ====
model = get_model(
    model_name='vit_base_patch16_224',
    num_classes=num_classes,
    num_frames=num_frames,
    img_size=img_size,
    tubelet_size=2,
    use_flash=True  # FlashAttention有効化
)

# 重みロード（head除外）
checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
state_dict = checkpoint.get('model', checkpoint)
for k in ['head.weight', 'head.bias']:
    if k in state_dict:
        del state_dict[k]
msg = model.load_state_dict(state_dict, strict=False)
print("Loaded checkpoint:", msg)

model.to(device)

# ==== 損失・最適化・スケジューラ ====
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
scaler = GradScaler()
steps_per_epoch = math.ceil(len(train_loader.dataset) / batch_size)
lr_scheduler = CosineLRScheduler(optimizer, t_initial=num_epochs * steps_per_epoch)

# ==== 評価関数（F1付き） ====
@torch.no_grad()
def evaluate(model, data_loader, tag="Eval"):
    model.eval()
    all_preds = []
    all_targets = []

    for videos, targets in data_loader:
        videos = videos.to(device)
        targets = targets.to(device)
        with torch.cuda.amp.autocast():
            outputs = model(videos)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()

    # 2値化（例: ラベル > 0 を1とする簡易化）
    y_pred_bin = (y_pred > 0).astype(int)
    y_true_bin = (y_true > 0).astype(int)

    precision, recall, f1 = compute_precision_recall_f1(y_pred_bin, y_true_bin)
    print(f"[{tag}] Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    return precision, recall, f1

# ==== 学習ループ ====
for epoch in range(num_epochs):
    print(f"\n[Epoch {epoch+1}/{num_epochs}]")
    _ = train_one_epoch(
        model=model,
        criterion=criterion,
        data_loader=train_loader,
        optimizer=optimizer,
        device=device,
        epoch=epoch,
        loss_scaler=scaler,
        model_ema=None,
        mixup_fn=None,
        log_writer=None,
        start_steps=epoch * steps_per_epoch,
        lr_schedule_values=lr_scheduler.get_epoch_values(epoch * steps_per_epoch),
        wd_schedule_values=None,
        num_training_steps_per_epoch=steps_per_epoch,
        update_freq=update_freq
    )

    # === F1スコアをTrain/Val両方で表示 ===
    evaluate(model, train_loader, tag="Train")
    evaluate(model, val_loader, tag="Val")

# ==== モデル保存 ====
save_path = f"vitb_flash_finetuned_ego4d_f1_{num_epochs}ep.pth"
torch.save(model.state_dict(), save_path)
print(f"Model saved to: {save_path}")
