# -*- coding: utf-8 -*-
import os
import sys
import math
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from torchvision import transforms
from timm.scheduler import CosineLRScheduler

# パス設定
sys.path.append("/home/ollo/VideoMAE/")
from engine_for_pretraining import train_one_epoch as pretrain_one_epoch
from engine_for_finetuning import train_one_epoch as finetune_one_epoch
from AVION.avion.models.model_videomae import PretrainVisionTransformer, VisionTransformer
from AVION.avion.data.transforms import TubeMaskingGenerator

# === モデル定義 ===
def get_pretrain_model():
    return PretrainVisionTransformer(
        img_size=224,
        patch_size=2,
        num_frames=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        use_flash_attn=True
    )

def get_finetune_model(num_classes):
    return VisionTransformer(
        img_size=224,
        patch_size=2,
        num_frames=16,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        use_flash_attn=True
    )

# === 評価 ===
def compute_precision_recall_f1(y_pred, y_target):
    tp = np.sum(y_pred * y_target)
    fp = np.sum(y_pred * (1 - y_target))
    fn = np.sum((1 - y_pred) * y_target)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 1e-4 else 0
    return precision, recall, f1

@torch.no_grad()
def evaluate(model, data_loader, device, tag="Val"):
    model.eval()
    all_preds, all_targets = [], []
    for videos, targets in data_loader:
        videos, targets = videos.to(device), targets.to(device)
        with torch.cuda.amp.autocast():
            outputs = model(videos)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()
    y_pred_bin = (y_pred > 0).astype(int)
    y_true_bin = (y_true > 0).astype(int)
    precision, recall, f1 = compute_precision_recall_f1(y_pred_bin, y_true_bin)
    print(f"[{tag}] Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    return precision, recall, f1

# === Dataset ===
class PretrainDataset(Dataset):
    def __init__(self, json_path, root_dir, transform=None, mask_ratio=0.9):
        with open(json_path) as f:
            self.annots = json.load(f)
        self.root = root_dir
        self.transform = transform
        self.mask_gen = TubeMaskingGenerator((16, 224//2, 224//2), mask_ratio)
    def __len__(self): return len(self.annots)
    def __getitem__(self, idx):
        a = self.annots[idx]
        x = torch.load(os.path.join(self.root, a["video_id"]))
        return self.transform(x) if self.transform else x, self.mask_gen()

class FinetuneDataset(Dataset):
    def __init__(self, json_path, root_dir, transform=None):
        with open(json_path) as f:
            self.annots = json.load(f)
        self.root = root_dir
        self.transform = transform
    def __len__(self): return len(self.annots)
    def __getitem__(self, idx):
        a = self.annots[idx]
        x = torch.load(os.path.join(self.root, a["video_id"]))
        y = a["label"]
        return self.transform(x) if self.transform else x, y

# === 共通設定 ===
annotation_dir = "/home/ollo/VideoMAE/videomae-clean"
video_root = "/srv/shared/data/ego4d/short_clips/verb_annotation_simple"
train_json = os.path.join(annotation_dir, "20250512_annotations_train.json")
val_json   = os.path.join(annotation_dir, "20250512_annotations_val.json")
pretrained_ckpt = "/home/ollo/VideoMAE/checkpoints/vit_b_hybrid_pt_800e_k710_ft.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize([0.5], [0.5])
])

# === ① Pretraining ===
model = get_pretrain_model()
ckpt = torch.load(pretrained_ckpt, map_location="cpu")
state_dict = ckpt.get("model", ckpt)
for k in ['head.weight', 'head.bias']:
    state_dict.pop(k, None)
model.load_state_dict(state_dict, strict=False)
model.to(device)

pre_dataset = PretrainDataset(train_json, video_root, transform)
pre_loader = DataLoader(pre_dataset, batch_size=4, shuffle=True, num_workers=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, betas=(0.9, 0.95), weight_decay=0.05)
scaler = GradScaler()
epochs = 20
lr_scheduler = CosineLRScheduler(optimizer, t_initial=epochs * len(pre_loader))

for ep in range(epochs):
    print(f"[Pretrain Epoch {ep+1}/{epochs}]")
    pretrain_one_epoch(model, pre_loader, optimizer, device, ep, scaler,
        patch_size=2, start_steps=ep*len(pre_loader),
        lr_schedule_values=lr_scheduler.get_epoch_values(ep*len(pre_loader)),
        wd_schedule_values=None, lr_scheduler=lr_scheduler)

torch.save(model.state_dict(), "vitb_flash_pretrained.pth")

# === ② Finetuning ===
model = get_finetune_model(num_classes=57)
model.load_state_dict(torch.load("vitb_flash_pretrained.pth"), strict=False)
model.to(device)

train_set = FinetuneDataset(train_json, video_root, transform)
val_set   = FinetuneDataset(val_json, video_root, transform)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()
lr_scheduler = CosineLRScheduler(optimizer, t_initial=epochs * len(train_loader))

for ep in range(epochs):
    print(f"[Finetune Epoch {ep+1}/{epochs}]")
    finetune_one_epoch(model, criterion, train_loader, optimizer, device, ep, scaler,
        model_ema=None, mixup_fn=None, log_writer=None,
        start_steps=ep*len(train_loader),
        lr_schedule_values=lr_scheduler.get_epoch_values(ep*len(train_loader)),
        wd_schedule_values=None,
        num_training_steps_per_epoch=len(train_loader),
        update_freq=1)
    evaluate(model, train_loader, device, tag="Train")
    evaluate(model, val_loader, device, tag="Val")

torch.save(model.state_dict(), "vitb_flash_finetuned.pth")
print("✔ All training complete.")
