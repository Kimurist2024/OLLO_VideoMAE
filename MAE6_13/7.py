import os
import sys
import json
import yaml
import torch
import numpy as np
import warnings
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.io import read_video
from tqdm import tqdm

# --- Warning suppression ---
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Path setup ---
sys.path.append("/home/ollo/VideoMAE/AVION")
sys.path.append("/home/ollo/VideoMAE/AVION/scripts")
sys.path.append('/home/ollo/VideoMAE/')
# --- AVION imports ---
from AVION.avion.models.model_videomae import VisionTransformer
from AVION.avion.optim.schedulers import cosine_scheduler
from AVION.avion.utils.meters import AverageMeter
from AVION.avion.utils.misc import set_seed

# --- Pretraining function imports ---
from AVION.scripts.main_lavila_pretrain import (
    build_pretraining_model,
    build_transform,
    build_dataset,
    pretrain_one_epoch
)

# --- Custom Dataset ---
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
            video, _, _ = read_video(video_path, pts_unit='sec')
        except Exception:
            video = torch.zeros(self.num_frames, 224, 224, 3)

        T_len = video.shape[0]
        if T_len < self.num_frames:
            indices = torch.linspace(0, T_len - 1, self.num_frames).long()
            video = video[indices]
        else:
            start_idx = torch.randint(0, max(1, T_len - self.num_frames), (1,)).item() if self.is_training else 0
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

# --- Focal Loss ---
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

# --- F1 score ---
def compute_precision_recall_f1(y_pred, y_target):
    tp = np.sum(y_pred * y_target)
    fp = np.sum(y_pred * (1 - y_target))
    fn = np.sum((1 - y_pred) * y_target)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 1e-4 else 0
    return precision, recall, f1

# --- Load checkpoint ---
def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)

# --- Main ---
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    annotation_dir = "/home/ollo/VideoMAE/videomae-clean"
    video_root = "/srv/shared/data/ego4d/short_clips/verb_annotation_simple"
    train_json = os.path.join(annotation_dir, "20250512_annotations_train.json")
    val_json = os.path.join(annotation_dir, "20250512_annotations_val.json")
    pretrain_config_path = "/home/ollo/VideoMAE/AVION/configs/lavila_pretrain.yaml"

    # --- Pretraining ---
    with open(pretrain_config_path, "r") as f:
        config = yaml.safe_load(f)

    print("=== Pretraining Phase ===")
    model = build_pretraining_model(config).to(device)
    transform = build_transform(is_train=True, config=config)
    dataset = build_dataset(is_train=True, transform=transform, config=config)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scaler = GradScaler()
    lr_schedule = cosine_scheduler(
        base_value=config['lr'],
        final_value=1e-6,
        epochs=config['epochs'],
        niter_per_ep=len(dataloader),
        warmup_epochs=2
    )

    for epoch in range(config['epochs']):
        pretrain_one_epoch(model, dataloader, optimizer, lr_schedule, epoch, device, scaler)

    torch.save(model.state_dict(), "avion_pretrained_final.pt")
    print("✓ Pretraining Done")

    # --- Finetuning ---
    print("=== Finetuning Phase ===")
    model = VisionTransformer(num_classes=58, use_flash_attn=True, drop_path_rate=0.3).to(device)
    load_checkpoint(model, "avion_pretrained_final.pt", device)

    transform_train = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_val = T.Compose([
        T.Resize((224, 224)),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ImprovedEgo4DFlashDataset(train_json, video_root, transform_train, is_training=True)
    val_dataset = ImprovedEgo4DFlashDataset(val_json, video_root, transform_val, is_training=False)
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    criterion = FocalLoss(alpha=1, gamma=2)
    scaler = GradScaler()
    best_f1 = 0.0

    for epoch in range(10):
        model.train()
        loss_meter = AverageMeter("train_loss")

        for videos, targets in tqdm(train_loader):
            videos, targets = videos.to(device), targets.to(device)
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

        # --- Validation ---
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
            torch.save(model.state_dict(), f"best_finetune_f1_{f1:.4f}.pt")
            print(f"★ Best model saved: best_finetune_f1_{f1:.4f}.pt")

if __name__ == "__main__":
    main()
