# wandb・Compose自作・engine削除済みAVIONファインチューニングスクリプト
import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as tv_transforms
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb

# AVIONパス追加
sys.path.append('/home/ollo/VideoMAE')
sys.path.append('/home/ollo/VideoMAE/AVION')
sys.path.append('/home/ollo/VideoMAE/')
from AVION.avion.models.transformer import VisionTransformer
from AVION.avion.utils.meters import AverageMeter
from AVION.avion.optim.schedulers import cosine_scheduler

# Compose定義（transforms.pyに含まれていないため自作）
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip

class ImprovedEgo4DFlashDataset(torch.utils.data.Dataset):
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
        import torchvision.io as io
        ann = self.annotations[idx]
        video_path = os.path.join(self.video_root, ann["video_url"])
        try:
            video, _, _ = io.read_video(video_path, pts_unit='sec')
        except:
            video = torch.zeros(self.num_frames, 224, 224, 3)
        T = video.shape[0]
        if T < self.num_frames:
            indices = torch.linspace(0, T - 1, self.num_frames).long()
        else:
            start = torch.randint(0, max(1, T - self.num_frames), (1,)).item() if self.is_training else 0
            indices = torch.arange(start, start + self.num_frames)
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

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean() if self.reduction == 'mean' else loss.sum()

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

def main():
    train_json = "/home/ollo/VideoMAE/videomae-clean/20250512_annotations_train.json"
    val_json = "/home/ollo/VideoMAE/videomae-clean/20250512_annotations_val.json"
    video_root = "/srv/shared/data/ego4d/short_clips/verb_annotation_simple"
    ckpt_path = "/home/ollo/VideoMAE/checkpoints/avion_finetune_cls_lavila_vitb_best_converted.pt"
    epochs = 10
    batch_size = 6
    lr = 5e-4
    output_dir = "."
    use_wandb = False

    if use_wandb:
        wandb.init(project="ego4d-avion", config=locals())

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTransformer(
        image_size=224, patch_size=16, width=768, layers=12, heads=12,
        mlp_ratio=4.0, num_frames=16, output_dim=58,
        use_flash_attn=True, drop_path_rate=0.3, global_average_pool=False
    ).to(device)

    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
        print("[\u2713] Checkpoint loaded")
    except Exception as e:
        print(f"[!] Failed to load checkpoint: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    lr_schedule = cosine_scheduler(
        base_value=lr, final_value=1e-6, epochs=epochs,
        niter_per_ep=len(train_loader), warmup_epochs=2, start_warmup_value=1e-7
    )
    criterion = FocalLoss()
    scaler = GradScaler()
    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        loss_meter = AverageMeter(name="loss")

        for it, (videos, targets) in enumerate(tqdm(train_loader)):
            global_step = epoch * len(train_loader) + it
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[global_step]

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

        if use_wandb:
            wandb.log({"val/precision": precision, "val/recall": recall, "val/f1": f1})

        print(f"[Epoch {epoch+1}] Val Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(output_dir, f"avion_best_f1_{f1:.4f}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"★ 保存: {save_path}")

if __name__ == "__main__":
    main()
