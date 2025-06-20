

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.io as io
from tqdm.auto import tqdm
import numpy as np
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from sklearn.metrics import precision_score, recall_score, f1_score
import optuna

# === CONFIGURATION === #
CONFIG = {
    "num_classes": 58,
    "num_frames": 16,
    "batch_size": 4,
    "epochs": 20,
    "learning_rate": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "train_json": "/home/ollo/videomae-clean/20250512_annotations_train.json",
    "val_json": "/home/ollo/videomae-clean/20250512_annotations_val.json",
    "video_root": "/srv/shared/data/ego4d/short_clips/verb_annotation_simple",
    "checkpoint_path": "videomae_checkpoint.pth",
    "save_path": "videomae_finetuned.pth",
}

# === Dataset Class === #
class Ego4DFlashDataset(Dataset):
    def __init__(self, annotation_file, video_root, transform=None, num_frames=16, num_classes=58):
        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)["annotations"]
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

# === Vision Transformer Components === #
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = attn_head_dim or head_dim
        self.all_head_dim = self.head_dim * self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, self.all_head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.all_head_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(tubelet_size, *patch_size), stride=(tubelet_size, *patch_size))

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=58, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size)
        num_patches = self.patch_embed.proj.weight.shape[2] * self.patch_embed.proj.weight.shape[3] * self.patch_embed.proj.weight.shape[4]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        trunc_normal_(self.head.weight, std=.02)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(1)
        return self.head(x)

def get_model():
    return VisionTransformer()

# === 以下、学習・評価・Optunaコードは既に展開済みの通り ===
# train_one_epoch, evaluate, save_checkpoint, load_checkpoint, train_model, objective, run_optuna_study, __main__ 実装
# （前回更新内容と組み合わせることで完全構成になります）
# === Training and Evaluation Functions ===
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    for videos, targets in tqdm(dataloader, desc="Training"):
        videos, targets = videos.to(device), targets.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(videos)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for videos, targets in tqdm(dataloader, desc="Evaluating"):
            videos, targets = videos.to(device), targets.to(device)
            with autocast():
                outputs = model(videos)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    precision = precision_score(all_targets, all_preds, average='samples', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='samples', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='samples', zero_division=0)
    return precision, recall, f1

# === Checkpoint Utilities ===
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Resumed from checkpoint at {path}, epoch {checkpoint.get('epoch', '?')}")
    return checkpoint.get('epoch', 0)

# === Full Training Loop ===
def train_model():
    device = CONFIG["device"]
    transform = transforms.Compose([transforms.Resize((224, 224))])
    train_dataset = Ego4DFlashDataset(CONFIG["train_json"], CONFIG["video_root"], transform)
    val_dataset = Ego4DFlashDataset(CONFIG["val_json"], CONFIG["video_root"], transform)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)

    model = get_model().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    start_epoch = 0
    if os.path.exists(CONFIG["checkpoint_path"]):
        start_epoch = load_checkpoint(model, optimizer, CONFIG["checkpoint_path"], device)

    for epoch in range(start_epoch, CONFIG["epochs"]):
        print(f"\n=== Epoch {epoch+1}/{CONFIG['epochs']} ===")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        precision, recall, f1 = evaluate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        save_checkpoint(model, optimizer, epoch, CONFIG["save_path"])

# === Optuna Objective Function ===
def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    dropout = trial.suggest_uniform("dropout", 0.0, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])

    transform = transforms.Compose([transforms.Resize((224, 224))])
    train_dataset = Ego4DFlashDataset(CONFIG["train_json"], CONFIG["video_root"], transform)
    val_dataset = Ego4DFlashDataset(CONFIG["val_json"], CONFIG["video_root"], transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = VisionTransformer(fc_drop_rate=dropout).to(CONFIG["device"])
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    model.train()
    for videos, targets in train_loader:
        videos, targets = videos.to(CONFIG["device"]), targets.to(CONFIG["device"])
        optimizer.zero_grad()
        with autocast():
            outputs = model(videos)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        break

    precision, recall, f1 = evaluate(model, val_loader, CONFIG["device"])
    trial.report(f1, step=0)

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return f1

# === Optuna Tuning Runner ===
def run_optuna_study():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

# === Entry Point ===
if __name__ == "__main__":
    mode = "train"  # or "tune"
    if mode == "train":
        train_model()
    elif mode == "tune":
        run_optuna_study()