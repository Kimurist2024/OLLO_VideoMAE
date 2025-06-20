import os
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
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

# --- Core Modules ---
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None, use_flash_attn=False):
        super().__init__()
        self.num_heads = num_heads
        self.use_flash_attn = use_flash_attn
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        self.head_dim = head_dim
        self.all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, self.all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(self.all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(self.all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_flash_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            x = x.transpose(1, 2).reshape(B, N, self.all_head_dim)
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, self.all_head_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, use_flash_attn=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, attn_head_dim, use_flash_attn)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1 = self.gamma_2 = None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = tubelet_size
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (num_frames // tubelet_size)
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
                              stride=(tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0.,
                 all_frames=16, tubelet_size=2, use_flash_attn=True):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, all_frames, tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[i], init_values,
                  norm_layer=norm_layer, act_layer=nn.GELU, use_flash_attn=use_flash_attn)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed.to(x.device)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

class VisionTransformerWithLSTM(nn.Module):
    def __init__(self, vit_model, hidden_dim=512, lstm_layers=1, bidirectional=False, num_classes=58):
        super().__init__()
        self.vit = vit_model
        self.lstm = nn.LSTM(
            input_size=vit_model.norm.normalized_shape[0],
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x):
        x = self.vit(x)  # [B, T', D]
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1]  # 最終ステップ
        return self.classifier(out)

# --- Dataset and Training ---
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

def compute_precision_recall_f1(y_pred, y_target):
    tp = np.sum(y_pred * y_target)
    fp = np.sum(y_pred * (1 - y_target))
    fn = np.sum((1 - y_pred) * y_target)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 1e-4 else 0
    return precision, recall, f1

def get_model():
    vit = VisionTransformer()
    return VisionTransformerWithLSTM(vit, hidden_dim=512, bidirectional=True, num_classes=58)

def train_model_interactive():
    annotation_dir = "/home/ollo/videomae-clean"
    video_root = "/srv/shared/data/ego4d/short_clips/verb_annotation_simple"
    train_json = os.path.join(annotation_dir, "20250512_annotations_train.json")
    val_json = os.path.join(annotation_dir, "20250512_annotations_val.json")

    transform = transforms.Compose([transforms.Resize((224, 224))])
    train_dataset = Ego4DFlashDataset(train_json, video_root, transform)
    val_dataset = Ego4DFlashDataset(val_json, video_root, transform)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    for epoch in range(20):
        model.train()
        for videos, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            videos, targets = videos.to(device), targets.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(videos)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        all_preds = []
        all_targets = []
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
        print(f"[Epoch {epoch+1}] Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    torch.save(model.state_dict(), "videomae_lstm_finetuned.pth")
    print("Model saved.")

if __name__ == "__main__":
    train_model_interactive()