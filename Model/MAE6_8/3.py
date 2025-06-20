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
import warnings

# 警告を無視
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append('/home/ollo/VideoMAE/videomae-clean')
sys.path.append('/home/ollo/VideoMAE')
sys.path.append('/home/ollo/VideoMAE/videomae-clean/MAE6_7')

from AVION.avion.models.model_videomae import VisionTransformer
from mixup import Mixup
import volume_transforms as volume_transforms

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip

volume_transforms.Compose = Compose

class Ego4DFlashDataset(Dataset):
    def __init__(self, annotation_file, video_root, transform=None, num_frames=16, num_classes=58):
        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)["annotations"]
        self.video_root, self.transform, self.num_frames, self.num_classes = video_root, transform, num_frames, num_classes

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        video_path = os.path.join(self.video_root, ann["video_url"])
        video, _, _ = io.read_video(video_path, pts_unit='sec')
        T = video.shape[0]
        if T < self.num_frames:
            video = video.repeat((self.num_frames + T - 1) // T, 1, 1, 1)
        indices = torch.linspace(0, T - 1, self.num_frames).long()
        video = video[indices].permute(0, 3, 1, 2).float() / 255.0
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

def compute_precision_recall_f1(y_pred, y_target):
    tp = np.sum(y_pred * y_target)
    fp = np.sum(y_pred * (1 - y_target))
    fn = np.sum((1 - y_pred) * y_target)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 1e-4 else 0
    return precision, recall, f1

def load_checkpoint_with_key_matching(model, checkpoint_path, device):
    """チェックポイントをロードし、キーの不一致を処理する"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    ckpt_raw = torch.load(checkpoint_path, map_location=device)
    ckpt = ckpt_raw.get("model", ckpt_raw)
    
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(ckpt.keys())
    
    print(f"モデルのキー数: {len(model_keys)}")
    print(f"チェックポイントのキー数: {len(ckpt_keys)}")
    
    # 共通するキーを見つける
    common_keys = model_keys.intersection(ckpt_keys)
    missing_keys = model_keys - ckpt_keys
    unexpected_keys = ckpt_keys - model_keys
    
    print(f"共通キー数: {len(common_keys)}")
    print(f"不足キー数: {len(missing_keys)}")
    print(f"余分キー数: {len(unexpected_keys)}")
    
    if missing_keys:
        print("不足しているキー（最初の10個）:")
        for key in list(missing_keys)[:10]:
            print(f"  - {key}")
    
    if unexpected_keys:
        print("余分なキー（最初の10個）:")
        for key in list(unexpected_keys)[:10]:
            print(f"  - {key}")
    
    # 共通するキーのみを使用してロード
    filtered_ckpt = {k: v for k, v in ckpt.items() if k in common_keys}
    
    # strict=Falseでロード（不足するキーは無視）
    missing_keys_load, unexpected_keys_load = model.load_state_dict(filtered_ckpt, strict=False)
    
    print(f"ロード完了: {len(filtered_ckpt)}個のパラメータがロードされました")
    if missing_keys_load:
        print(f"初期化されたパラメータ: {len(missing_keys_load)}個")
    
    return len(filtered_ckpt) > 0

def main():
    annotation_dir = "/home/ollo/VideoMAE/videomae-clean"
    video_root = "/srv/shared/data/ego4d/short_clips/verb_annotation_simple"
    converted_ckpt_path = "/home/ollo/VideoMAE/checkpoints/avion_finetune_cls_lavila_vitb_best_converted.pt"
    train_json = os.path.join(annotation_dir, "20250512_annotations_train.json")
    val_json = os.path.join(annotation_dir, "20250512_annotations_val.json")

    # データ変換の定義
    transform = volume_transforms.Compose([
        volume_transforms.Resize((224, 224)),
        volume_transforms.RandomHorizontalFlip(p=0.5),
        volume_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # データセットとデータローダーの作成
    train_dataset = Ego4DFlashDataset(train_json, video_root, transform)
    val_dataset = Ego4DFlashDataset(val_json, video_root, transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # モデルの作成
    try:
        model = VisionTransformer(
            num_classes=58, 
            use_flash_attn=True, 
            drop_path_rate=0.2
        ).to(device)
        print("モデル作成完了")
    except Exception as e:
        print(f"モデル作成エラー: {e}")
        # フォールバック: flash attentionを無効にする
        model = VisionTransformer(
            num_classes=58, 
            use_flash_attn=False, 
            drop_path_rate=0.2
        ).to(device)
        print("フォールバックモデルで作成完了")

    # チェックポイントのロード
    try:
        success = load_checkpoint_with_key_matching(model, converted_ckpt_path, device)
        if success:
            print("チェックポイントのロードが完了しました")
        else:
            print("警告: チェックポイントからパラメータをロードできませんでした。ランダム初期化で続行します。")
    except Exception as e:
        print(f"チェックポイントロードエラー: {e}")
        print("ランダム初期化で続行します。")

    # トレーニング設定
    mixup_fn = Mixup(
        mixup_alpha=0.8, 
        cutmix_alpha=1.0, 
        prob=1.0, 
        switch_prob=0.5, 
        mode='batch', 
        label_smoothing=0.1, 
        num_classes=58
    )
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=1e-3, 
        betas=(0.9, 0.999), 
        weight_decay=0.05
    )
    
    scheduler = LambdaLR(
        optimizer, 
        lr_lambda=lambda epoch: (epoch + 1) / 5 if epoch < 5 else 0.5 * (1 + np.cos(np.pi * (epoch - 5) / 5))
    )
    
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    # トレーニングループ
    num_epochs = 1
    for epoch in range(num_epochs):
        # トレーニングフェーズ
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        print(f"エポック {epoch+1}/{num_epochs} 開始")
        
        for videos, targets in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            try:
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
                num_batches += 1
                
            except Exception as e:
                print(f"バッチ処理エラー: {e}")
                continue
        
        scheduler.step()
        avg_train_loss = total_loss / max(num_batches, 1)
        print(f"[Epoch {epoch+1}] Training Loss: {avg_train_loss:.4f}")

        # 検証フェーズ
        model.eval()
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for videos, targets in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                try:
                    videos, targets = videos.to(device), targets.to(device)
                    
                    with autocast():
                        outputs = model(videos)
                    
                    preds = (torch.sigmoid(outputs) > 0.3).float()
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
                    
                except Exception as e:
                    print(f"検証バッチエラー: {e}")
                    continue
        
        if all_preds and all_targets:
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            precision, recall, f1 = compute_precision_recall_f1(all_preds, all_targets)
            print(f"[Epoch {epoch+1}] Val Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        else:
            print(f"[Epoch {epoch+1}] 検証データの処理に失敗しました")

    # モデルの保存
    try:
        save_path = "avion_finetuned_from_converted_ckpt.pth"
        torch.save(model.state_dict(), save_path)
        print(f"モデルを {save_path} に保存しました")
    except Exception as e:
        print(f"モデル保存エラー: {e}")

if __name__ == "__main__":
    main()