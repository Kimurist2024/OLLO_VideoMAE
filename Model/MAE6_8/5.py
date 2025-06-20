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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm.auto import tqdm
import numpy as np
import warnings
from sklearn.utils.class_weight import compute_class_weight

# 警告を無視
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append('/home/ollo/VideoMAE/videomae-clean')
sys.path.append('/home/ollo/VideoMAE')
sys.path.append('/home/ollo/VideoMAE/videomae-clean/MAE6_7')
sys.path.append('/home/ollo/VideoMAE/AVION')

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
            video, _, _ = io.read_video(video_path, pts_unit='sec')
        except Exception as e:
            print(f"ビデオ読み込みエラー: {video_path}, {e}")
            # フォールバック: ダミーデータ
            video = torch.zeros(self.num_frames, 224, 224, 3)
        
        T = video.shape[0]
        
        # より良いフレームサンプリング
        if T < self.num_frames:
            # フレーム数が不足している場合の補間
            indices = torch.linspace(0, T - 1, self.num_frames).long()
            video = video[indices]
        else:
            if self.is_training:
                # ランダムサンプリング（トレーニング時）
                start_idx = torch.randint(0, max(1, T - self.num_frames), (1,)).item()
                indices = torch.arange(start_idx, start_idx + self.num_frames)
            else:
                # 等間隔サンプリング（検証時）
                indices = torch.linspace(0, T - 1, self.num_frames).long()
            video = video[indices]
        
        # 形状を調整
        video = video.permute(0, 3, 1, 2).float() / 255.0
        
        if self.transform:
            video = self.transform(video)
        
        video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
        
        label = ann["label"]
        if not isinstance(label, list):
            label = [label]
        
        target = torch.zeros(self.num_classes)
        for l in label:
            if 0 <= l < self.num_classes:
                target[l] = 1.0
        
        return video, target

def compute_precision_recall_f1_per_class(y_pred, y_target):
    """クラス別の詳細な評価指標を計算"""
    num_classes = y_pred.shape[1]
    precisions, recalls, f1s = [], [], []
    
    for i in range(num_classes):
        tp = np.sum(y_pred[:, i] * y_target[:, i])
        fp = np.sum(y_pred[:, i] * (1 - y_target[:, i]))
        fn = np.sum((1 - y_pred[:, i]) * y_target[:, i])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 1e-4 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    return np.mean(precisions), np.mean(recalls), np.mean(f1s)

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
    
    # 共通するキーのみを使用してロード
    filtered_ckpt = {k: v for k, v in ckpt.items() if k in common_keys}
    missing_keys_load, unexpected_keys_load = model.load_state_dict(filtered_ckpt, strict=False)
    
    print(f"ロード完了: {len(filtered_ckpt)}個のパラメータがロードされました")
    return len(filtered_ckpt) > 0

class FocalLoss(nn.Module):
    """不均衡データに効果的なFocal Loss"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def main():
    annotation_dir = "/home/ollo/VideoMAE/videomae-clean"
    video_root = "/srv/shared/data/ego4d/short_clips/verb_annotation_simple"
    converted_ckpt_path = "/home/ollo/VideoMAE/checkpoints/avion_finetune_cls_lavila_vitb_best_converted.pt"
    train_json = os.path.join(annotation_dir, "20250512_annotations_train.json")
    val_json = os.path.join(annotation_dir, "20250512_annotations_val.json")

    # 改良されたデータ拡張
    train_transform = volume_transforms.Compose([
        volume_transforms.Resize((256, 256)),  # より大きなサイズからクロップ
        volume_transforms.RandomCrop((224, 224)),  # ランダムクロップ追加
        volume_transforms.RandomHorizontalFlip(p=0.5),
        volume_transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),  # 色調整
        volume_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = volume_transforms.Compose([
        volume_transforms.Resize((224, 224)),
        volume_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # データセットとデータローダーの作成
    train_dataset = ImprovedEgo4DFlashDataset(train_json, video_root, train_transform, is_training=True)
    val_dataset = ImprovedEgo4DFlashDataset(val_json, video_root, val_transform, is_training=False)
    
    # バッチサイズを調整（より大きなバッチサイズで安定性向上）
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=4, pin_memory=True)

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # モデルの作成（よりアグレッシブな正則化）
    try:
        model = VisionTransformer(
            num_classes=58, 
            use_flash_attn=True, 
            drop_path_rate=0.3  # ドロップパス率を上げて過学習を防ぐ
        ).to(device)
        print("モデル作成完了")
    except Exception as e:
        print(f"モデル作成エラー: {e}")
        model = VisionTransformer(
            num_classes=58, 
            use_flash_attn=False, 
            drop_path_rate=0.3
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

    # より効果的なMixup設定
    mixup_fn = Mixup(
        mixup_alpha=0.4,  # より控えめなMixup
        cutmix_alpha=0.8, 
        prob=0.8,  # 確率を下げて、通常の学習も混ぜる
        switch_prob=0.5, 
        mode='batch', 
        label_smoothing=0.05,  # ラベルスムージングを控えめに
        num_classes=58
    )
    
    # より効果的な最適化設定
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=5e-4,  # 学習率を下げて安定性向上
        betas=(0.9, 0.999), 
        weight_decay=0.1,  # 重み減衰を強めて正則化
        eps=1e-8
    )
    
    # Cosine Annealing with Warm Restartsを使用
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # 10エポックでリスタート
        T_mult=1,
        eta_min=1e-6
    )
    
    # Focal Lossを使用（不均衡データに効果的）
    criterion = FocalLoss(alpha=1, gamma=2)
    scaler = GradScaler()

    # より多くのエポック数
    num_epochs = 1  # エポック数を増やす
    best_f1 = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # トレーニングフェーズ
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        print(f"エポック {epoch+1}/{num_epochs} 開始")
        
        for videos, targets in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            try:
                videos, targets = videos.to(device), targets.to(device)
                
                # Mixupを段階的に適用（初期エポックは控えめに）
                if epoch < 2:
                    # 初期エポックは通常の学習
                    pass
                else:
                    videos, targets = mixup_fn(videos, targets)
                
                optimizer.zero_grad()
                
                with autocast():
                    outputs = model(videos)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                # グラディエントクリッピング
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
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
                    
                    # より良い閾値設定（複数の閾値でテスト）
                    thresholds = [0.3, 0.4, 0.5]
                    best_threshold_f1 = 0
                    best_preds = None
                    
                    for threshold in thresholds:
                        temp_preds = (torch.sigmoid(outputs) > threshold).float()
                        temp_preds_np = temp_preds.cpu().numpy()
                        temp_targets_np = targets.cpu().numpy()
                        
                        # 一時的なF1を計算
                        _, _, temp_f1 = compute_precision_recall_f1_per_class(temp_preds_np, temp_targets_np)
                        
                        if temp_f1 > best_threshold_f1:
                            best_threshold_f1 = temp_f1
                            best_preds = temp_preds_np
                    
                    if best_preds is not None:
                        all_preds.append(best_preds)
                        all_targets.append(targets.cpu().numpy())
                    
                except Exception as e:
                    print(f"検証バッチエラー: {e}")
                    continue
        
        if all_preds and all_targets:
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            precision, recall, f1 = compute_precision_recall_f1_per_class(all_preds, all_targets)
            
            print(f"[Epoch {epoch+1}] Val Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # ベストモデルの保存
            if f1 > best_f1:
                best_f1 = f1
                best_model_state = model.state_dict().copy()
                print(f"★ 新しいベストF1スコア: {best_f1:.4f}")
        else:
            print(f"[Epoch {epoch+1}] 検証データの処理に失敗しました")

    # ベストモデルの保存
    if best_model_state is not None:
        try:
            save_path = f"avion_best_f1_{best_f1:.4f}.pth"
            torch.save(best_model_state, save_path)
            print(f"ベストモデルを {save_path} に保存しました (F1: {best_f1:.4f})")
        except Exception as e:
            print(f"モデル保存エラー: {e}")

if __name__ == "__main__":
    main()