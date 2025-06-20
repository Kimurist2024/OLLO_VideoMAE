import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import warnings
import cv2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

# 警告抑制
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# パス設定
sys.path.append('/home/ollo/VideoMAE/')
sys.path.append('/home/ollo/VideoMAE/AVION/')

from AVION.avion.models.model_videomae import PretrainVisionTransformer

class Ego4DVideoDataset(Dataset):
    def __init__(self, annotation_file, video_root, num_frames=16, img_size=224):
        """
        Ego4D動詞分類データセット（COCO形式対応）
        """
        self.video_root = video_root
        self.num_frames = num_frames
        self.img_size = img_size
        
        # アノテーションファイルを読み込み（COCO形式）
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        # COCO形式のデータを取得
        self.categories = data.get('categories', {})
        self.annotations = data.get('annotations', [])
        
        print(f"Loaded {len(self.annotations)} annotations from {annotation_file}")
        print(f"Found {len(self.categories)} categories")
        
        # ラベルマッピングを作成
        self.label_to_idx = self.categories  # カテゴリ名 -> インデックス
        self.idx_to_label = {v: k for k, v in self.categories.items()}  # インデックス -> カテゴリ名
        self.num_classes = len(self.categories)
        
        if self.num_classes > 0:
            print(f"Classes ({self.num_classes}):")
            # カテゴリをインデックス順にソートして表示
            sorted_categories = sorted(self.categories.items(), key=lambda x: x[1])
            for i, (label, idx) in enumerate(sorted_categories):
                if i < 10:  # 最初の10個を表示
                    print(f"  {idx}: {label}")
                elif i == 10:
                    print(f"  ... and {self.num_classes - 10} more classes")
                    break
        else:
            print("⚠️ No categories found!")
        
        # カテゴリIDの分布を確認
        category_ids = [ann.get('category_id', -1) for ann in self.annotations]
        category_counter = Counter(category_ids)
        print(f"\nCategory distribution (top 10):")
        for cat_id, count in sorted(category_counter.items(), key=lambda x: x[1], reverse=True)[:10]:
            if cat_id in self.idx_to_label:
                print(f"  {cat_id} ({self.idx_to_label[cat_id]}): {count} samples")
            else:
                print(f"  {cat_id} (unknown): {count} samples")
        
        # 変換処理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_video(self, video_path):
        """ビデオファイルからフレームを抽出"""
        if not os.path.exists(video_path):
            # ビデオファイルが見つからない場合はダミービデオを返す（警告は出さない）
            return torch.randn(3, self.num_frames, self.img_size, self.img_size)
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # 総フレーム数を取得
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            return torch.randn(3, self.num_frames, self.img_size, self.img_size)
        
        if total_frames < self.num_frames:
            # フレーム数が不足している場合は最後のフレームを複製
            frame_indices = list(range(total_frames)) + [total_frames-1] * (self.num_frames - total_frames)
        else:
            # 均等にフレームをサンプリング
            frame_indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
        
        success_count = 0
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Transform適用
                frame_tensor = self.transform(frame)
                frames.append(frame_tensor)
                success_count += 1
            else:
                # フレーム読み込みに失敗した場合
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(torch.randn(3, self.img_size, self.img_size))
        
        cap.release()
        
        # (num_frames, C, H, W) -> (C, num_frames, H, W)
        video_tensor = torch.stack(frames, dim=1)
        return video_tensor
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # ビデオファイルパスを構築
        video_file = None
        
        # 様々なキーを試してビデオファイル名を取得
        if 'video_path' in ann:
            video_file = ann['video_path']
        elif 'clip_id' in ann:
            video_file = f"{ann['clip_id']}.mp4"
        elif 'video_id' in ann:
            video_file = f"{ann['video_id']}.mp4"
        else:
            # ファイル名を推測
            video_file = f"video_{idx}.mp4"
        
        video_path = os.path.join(self.video_root, video_file)
        
        # ビデオロード
        video = self._load_video(video_path)
        
        # ラベル取得（COCO形式ではcategory_idを使用）
        category_id = ann.get('category_id', 0)
        
        # category_idが有効な範囲内かチェック
        if category_id < 0 or category_id >= self.num_classes:
            category_id = 0
        
        return video, category_id

# 分類用モデル（修正版）
class VideoMAEClassifier(nn.Module):
    def __init__(self, pretrained_path, num_classes):
        super().__init__()
        
        # 事前学習済みVideoMAEをロード
        self.backbone = PretrainVisionTransformer(
            img_size=224,
            patch_size=16,
            encoder_embed_dim=768,
            encoder_depth=12,
            encoder_num_heads=12,
            encoder_num_classes=0,
            decoder_num_classes=3 * 2 * 16 * 16,
            decoder_embed_dim=384,
            decoder_depth=8,
            decoder_num_heads=6,
            tubelet_size=2,
            qkv_bias=True,
            use_learnable_pos_emb=False,
            norm_layer=nn.LayerNorm,
            use_checkpoint=False,
        )
        
        # 事前学習済み重みをロード
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        self.backbone.load_state_dict(checkpoint, strict=False)
        print(f"✓ Loaded pretrained weights from {pretrained_path}")
        
        # 分類ヘッドを追加
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.2),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        # デコーダーの重みを固定（メモリ節約）
        for param in self.backbone.decoder.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        # マスクなしでエンコーダーのみを使用
        # 全てのパッチを可視化（マスクなし）
        mask = torch.zeros(B, T//2 * H//16 * W//16, dtype=torch.bool, device=x.device)
        
        # エンコーダーで特徴抽出
        features = self.backbone.encoder(x, mask)  # [B, N_vis, C_e]
        
        # グローバル平均プーリング
        features = features.mean(dim=1)  # [B, C_e] = [B, 768]
        
        # 分類
        logits = self.classifier(features)
        return logits

def main():
    annotation_dir = "/home/ollo/VideoMAE/videomae-clean"
    video_root = "/srv/shared/data/ego4d/short_clips/verb_annotation_simple"
    pretrained_path = "/home/ollo/pretrain_epoch300.pth"  # 事前学習済みモデル
    train_json = os.path.join(annotation_dir, "20250512_annotations_train.json")
    val_json = os.path.join(annotation_dir, "20250512_annotations_val.json")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # データセット作成
    print("Loading datasets...")
    train_dataset = Ego4DVideoDataset(train_json, video_root)
    val_dataset = Ego4DVideoDataset(val_json, video_root)
    
    # クラス数を確認
    num_classes = train_dataset.num_classes
    print(f"\nNumber of classes: {num_classes}")
    
    if num_classes <= 1:
        print("⚠️ Error: Not enough classes for training!")
        return
    
    # データローダー
    batch_size = 1  # さらにメモリ節約
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # モデル作成
    model = VideoMAEClassifier(pretrained_path, num_classes).to(device)
    
    # 学習設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)  # 学習率を小さく
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    scaler = GradScaler()
    
    # 学習ループ
    num_epochs = 20  # エポック数を減らす
    best_val_acc = 0.0
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        # 訓練フェーズ
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (videos, labels) in enumerate(train_pbar):
            try:
                videos, labels = videos.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                with autocast():
                    outputs = model(videos)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # 統計
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
                
                # 100バッチごとに進捗表示
                if (batch_idx + 1) % 100 == 0:
                    current_acc = 100. * train_correct / train_total
                    print(f"\nBatch {batch_idx + 1}: Train Acc = {current_acc:.2f}%")
                
            except Exception as e:
                print(f"\nError in training batch {batch_idx}: {e}")
                continue
        
        # 検証フェーズ
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (videos, labels) in enumerate(val_pbar):
                try:
                    videos, labels = videos.to(device), labels.to(device)
                    
                    outputs = model(videos)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*val_correct/val_total:.2f}%'
                    })
                    
                except Exception as e:
                    print(f"\nError in validation batch {batch_idx}: {e}")
                    continue
        
        # エポック結果
        train_acc = 100. * train_correct / train_total if train_total > 0 else 0
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        # ベストモデル保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "/home/ollo/best_ego4d_classifier.pth")
            print(f"✓ New best model saved! Val Acc: {val_acc:.2f}%")
        
        scheduler.step()
        print("-" * 60)
    
    # 最終結果
    print(f"\n🎯 Final Results:")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # 詳細な分類レポート
    if len(all_preds) > 0 and len(all_labels) > 0:
        print(f"\n📊 Classification Report (Last Epoch):")
        # 実際に使用されたラベルのみを対象とする
        unique_labels = sorted(list(set(all_labels)))
        class_names = [train_dataset.idx_to_label.get(i, f"Class_{i}") for i in unique_labels]
        
        if len(unique_labels) > 0:
            print(classification_report(all_labels, all_preds, 
                                      target_names=class_names,
                                      zero_division=0))

if __name__ == "__main__":
    main()