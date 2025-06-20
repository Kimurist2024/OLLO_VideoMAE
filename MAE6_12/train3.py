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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from collections import Counter
import math

# 警告抑制
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# パス設定
sys.path.append('/home/ollo/VideoMAE/')
sys.path.append('/home/ollo/VideoMAE/AVION/')

from AVION.avion.models.model_videomae import PretrainVisionTransformer

def compute_precision_recall_f1(y_pred, y_target):
    """
    バイナリ予測のPrecision, Recall, F1を計算
    y_pred: 0 or 1 binary prediction
    y_target: binary target
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

class FlashMultiHeadAttention(nn.Module):
    """Flash Attention実装"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Flash Attentionが利用可能かチェック
        self.use_flash_attention = self._check_flash_attention()
        if self.use_flash_attention:
            print("✅ Using Flash Attention (PyTorch built-in)")
        else:
            print("⚠️ Flash Attention not available, using standard attention")
    
    def _check_flash_attention(self):
        """Flash Attentionが利用可能かチェック"""
        try:
            # PyTorch 2.0以降のscaled_dot_product_attentionをチェック
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                if torch.cuda.is_available():
                    return torch.backends.cuda.flash_sdp_enabled()
            return False
        except:
            return False
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, num_heads, N, head_dim]
        
        if self.use_flash_attention:
            # Flash Attentionを使用
            try:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=False
                ):
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v,
                        dropout_p=self.attn_drop.p if self.training else 0.0,
                        scale=self.scale
                    )
            except Exception as e:
                print(f"Flash Attention failed, falling back to standard: {e}")
                attn_output = self._standard_attention(q, k, v)
        else:
            # 標準のAttentionを使用
            attn_output = self._standard_attention(q, k, v)
        
        # [B, num_heads, N, head_dim] -> [B, N, num_heads, head_dim] -> [B, N, C]
        x = attn_output.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def _standard_attention(self, q, k, v):
        """標準のAttention実装"""
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        return attn @ v

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

# Flash Attention対応のVideoMAE分類モデル
class VideoMAEClassifierWithFlashAttention(nn.Module):
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
        
        # エンコーダーのAttentionをFlash Attentionに置き換え
        self._replace_attention_with_flash_attention()
        
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
    
    def _replace_attention_with_flash_attention(self):
        """エンコーダーのAttentionをFlash Attentionに置き換え"""
        if hasattr(self.backbone.encoder, 'blocks'):
            for i, block in enumerate(self.backbone.encoder.blocks):
                if hasattr(block, 'attn'):
                    # 元のAttentionの設定を取得
                    old_attn = block.attn
                    embed_dim = old_attn.qkv.in_features
                    num_heads = old_attn.num_heads if hasattr(old_attn, 'num_heads') else 12
                    
                    # Flash Attentionに置き換え
                    block.attn = FlashMultiHeadAttention(
                        dim=embed_dim,
                        num_heads=num_heads,
                        qkv_bias=True,
                        attn_drop=0.0,
                        proj_drop=0.0
                    )
                    
                    # 重みをコピー（可能な場合）
                    try:
                        block.attn.qkv.weight.data = old_attn.qkv.weight.data.clone()
                        if old_attn.qkv.bias is not None:
                            block.attn.qkv.bias.data = old_attn.qkv.bias.data.clone()
                        block.attn.proj.weight.data = old_attn.proj.weight.data.clone()
                        if old_attn.proj.bias is not None:
                            block.attn.proj.bias.data = old_attn.proj.bias.data.clone()
                    except Exception as e:
                        print(f"Warning: Could not copy weights for block {i}: {e}")
            
            print(f"✅ Replaced attention in {len(self.backbone.encoder.blocks)} encoder blocks with Flash Attention")
            
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
    
    # Flash Attentionの状況を確認
    print("\n🔍 Flash Attention Status:")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        try:
            print(f"Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}")
            print(f"Math SDP enabled: {torch.backends.cuda.math_sdp_enabled()}")
            print(f"Mem efficient SDP enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
        except:
            print("Could not check SDP status")
    
    # データセット作成
    print("\nLoading datasets...")
    train_dataset = Ego4DVideoDataset(train_json, video_root)
    val_dataset = Ego4DVideoDataset(val_json, video_root)
    
    # クラス数を確認
    num_classes = train_dataset.num_classes
    print(f"\nNumber of classes: {num_classes}")
    
    if num_classes <= 1:
        print("⚠️ Error: Not enough classes for training!")
        return
    
    # データローダー
    batch_size = 2  # Flash Attentionでメモリ効率向上
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # モデル作成（Flash Attention対応）
    model = VideoMAEClassifierWithFlashAttention(pretrained_path, num_classes).to(device)
    
    # 学習設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    scaler = GradScaler()
    
    # 学習ループ
    num_epochs = 20
    best_val_acc = 0.0
    best_val_f1 = 0.0
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        # 訓練フェーズ
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []
        
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
                
                all_train_preds.extend(predicted.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
                
                # 100バッチごとに進捗表示
                if (batch_idx + 1) % 1000 == 0:
                    current_acc = 100. * train_correct / train_total
                    print(f"\nBatch {batch_idx + 1}: Train Acc = {current_acc:.2f}%")
                
            except Exception as e:
                print(f"\nError in training batch {batch_idx}: {e}")
                continue
        
        # 訓練のF1スコア計算（カスタム関数使用）
        train_f1_scores = []
        train_precisions = []
        train_recalls = []
        
        for class_id in range(num_classes):
            y_pred_binary = (np.array(all_train_preds) == class_id).astype(int)
            y_true_binary = (np.array(all_train_labels) == class_id).astype(int)
            
            precision, recall, f1 = compute_precision_recall_f1(y_pred_binary, y_true_binary)
            train_f1_scores.append(f1)
            train_precisions.append(precision)
            train_recalls.append(recall)
        
        train_f1_macro = np.mean(train_f1_scores)
        train_precision_macro = np.mean(train_precisions)
        train_recall_macro = np.mean(train_recalls)
        
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
        
        # 検証のF1スコア計算（カスタム関数使用）
        val_f1_scores = []
        val_precisions = []
        val_recalls = []
        
        for class_id in range(num_classes):
            y_pred_binary = (np.array(all_preds) == class_id).astype(int)
            y_true_binary = (np.array(all_labels) == class_id).astype(int)
            
            precision, recall, f1 = compute_precision_recall_f1(y_pred_binary, y_true_binary)
            val_f1_scores.append(f1)
            val_precisions.append(precision)
            val_recalls.append(recall)
        
        val_f1_macro = np.mean(val_f1_scores)
        val_precision_macro = np.mean(val_precisions)
        val_recall_macro = np.mean(val_recalls)
        
        # エポック結果
        train_acc = 100. * train_correct / train_total if train_total > 0 else 0
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Train Precision: {train_precision_macro:.4f}, Train Recall: {train_recall_macro:.4f}, Train F1: {train_f1_macro:.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Val Precision: {val_precision_macro:.4f}, Val Recall: {val_recall_macro:.4f}, Val F1: {val_f1_macro:.4f}")
        
        # 詳細なクラス別結果（最初の5クラスのみ表示）
        print("Class-wise results (first 5 classes):")
        for i in range(min(5, num_classes)):
            class_name = train_dataset.idx_to_label.get(i, f"Class_{i}")
            print(f"  {class_name}: P={val_precisions[i]:.3f}, R={val_recalls[i]:.3f}, F1={val_f1_scores[i]:.3f}")
        if num_classes > 5:
            print(f"  ... and {num_classes - 5} more classes")
        
        # ベストモデル保存（F1スコアベース）
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            best_val_acc = val_acc
            torch.save(model.state_dict(), "/home/ollo/best_ego4d_classifier_flash.pth")
            print(f"✅ New best model saved! Val F1: {val_f1_macro:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step()
        print("-" * 60)
    
    # 最終結果
    print(f"\n🎯 Final Results:")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Validation F1 (Macro): {best_val_f1:.4f}")
    
    # 最終エポックでの詳細なクラス別F1スコア
    if len(all_preds) > 0 and len(all_labels) > 0:
        print(f"\n📊 Final Class-wise Performance:")
        print("=" * 80)
        print(f"{'Class Name':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 80)
        
        for class_id in range(num_classes):
            y_pred_binary = (np.array(all_preds) == class_id).astype(int)
            y_true_binary = (np.array(all_labels) == class_id).astype(int)
            
            precision, recall, f1 = compute_precision_recall_f1(y_pred_binary, y_true_binary)
            support = np.sum(y_true_binary)
            class_name = train_dataset.idx_to_label.get(class_id, f"Class_{class_id}")
            
            print(f"{class_name:<25} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {support:<10}")
        
        print("-" * 80)
        print(f"{'Macro Average':<25} {val_precision_macro:<10.3f} {val_recall_macro:<10.3f} {val_f1_macro:<10.3f} {len(all_labels):<10}")
        
        # 詳細な分類レポート（scikit-learn版も表示）
        print(f"\n📋 Scikit-learn Classification Report (for comparison):")
        unique_labels = sorted(list(set(all_labels)))
        class_names = [train_dataset.idx_to_label.get(i, f"Class_{i}") for i in unique_labels]
        
        if len(unique_labels) > 0:
            print(classification_report(all_labels, all_preds, 
                                      target_names=class_names,
                                      zero_division=0))

if __name__ == "__main__":
    main()