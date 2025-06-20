import os
import sys
import json
import torch
from torch.utils.data import DataLoader

sys.path.append('/home/ollo/VideoMAE')
sys.path.append('/home/ollo/VideoMAE/AVION')

# train_videoMAE_ego4d.pyからインポート
from train_videoMAE_ego4d import Ego4DVideoDataset, build_dataloader

# パス設定
annotation_dir = "/home/ollo/VideoMAE/videomae-clean"
video_root = "/srv/shared/data/ego4d/short_clips/verb_annotation_simple"
train_json = os.path.join(annotation_dir, "20250512_annotations_train.json")
val_json = os.path.join(annotation_dir, "20250512_annotations_val.json")

print("=== Testing Ego4D DataLoader ===")

# データセットの作成
print("\nCreating dataset...")
dataset = Ego4DVideoDataset(
    annotation_file=train_json,
    video_root=video_root,
    num_frames=16,
    image_size=224
)

print(f"Dataset size: {len(dataset)}")

# 最初のサンプルを読み込んでみる
print("\nLoading first sample...")
try:
    video, label, path, idx = dataset[0]
    print(f"Video shape: {video.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Label sum: {label.sum().item()} (number of positive labels)")
    print(f"Video path: {path}")
    print("First sample loaded successfully!")
except Exception as e:
    print(f"Error loading first sample: {e}")
    import traceback
    traceback.print_exc()

# データローダーのテスト
print("\n\nTesting DataLoader...")
try:
    train_loader, val_loader = build_dataloader(
        train_json=train_json,
        val_json=val_json,
        video_root=video_root,
        batch_size=4,
        num_workers=0  # デバッグ時は0に設定
    )
    
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")
    
    # 最初のバッチを取得
    print("\nGetting first batch...")
    for batch_idx, (videos, labels, paths, indices) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Videos shape: {videos.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels per sample: {labels.sum(dim=1).tolist()}")
        break
    
    print("\nDataLoader test successful!")
    
except Exception as e:
    print(f"Error with DataLoader: {e}")
    import traceback
    traceback.print_exc()