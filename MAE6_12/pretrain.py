
#/home/ollo/pretrain_epoch10.pthを作成した






import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import warnings

# 警告抑制
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# パス設定
sys.path.append('/home/ollo/VideoMAE/')
sys.path.append('/home/ollo/VideoMAE/AVION/')

from AVION.avion.models.model_videomae import PretrainVisionTransformer
from AVION.avion.data.transforms import TubeMaskingGenerator
from AVION.avion.utils.meters import AverageMeter
from AVION.avion.optim.schedulers import cosine_scheduler

# ダミーデータセット
class DummyVideoDataset(torch.utils.data.Dataset):
    def __init__(self, length=100):
        self.length = length
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        video = torch.randn(3, 16, 224, 224)  # C, T, H, W
        return video

# MSE loss (マスクを考慮)
def reconstruction_loss(pred, target, mask):
    # 形状を確認
    print(f"pred shape: {pred.shape}")
    print(f"target shape: {target.shape}")
    print(f"mask shape: {mask.shape}")
    
    # マスクされた部分のみでlossを計算
    loss = ((pred - target) ** 2) * mask.unsqueeze(-1)
    loss = loss.sum() / mask.sum()
    return loss

# メイン関数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=3 * 2 * 16 * 16,  # 1536
        decoder_embed_dim=384,
        decoder_depth=8,
        decoder_num_heads=6,
        tubelet_size=2,
        qkv_bias=True,
        use_learnable_pos_emb=False,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
    ).to(device)
    
    # 重み読み込み
    ckpt_path = "/home/ollo/VideoMAE/checkpoints/avion_finetune_cls_lavila_vitb_best_converted.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    print("✓ Checkpoint loaded with strict=False")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scaler = GradScaler()
    
    dataset = DummyVideoDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    mask_generator = TubeMaskingGenerator(input_size=(8, 14, 14), mask_ratio=0.9)
    
    model.train()
    
    for epoch in range(300):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for videos in pbar:
            videos = videos.to(device)
            B, C, T, H, W = videos.shape
            
            # マスク生成: numpy → torch.bool
            np_masks = np.stack([mask_generator() for _ in range(B)])  # shape: [B, T*H*W]
            masks = torch.from_numpy(np_masks).to(device=device, dtype=torch.bool)
            
            with autocast():
                pred = model(videos, masks)
                
                # モデルの出力形状を確認して、それに合わせてターゲットを作成
                print(f"Original video shape: {videos.shape}")
                print(f"Prediction shape: {pred.shape}")
                print(f"Mask shape: {masks.shape}")
                
                # ターゲットの形状をpredに合わせて調整
                # predの形状に基づいてターゲットを作成
                B, num_patches, feature_dim = pred.shape
                
                # 簡単なターゲット作成（実際のパッチ化は複雑なので、とりあえず動作確認用）
                # ビデオをパッチに分割してからreshape
                tubelet_size = 2
                patch_size = 16
                
                # (B, C, T, H, W) -> パッチ化
                B, C, T, H, W = videos.shape
                T_patches = T // tubelet_size  # 8
                H_patches = H // patch_size    # 14
                W_patches = W // patch_size    # 14
                
                # ビデオをパッチに分割
                patches = videos.view(B, C, T_patches, tubelet_size, H_patches, patch_size, W_patches, patch_size)
                patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7)  # (B, T_patches, H_patches, W_patches, C, tubelet_size, patch_size, patch_size)
                target = patches.reshape(B, T_patches * H_patches * W_patches, C * tubelet_size * patch_size * patch_size)
                
                # predとtargetの形状を揃える
                if target.shape[1] != pred.shape[1]:
                    # マスクの形状に合わせてターゲットをトリミングまたはパディング
                    min_patches = min(target.shape[1], pred.shape[1])
                    target = target[:, :min_patches, :]
                    pred = pred[:, :min_patches, :]
                    masks = masks[:, :min_patches]
                
                print(f"Adjusted target shape: {target.shape}")
                print(f"Adjusted pred shape: {pred.shape}")
                print(f"Adjusted mask shape: {masks.shape}")
                
                loss = reconstruction_loss(pred, target, masks)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix(loss=loss.item())
        
        torch.save(model.state_dict(), f"pretrain_epoch{epoch+1}.pth")
        print(f"✓ Saved model for epoch {epoch+1}")

if __name__ == "__main__":
    main()