import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import warnings
from timm.models.vision_transformer import VisionTransformer
from functools import partial

warnings.filterwarnings("ignore")

# ✅ ViT-Baseモデル定義（マルチラベル出力）
def vit_base_patch16_224(num_classes=58):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    return model

# ✅ 仮のマルチラベルデータセット（要差し替え）
class DummyMultiLabelDataset(Dataset):
    def __init__(self, num_samples=100):
        self.data = torch.rand(num_samples, 3, 224, 224)
        self.labels = torch.randint(0, 2, (num_samples, 58)).float()  # バイナリマルチラベル
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ モデル初期化と読み込み
    model = vit_base_patch16_224(num_classes=58).to(device)
    ckpt_path = "/home/ollo/VideoMAE/checkpoints/avion_finetune_cls_lavila_vitb_best_converted.pt"
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    msg = model.load_state_dict(state_dict, strict=False)
    print("✓ Checkpoint loaded with strict=False")

    # ✅ データとロス
    dataset = DummyMultiLabelDataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ✅ トレーニングループ
    model.train()
    for epoch in range(1):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            preds = model(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

if __name__ == "__main__":
    main()
