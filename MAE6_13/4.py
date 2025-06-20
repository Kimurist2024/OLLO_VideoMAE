import torch

ckpt_path = "/home/ollo/VideoMAE/checkpoints/avion_finetune_cls_lavila_vitb_best_converted.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

print("=== Checkpoint Top-level Keys ===")
print(ckpt.keys())

# 任意で各キーの中身の型も確認
for k in ckpt.keys():
    print(f"{k}: type = {type(ckpt[k])}")
