import torch
from modeling_finetune import VisionTransformer

def check_model_loadable(checkpoint_path):
    try:
        # 推論と同じ構造でモデルを構築
        model = VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=58,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            fc_drop_rate=0.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=torch.nn.LayerNorm,
            init_values=0.0,
            use_learnable_pos_emb=False,
            all_frames=16,
            tubelet_size=2,
            use_checkpoint=False,
            use_mean_pooling=True,
            use_flash_attn=True
        )

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 重みの読み込み（strict=False でマージンを持たせる）
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

        print(f"✅ モデルの読み込みに成功しました: {checkpoint_path}")
        print(f" - Missing keys: {missing_keys}")
        print(f" - Unexpected keys: {unexpected_keys}")
        print(f" - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"❌ モデルの読み込みに失敗しました: {checkpoint_path}")
        print(f" - エラー内容: {str(e)}")

if __name__ == "__main__":
    checkpoint_path = "/home/ollo/VideoMAE/video.pth"  # あなたのパスに置き換えてください
    check_model_loadable(checkpoint_path)
