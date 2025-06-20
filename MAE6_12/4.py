import os
import sys
import torch
import torch.nn as nn

# パス設定
sys.path.append('/home/ollo/VideoMAE/')
sys.path.append('/home/ollo/VideoMAE/AVION/')

from AVION.avion.models.model_videomae import PretrainVisionTransformer

def debug_model_structure():
    """モデルの実際の構造を確認"""
    print("🔍 Debugging PretrainVisionTransformer structure...")
    
    model = PretrainVisionTransformer(
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
    
    print("📋 Model attributes:")
    for name, module in model.named_children():
        print(f"  - {name}: {type(module).__name__}")
    
    print("\n🔍 Looking for patch embedding related attributes:")
    patch_embed_attrs = [attr for attr in dir(model) if 'patch' in attr.lower()]
    print(f"Patch-related attributes: {patch_embed_attrs}")
    
    print("\n🔍 Looking for embedding related attributes:")
    embed_attrs = [attr for attr in dir(model) if 'embed' in attr.lower()]
    print(f"Embed-related attributes: {embed_attrs}")
    
    print("\n🔍 All main attributes:")
    main_attrs = [attr for attr in dir(model) if not attr.startswith('_') and not callable(getattr(model, attr))]
    print(f"Main attributes: {main_attrs}")
    
    # テスト入力でforward pathを確認
    print("\n🧪 Testing forward path...")
    test_input = torch.randn(1, 3, 16, 224, 224)
    test_mask = torch.ones(1, 8*14*14, dtype=torch.bool)
    
    try:
        output = model(test_input, test_mask)
        print(f"✅ Forward pass successful! Output shape: {output.shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        
        # エラーの詳細を調べる
        print("\n🔍 Trying to understand the forward method...")
        import inspect
        forward_source = inspect.getsource(model.forward)
        print("Forward method source (first 20 lines):")
        print('\n'.join(forward_source.split('\n')[:20]))

if __name__ == "__main__":
    debug_model_structure()