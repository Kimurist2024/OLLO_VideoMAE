import torch
import os
from collections import OrderedDict

def investigate_checkpoint(checkpoint_path):
    """チェックポイントファイルの詳細な調査"""
    print(f"=== チェックポイント調査: {checkpoint_path} ===")
    
    # ファイルの存在確認
    if not os.path.exists(checkpoint_path):
        print(f"エラー: ファイルが存在しません: {checkpoint_path}")
        return
    
    # ファイルサイズ確認
    file_size = os.path.getsize(checkpoint_path)
    print(f"ファイルサイズ: {file_size / (1024*1024):.2f} MB")
    
    if file_size < 1024:  # 1KB未満の場合
        print("警告: ファイルサイズが異常に小さいです")
    
    try:
        # チェックポイントの読み込み
        print("\n=== チェックポイント読み込み ===")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # トップレベルの構造を確認
        print(f"トップレベルキー: {list(ckpt.keys())}")
        print(f"トップレベルキー数: {len(ckpt.keys())}")
        
        # 各キーの詳細を調査
        for key in ckpt.keys():
            print(f"\n--- キー '{key}' の詳細 ---")
            value = ckpt[key]
            print(f"タイプ: {type(value)}")
            
            if isinstance(value, dict):
                print(f"辞書サイズ: {len(value)}")
                if len(value) > 0:
                    print(f"辞書のキー例（最初の10個）: {list(value.keys())[:10]}")
                    
                    # 各辞書エントリーの詳細
                    for i, (sub_key, sub_value) in enumerate(value.items()):
                        if i >= 5:  # 最初の5個のみ表示
                            print(f"... (残り{len(value)-5}個)")
                            break
                        print(f"  {sub_key}: {type(sub_value)}, shape: {getattr(sub_value, 'shape', 'N/A')}")
                        
            elif isinstance(value, torch.Tensor):
                print(f"テンソル形状: {value.shape}")
                print(f"テンソルデータ型: {value.dtype}")
            else:
                print(f"値: {value}")
        
        # モデルパラメータの確認
        print("\n=== モデルパラメータ確認 ===")
        
        # 'model'キーがある場合
        if 'model' in ckpt:
            model_dict = ckpt['model']
            print(f"'model'キー内のパラメータ数: {len(model_dict)}")
            
            if len(model_dict) > 0:
                print("モデルパラメータ例（最初の10個）:")
                for i, (param_name, param_tensor) in enumerate(model_dict.items()):
                    if i >= 10:
                        break
                    print(f"  {param_name}: {param_tensor.shape if hasattr(param_tensor, 'shape') else type(param_tensor)}")
        
        # 'state_dict'キーがある場合
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
            print(f"'state_dict'キー内のパラメータ数: {len(state_dict)}")
            
            if len(state_dict) > 0:
                print("状態辞書パラメータ例（最初の10個）:")
                for i, (param_name, param_tensor) in enumerate(state_dict.items()):
                    if i >= 10:
                        break
                    print(f"  {param_name}: {param_tensor.shape if hasattr(param_tensor, 'shape') else type(param_tensor)}")
        
        # 直接的なパラメータがある場合
        else:
            # テンソルのキーのみを抽出
            tensor_keys = [k for k, v in ckpt.items() if isinstance(v, torch.Tensor)]
            print(f"直接的なテンソルキー数: {len(tensor_keys)}")
            
            if len(tensor_keys) > 0:
                print("直接テンソルパラメータ例（最初の10個）:")
                for i, key in enumerate(tensor_keys[:10]):
                    tensor = ckpt[key]
                    print(f"  {key}: {tensor.shape}")
    
    except Exception as e:
        print(f"エラー: チェックポイント読み込み失敗: {e}")
        
        # ファイルの先頭を確認（バイナリ）
        try:
            with open(checkpoint_path, 'rb') as f:
                header = f.read(100)
                print(f"ファイルヘッダー（最初の100バイト）: {header}")
        except Exception as e2:
            print(f"ファイル読み込みエラー: {e2}")

def compare_with_model(checkpoint_path, model):
    """チェックポイントとモデルの詳細比較"""
    print(f"\n=== モデルとの比較 ===")
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        ckpt_data = ckpt.get('model', ckpt)
        
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(ckpt_data.keys())
        
        print(f"モデルキー数: {len(model_keys)}")
        print(f"チェックポイントキー数: {len(ckpt_keys)}")
        
        # 共通キー
        common_keys = model_keys.intersection(ckpt_keys)
        print(f"共通キー数: {len(common_keys)}")
        
        # 不足キー
        missing_keys = model_keys - ckpt_keys
        print(f"不足キー数: {len(missing_keys)}")
        if len(missing_keys) > 0:
            print("不足キー例（最初の10個）:")
            for key in list(missing_keys)[:10]:
                print(f"  - {key}")
        
        # 余分キー
        unexpected_keys = ckpt_keys - model_keys
        print(f"余分キー数: {len(unexpected_keys)}")
        if len(unexpected_keys) > 0:
            print("余分キー例（最初の10個）:")
            for key in list(unexpected_keys)[:10]:
                print(f"  + {key}")
        
        # キー名パターンの分析
        print("\n=== キー名パターン分析 ===")
        print("モデルキーのパターン例:")
        for key in list(model_keys)[:10]:
            print(f"  {key}")
        
        print("チェックポイントキーのパターン例:")
        for key in list(ckpt_keys)[:10]:
            print(f"  {key}")
            
    except Exception as e:
        print(f"比較エラー: {e}")

# 使用例
if __name__ == "__main__":
    checkpoint_path = "/home/ollo/VideoMAE/checkpoints/avion_finetune_cls_lavila_vitb_best_converted.pt"
    
    # チェックポイントの詳細調査
    investigate_checkpoint(checkpoint_path)
    
    # モデルとの比較（モデルが利用可能な場合）
    try:
        import sys
        sys.path.append('/home/ollo/VideoMAE/AVION')
        from AVION.avion.models.model_videomae import VisionTransformer
        
        model = VisionTransformer(num_classes=58, use_flash_attn=False, drop_path_rate=0.2)
        compare_with_model(checkpoint_path, model)
        
    except Exception as e:
        print(f"モデル比較をスキップ: {e}")
    
    # 同じディレクトリの他のファイルも確認
    checkpoint_dir = "/home/ollo/VideoMAE/checkpoints/"
    if os.path.exists(checkpoint_dir):
        print(f"\n=== チェックポイントディレクトリ内容 ===")
        files = os.listdir(checkpoint_dir)
        for file in files:
            if file.endswith(('.pt', '.pth', '.ckpt')):
                file_path = os.path.join(checkpoint_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"{file}: {file_size / (1024*1024):.2f} MB")