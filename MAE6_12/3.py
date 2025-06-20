import json
import os

def debug_json_files():
    """JSONファイルの構造を確認"""
    
    annotation_dir = "/home/ollo/VideoMAE/videomae-clean"
    train_json = os.path.join(annotation_dir, "20250512_annotations_train.json")
    val_json = os.path.join(annotation_dir, "20250512_annotations_val.json")
    
    print("🔍 Debugging JSON files...")
    print("=" * 400)
    
    for json_file in [train_json, val_json]:
        print(f"\n📁 File: {json_file}")
        
        if not os.path.exists(json_file):
            print(f"❌ File not found!")
            continue
            
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            print(f"✅ File loaded successfully")
            print(f"📊 Data type: {type(data)}")
            print(f"📏 Data length/size: {len(data) if hasattr(data, '__len__') else 'N/A'}")
            
            # データの構造を詳しく確認
            if isinstance(data, dict):
                print(f"🗝️  Dictionary keys: {list(data.keys())}")
                for key, value in list(data.items())[:5]:  # 最初の3つのキーを表示
                    print(f"   {key}: {type(value)} - {str(value)[:100]}...")
                    
            elif isinstance(data, list):
                print(f"📋 List with {len(data)} items")
                if len(data) > 0:
                    print(f"🔬 First item type: {type(data[0])}")
                    print(f"🔬 First item: {data[0]}")
                    
                    if len(data) > 1:
                        print(f"🔬 Second item: {data[1]}")
            else:
                print(f"❓ Unknown data type: {type(data)}")
                print(f"📄 Content: {str(data)[:300]}...")
                
        except Exception as e:
            print(f"❌ Error loading file: {e}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    debug_json_files()