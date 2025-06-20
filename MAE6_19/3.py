#!/usr/bin/env python3
"""
単一動画の推論と可視化を簡単に実行するスクリプト
"""

import os
import sys

def main():
    # 設定 - 必要に応じて変更してください
    MODEL_PATH = "/home/ollo/VideoMAE/videomae-clean/MAE6_18/videomae_finetuned_interactive.pth"
    VIDEO_DIR = "/home/ollo/Ollo_video"
    OUTPUT_DIR = "./inference_results"
    
    # 使用方法の表示
    if len(sys.argv) != 2:
        print("Usage: python infer_single_video.py <video_filename>")
        print(f"Available videos in {VIDEO_DIR}:")
        
        # 利用可能な動画ファイルを表示
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        if os.path.exists(VIDEO_DIR):
            for file in os.listdir(VIDEO_DIR):
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(file)
        
        if video_files:
            for i, video in enumerate(video_files, 1):
                print(f"  {i}. {video}")
        else:
            print("  No video files found.")
        
        sys.exit(1)
    
    video_filename = sys.argv[1]
    video_path = os.path.join(VIDEO_DIR, video_filename)
    
    # ファイルの存在確認
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 出力ファイル名の生成
    name_without_ext = os.path.splitext(video_filename)[0]
    output_path = os.path.join(OUTPUT_DIR, f"{name_without_ext}_visualization.mp4")
    
    # 推論コマンドの実行
    cmd = f"""python video_inference_visualization.py \
        --model_path "{MODEL_PATH}" \
        --video_path "{video_path}" \
        --output_path "{output_path}" \
        --num_classes 58 \
        --fps 10 \
        --window_size 16 \
        --overlap 8"""
    
    print(f"Processing video: {video_filename}")
    print(f"Output will be saved to: {output_path}")
    print()
    print("Running inference...")
    
    # コマンド実行
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        print(f"\n✓ Successfully processed: {video_filename}")
        print(f"Visualization saved to: {output_path}")
    else:
        print(f"\n✗ Failed to process: {video_filename}")
        sys.exit(1)

if __name__ == "__main__":
    main()