
import argparse
import json
import pickle
import cv2
import os
import numpy as np
from tqdm import tqdm
from inference_utils import bisect_nearest
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/home/ollo/VideoMAE/data/fonts/static/NotoSansJP-Regular.ttf"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def cv2_put_japanese_text(
    cv2_image: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_path: str,
    font_size: int = 32,
    color: tuple = (0, 0, 0)
) -> np.ndarray:
    """OpenCV の画像に日本語文字列を描画する"""
    b, g, r = color
    pil_color = (r, g, b)
    pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    draw.text((x, y), text, font=font, fill=pil_color)
    cv2_image_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return cv2_image_with_text


def get_confidence_color(confidence):
    """信頼度に基づいて色を返す"""
    if confidence >= 0.7:
        return (0, 0, 255)  # Red (BGR)
    elif confidence >= 0.4:
        return (0, 165, 255)  # Orange (BGR)
    else:
        return (0, 255, 0)  # Green (BGR)


def visualize_factory_analysis(
    factory_analysis_path,
    read_video_path, 
    visualize_dir, 
    conf_threshold=0.01,
    top_n_labels=5
):
    """factory_analysis.jsonを使用して動画を可視化"""
    
    # Load factory analysis results
    factory_data = load_json(factory_analysis_path)
    all_detections = factory_data["factory_analysis"]["all_detections"]
    
    # フレームごとの検出結果を整理
    frame2detections = {}
    for detection_info in all_detections:
        frame = detection_info["frame"]
        detections = detection_info["detections"]
        
        # 信頼度でフィルタリング
        filtered_detections = [d for d in detections if d["confidence"] >= conf_threshold]
        
        # 信頼度でソート
        filtered_detections.sort(key=lambda x: x["confidence"], reverse=True)
        
        # 上位N個のみ
        frame2detections[frame] = filtered_detections[:top_n_labels]
    
    print(f"Loaded {len(frame2detections)} frames with detections")
    
    # 出力ファイル名
    original_video_name = os.path.splitext(os.path.basename(read_video_path))[0]
    save_video_path = os.path.join(visualize_dir, f"visualize_{original_video_name}_factory_top{top_n_labels}.mp4")
    
    # Video processing
    read_video_cap = cv2.VideoCapture(read_video_path)
    if not read_video_cap.isOpened():
        raise ValueError(f"{read_video_path} is not opened")
    
    total_frames = int(read_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = read_video_cap.get(cv2.CAP_PROP_FPS)
    
    video_writer = None
    os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
    
    # analysis_fpsを取得（10fps）
    analysis_fps = factory_data["analysis_config"]["analysis_fps"]
    video_analysis_skipper = int(round(fps / analysis_fps))
    
    # Process each frame
    for video_frame in tqdm(range(total_frames)):
        grabbed, image = read_video_cap.read()
        if not grabbed:
            break
        
        # video_frameからanalysis_frameを計算
        analysis_frame = video_frame // video_analysis_skipper
        
        # フレーム情報を右上に表示
        cv2.putText(image, f"v:{video_frame} a:{analysis_frame}", 
                   (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 予測結果を描画
        if analysis_frame in frame2detections:
            detections = frame2detections[analysis_frame]
            
            # 背景を半透明で描画
            if detections:
                overlay = image.copy()
                box_height = len(detections) * 25 + 20
                cv2.rectangle(overlay, (5, 35), (450, 35 + box_height), (0, 0, 0), -1)
                image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
            
            # 各検出を描画
            for i, detection in enumerate(detections):
                text = f"{detection['action']} ({detection['confidence']:.3f})"
                color = get_confidence_color(detection['confidence'])
                
                # 日本語テキストを描画
                try:
                    image = cv2_put_japanese_text(
                        cv2_image=image,
                        text=text,
                        x=10,
                        y=40 + i * 25,
                        font_path=FONT_PATH,
                        font_size=20,
                        color=color
                    )
                except Exception as e:
                    # エラーの場合は英語で表示
                    cv2.putText(image, f"Class_{detection['class_id']} ({detection['confidence']:.3f})", 
                               (10, 40 + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if video_writer is None:
            video_shape = image.shape[:2][::-1]
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            video_writer = cv2.VideoWriter(save_video_path, fourcc, fps, video_shape)
        
        video_writer.write(image)
    
    if video_writer is not None:
        video_writer.release()
    
    print(f"\nVideo saved to: {save_video_path}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory_analysis_path", type=str, required=True, help="factory_analysis.jsonのパス")
    parser.add_argument("--read_video_path", type=str, required=True, help="元の動画ファイル")
    parser.add_argument("--visualize_dir", type=str, required=True, help="出力ディレクトリ")
    parser.add_argument("--conf_threshold", type=float, default=0.01, help="信頼度閾値")
    parser.add_argument("--top_n_labels", type=int, default=5, help="表示する最大ラベル数")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    visualize_factory_analysis(
        factory_analysis_path=args.factory_analysis_path,
        read_video_path=args.read_video_path,
        visualize_dir=args.visualize_dir,
        conf_threshold=args.conf_threshold,
        top_n_labels=args.top_n_labels
    )


