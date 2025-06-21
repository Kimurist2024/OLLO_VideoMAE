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
    """
    OpenCV の画像(cv2_image)に日本語文字列(text)を描画する。

    Parameters
    ----------
    cv2_image : np.ndarray
        OpenCV 形式の画像 (BGR)
    text : str
        描画したい文字列(日本語含む)
    x : int
        描画先の左上 X 座標
    y : int
        描画先の左上 Y 座標
    font_path : str
        TrueType フォント(.ttfなど)のファイルパス
    font_size : int, default=32
        フォントサイズ
    color : tuple, default=(0, 0, 0)
        文字色 (B, G, R) のタプル

    Returns
    -------
    np.ndarray
        文字を描画したあとの OpenCV 画像 (BGR)
    """
    # Pillow は RGB, OpenCV は BGR なので色を変換
    b, g, r = color
    pil_color = (r, g, b)

    # OpenCV 画像 -> Pillow 画像 (RGB)
    pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

    # Pillow で文字を描画
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(font_path, font_size)
    draw.text((x, y), text, font=font, fill=pil_color)

    # Pillow 画像 -> OpenCV 画像 (BGR)
    cv2_image_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return cv2_image_with_text


def visualize_extracted_video_output(model_output_path, video_meta_path, annot_path, read_video_path, visualize_dir, conf_threshold=0.0001):
    # extract_feat_production の出力を可視化する
    # ディレクトリ以下の構成
    # output_dir/centric_feature_list/pid_feature_centric_v*-*.pkl
    # annot_path = モデル学習用のjson path


    # model の output が (frame_list, num_class) で、各クラスは1クラス分類のbceで学習している
    # デフォルトでは conf >= 0.5 のクラスを可視化する -> コマンドラインから指定可能
    if len(annot_path):
        annotations = load_json(annot_path)
        categories = annotations["categories"]

        index2name = {}
        for name, index in categories.items():
            if name == "empty":
                continue
            index2name[index] = name

    else:
        index2name = None

    # 元の動画ファイル名を取得して、結果の動画ファイル名に含める
    original_video_name = os.path.splitext(os.path.basename(read_video_path))[0]
    save_video_path = os.path.join(visualize_dir, f"visualize_{original_video_name}_conf{conf_threshold:.1f}.mp4")

    video_meta = load_json(path=video_meta_path)
    # video_frame_list = video_meta["video_frame_list"]
    # analysis_frame_list = video_meta["analysis_frame_list"]

    # video_analysis_converter = frame_num_converter.Video2AnalysisConverter(
    #     video_frame_list=video_meta["video_frame_list"],
    #     analysis_frame_list=video_meta["analysis_frame_list"]
    # )

    feature_info = load_pickle(model_output_path)
    """
    {
        "pid2pick_person_info_list": [
            pid: {
                features: np.ndarray(length, dim,),
                frame_list: list[int] (length,)
            }
        ]
    }
    """

    pid2pick_person_info_list = feature_info["pid2pick_person_info_list"]

    features = pid2pick_person_info_list[-100]["features"]
    frame_list = pid2pick_person_info_list[-100]["frame_list"]

    frame2pred_name_list = {}
    for frame, pred_prob in zip(frame_list, features):
        pred_indices = np.where(pred_prob >= conf_threshold)[0]

        # 名前と確率をペアで保存するように変更
        pred_info_list = []
        for index in pred_indices:
            pred_info_list.append({
                "name": index2name[index],
                "conf": pred_prob[index]
            })
    
        frame2pred_name_list[frame] = pred_info_list

        # pred_name_list = [index2name[index] for index in pred_indices]
        # frame2pred_name_list[frame] = pred_name_list

    fps = video_meta["analysis_fps"]

    read_video_cap = cv2.VideoCapture(read_video_path)
    if not read_video_cap.isOpened():
        raise ValueError(f"{read_video_path} is not opened")
    
    read_video_frame_count = int(read_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # read_video_cap の a index 目が対応するanalyais_frame を算出する

    analysis_frame_list_on_video = []
    for video_frame in range(read_video_frame_count):
        _, analysis_frame_index = bisect_nearest.pick_nearest(
            frame_list_sorted=video_meta["video_frame_list"],
            pick_frame=video_frame,
            get_index=True
        )
        analysis_frame = video_meta["analysis_frame_list"][analysis_frame_index]
        analysis_frame_list_on_video.append(analysis_frame)
    


    read_video_cap = cv2.VideoCapture(read_video_path)
    if not read_video_cap.isOpened():
        raise ValueError(f"Failed to open video: {read_video_path}")
    


    video_writer = None
    fps = read_video_cap.get(cv2.CAP_PROP_FPS)
    fps = video_meta["analysis_fps"]


    os.makedirs(os.path.dirname(save_video_path), exist_ok=True)

    now_read_video_frame = -1

    for video_frame, analysis_frame in tqdm(zip(video_meta["video_frame_list"], video_meta["analysis_frame_list"]), total=len(video_meta["video_frame_list"])):

        while now_read_video_frame < video_frame:
            grabbed, image = read_video_cap.read()
            if not grabbed:
                break
            now_read_video_frame += 1

        if now_read_video_frame > video_frame:
            continue

        if not grabbed:
            break

        if analysis_frame in frame2pred_name_list:
            pred_info_list = frame2pred_name_list[analysis_frame]

            for i, pred_info in enumerate(pred_info_list):
                # クラス名と信頼度を組み合わせたテキストを作成
                text = f"{pred_info['name']} ({pred_info['conf']:.2f})"
                image = cv2_put_japanese_text(
                    cv2_image=image,
                    text=text,
                    x=10,
                    y=30 + i * 25,
                    font_path=FONT_PATH,
                    font_size=20,
                    color=(0, 255, 0) # BGR
                )

        else:
            continue

        cv2.putText(image, "v:{} a:{}".format(video_frame, analysis_frame), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if video_writer is None:
            video_shape = image.shape[:2][::-1]
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            video_writer = cv2.VideoWriter(save_video_path, fourcc,
                                          fps, video_shape)
            
        video_writer.write(image)

        # if video_frame >= 1000:
        #     break

    if video_writer is not None:
        video_writer.release()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output_path", type=str, required=True, help="extract_feature.py の出力ディレクトリ")
    parser.add_argument("--video_meta_path", type=str, required=True, help="extract_feature.py の出力ディレクトリにある video_meta.json ファイル")
    parser.add_argument("--annot_path", type=str, required=True, help="学習に使ったannotatoin file")
    parser.add_argument("--read_video_path", type=str, required=True, help="可視化したい動画ファイル")
    parser.add_argument("--visualize_dir", type=str, required=True, help="可視化した結果を保存するディレクトリ")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="予測確率の閾値（0.0〜1.0）")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # example
    # poetry run python3 visualize_model_output.py --model_output_path ./data/pose_data/test_short/centric_feature_list/feature_v-0.033-19.267.pickle --video_meta_path ./data/pose_data/test_short/video_meta_list/video_meta_v-0.033-19.267.json --annot_path /f195f550-4024-45d9-9c41-024e57ddf837/OlloFactoryAnalysis/data/annotations/20250523_annotations_split_0.json --read_video_path /f195f550-4024-45d9-9c41-024e57ddf837/OlloFactoryAnalysis/data/videos/test_short/video.mp4 --visualize_dir check/vis
    visualize_extracted_video_output(
        model_output_path=args.model_output_path,
        video_meta_path=args.video_meta_path,
        annot_path=args.annot_path,
        read_video_path=args.read_video_path,
        visualize_dir=args.visualize_dir,
        conf_threshold=args.conf_threshold
    )