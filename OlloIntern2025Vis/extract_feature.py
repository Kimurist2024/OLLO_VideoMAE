# ruff: noqa: E402

import collections
import sys
import time
import argparse

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torchvision import transforms as torch_transforms

from models import test_net

from inference_utils import input_maker, detector, data_writer

# 追加
import decord
from decord import VideoReader

import sys
sys.path.append('/home/ollo/VideoMAE')

import modeling_finetune
import video_transforms as video_transforms
import volume_transforms as volume_transforms

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_path", type=str, required=True, help="可視化したい動画ファイル")
    parser.add_argument("--output_dir", type=str, required=True, help="model の予測結果を保存するディレクトリ")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="読み込むモデルチェックポイントのパス")
    parser.add_argument("--analysis_fps", type=int, required=True, help="10")
    parser.add_argument("--inference_batch_size", type=int, required=True, help="推論のバッチサイズ")
    parser.add_argument("--window_size", type=int, required=True, help="model 学習時の model の windowsize")
    parser.add_argument("--input_size", type=int, required=True, help="model 学習時の model の input size")
    parser.add_argument("--num_classes", type=int, required=True, help="model 学習時の model の num_classes")
    parser.add_argument("--tubelet_size", type=int, required=True, help="model 学習時の model の tubelet_size")

    return parser.parse_args()


def main(args, logger=None):
    start = time.time()
    if logger is None:
        print_ = print
    else:
        print_ = logger.info


    msg = "####### EXTRACT FEATURE #######"
    print_("#" * len(msg))
    print_(msg)
    print_("#" * len(msg))

    #################################
    #################################
    start_frame = 0
    end_frame = 10000000

    ####### decord で学習した場合はdecord に書き換える ##########
    # GPUを使用したい場合は以下のコメントを解除
    decord.bridge.set_bridge('torch')
    read_cap = VideoReader(args.video_path)
    ########################################################

    # cv2の場合
    # fps = int(round(read_cap.get(cv2.CAP_PROP_FPS)))
    # total_frame_num = int(read_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # decordの場合
    fps = int(round(read_cap.get_avg_fps()))
    total_frame_num = len(read_cap)

    end_frame = min(end_frame, total_frame_num)

    if end_frame == -1:
        end_frame = total_frame_num

    # os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICES

    cudnn.benchmark = True



    # 短すぎる動画に対してskip_lengthが大きいと推論できないため短い場合は1にする
    # 1 ~ 100 frame の範囲の動画を推論するときに、window_size = 16 として
    # [1 ~ 16], [1 + slide_size ~ 16 + slide_size], [1 + 2 * slide_size ~ 16 + 2 * slide_size], ...
    # というように、slide_size ずつずらして推論する
    # [1 ~ 16] の画像16枚を 1 ~ 16 frame の予測値として採用するので
    # model の output (1, num_classes) を (window_size, num_classes) に拡張する必要がある
    # skip length は推論を analysis_fps に対して何フレームに1回分のデータにするのかを指定する
    # analysis_fps = 10 で skip_lenght = 2 とすると
    # [0.1, 0.3, 0.5, ..., 3.1] 秒での画像で input を作成して推論する

    if total_frame_num > 1000:
        skip_length = 1
        slide_size = 2
    else:
        skip_length = 1
        slide_size = 1

    # transform = torch_transforms.Compose(
    #     [
    #         torch_transforms.ToPILImage(),
    #         # torch_transforms.Grayscale(num_output_channels=3),
    #         torch_transforms.ToTensor(),
    #         torch_transforms.Normalize(
    #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #         ),
    #     ]
    # )

    # custom.pyからvalidation時のtransformを実装
    transform = video_transforms.Compose([
                video_transforms.Resize(256, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(224, 224)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])

    model_name2data_transform = {
        "main": transform,
    }

    input_preprocessor = input_maker.Preprocessor(
        logger=logger,
        model_name2data_transform=model_name2data_transform,
        analysis_img_wh=(args.input_size, args.input_size),
        window_size=args.window_size,
        skipper=skip_length,
        slide_size=slide_size,
        is_keep_aspratio=False,
        resize_img_wh=(args.input_size, args.input_size),
        centercrop=False,
    )

    ###### ここを書き換える ######
    # モデルの初期化
    model = modeling_finetune.vit_base_patch16_224(
        all_frames=args.window_size,
        img_size=args.input_size,
        num_classes=args.num_classes,
        tubelet_size=args.tubelet_size,
        use_checkpoint=True
    )
    
    # チェックポイントの読み込み
    checkpoint_path = args.checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]
    
    # head層も含めてすべての重みを読み込む
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if logger:
        logger.info(f"Model weights loaded from {checkpoint_path}")
        logger.info(f"Missing keys: {missing_keys}")
        logger.info(f"Unexpected keys: {unexpected_keys}")
    else:
        print(f"Model weights loaded from {checkpoint_path}")
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    #################################
    #################################

    model_detector = detector.Detector(
        logger=logger,
        model_name2model={
            "main": model,
        },
        model_name2device_id={
            "main": 0,
        },
        inference_batch_size=args.inference_batch_size,
        use_trt_model=False,
        input_data=None,
    )

    output_dir = args.output_dir
    analysis_fps = args.analysis_fps

    video_fps = fps

    data_write_monitor = data_writer.DataWriter(
        save_dir=output_dir,
        analysis_fps=analysis_fps,
        video_fps=video_fps,
        save_freq_second=900,
        logger=logger,
        is_generate_video_meta_data=True
    )



    analysis_frame_set = set()


    print_(f"start frame = {start_frame}, end frame = {end_frame}")

    time_info_history = collections.defaultdict(list)

    progress_bar = tqdm(total=end_frame - start_frame + 1)
    prev_video_index = 0
    next_send_progress_percentage = 0
    progress_freq_percentage = 10
    iter_st = time.time()

    video_analysis_skipper = int(round(video_fps / analysis_fps))

    first_analysis_frame = 0
    now_timestamp = 0.0


    for fi in range(start_frame, end_frame + 1):
        # cv2の場合
        # grabbed, img = read_cap.read()
        # if not grabbed:
        #     break

        # decordの場合
        try:
            img = read_cap[fi].numpy()  # NumPy配列に変換
        except (IndexError, decord.DECORDError):
            break

        video_frame = fi

        diff_frame = fi - prev_video_index
        progress_bar.update(diff_frame)

        prev_video_index = fi

        now_timestamp += 1 / fps
        is_infer = fi == end_frame

        analysis_frame = video_frame // video_analysis_skipper

        if not is_infer:
            if (analysis_frame - first_analysis_frame) % skip_length != 0:
                continue

        if analysis_frame in analysis_frame_set:
            continue

        analysis_frame_set.add(analysis_frame)


        preprocess_st = time.time()
        (
            model_name2x_tensor_list,
            batch_info_list,
            pop_pid_list,
            time_info,
        ) = input_preprocessor.update(
            orig_img=img,
            crop_area_bbox=None,
            frame=analysis_frame,
            timestamp=now_timestamp,
            is_last=is_infer,
        )
        preprocess_time = time.time() - preprocess_st
        time_info["preprocess_time"] = preprocess_time

        # logger.info(time_info)
        # if len(batch_info_list):
        # print(batch_info_list[0]["frame_list"])
        #     x_tensor = model_name2x_tensor_list["main"][0]
        #     print(x_tensor[0, :3, :3, :3])

        # for batch_info in batch_info_list:
        # frame_list = batch_info["frame_list"]
        # print(batch_info["pid"], frame_list[0], frame_list[-1], len(frame_list))

        # if len(batch_info_list):
        #     print(time_info)
        # print(len(model_name2x_tensor_list["main"]))

        model_forward_st = time.time()
        output_result = model_detector.update(
            detected_model_name2x_tensor_list=model_name2x_tensor_list,
            detected_batch_info_list=batch_info_list,
            is_last=False,
            now_video_frame=fi,
            now_timestamp=now_timestamp,
        )
        # for output_info in output_result:
        #     print(output_info["frame_list"])
        #     print(output_info["features"][:3])
        model_forward_time = time.time() - model_forward_st
        time_info["model_forward_time"] = model_forward_time

        write_st = time.time()
        data_write_monitor.update(
            output_result_list=output_result,
            now_analysis_frame=analysis_frame,
            now_video_frame=fi,
            now_timestamp=now_timestamp,
            image=img,
        )
        write_duration = time.time() - write_st
        time_info["write_duration"] = write_duration

        duration = time.time() - iter_st
        time_info["iteration_duration"] = duration
        iter_st = time.time()

        # if len(model_name2x_tensor_list):
        #     print("b=",  video_model_detector.inference_batch_size, time_info)

        for key, value in time_info.items():
            time_info_history[key].append(value)

        # if fi >= 1000:
        #     break


    time_info_summary = {}
    for key, value in time_info_history.items():
        mean = np.mean(value)
        sum = np.sum(value)

        time_info_summary[key] = {"mean": mean, "sum": sum}

    print_(f"time_info_summary: {time_info_summary}")

    # import pprint
    # pprint.pprint(time_info_summary)

    # 動画の最後の数フレームがframe_numのせいで推論されない。
    # その分を最後の画像をコピーすることで対応
    # 例えば 0 ~ 255 までのフレームがあって、window_size=16だと
    # 最後のバッチが 240 ~ 255になり、240フレーム目までしか存在しなくなる。
    # 241 ~ 255の分を作成する

    last_update_generator = input_preprocessor.last_update(
        inference_length=args.window_size
    )
    for last_output in last_update_generator:
        (
            model_name2x_tensor_list,
            batch_info_list,
            pop_pid_list,
            time_info,
            fi,
        ) = last_output

        video_frame = end_frame + fi * skip_length + 1

        # print("video frame=", video_frame)
        # print("batch_info_list:")
        # for batch_info in batch_info_list:
        #     print(batch_info)

        if len(model_name2x_tensor_list):
            output_result = model_detector.update(
                detected_model_name2x_tensor_list=model_name2x_tensor_list,
                detected_batch_info_list=batch_info_list,
                now_video_frame=video_frame,
                is_last=False,
            )

        else:
            output_result = []

        data_write_monitor.update(
            output_result_list=output_result,
            now_analysis_frame=analysis_frame,
            now_video_frame=video_frame,
            image=None,
        )

        now_timestamp = video_frame / fps

    output_result = model_detector.update(
        model_name="main",
        is_last=True,
        detected_model_name2x_tensor_list={},
        detected_batch_info_list=[],
        now_video_frame=video_frame,
    )

    data_write_monitor.update(
        output_result_list=output_result,
        now_analysis_frame=analysis_frame,
        now_video_frame=video_frame,
        image=None,
    )
    data_write_monitor.process_one_save_batch(
        now_video_frame=video_frame, now_timestamp=now_timestamp
    )

    total_time = time.time() - start
    print_(f"total_time: {total_time:.2f} sec")


if __name__ == "__main__":
    # opts = get_args()
    args = get_args()
    """
    example
    poetry run python3 extract_feature.py --video_path /f195f550-4024-45d9-9c41-024e57ddf837/OlloFactoryAnalysis/data/videos/test_short/video.mp4 --output_dir ./data/pose_data/test_short --analysis_fps 10 --inference_batch_size 8 --window_size 16 --input_size 224 --num_classes 58 --tubelet_size 2
    """

    logger = None
    # try:
        # main_async(args, logger=logger)
    main(args, logger=logger)
    # except Exception as e:
    #     logger.error(e, exc_info=True)
    #     sys.exit(1)
