from time import perf_counter as t

import cv2
import numpy as np



class Preprocessor:
    """動画や画像データの前処理を行うクラス。

    人物検出モード(MODEL_MODE_PERSON)とシーン中心モード(MODEL_MODE_CENTRIC)の2つの処理モードをサポートします。
    入力データをスライディングウィンドウで処理し、モデル入力用のテンソルを生成します。

    Attributes:
        window_size (int): スライディングウィンドウのサイズ
        slide_size (int): スライドのステップサイズ
        model_name2data_transform (dict): モデル名とその前処理変換のマッピング
        model_name2model_type (dict): モデル名とモデルタイプのマッピング
        is_person_crop (bool): 人物検出モードかどうか
        analysis_img_wh (tuple): 分析用画像サイズ(幅, 高さ)
        resize_img_wh (tuple): リサイズ後の画像サイズ(幅, 高さ)
        is_keep_aspratio (bool): アスペクト比を維持するかどうか
        centercrop (bool): 中心切り抜きを行うかどうか
        save_freq (int): 保存頻度(秒)
        video_fps (int): 動画のFPS

    Args:
        logger: ロガーオブジェクト
        model_name2data_transform (dict): モデル名と前処理変換のマッピング
        analysis_img_wh (tuple): 分析用画像サイズ
        model_name2model_type (dict, optional): モデル名とモデルタイプのマッピング
        window_size (int, optional): スライディングウィンドウサイズ。デフォルト16
        skipper (int, optional): フレームスキップ数。デフォルト1
        slide_size (int, optional): スライドステップサイズ。デフォルト8
        is_keep_aspratio (bool, optional): アスペクト比維持フラグ。デフォルトFalse
        resize_img_wh (tuple, optional): リサイズサイズ。デフォルトFalse
        model_mode (str, optional): 処理モード。デフォルトMODEL_MODE_PERSON
        centercrop (bool, optional): 中心切り抜きフラグ。デフォルトFalse
        save_dir (str, optional): 保存ディレクトリ。デフォルト""
        save_freq (int, optional): 保存頻度(秒)。デフォルト15*60
        video_fps (int, optional): 動画FPS。デフォルト30
    """

    def __init__(
        self,
        logger,
        model_name2data_transform,
        analysis_img_wh,
        resize_img_wh: tuple,
        window_size: int = 16,
        skipper: int = 1,
        slide_size: int = 8,
        is_keep_aspratio: bool = False,
        centercrop: bool = False,
    ) -> None:
        self.logger = logger
        self.window_size = window_size
        self.slide_size = slide_size
        self.model_name2data_transform = model_name2data_transform
        self._model_name_list = list(model_name2data_transform.keys())

        self.centercrop = centercrop

        self.analysis_img_wh = analysis_img_wh
        self.resize_img_wh = resize_img_wh

        # frameに対応するorig_imgを保存しておく
        self.frame2orig_or_norm_img: dict = {}
        self.pid2info: dict = {}

        # window_size内で離れすぎていないか確認する
        self.ok_frame_duration = self.window_size * skipper * 2

        self.img_w, self.img_h = None, None

        self.is_keep_aspratio = is_keep_aspratio

        # 検出されたframeを保存しておくlist
        self.latest_frame_list: list = []
        self.latest_image_list: list = []
        self.latest_timestamp_list: list = []

        self.pid2latest_frame_list: dict = {}

        self.mean_array = np.array([0.485, 0.456, 0.406])
        self.std_array = np.array([0.229, 0.224, 0.225])
        self.std_multiple_array = 1.0 / self.std_array

        # 事前に固定サイズの配列を確保
        self.preallocated_buffer = np.zeros(
            (self.window_size, 3, 224, 224), dtype=np.float32
        )

    def last_update(
        self, inference_length: int = 16
    ):
        if self.img_w is None and self.img_h is None:
            inference_length = 1
        # output_list = []
        for fi in range(inference_length - 1):

            (
                model_name2x_tensor_list,
                batch_info_list,
                pop_pid_list,
                time_info,
            ) = self._update_image(
                orig_img=None,
                crop_area_bbox=None,
                frame=None,
                timestamp=None,
                is_last=True,
            )

            output = (
                model_name2x_tensor_list,
                batch_info_list,
                pop_pid_list,
                time_info,
                fi,
            )

            yield output

    def update(
        self,
        orig_img: np.ndarray,
        crop_area_bbox: list,
        frame: int,
        timestamp: int,
        is_last: bool = False,
    ):
        output = self._update_image(
            orig_img=orig_img,
            crop_area_bbox=crop_area_bbox,
            frame=frame,
            timestamp=timestamp,
            is_last=is_last,
        )

        return output

    def _update_image(
        self, orig_img, crop_area_bbox: list, frame: int, timestamp: int, is_last: bool
    ):
        time_info = {}
        if crop_area_bbox is None:
            pass
        else:
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_area_bbox
            orig_img = orig_img[crop_y1:crop_y2, crop_x1:crop_x2]

        if self.centercrop and orig_img is not None:
            orig_img_h, orig_img_w = orig_img.shape[:2]
            crop_x1 = (orig_img_w - orig_img_h) // 2
            crop_x2 = crop_x1 + orig_img_h

            orig_img = orig_img[:, crop_x1:crop_x2]

        if self.img_w is None and self.img_h is None:
            if is_last:
                raise NotImplementedError("error")
            self.img_h, self.img_w, _ = orig_img.shape

        if is_last:
            transformed_img = self.latest_image_list[-1]
            frame = self.latest_frame_list[-1]
            timestamp = self.latest_timestamp_list[-1]

        else:
            # print("orig_img.shape", orig_img.shape, orig_img.dtype, self.resize_img_wh)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

            orig_img = cv2.resize(orig_img, self.resize_img_wh)

            transform_st = t()

            transpose_st = t()
            orig_img = orig_img.transpose(2, 0, 1)
            transpose_time = t() - transpose_st
            if "transpose_time" not in time_info:
                time_info["transpose_time"] = 0
            time_info["transpose_time"] += transpose_time

            div_st = t()
            transformed_img = orig_img.astype(np.float32) / 255.0
            div_time = t() - div_st
            if "div_time" not in time_info:
                time_info["div_time"] = 0
            time_info["div_time"] += div_time

            # sub_image = transformed_img - self.mean_array
            sub_div_st = t()
            for i in range(transformed_img.shape[0]):
                transformed_img[i] -= self.mean_array[i]
                transformed_img[i] *= self.std_multiple_array[i]
            sub_div_time = t() - sub_div_st
            if "sub_div_time" not in time_info:
                time_info["sub_div_time"] = 0
            time_info["sub_div_time"] += sub_div_time

            # transformed_img = (transformed_img - self.mean_array) / self.std_array

            transform_time = t() - transform_st

            if "transform_time" not in time_info:
                time_info["transform_time"] = 0

            time_info["transform_time"] += transform_time

        self.latest_image_list.append(transformed_img)
        self.latest_frame_list.append(frame)
        self.latest_timestamp_list.append(timestamp)

        model_name2x_tensor_list = {}
        pid = -100
        batch_info_list = []

        start_preprocess = t()
        if len(self.latest_image_list) >= self.window_size:
            # preallocated buffer を用いる場合、iの値によっては最後の方に余分なものがあるので
            # length が self.preallocated_buffer の長さと同じになるようにする.

            insert_st = t()
            for i, img in enumerate(self.latest_image_list[-self.window_size :]):
                self.preallocated_buffer[i] = img
            insert_time = t() - insert_st
            if "insert_time" not in time_info:
                time_info["insert_time"] = 0
            time_info["insert_time"] += insert_time

            # 軸の入れ替えを1回だけ行う
            transpose_window_st = t()
            input_images = self.preallocated_buffer.copy().transpose(1, 0, 2, 3)
            transpose_window_time = t() - transpose_window_st
            if "transpose_window_time" not in time_info:
                time_info["transpose_window_time"] = 0
            time_info["transpose_window_time"] += transpose_window_time

            model_name2x_tensor_list["main"] = [input_images]

            insert_frame_list = self.latest_frame_list[-self.window_size :]
            insert_timestamp_list = self.latest_timestamp_list[-self.window_size :]

            batch_info_list.append(
                {
                    "pid": pid,
                    "frame_list": insert_frame_list,
                    "timestamp_list": insert_timestamp_list,
                }
            )

            # スライドサイズ分だけ破棄
            self.latest_image_list = self.latest_image_list[self.slide_size :]
            self.latest_frame_list = self.latest_frame_list[self.slide_size :]
            self.latest_timestamp_list = self.latest_timestamp_list[self.slide_size :]

            # メモリを逐一動的確保する
            # image_list = self.latest_image_list[-self.window_size :]
            # frame_list = self.latest_frame_list[-self.window_size :]
            # timestamp_list = self.latest_timestamp_list[-self.window_size :]

            # # input_images = torch.stack(image_list, dim=1)
            # # np.float32にしないとmodelのoutputがnanになる, stackするとnp.float64になる
            # input_images = np.stack(image_list, axis=1).astype(np.float32)

            # model_name2x_tensor_list["main"] = [input_images]

            # self.latest_image_list = self.latest_image_list[self.slide_size :]
            # self.latest_frame_list = self.latest_frame_list[self.slide_size :]
            # self.latest_timestamp_list = self.latest_timestamp_list[self.slide_size :]

            # batch_info_list.append(
            #     {"pid": pid, "frame_list": frame_list, "timestamp_list": timestamp_list}
            # )

        preprocess_time = t() - start_preprocess
        time_info["prepare_preprocess_time"] = preprocess_time

        return (model_name2x_tensor_list, batch_info_list, [], time_info)

