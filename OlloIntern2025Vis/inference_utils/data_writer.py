import json
import os
import pickle

import numpy as np

from inference_utils import bisect_nearest



class DataWriter:
    """データ保存を管理するクラス

    フレームベースまたはタイムスタンプベースでデータを保存する機能を提供します。

    Attributes:
        time_metric (str): 保存の時間メトリック（'frame' または 'timestamp'）
        save_freq_video_frame (int): フレームベースの保存間隔
        save_freq_second (float): タイムスタンプベースの保存間隔（秒）
    """

    def __init__(
        self,
        save_dir: str,
        analysis_fps: int,
        video_fps: int,
        save_freq_second: float = 900.0,
        logger=None,
        is_generate_video_meta_data=False,
    ) -> None:
        self.save_dir = save_dir
        self.analysis_fps = analysis_fps
        self.video_fps = video_fps
        self.save_freq_video_frame = int(video_fps * save_freq_second)
        self.save_freq_second = save_freq_second

        self.is_generate_video_meta_data = is_generate_video_meta_data

        self.pid2pick_person_info_list = {}

        self.save_start_frame: int | None = None
        self.save_end_frame: int | None = None

        self.logger = logger

        self.save_feature_path_template = os.path.join(
            self.save_dir,
            "centric_feature_list",
            "feature_v-{}-{}.pickle",
        )

        if self.logger is not None:
            self.logger.info(f"save path: {self.save_feature_path_template}")

        self.is_output_sequence = True

        if self.is_generate_video_meta_data:
            self.save_video_meta_data_path = os.path.join(
                self.save_dir,
                "video_meta_list",
                "video_meta_v-{}-{}.json",
            )
            os.makedirs(os.path.dirname(self.save_video_meta_data_path), exist_ok=True)

            self.pose_data_path = os.path.join(
                self.save_dir,
                "pose_list",
                "pose_v-{}-{}.pickle",
            )
            os.makedirs(os.path.dirname(self.pose_data_path), exist_ok=True)

            self.video_meta = {
                "video_frame_list": [],
                "analysis_frame_list": [],
                "timestamp_list": [],
                "video_fps": video_fps,
                "analysis_fps": analysis_fps,
                "image_width": -1,
                "image_height": -1,
            }

            self.pose_data = {}
        # print("data writer:", self.save_freq_video_frame)

    def update(
        self,
        output_result_list: list,
        now_analysis_frame: int,
        now_video_frame: int,
        image=None,
        now_timestamp: float | None = None,
    ):
        """
        # output_result_list: [
        #     {
        #         "pid": int,
        #         "features": np.ndarray(dim,)
        #         "frame_list": int
        #     }
        # ]
        出力結果を保持するもの

        Args:
            output_result_list (list): 以下の形式の出力結果のリスト
                {
                    "pid": int, # 人物ID
                    "features": np.ndarray(dim,), # 特徴量
                    "frame_list": int, # フレーム番号
                    "last_features": np.ndarray(dim,), # 前回の特徴量
                    "timestamp_list": list[float], # タイムスタンプのリスト
                    "other_features": dict, # その他の特徴量(オプション)
                }
            now_analysis_frame (int): 現在の分析フレーム番号
            now_video_frame (int): 現在のビデオフレーム番号
            image (np.ndarray, optional): 入力画像
            now_timestamp (float | None, optional): 現在のタイムスタンプ

        Raises:
            ValueError: タイムスタンプベースのメトリクスを使用時にタイムスタンプが指定されていない場合
            ValueError: タイムスタンプが負の値の場合
        """

        if self.save_start_frame is None:
            self.save_start_frame = now_video_frame
            self.save_start_timestamp = now_timestamp

        if image is None:
            image_w = self.image_width
            image_h = self.image_height

        else:
            self.image_width = image.shape[1]
            self.image_height = image.shape[0]

            image_w = image.shape[1]
            image_h = image.shape[0]

        if self.is_generate_video_meta_data:
            # print("now timestamp", now_timestamp)
            self.video_meta["video_frame_list"].append(now_video_frame)
            self.video_meta["analysis_frame_list"].append(now_analysis_frame)
            self.video_meta["timestamp_list"].append(now_timestamp)
            self.video_meta["image_width"] = image_w
            self.video_meta["image_height"] = image_h

        for output_result in output_result_list:
            pid = output_result["pid"]
            feature = output_result["features"]
            frame = output_result["frame_list"]

            if "last_features" in output_result:
                last_features = output_result["last_features"]
            else:
                last_features = None

            timestamp_list = output_result["timestamp_list"]

            key_value_info = {
                "frame_list": frame,
                "features": feature,
                "last_features": last_features,
                "timestamp_list": timestamp_list,
            }

            if "other_features" in output_result:
                key_value_info["other_features"] = output_result["other_features"]

            pick_person_list_info = self.pid2pick_person_info_list.get(pid, {})

            for key, value in key_value_info.items():
                if key == "other_features":
                    if key not in pick_person_list_info:
                        pick_person_list_info[key] = {}
                    for key2, value2 in value.items():
                        if key2 not in pick_person_list_info[key]:
                            pick_person_list_info[key][key2] = []

                        pick_person_list_info[key][key2].append(value2)

                else:
                    if key not in pick_person_list_info:
                        pick_person_list_info[key] = []
                    pick_person_list_info[key].append(value)

            self.pid2pick_person_info_list[pid] = pick_person_list_info

        is_save = (
            now_video_frame - self.save_start_frame >= self.save_freq_video_frame
        )


        if is_save:
            self.process_one_save_batch(
                now_video_frame=now_video_frame, now_timestamp=now_timestamp
            )

        return is_save

    def has_data(self):
        return len(self.pid2pick_person_info_list) > 0

    def _get_save_time_info(self, now_video_frame: int, now_timestamp: float) -> tuple:
        """保存時の時間情報を取得する

        Args:
            now_video_frame (int): 現在のビデオフレーム
            now_timestamp (float): 現在のタイムスタンプ

        Returns:
            tuple: (開始時間, 終了時間, 開始時間文字列, 終了時間文字列)
        """

        start_time = self.save_start_timestamp
        end_time = now_timestamp
        start_timestr = f"{start_time:.3f}"
        end_timestr = f"{end_time:.3f}"

        return start_time, end_time, start_timestr, end_timestr

    def _format_save_path(
        self, formatted_string: str, start_timestr: str, end_timestr: str
    ) -> str:
        """保存パスを生成する

        Args:
            start_timestr (str): 開始時間文字列
            end_timestr (str): 終了時間文字列

        Returns:
            str: フォーマットされた保存パス
        """
        return formatted_string.format(start_timestr, end_timestr)

    def process_one_save_batch(
        self, now_video_frame: int, now_timestamp: float, version: str = "v1"
    ) -> None:
        """1バッチの保存処理を実行する"""

        # self.save_end_frame = now_video_frame

        # self.save_start_frame = min(
        #     self.save_start_frame, self.trace_frame_list[0]
        # )

        save_start_frame = None
        save_end_frame = None
        for pid, pick_person_info_list in self.pid2pick_person_info_list.items():
            frame_list = pick_person_info_list["frame_list"]
            timestamp_list = pick_person_info_list["timestamp_list"]
            if self.is_output_sequence:
                # features_batch = list[feature (window size, dim)]
                # frame_list = list[frame (window_size)]
                # min_start_frame = np.min(frame_list)
                # max_start_frame = np.max(frame_list)
                features_batch = pick_person_info_list["features"]
                # unique_frame_list = [ for frames in frame_list for frame in frames]
                unique_frame_list = set()
                for frames in frame_list:
                    # print(frames)
                    unique_frame_list.update(frames)

                unique_frame_list = sorted(unique_frame_list)
                # print("save features_batch shape =", features_batch[0].shape)
                unique_features = np.zeros(
                    (len(unique_frame_list), features_batch[0].shape[1])
                )

                if "other_features" in pick_person_info_list:
                    other_feature_info = pick_person_info_list["other_features"]
                    unique_other_feature_info = {}
                    for key, value in other_feature_info.items():
                        new_shape = (len(unique_frame_list),) + value[0].shape[1:]
                        unique_other_feature_info[key] = np.zeros(new_shape)

                else:
                    other_feature_info = None
                    unique_other_feature_info = {}

                num_add_array = np.zeros((len(unique_frame_list), 1))

                # for features, frames in zip(features_batch, frame_list):
                for chunk_index, frames in enumerate(frame_list):
                    features = features_batch[chunk_index]

                    span_start_frame = frames[0]
                    span_end_frame = frames[-1]

                    span_start_index, span_end_index = bisect_nearest.pick_nearest_list(
                        frame_list_sorted=unique_frame_list,
                        pick_frame_list=[span_start_frame, span_end_frame],
                        get_index=True,
                    )
                    length = span_end_index - span_start_index + 1
                    # print("span start frame = {} ~ {}".format(span_start_frame, span_end_frame))
                    # print(unique_frame_list[span_start_index], unique_frame_list[span_end_index])
                    unique_features[span_start_index : span_end_index + 1] += features[
                        :length
                    ]
                    num_add_array[span_start_index : span_end_index + 1] += 1

                    if other_feature_info is not None:
                        for (
                            other_key,
                            other_feature_array_list,
                        ) in other_feature_info.items():
                            other_feature_array = other_feature_array_list[chunk_index]

                            unique_other_feature_info[other_key][
                                span_start_index : span_end_index + 1
                            ] += other_feature_array[:length]

                assert np.all(num_add_array > 0), np.sum(num_add_array == 0)
                unique_features = unique_features / num_add_array

                if other_feature_info is not None:
                    for (
                        other_key,
                        other_feature_array,
                    ) in unique_other_feature_info.items():
                        num_add_array = np.maximum(num_add_array, 1)
                        unique_other_feature_info[other_key] = (
                            unique_other_feature_info[other_key] / num_add_array
                        )

                start_frame = unique_frame_list[0]
                end_frame = unique_frame_list[-1]

                timestamp_list = np.unique(timestamp_list).tolist()

                self.pid2pick_person_info_list[pid]["timestamp_list"] = timestamp_list
                self.pid2pick_person_info_list[pid]["frame_list"] = unique_frame_list
                self.pid2pick_person_info_list[pid]["features"] = unique_features

            else:
                start_frame = frame_list[0]
                end_frame = frame_list[-1]

            if save_start_frame is None:
                save_start_frame = start_frame
                save_end_frame = end_frame

            else:
                save_start_frame = min(save_start_frame, start_frame)
                save_end_frame = max(save_end_frame, end_frame)

        # if save_start_frame is None:
        #     save_start_frame = self.save_start_frame
        #     save_end_frame = now_video_frame

        # else:
        #     (
        #         save_start_frame,
        #         save_end_frame,
        #     ) = self.video_analysis_converter.analysis2video_batch(
        #         analysis_frame_container=[save_start_frame, save_end_frame]
        #     )

        _, _, start_timestr, end_timestr = self._get_save_time_info(
            now_video_frame, now_timestamp
        )
        save_feature_path = self._format_save_path(
            self.save_feature_path_template, start_timestr, end_timestr
        )

        self.meta_data = {
            "fps": self.analysis_fps,
        }

        self._save_data(
            save_feature_path=save_feature_path,
            pid2pick_person_info_list=self.pid2pick_person_info_list,
        )

        if self.is_generate_video_meta_data:
            _, _, start_timestr, end_timestr = self._get_save_time_info(
                now_video_frame, now_timestamp
            )
            save_video_meta_data_path = self._format_save_path(
                self.save_video_meta_data_path, start_timestr, end_timestr
            )
            self._save_video_meta_data(
                save_video_meta_data_path=save_video_meta_data_path
            )

            pose_data_path = self._format_save_path(
                self.pose_data_path, start_timestr, end_timestr
            )
            self._save_pose_data(save_pose_data_path=pose_data_path)

            self.video_meta = {
                "video_frame_list": [],
                "analysis_frame_list": [],
                "timestamp_list": [],
                "video_fps": self.video_fps,
                "analysis_fps": self.analysis_fps,
                "image_width": -1,
                "image_height": -1,
            }

            self.pose_data = {}

        self.save_start_frame = None
        self.save_end_frame = None

        self.pid2pick_person_info_list = {}

    def _save_data(
        self, save_feature_path: str, pid2pick_person_info_list: dict = None
    ):
        if pid2pick_person_info_list is None:
            pid2pick_person_info_list = self.pid2pick_person_info_list

        for pid, person_list_info in pid2pick_person_info_list.items():
            for key, value_list in person_list_info.items():
                if isinstance(value_list, list):
                    person_list_info[key] = np.array(value_list)

        print("save to", save_feature_path)
        # print(pid2pick_person_info_list[-100]["last_features"].shape)
        # print(
        #     pid2pick_person_info_list[-100]["frame_list"][:3],
        #     pid2pick_person_info_list[-100]["frame_list"][-3:],
        # )

        pid2pick_person_info_list = {
            "pid2pick_person_info_list": pid2pick_person_info_list,
            "meta_data": self.meta_data,
        }

        os.makedirs(os.path.dirname(save_feature_path), exist_ok=True)

        with open(save_feature_path, "wb") as f:
            pickle.dump(pid2pick_person_info_list, f)

        print(f"save to {save_feature_path}")

    def _save_video_meta_data(self, save_video_meta_data_path: str):
        with open(save_video_meta_data_path, "w") as f:
            json.dump(self.video_meta, f)

    def _save_pose_data(self, save_pose_data_path: str):
        with open(save_pose_data_path, "wb") as f:
            pickle.dump(self.pose_data, f)
