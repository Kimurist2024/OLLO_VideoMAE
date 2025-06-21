import os
import pathlib
import pickle

import cv2
import numpy as np
import torch



current_dir = pathlib.Path(os.path.abspath(__file__)).parent


def visualize_debug(x_tensor, batch_info_list, model_name="main", num_write=0):
    x_tensor_numpy = x_tensor.numpy()  # B, C, T, H, W

    mean = np.array([0.485, 0.456, 0.406])[None, :, None, None, None]
    std = np.array([0.229, 0.224, 0.225])[None, :, None, None, None]

    x_tensor_numpy = x_tensor_numpy * std + mean
    x_tensor_numpy = (x_tensor_numpy * 255).astype(np.uint8)
    x_tensor_numpy = np.transpose(x_tensor_numpy, (0, 2, 3, 4, 1))  # B, T, H, W, C
    # x_tensor_numpy = np.concatenate()
    image_batch = []
    for index, seq_image in enumerate(x_tensor_numpy):
        batch_info = batch_info_list[index]
        print(batch_info["pid"], batch_info["frame_list"])
        seq_image = np.concatenate(seq_image, axis=1)
        seq_image = seq_image[:, :, ::-1]
        seq_image = np.ascontiguousarray(seq_image)
        x = 20
        y = 30
        id_msg = "pid: {}, frame: {}".format(
            batch_info["pid"], batch_info["frame_list"][0]
        )
        cv2.putText(
            seq_image,
            id_msg,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (127, 255, 0),
            thickness=2,
        )
        image_batch.append(seq_image)
    image_batch = np.concatenate(image_batch, axis=0)
    cv2.imwrite(f"./check/batch_{model_name}_{num_write}.jpg", image_batch)


class Detector:
    def __init__(
        self,
        logger,
        model_name2model: dict,
        model_name2device_id: dict,
        inference_batch_size=8,
        use_trt_model: bool = False,
        input_data: list = None,  # tensorrt model は model(*x) でしか受け付けないので input data で tensor　以外の引数を指定する
    ) -> None:
        """
        Args:
            logger: ログ出力用のロガー
            model_name2model (dict): モデル名をキーとしたモデルの辞書
            model_name2device_id (dict): モデル名をキーとしたデバイスIDの辞書
            inference_batch_size (int): 推論時のバッチサイズ。デフォルトは8
            use_trt_model (bool): TensorRTモデル使用フラグ。デフォルトはFalse
            input_data (list): TensorRTモデル用の入力データリスト。デフォルトはNone
        """
        if logger is None:
            self.print_ = print
        else:
            self.print_ = logger.info

        self.model_name2model = model_name2model

        self.model_name2device_id = model_name2device_id


        self.use_trt_model = use_trt_model
        # self.pretrained = pretrained
        # self.not_use_head = not_use_head
        # self.input_img_size = 448

        # gpu_list = os.environ["CUDA_VISIBLE_DEVICES"]
        # print("gpu list:", gpu_list)
        for model_name, device_id in model_name2device_id.items():
            # if device_id.index is None:
            #     device_index = 0
            # else:
            #     device_index = device_id.index
            # gpu_id = gpu_list.split(",")[device_index]
            self.print_(f"model {model_name} use device id={device_id}")

            self.model_name2model[model_name] = self.model_name2model[model_name].to(
                device_id
            )


        # window_size内で離れすぎていないか確認する
        # skip_per = self.fps / 10
        # self.ok_frame_duration = self.window_size * skip_per * 2

        self.img_w, self.img_h = None, None

        # nmsするときに使う前のtoruokuの情報
        # count系
        # 検出されたframeを保存しておくlist
        self.before_x_tensor = None
        self.before_heatmaps = None

        self.num_write = 0

        self.batch_input_list = []
        self.batch_info_list = []
        self.now_batch_size = 0
        self.inference_batch_size = inference_batch_size


        self.input_data = input_data

        # print("detector:", self.save_freq_video_frame)

    def batch_forward(
        self, model_name, input_tensor, batch_info_list, output_result
    ):
        device_id_here = self.model_name2device_id[model_name]
        # forward

        with torch.no_grad():
            input_tensor = input_tensor.to(device_id_here)
            model = self.model_name2model[model_name]

            output = model(input_tensor)
            output = output.sigmoid()

            output = output.detach().cpu().numpy()

            # print(output)

        for person_index, feature in enumerate(output):
            batch_info = batch_info_list[person_index]

            pid = batch_info["pid"]
            frame_list = batch_info["frame_list"]

            timestamp_list = batch_info["timestamp_list"]

            # featureに時間方向の次元を持たせる
            if len(feature.shape) == 1:
                feature = np.expand_dims(feature, axis=0)
                feature = np.repeat(feature, len(frame_list), axis=0)

            if len(feature) != len(frame_list):
                raise NotImplementedError(
                    f"feature len: {len(feature)} != frame_list len: {len(frame_list)}"
                )

            output_result.append(
                {
                    "pid": pid,
                    "features": feature,
                    "frame_list": frame_list,
                    "timestamp_list": timestamp_list,
                }
            )

    def update(
        self,
        detected_model_name2x_tensor_list: dict,
        detected_batch_info_list: list,
        is_last: bool = False,
        model_name: str = "main",
        now_timestamp=None,
        now_video_frame=None,
    ) -> list:
        # update
        # inference
        output_result = []
        for model_name, x_tensor_list in detected_model_name2x_tensor_list.items():
            # print(model_name, len(x_tensor_list), now_video_frame, self.now_batch_size, self.inference_batch_size)
            if len(x_tensor_list) == 0:
                continue

            self.batch_input_list.extend(x_tensor_list)
            self.batch_info_list.extend(detected_batch_info_list)
            self.now_batch_size += len(x_tensor_list)

            if self.now_batch_size < self.inference_batch_size:
                continue

            while self.now_batch_size >= self.inference_batch_size:
                if isinstance(self.batch_input_list[0], np.ndarray):
                    input_tensor = np.stack(
                        self.batch_input_list[: self.inference_batch_size], axis=0
                    )
                    input_tensor = torch.from_numpy(input_tensor)
                else:
                    input_tensor = torch.stack(
                        self.batch_input_list[: self.inference_batch_size], dim=0
                    )

                # mean = [0.485, 0.456, 0.406]
                # mean = torch.tensor(mean).view(1, 3, 1, 1, 1)
                # std = [0.229, 0.224, 0.225]
                # std = torch.tensor(std).view(1, 3, 1, 1, 1)

                # tensor_draw = input_tensor * std + mean
                # tensor_draw = (tensor_draw * 255).numpy().astype(np.uint8)

                # tensor_draw = np.transpose(
                #     tensor_draw, (0, 2, 3, 4, 1)
                # )  # B, T, H, W, C
                # # print(tensor_draw.shape)
                # x_tensor_numpy = np.concatenate(tensor_draw, axis=1)
                # x_tensor_numpy = np.concatenate(x_tensor_numpy, axis=1)
                # # print(x_tensor_numpy.shape)
                # os.makedirs("./check", exist_ok=True)
                # cv2.imwrite(
                #     "./check/batch_{}_{}_stream.jpg".format(model_name, self.num_write),
                #     x_tensor_numpy,
                # )
                # sdfa

                self.batch_input_list = self.batch_input_list[
                    self.inference_batch_size :
                ]

                self.now_batch_size -= len(input_tensor)

                batch_info_list = self.batch_info_list[: self.inference_batch_size]

                self.batch_info_list = self.batch_info_list[self.inference_batch_size :]
                self.batch_forward(
                    model_name=model_name,
                    input_tensor=input_tensor,
                    batch_info_list=batch_info_list,
                    output_result=output_result,
                )

        if is_last:
            while self.now_batch_size > 0:
                if isinstance(self.batch_input_list[0], np.ndarray):
                    input_tensor = np.stack(
                        self.batch_input_list[: self.inference_batch_size], axis=0
                    )
                    input_tensor = torch.from_numpy(input_tensor)
                else:
                    input_tensor = torch.stack(
                        self.batch_input_list[: self.inference_batch_size], dim=0
                    )
                self.batch_input_list = self.batch_input_list[
                    self.inference_batch_size :
                ]
                self.now_batch_size -= len(input_tensor)

                batch_info_list = self.batch_info_list[: self.inference_batch_size]
                self.batch_info_list = self.batch_info_list[self.inference_batch_size :]

                # print(len(input_tensor), len(batch_info_list), self.now_batch_size)

                self.batch_forward(
                    model_name=model_name,
                    input_tensor=input_tensor,
                    batch_info_list=batch_info_list,
                    output_result=output_result,
                )

        # if len(output_result):
        #     for person_index, single_output_result in enumerate(output_result[:4]):
        #         print("----- person index = {} -----".format(person_index))
        #         print("pid:", single_output_result["pid"], "frame list=", single_output_result["frame_list"])
        #         print(single_output_result["features"][:8])

        return output_result

