## コードの書き換え
- extract_feature.py
    - 現状だと cv2 で動画読み込みを行っているが 必要に応じて decord で動画を読むようにする
        - model の学習が decord で読み込んでいるなら decord にした方が精度は低下しない
    - preprocess を学習と揃える. transform や cv2.resize など
    - model 構造を学習で使ったものにする
    - checkpoint から load_state_dict を行う
- その他各々の学習したmodelの環境に応じて書き換える。推論と学習で状況や状態が一致するようにする

## model の推論 コマンド例
``` bash
poetry run python3 extract_feature.py --video_path /f195f550-4024-45d9-9c41-024e57ddf837/OlloFactoryAnalysis/data/videos/test_short/video.mp4 --output_dir ./data/pose_data/test_short --analysis_fps 10 --inference_batch_size 8 --window_size 16 --input_size 224 --num_classes 58 --tubelet_size 2
```

## 可視化のコマンド例
```bash
poetry run python3 visualize_model_output.py --model_output_path ./data/pose_data/test_short/centric_feature_list/feature_v-0.033-19.267.pickle --video_meta_path ./data/pose_data/test_short/video_meta_list/video_meta_v-0.033-19.267.json --annot_path /f195f550-4024-45d9-9c41-024e57ddf837/OlloFactoryAnalysis/data/annotations/20250523_annotations_split_0.json --read_video_path /f195f550-4024-45d9-9c41-024e57ddf837/OlloFactoryAnalysis/data/videos/test_short/video.mp4 --visualize_dir check/vis

```