BACKBONE='vit_b'
# PT_NAME='hybrid_pt_800e_k710_ft'
PT_NAME='k710_dl_from_giant'


# VIDEO_NAME='office_bike_namiki_sagyo'
# FEATURE_NAME='feature_v-0.033-181.033'
# META_NAME='video_meta_v-0.033-181.033'


# VIDEO_NAME='office_denshi'
# FEATURE_NAME='feature_v-0.033-209.767'
# META_NAME='video_meta_v-0.033-209.767'


VIDEO_NAME='video'
FEATURE_NAME='feature_v-0.067-102.200'
META_NAME='video_meta_v-0.067-102.200'


OUTPUT_PATH="/home/ollo/VideoMAE/data/test_videos/results/${BACKBONE}/${PT_NAME}/${VIDEO_NAME}/centric_feature_list/${FEATURE_NAME}.pickle"
META_PATH="/home/ollo/VideoMAE/data/test_videos/results/${BACKBONE}/${PT_NAME}/${VIDEO_NAME}/video_meta_list/${META_NAME}.json"
ANNOT_PATH='/home/ollo/VideoMAE/data/annotations/20250512_annotations_split_0.json'
VIDEO_PATH="/home/ollo/VideoMAE/data/test_videos/videos/${VIDEO_NAME}.mp4"
VISUALIZE_DIR="/home/ollo/VideoMAE/data/test_videos/visualization/${BACKBONE}/${PT_NAME}"

CUDA_VISIBLE_DEVICES=0 poetry run python /home/ollo/VideoMAE/OlloIntern2025Vis/visualize_model_output.py \
    --model_output_path ${OUTPUT_PATH} \
    --video_meta_path ${META_PATH} \
    --annot_path ${ANNOT_PATH} \
    --read_video_path ${VIDEO_PATH} \
    --visualize_dir ${VISUALIZE_DIR} \
    --conf_threshold 0.1 \
