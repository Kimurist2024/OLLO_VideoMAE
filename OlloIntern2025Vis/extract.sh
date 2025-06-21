# BACKBONE='vit_b'
BACKBONE='vit_s'

# PT_NAME='hybrid_pt_800e_k710_ft'
PT_NAME='k710_dl_from_giant'


# VIDEO_NAME='office_bike_namiki_sagyo'
# VIDEO_NAME='office_denshi'
VIDEO_NAME='video'


VIDEO_PATH="/home/ollo/VideoMAE/data/test_videos/videos/${VIDEO_NAME}.mp4"
OUTPUT_DIR="/home/ollo/VideoMAE/data/test_videos/results/${BACKBONE}/${PT_NAME}/${VIDEO_NAME}"
CKPT_PATH="/home/ollo/VideoMAE/output/finetuned_models/vit_b/${PT_NAME}/checkpoint-best.pth"

CUDA_VISIBLE_DEVICES=0 poetry run python3 /home/ollo/VideoMAE/OlloIntern2025Vis/extract_feature.py \
    --video_path ${VIDEO_PATH} \
    --checkpoint_path ${CKPT_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --analysis_fps 10 \
    --inference_batch_size 16 \
    --window_size 16 \
    --input_size 224 \
    --num_classes 58 \
    --tubelet_size 2 \