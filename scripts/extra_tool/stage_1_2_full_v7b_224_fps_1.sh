#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate llamavid

echo $CONDA_DEFAULT_ENV

cd ~/data/LLaMA-VID

python -m scripts.extra_tool.extract_training_features.py \
    --model_name_or_path model_zoo/LLM/vicuna/7B-V1.5 \
    --version plain_guided \
    --data_path ./data/LLaMA-VID-Pretrain/llava_with_webvid.json \
    --image_folder ./data/LLaMA-VID-Pretrain/ \
    --video_folder ./data/LLaMA-VID-Pretrain/ \
    --vision_tower ./model_zoo/LAVIS/eva_vit_g.pth \
    --image_processor ./llamavid/processor/clip-patch14-224 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --video_fps 1 \
    --bf16 True \
    --feat_dir ./data/llama-vid-clip-text-token-reduction-pretrain \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True

python -m scripts.extra_tool.extract_training_features.py \
    --model_name_or_path model_zoo/LLM/vicuna/7B-V1.5 \
    --version imgsp_v1 \
    --data_path ./data/LLaMA-VID-Finetune/llava_v1_5_mix665k_with_video_chatgpt_maxtime_5min.json \
    --image_folder ./data/LLaMA-VID-Finetune \
    --video_folder ./data/LLaMA-VID-Finetune \
    --vision_tower ./model_zoo/LAVIS/eva_vit_g.pth \
    --image_processor ./llamavid/processor/clip-patch14-224 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --video_fps 1 \
    --bf16 True \
    --feat_dir ./data/llama-vid-clip-text-token-reduction-full  \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True
