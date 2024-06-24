#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate llamavid

echo $CONDA_DEFAULT_ENV

cd ~/data/LLaMA-VID

python -m scripts.extra_tool.extract_training_features \
    --model_name_or_path model_zoo/LLM/vicuna/7B-V1.5 \
    --text_model_name_or_path openai/clip-vit-large-patch14 \
    --model_max_length 4096 \
    --version plain_guided \
    --mm_use_im_patch_token False \
    --mm_use_im_start_end False \
    --vision_tower ./model_zoo/LAVIS/eva_vit_g.pth \
    --image_processor ./llamavid/processor/clip-patch14-224 \
    --bf16 True \
    --feat_dir ./data/LLaMA-VID-Pretrain/lama-vid-clip-text-token-reduction-pretrain-features \
    --video_fps 1 \
    --video_folder ./data/LLaMA-VID-Pretrain/ \
    --image_folder ./data/LLaMA-VID-Pretrain/ \
    --data_path ./data/LLaMA-VID-Pretrain/llava_with_webvid.json \
    --mm_redundant_token_selection sum \
    --cache_dir ~/.cache \
    --device cuda:0 \
    --refine_prompt False \
    --is_multimodal True

python -m scripts.extra_tool.extract_training_features \
    --model_name_or_path model_zoo/LLM/vicuna/7B-V1.5 \
    --text_model_name_or_path openai/clip-vit-large-patch14 \
    --model_max_length 8192 \
    --version  imgsp_v1 \
    --mm_use_im_patch_token False \
    --mm_use_im_start_end False \
    --vision_tower ./model_zoo/LAVIS/eva_vit_g.pth \
    --image_processor ./llamavid/processor/clip-patch14-224 \
    --bf16 True \
    --feat_dir ./data/LLaMA-VID-Finetune/llama-vid-clip-text-token-reduction-full-features \
    --video_fps 1 \
    --video_folder ./data/LLaMA-VID-Finetune/ \
    --image_folder ./data/LLaMA-VID-Finetune/ \
    --data_path ./data/LLaMA-VID-Pretrain/llava_v1_5_mix665k_with_video_chatgpt_maxtime_5min.json \
    --mm_redundant_token_selection sum \
    --cache_dir ~/.cache \
    --device cuda:0 \
    --refine_prompt False \
    --is_multimodal True