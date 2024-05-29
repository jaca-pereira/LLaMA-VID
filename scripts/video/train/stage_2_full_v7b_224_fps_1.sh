#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate llamavid

echo $CONDA_DEFAULT_ENV

cd ~/data/LLaMA-VID

wandb login a5569ff69ae6ef0b1d94c04c83e390de9c453efd

deepspeed llamavid/train/train_mem.py \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path model_zoo/LLM/vicuna/7B-V1.5 \
    --version imgsp_v1 \
    --data_path ./data/LLaMA-VID-Finetune/subsample_llava_v1_5_mix665k_with_video_chatgpt_maxtime_5min.json \
    --image_folder ./data/LLaMA-VID-Finetune \
    --video_folder ./data/LLaMA-VID-Finetune \
    --vision_tower ./model_zoo/LAVIS/eva_vit_g.pth \
    --image_processor ./llamavid/processor/clip-patch14-224 \
    --pretrain_mm_mlp_adapter ./work_dirs/llama-vid-7b-pretrain-224-video-fps-1/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --video_fps 1 \
    --bf16 True \
    --output_dir ./work_dirs/llama-vid-7b-full-224-video-fps-1  \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
