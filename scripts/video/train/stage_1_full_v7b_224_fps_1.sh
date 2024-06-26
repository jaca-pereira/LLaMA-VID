#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate llamavid

echo $CONDA_DEFAULT_ENV

cd ~/data/LLaMA-VID

wandb login a5569ff69ae6ef0b1d94c04c83e390de9c453efd

deepspeed --module llamavid.train.train_mem \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path model_zoo/LLM/vicuna/7B-V1.5 \
    --version plain_guided \
    --data_path ./data/LLaMA-VID-Pretrain/llava_with_webvid.json \
    --image_folder ./data/LLaMA-VID-Pretrain/ \
    --video_folder ./data/LLaMA-VID-Pretrain/ \
    --vision_tower ./model_zoo/LAVIS/eva_vit_g.pth \
    --image_processor ./llamavid/processor/clip-patch14-224 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --video_fps 1 \
    --bf16 True \
    --output_dir ./work_dirs/llama-vid-clip-text-token-reduction-7b-pretrain-224-video-fps-1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --freeze_backbone True \
    --report_to wandb
