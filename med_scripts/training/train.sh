#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH=/home/prateek/projects/MedVILA

n_node=1
n_gpus=2
port_num=25002
OUTPUT_DIR=/home/prateek/projects/MedVILA/med_results
MODEL_NAME_OR_PATH=/models_vlm/VILA1.5-3b
VISION_TOWER=Efficient-Large-Model/paligemma-siglip-so400m-patch14-448

HEALTHCARE_DS=$(for i in {1..4}; do echo -n slake+; done)
HEALTHCARE_DS=${HEALTHCARE_DS%+}

batchsize=4

#Following M3 codebase on MONAI
chat_version=v1

torchrun --nnodes $n_node --nproc_per_node $n_gpus --rdzv_id 42 \
--rdzv_backend c10d --rdzv_endpoint localhost:$port_num \
llava/train/med_train.py --output_dir $OUTPUT_DIR \
--deepspeed ./scripts/zero3.json \
--model_name_or_path $MODEL_NAME_OR_PATH \
--model_max_length 4096 \
--vision_tower $VISION_TOWER \
--version $chat_version \
--data_mixture ${HEALTHCARE_DS} \
--mm_vision_select_feature cls_patch \
--mm_projector mlp_downsample \
--tune_vision_tower True \
--tune_mm_projector True \
--tune_language_model True \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio resize \
--output_dir $OUTPUT_DIR \
--num_train_epochs 2 \
--per_device_train_batch_size $batchsize \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 100 \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 4096 \
--gradient_checkpointing True \
--dataloader_num_workers 1 \
--lazy_preprocess True \
--vflan_no_system_prompt True
  
# --bf16 True
    # --report_to wandb