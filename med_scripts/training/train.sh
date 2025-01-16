#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH=/home/prateek/projects/MedVILA

n_node=1
n_gpus=1
port_num=25002
OUTPUT_DIR=/home/prateek/projects/MedVILA/med_results
MODEL_NAME_OR_PATH=/models_vlm/VILA1.5-3b
VISION_TOWER=Efficient-Large-Model/paligemma-siglip-so400m-patch14-448

HEALTHCARE_DS=$(for i in {1..4}; do echo -n slake+; done)
HEALTHCARE_DS=${HEALTHCARE_DS%+}


#Following M3 codebase on MONAI
chat_version=v1

torchrun --nnodes $n_node --nproc_per_node $n_gpus --rdzv_id 42 \
--rdzv_backend c10d --rdzv_endpoint localhost:$port_num \
llava/train/med_train.py --output_dir $OUTPUT_DIR \
--model_name_or_path $MODEL_NAME_OR_PATH \
--model_max_length 4096 \
--vision_tower $VISION_TOWER \
--tune_vision_tower False \
--tune_mm_projector True \
--tune_language_model False \
--version $chat_version \
--data_mixture ${HEALTHCARE_DS}

# --mm_vision_select_feature cls_patch \
# --mm_projector mlp_downsample \
# --mm_vision_select_layer -2 \
# --mm_use_im_start_end False \
# --mm_use_im_patch_token False \
# --image_aspect_ratio resize