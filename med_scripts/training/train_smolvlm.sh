#!/bin/bash

export PYTHONPATH=/home/prateek/projects/MedVILA

n_node=1
n_gpus=8
port_num=25004

torchrun --nnodes $n_node --nproc_per_node $n_gpus --rdzv_id 42 \
--rdzv_backend c10d --rdzv_endpoint localhost:$port_num \
llava/train/smolvlm_train.py --deepspeed ./scripts/zero3.json

