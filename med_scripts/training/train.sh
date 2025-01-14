#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH=/home/prateek/projects/MedVILA

n_node=1
n_gpus=1
port_num=25001

torchrun --nnodes $n_node --nproc_per_node $n_gpus --rdzv_id 42 \
--rdzv_backend c10d --rdzv_endpoint localhost:$port_num \
llava/train/train_mem.py