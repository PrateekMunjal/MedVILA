# export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH=/home/prateek/projects/MedVILA

torchrun --nnodes 1 --nproc_per_node 1 --rdzv_id 42 \
--rdzv_backend c10d --rdzv_endpoint localhost:23454 \
llava/eval/slake_eval.py \
--model_path "Efficient-Large-Model/VILA1.5-3b" \
--conv-mode "v1" \
--model_max_length 4096 --batch_size 100 \
--output_dir "med_benchmarks/pretrained/VILA1.5-3b"