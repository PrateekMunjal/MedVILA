export CUDA_VISIBLE_DEVICES=0,1,2,3

export PYTHONPATH=/home/prateek/projects/MedVILA

n_node=1
n_gpus=1
port_num=25002
# # MODEL_NAME_OR_PATH=/models_vlm/Llama-3-VILA1.5-8B
# # MODEL_NAME_OR_PATH=/home/prateek/projects/MedVILA/med_results_old
# MODEL_NAME_OR_PATH=Efficient-Large-Model/VILA1.5-3b
# MODEL_NAME_OR_PATH=/models_vlm/temp_VILA1.5-3b
# # MODEL_NAME_OR_PATH="/home/prateek/projects/MedVILA/med_results"

MODEL_NAME_OR_PATH=/home/prateek/projects/MedVILA/med_results_vila1.5_3b

batchsize=4
chat_version=vicuna_v1

torchrun --nnodes $n_node --nproc_per_node $n_gpus --rdzv_id 42 \
--rdzv_backend c10d --rdzv_endpoint localhost:$port_num \
llava/eval/inference.py \
--model_path $MODEL_NAME_OR_PATH \
--conv_mode $chat_version \
--text "What is the modality of this image?" \
--media /home/prateek/projects/MedVILA/temp.png

# --text "Please describe the image in detail" \

# vila-infer \
# --model-path Efficient-Large-Model/VILA1.5-3b \
# --conv-mode vicuna_v1 \
# --text "Please describe the image" \
# --media /home/prateek/projects/MedVILA/temp.png
