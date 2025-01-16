cd /home/prateek/projects/MedVILA/med_data_prepare

slake_ds_path=/data/vlm/original/slake/Slake1.0

output_path=/data/vlm/preprocessed/slake

mkdir -p $output_path

python slake_instruct_data_generate.py \
    --input_paths $slake_ds_path/train.json $slake_ds_path/validate.json \
    --output_path $output_path/slake_train_val_instruct.json