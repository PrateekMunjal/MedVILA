n_nodes=1
n_gpus=8

port_num=25001

align_output_dir=med_results_vila1.5_3b/align

# Alignment
tune_visiontower=False
tune_projector=True
tune_lang_model=False

train_from="Efficient-Large-Model/VILA1.5-3b"

sh med_scripts/training/train.sh $n_nodes $n_gpus \
    $port_num $align_output_dir $train_from \
    $tune_vision_tower $tune_mm_projector $tune_lang_model

#########################################
#        Alignment stage 1.5
#########################################
tune_visiontower=True
tune_projector=True
tune_lang_model=False

align15_output_dir=med_results_vila1.5_3b/align1.5
train_from=$align_output_dir

sh med_scripts/training/train.sh $n_nodes $n_gpus \
    $port_num $align15_output_dir $train_from \
    $tune_vision_tower $tune_mm_projector $tune_lang_model

#########################################
#        Pretraining stage
#########################################
tune_visiontower=False
tune_projector=True
tune_lang_model=True

pretrain_output_dir=med_results_vila1.5_3b/pretrain
train_from=$align15_output_dir

sh med_scripts/training/train.sh $n_nodes $n_gpus \
    $port_num $pretrain_output_dir $train_from \
    $tune_vision_tower $tune_mm_projector $tune_lang_model


#########################################
#        SFT stage
#########################################
tune_visiontower=True
tune_projector=True
tune_lang_model=True

sft_output_dir=med_results_vila1.5_3b/sft
train_from=$pretrain_output_dir

sh med_scripts/training/train.sh $n_nodes $n_gpus \
    $port_num $sft_output_dir $train_from \
    $tune_vision_tower $tune_mm_projector $tune_lang_model

echo "----------------------------------------------"
echo "Training saved successfully...!!"
echo "Results are in following directories: "
echo "Align: ",$align_output_dir
echo "Align1.5: ",$align15_output_dir
echo "Pretrain: ",$pretrain_output_dir
echo "SFT: ",$sft_output_dir
echo "----------------------------------------------"