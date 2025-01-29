from dataclasses import dataclass
from transformers import PretrainedConfig
import torch

def prepare_config_for_training(
    config: PretrainedConfig, model_args: dataclass, training_args: dataclass, data_args: dataclass
) -> None:
    assert model_args.vision_tower is not None, "requires vision tower"

    ## set module configurations
    if getattr(config, "llm_cfg", None) is None:
        config.llm_cfg = model_args.model_name_or_path
    if getattr(config, "vision_tower_cfg", None) is None:
        config.vision_tower_cfg = model_args.vision_tower
    if getattr(config, "mm_projector_cfg", None) is None:
        config.mm_projector_cfg = model_args.mm_projector
    
    ## set default dtype
    config.model_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    config.model_dtype = config.model_dtype.__str__()
    
    ## set tuning modules
    config.tune_language_model = training_args.tune_language_model
    config.tune_vision_tower = training_args.tune_vision_tower
    config.tune_mm_projector = training_args.tune_mm_projector
    
    ## set data args
    config.image_aspect_ratio = data_args.image_aspect_ratio
    
    ## extra vision tower configuration
    if getattr(config, "vision_tower_cfg", None) is not None:
        config.mm_vision_select_layer = model_args.mm_vision_select_layer
        config.mm_vision_select_feature = model_args.mm_vision_select_feature
        
        ## vision tower configurations
        config.vision_resolution = model_args.vision_resolution
        config.interpolate_mode = model_args.interpolate_mode
        config.drop_path_rate = model_args.drop_path_rate
        config.s2 = model_args.s2
        config.s2_scales = model_args.s2_scales
        config.s2_max_split_size = model_args.s2_max_split_size