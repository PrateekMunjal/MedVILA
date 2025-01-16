import os
import sys
import torch
from transformers import HfArgumentParser, set_seed
from llava.train.args import ModelArguments, DataArguments, TrainingArguments
from loguru import logger

from llava.data import make_supervised_data_module
# from helper import make_supervised_data_module

from llava.model import LlavaLlamaModel, LlavaLlamaConfig
from llava.train.utils import prepare_config_for_training, vision_resolution_elevation, need_to_modify_do_sample
from llava.train.helper import get_nb_trainable_parameters
from llava import conversation as conversation_lib
from llava.orig_constanst import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
from llava.train.llava_trainer import LLaVATrainer
import math

def train():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    logger.info(f"Set Seed to {training_args.seed}")
    set_seed(training_args.seed)

    #TODO: Check for sequence parallelism here -- llava/train/train.py (search for sp_degree) and llava/train/sequence_parallel/globals.py

    # TODO: Add support for quantized model
    assert model_args.quantize_model == "false"

    logger.info(f"Quantization Status: {model_args.quantize_model}")
    logger.info(f"HF Model: {model_args.model_name_or_path}")
    logger.info(f"Cache directory: {training_args.cache_dir}")

    model_cls = LlavaLlamaModel
    config = LlavaLlamaConfig.from_pretrained(model_args.model_name_or_path)

    prepare_config_for_training(config, model_args, training_args, data_args)

    #TODO: Change it later with actual path for resuming training
    config.resume_path = model_args.model_name_or_path 
    resume_from_checkpoint = False
    model = model_cls(
        config=config,
        attn_implementation="flash_attention_2",
        model_max_length = training_args.model_max_length,
        cache_dir=training_args.cache_dir
    )

    logger.info("Model loaded successfully")
    # logger.info(f"VILA MODEL \n\n {model}")

    # Adjust for resolutions other than the vision tower's default (384x384)
    vision_resolution_elevation(model, config)

    # TODO: Check cache usage - why is it set to False? -- seems it is to avoid any confusion which can happen with multimodal data
    model.llm.config.use_cache = False

    if need_to_modify_do_sample(model.llm.generation_config):
        model.llm.generation_config.do_sample = True

    logger.info(f"Training LLM: {config.tune_language_model}")
    logger.info(f"Training Vision Tower: {config.tune_vision_tower}")
    logger.info(f"Training MultiModal Projector: {config.tune_mm_projector}")

    # TODO: Handle PEFT Methods later
    model.get_llm().requires_grad_(config.tune_language_model)
    model.get_vision_tower().requires_grad_(config.tune_vision_tower)
    model.get_mm_projector().requires_grad_(config.tune_mm_projector)
    
    n_train_params, all_params = get_nb_trainable_parameters(model)
    logger.info(f"{(n_train_params / all_params)*100:.2f}% are trainable (i.e {n_train_params/10**9:.3f} Billions) out of total {all_params/10**9:.3f} Billions")

    # Check with @Clement to compare with original codebase for tokenizer config
    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.unk_token

    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True
        model.config.fps = 0.0
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.vision_tower_lr = training_args.vision_tower_lr
        model.config.num_time_tokens = data_args.num_time_tokens = model_args.num_time_tokens
        model.config.time_token_format = data_args.time_token_format = model_args.time_token_format

        # DISCUSS THIS WITH @Clement
        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        # assert not model_args.mm_use_im_patch_token
        assert model_args.mm_use_im_patch_token == False, "The model should not use im_patch_token."
        assert model_args.num_time_tokens == 0, "Non zero number of tokens are not handled at the moment"
        # TODO: Handle when user passes model_args.num_time_tokens > 0
        model.config.time_token_ids = []

        model.config.soft_ce_std = model_args.soft_ce_std

        num_patches = model.get_vision_tower().num_patches
        downsample_rate = model.get_mm_projector().downsample_rate
        num_image_tokens = math.ceil(num_patches**0.5 / downsample_rate) ** 2

        data_args.num_image_tokens = num_image_tokens

        logger.info(f"Number of patches processed: {num_patches} and number of image tokens after MM Projector: {num_image_tokens}")

        data_args.s2_scales = list(map(int, model_args.s2_scales.split(",")))

        # breakpoint()
        data_module = make_supervised_data_module(
            tokenizer=tokenizer,
            data_args=data_args,
            training_args=training_args,
        )

        # trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

        # breakpoint()

        # trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # breakpoint()

if __name__ == "__main__":
    train()