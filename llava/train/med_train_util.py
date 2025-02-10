import os
import sys
import torch
from transformers import HfArgumentParser, set_seed
from llava.train.args import ModelArguments, DataArguments, TrainingArguments
from loguru import logger

# from llava.data import make_supervised_data_module
# from helper import make_supervised_data_module
from llava.data.collate import DataCollator
from torch.utils.data import DataLoader
from llava.model import LlavaLlamaModel, LlavaLlamaConfig
# from llava.train.utils import prepare_config_for_training, vision_resolution_elevation, need_to_modify_do_sample

from llava.train.utils import vision_resolution_elevation, need_to_modify_do_sample
from llava.train.helper import get_nb_trainable_parameters
from llava import conversation as conversation_lib
from llava.orig_constanst import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
from llava.train.llava_trainer import LLaVATrainer
# from llava.train.llava_trainer_legacy import LLaVATrainer
import math
from llava.train.med_data import SLAKE
from utils import smart_tokenizer_and_embedding_resize
import transformers
from llava.train.med_utils import prepare_config_for_training as modify_config_for_training
from llava.conversation import auto_set_conversation_mode

    
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir, _internal_call=True)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def train():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    
    set_seed(training_args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if local_rank == 0: logger.info(f"Set Seed to {training_args.seed}")

    #TODO: Check for sequence parallelism here -- llava/train/train.py (search for sp_degree) and llava/train/sequence_parallel/globals.py

    # TODO: Add support for quantized model
    assert model_args.quantize_model == "false"
    
    if local_rank == 0:
        logger.info(f"Quantization Status: {model_args.quantize_model}")
        logger.info(f"HF Model: {model_args.model_name_or_path}")
        logger.info(f"Cache directory: {training_args.cache_dir}")

    model_cls = LlavaLlamaModel
    resume_from_checkpoint = False
    
    auto_set_conversation_mode(model_args.model_name_or_path)
    config = LlavaLlamaConfig.from_pretrained(model_args.model_name_or_path, resume=resume_from_checkpoint)

    # prepare_config_for_training(config, model_args, training_args, data_args)
    modify_config_for_training(config, model_args, training_args, data_args)

    #TODO: Change it later with actual path for resuming training
    config.resume_path = model_args.model_name_or_path

    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    
    model = model_cls(
        config=config,
        attn_implementation="flash_attention_2",
        model_max_length = training_args.model_max_length,
        cache_dir=training_args.cache_dir
    )
    # model.get_llm().model.embed_tokens = torch.nn.Embedding(32000, 2560, padding_idx=0)
    if local_rank == 0:
        logger.info("Model loaded successfully")
    # logger.info(f"VILA MODEL \n\n {model}")

    # Adjust for resolutions other than the vision tower's default (384x384)
    vision_resolution_elevation(model, config)

    # TODO: Check cache usage - why is it set to False? -- seems it is to avoid any confusion which can happen with multimodal data
    model.llm.config.use_cache = False

    if need_to_modify_do_sample(model.llm.generation_config):
        model.llm.generation_config.do_sample = True

    if local_rank == 0:
        logger.info(f"Training LLM: {config.tune_language_model}")
        logger.info(f"Training Vision Tower: {config.tune_vision_tower}")
        logger.info(f"Training MultiModal Projector: {config.tune_mm_projector}")

    # TODO: Handle PEFT Methods later
    model.get_llm().requires_grad_(config.tune_language_model)
    model.get_vision_tower().requires_grad_(config.tune_vision_tower)
    model.get_mm_projector().requires_grad_(config.tune_mm_projector)
    model.train()

    model.llm.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    assert training_args.gradient_checkpointing == False, "Wooh! We will handle gradient checkpointing later!"
    assert training_args.lora_enable == False, "Currently we don't handle PEFT based methods"

    # breakpoint()
    # model.get_llm().model.embed_tokens = torch.nn.Embedding(32000, 2560, padding_idx=0)

    model = model.to("cuda")
    
    n_train_params, all_params = get_nb_trainable_parameters(model)
    if local_rank == 0:
        logger.info(f"{(n_train_params / all_params)*100:.2f}% are trainable (i.e {n_train_params/10**9:.3f} Billions) out of total {all_params/10**9:.3f} Billions")

    # Check with @Clement to compare with original codebase for tokenizer config
    tokenizer = model.tokenizer
 
    if tokenizer.bos_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(bos_token="[BOS]"),
            tokenizer=tokenizer,
            model=model.llm,
        )

    ############################################################
    ##  Handling padding token behavior for different models  ## 
    ############################################################

    # - For model_id = "Efficient-Large-Model/VILA1.5-3B":
    #   - `tokenizer.unk_token` is NOT None, so no need to add a padding token explicitly.
    # - For model_id = "Efficient-Large-Model/Llama-3-VILA1.5-8B":
    #   - `tokenizer.unk_token` is None, meaning valid `input_ids` require a padding token.
    #   - Without a padding token, input sequences might be incorrectly processed.

    # To ensure consistent behavior across models, we explicitly set `pad_token` to None first
    # tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token = None  
    
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(special_tokens_dict=dict(pad_token="[PAD]"),tokenizer=tokenizer,model=model.llm)

    model.llm.pad_token_id = model.tokenizer.pad_token_id
    model.llm.config.tokenizer_padding_side = model.tokenizer.padding_side
    model.llm.config.tokenizer_model_max_length = model.tokenizer.model_max_length

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

        assert model_args.mm_use_im_patch_token == False, "The model should not use im_patch_token."
        assert model_args.num_time_tokens == 0, "Non zero number of tokens are not handled at the moment"
        
        # TODO: Handle when user passes model_args.num_time_tokens > 0
        model.config.time_token_ids = []

        model.config.soft_ce_std = model_args.soft_ce_std

        num_patches = model.get_vision_tower().num_patches
        downsample_rate = model.get_mm_projector().downsample_rate
        num_image_tokens = math.ceil(num_patches**0.5 / downsample_rate) ** 2
        data_args.num_image_tokens = num_image_tokens

        data_args.s2_scales = list(map(int, model_args.s2_scales.split(",")))
        
        dataset = SLAKE(tokenizer, "/data/vlm/preprocessed/slake/slake_train_val_instruct.json", data_args, "/data/vlm/original/slake/Slake1.0/imgs")
        
        if local_rank == 0:
            logger.info(f"Number of patches processed: {num_patches} and number of image tokens after MM Projector: {num_image_tokens}")
            logger.info(f"Length of datatset {len(dataset)}")
        
        collate_fn = DataCollator(tokenizer=tokenizer)

        # TODO: Remove hardcoded batch szie
        # data_loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
        model.train()

        # for batch in data_loader:

        #     breakpoint()

        #     batch['input_ids']      = batch['input_ids'].to("cuda")
        #     batch['labels']         = batch['labels'].to("cuda")
        #     batch['attention_mask'] = batch['attention_mask'].to("cuda")
            
        #     for i in range(len(batch['media']['image'])):
        #         batch['media']['image'][i] = batch['media']['image'][i].to(compute_dtype).to("cuda")
            
        #     temp_output = model(**batch)

        ## Working code to train VLM with Llava Trainer

        data_module = dict(
            train_dataset=dataset,
            data_collator=collate_fn,
        )

        training_args.sample_lens = [len(dataset)]
        trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        trainer.save_state()

        model.llm.config.use_cache = True
        model.config.resume_path = model.config._name_or_path = training_args.output_dir

        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()