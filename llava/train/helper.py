import os, sys
from transformers import PreTrainedTokenizer
from llava.train.args import DataArguments, TrainingArguments
from typing import Dict, Sequence
import llava.data.datasets_mixture as datasets_mixture
from llava.train.sequence_parallel import (
    get_pg_manager,
)
from llava.data.dataset import DataCollatorForSupervisedDatasetSeqParallel
from llava.data.datasets_mixture import Dataset as DataSetClass


def get_nb_trainable_parameters(model) -> tuple[int, int]:
    r"""
    Returns the number of trainable parameters and the number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "element_size"):
                num_bytes = param.element_size()
            elif not hasattr(param, "quant_storage"):
                num_bytes = 1
            else:
                num_bytes = param.quant_storage.itemsize
            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def add_medical_dataset(ds_name, json_filepath, image_path, dataset_type="torch"):
    temp_dataset = DataSetClass(
        dataset_name = ds_name,
        dataset_type = dataset_type,
        data_path = json_filepath,
        image_path = image_path
    )

    datasets_mixture.add_dataset(temp_dataset)

def make_supervised_data_module(
    tokenizer: PreTrainedTokenizer,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning.
    This function is originally implemented by the LLaVA team and
    modified by Jason Lu, Haotian Tang and Ligeng Zhu."""

    add_medical_dataset("slake", "/data/vlm/preprocessed/slake/slake_train_val_instruct.json", "/data/vlm/original/slake/Slake1.0/imgs")

    datasets_mixture.register_datasets_mixtures()

    from llava.data.builder import build_dataset
    from llava.data.collate import DataCollator

    train_dataset = build_dataset(data_args.data_mixture, data_args, training_args, tokenizer)
    training_args.sample_lens = [len(d) for d in train_dataset.datasets]

    PROCESS_GROUP_MANAGER = get_pg_manager()
    if PROCESS_GROUP_MANAGER is None:
        data_collator = DataCollator(tokenizer=tokenizer)
    else:
        sp_degree = training_args.seq_parallel_size
        sp_rank = PROCESS_GROUP_MANAGER.sp_rank
        ring_degree = PROCESS_GROUP_MANAGER.ring_degree
        ring_type = PROCESS_GROUP_MANAGER.ring_type
        data_collator = DataCollatorForSupervisedDatasetSeqParallel(
            tokenizer=tokenizer,
            data_args=data_args,
            training_args=training_args,
            sp_degree=sp_degree,
            sp_rank=sp_rank,
            ring_degree=ring_degree,
            ring_type=ring_type,
        )

    return dict(
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
