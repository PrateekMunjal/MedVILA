import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Any, Dict, Sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch
from transformers import PreTrainedTokenizer
from llava.data.collate import DataCollator


import os
import sys
import torch
from transformers import HfArgumentParser, set_seed
from llava.train.args import ModelArguments, DataArguments, TrainingArguments
from loguru import logger

from llava.data import make_supervised_data_module
# from helper import make_supervised_data_module
from llava.data.collate import DataCollator

from llava.model import LlavaLlamaModel, LlavaLlamaConfig
from llava.train.utils import prepare_config_for_training, vision_resolution_elevation, need_to_modify_do_sample
from llava.train.helper import get_nb_trainable_parameters
from llava import conversation as conversation_lib
from llava.orig_constanst import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
from llava.train.llava_trainer import LLaVATrainer
import math

class SLAKEDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data: Sequence[Dict[str, Any]]):
        """
        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer used for encoding.
            data (Sequence[Dict[str, Any]]): Sequence of dictionaries containing `input_ids`, `labels`, and other data.
        """
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        return self.data[idx]

if __name__ == "__main__":

    def just_load_tokenizer():
        
        hf_id = "/models_vlm/VILA1.5-3b"

        model_cls = LlavaLlamaModel
        config = LlavaLlamaConfig.from_pretrained(hf_id)
        config.resume_path = hf_id

        model = model_cls(
            config=config,
            attn_implementation="flash_attention_2",
            model_max_length = 4096,
            cache_dir=None
        )

        return model
    
    model = just_load_tokenizer()
    tokenizer = model.tokenizer

    conversation_lib.default_conversation = conversation_lib.conv_templates["v1"]
    tokenizer.pad_token = tokenizer.unk_token
  
    # Example tokenizer (replace with your specific tokenizer)
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # tokenizer = AutoTokenizer.from_pretrained("/models_vlm/VILA1.5-3b")

    # Example data (replace with your actual data)
    data = [
        {
            "input_ids": torch.tensor([101, 102, 103]),
            "labels": torch.tensor([1, 2, 3]),
            "media": {"image": [torch.tensor([1.0, 2.0])]},
        },
        # Add more instances as needed
    ]

    # Instantiate the custom dataset
    dataset = CustomDataset(tokenizer=tokenizer, data=data)

    # Instantiate the DataCollator
    data_collator = DataCollator(tokenizer=tokenizer)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=data_collator)

    # Iterate through the DataLoader
    for batch in dataloader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        media = batch["media"]
        attention_mask = batch["attention_mask"]

        print(input_ids)
        print(labels)
        print(media)
        print(attention_mask)
        # Your training logic goes here
