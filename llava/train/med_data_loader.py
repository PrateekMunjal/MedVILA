import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Any, Dict, Sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset


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
from transformers import AutoProcessor
import math

def get_style(system_message, datapoint):
    formatted_text_data = {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {
                        "type": "text",
                        "text": datapoint["conversations"][0]['value'],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": datapoint["conversations"][1]['value']}],
            },
        ]
    }

    return formatted_text_data

class SLAKEDataset(Dataset):
    def __init__(self, data: Sequence[Dict[str, Any]]):
        """
        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer used for encoding.
            data (Sequence[Dict[str, Any]]): Sequence of dictionaries containing `input_ids`, `labels`, and other data.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        return self.data[idx]

def load_model(hf_id):

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


def collate_fn(minibatch):

    hf_id = "/models_vlm/VILA1.5-3b"
    # hf_id = "Efficient-Large-Model/VILA1.5-40b"
    llm_processor = AutoProcessor.from_pretrained(f"{hf_id}/llm")
    system_message = "You are a helpful radiologist assistant"

    for datapoint in minibatch:

        temp = get_style(system_message, datapoint)
        breakpoint()
        formatted_text = llm_processor.apply_chat_template(temp["messages"], tokenize=False).strip()

        breakpoint()



if __name__ == "__main__":

    data_src = "/data/vlm/preprocessed/slake"
    data_files = {
        'train': os.path.join(data_src, "slake_train_val_instruct.json"),
        'test' : os.path.join(data_src, "slake_test_instruct.json")
    }    

    dataset = load_dataset("json", data_files=data_files)
    train_dataset = dataset['train']
    test_dataset  = dataset['test'] 

    data_loader = DataLoader(train_dataset, batch_size=4, collate_fn=collate_fn)

    for batch in data_loader:
        print(batch)
        breakpoint()


    # hf_id = "/models_vlm/VILA1.5-3b"
    # model = load_model(hf_id)
    # tokenizer = model.tokenizer

    # breakpoint()

    # conversation_lib.default_conversation = conversation_lib.conv_templates["v1"]
    # tokenizer.pad_token = tokenizer.unk_token
  
    # tokenizer = AutoTokenizer.from_pretrained("/models_vlm/VILA1.5-3b")

    # breakpoint()

    # # Example data (replace with your actual data)
    # data = [
    #     {
    #         "input_ids": torch.tensor([101, 102, 103]),
    #         "labels": torch.tensor([1, 2, 3]),
    #         "media": {"image": [torch.tensor([1.0, 2.0])]},
    #     },
    #     # Add more instances as needed
    # ]

    # # Instantiate the custom dataset
    # dataset = CustomDataset(tokenizer=tokenizer, data=data)

    # # Instantiate the DataCollator
    # data_collator = DataCollator(tokenizer=tokenizer)

    # # Create the DataLoader
    # dataloader = DataLoader(dataset, batch_size=2, collate_fn=data_collator)
