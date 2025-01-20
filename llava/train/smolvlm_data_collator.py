import torch
import os
from torch.utils.data import DataLoader
from llava.train.smolvlm_utils import system_message
from datasets import load_dataset
from loguru import logger
from transformers import AutoProcessor

def collate_fn(minibatch):

    vlm_processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")

    text_inputs = []
    image_inputs = []

    for datapoint in minibatch:

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
                            "text": datapoint["query"],
                        },
                    ],
                },
            ]
        }

        formatted_text_data = vlm_processor.apply_chat_template(formatted_text_data["messages"]).strip()
        text_inputs.append(formatted_text_data)

        image = datapoint["image"]
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_inputs.append([image])

    # Preprocess the image and text using VLM processor
    batch = vlm_processor(text=text_inputs, images=image_inputs, return_tensors="pt", padding=True)

    # clone the labels
    labels = batch["input_ids"].clone()
    labels [labels == vlm_processor.tokenizer.pad_token_id] = -100 # Mask padding tokens in labels

    # Ensure image_token is converted to string if it is an AddedToken
    # In some processor, processor.image_token return a list for each image.
    image_token_id = vlm_processor.tokenizer.convert_tokens_to_ids(str(vlm_processor.image_token))

    labels[labels == image_token_id] = -100 # mask image token IDs in the labels

    batch["labels"] = labels

    return batch

dataset_id = "HuggingFaceM4/ChartQA"
dataset = load_dataset(dataset_id)

dataset = dataset["train"]

data_loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

for batch in data_loader:
    print(batch)
    breakpoint()
