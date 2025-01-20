import torch
import os
from torch.utils.data import DataLoader
from llava.train.smolvlm_utils import system_message
from datasets import load_dataset
from loguru import logger
from transformers import AutoProcessor, AutoModelForVision2Seq
from trl import SFTConfig, SFTTrainer

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
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": datapoint['label'][0]}],
                },
            ]
        }

        formatted_text_data = vlm_processor.apply_chat_template(formatted_text_data["messages"], tokenize=False).strip()

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

def get_training_args():
    training_args = SFTConfig(
        output_dir="smolvlm_output",  # Directory to save the model
        num_train_epochs=10,                     # number of training epochs
        per_device_train_batch_size=8,          # batch size per device during training
        gradient_accumulation_steps=1,         # number of steps before performing a backward/update pass
        gradient_checkpointing=False,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=5,                        # log every 10 steps
        save_strategy="epoch",                  # save checkpoint every epoch
        learning_rate=2e-4,                     # learning rate, based on QLoRA paper
        bf16=True,                              # use bfloat16 precision
        tf32=True,                              # use tf32 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",           # use constant learning rate scheduler
        push_to_hub=False,                       # push model to hub
        gradient_checkpointing_kwargs = {"use_reentrant": False}, # use reentrant checkpointing
        # dataloader_num_workers=16, 
        dataset_text_field="", # need a dummy field for collator
        dataset_kwargs = {"skip_prepare_dataset": True}, # important for collator
        remove_unused_columns = False                    # necessary else features except label will be removed
    )

    return training_args


def get_model(model_id):

    model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
    return model


if __name__ == "__main__":

    device = "cuda"
    
    model = get_model("HuggingFaceTB/SmolVLM-Instruct")    
    model = model.to(device)

    model.model.vision_model.requires_grad_(False)
    model.model.connector.requires_grad_(True)
    model.model.text_model.requires_grad_(True)
    model.lm_head.requires_grad_(True)

    # breakpoint()

    dataset_id = "HuggingFaceM4/ChartQA"
    dataset = load_dataset(dataset_id)

    data_loader = DataLoader(dataset["train"], batch_size=4, collate_fn=collate_fn)
    for batch in data_loader:
        print(batch)
        breakpoint()

    # training_args = get_training_args()
    # vlm_processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")

    # trainer = SFTTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset["train"],
    #     eval_dataset=dataset["test"],
    #     data_collator=collate_fn,
    #     tokenizer=vlm_processor.tokenizer,
    # )

    # trainer.train()

    # logger.info("Saving Model...!!")
    # trainer.save("./smolvlm_finetuning_checkpoint")

