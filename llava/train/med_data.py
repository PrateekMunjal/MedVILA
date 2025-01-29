from torch.utils.data import Dataset, default_collate
import json, os
from PIL import Image
import torch
import copy
from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
# from llava.utils.tokenizer import preprocess_conversation
from llava.utils.med_tokenizer import preprocess_conversation as med_preprocess_conversation
from llava.constants import IGNORE_INDEX, SENTINEL_TOKEN

## On nebius
# image_folder: /data/vlm/original/slake/Slake1.0/imgs

class SLAKE(Dataset):

    def __init__(self, tokenizer, json_file_path, data_args, image_folder):
        super().__init__()

        self.json_file_path = json_file_path
        self.list_data_dict = json.load(open(json_file_path, "r"))
        self.image_folder = image_folder
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.image_processor = data_args.image_processor

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, index):
        
        sources = self.list_data_dict[index]

        if "image" in sources:
            image_fpath = os.path.join(self.image_folder, sources['image'])
            img = Image.open(image_fpath).convert("RGB")

            crop_size = self.data_args.image_processor.size
            img = img.resize((crop_size["width"], crop_size["height"]))

            img = self.image_processor(img, return_tensors="pt")["pixel_values"][0]

        # MED-VILA CODE
        sources = [self.list_data_dict[index]['conversations']]
        data_dict = default_collate([med_preprocess_conversation(conversation, self.tokenizer) for conversation in sources])
        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        data_dict["image"] = img.unsqueeze(0)

        return data_dict
    
        ### ORIGINAL CODE
        # sources = [sources]
        # sources = self.preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        # data_dict = default_collate([
        #     preprocess_conversation(conversation, self.tokenizer, no_system_prompt=False) for conversation in sources
        # ])

    