import os, sys
import torch, argparse
import llava
from llava import conversation as clib
from llava.model import LlavaLlamaModel, LlavaLlamaConfig
from loguru import logger
from llava.train.utils import vision_resolution_elevation, need_to_modify_do_sample
from llava.train.utils import smart_tokenizer_and_embedding_resize
from llava.train.med_data import SLAKE
from llava.data.collate import DataCollator
from torch.utils.data import DataLoader
from collections import defaultdict
import json
from tqdm import tqdm
import gc

def get_question_answer(ds, index):
    questions = []
    answers = []
    for idx in index:

        curr_question = ds.list_data_dict[idx]['conversations'][0]['value'].strip()
        curr_answer   = ds.list_data_dict[idx]['conversations'][1]['value'].strip()

        questions.append(curr_question)
        answers.append(curr_answer)

    return questions, answers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, required=True)
    parser.add_argument("--conv-mode", "-c", type=str, default="auto")
    parser.add_argument("--fp16", default=False)
    parser.add_argument("--bf16", default=False)
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()

    compute_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    config = LlavaLlamaConfig.from_pretrained(args.model_path, resume=False)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    config.resume_path = args.model_path

    # Set conversation mode
    clib.default_conversation = clib.conv_templates[args.conv_mode].copy()

    # Load model
    model = LlavaLlamaModel(
        config=config,
        attn_implementation="flash_attention_2",
        model_max_length=args.model_max_length,
        cache_dir="/home/prateek/projects/MedVILA/temp_cache",
    )
    model.llm.config.torch_dtype = compute_dtype

    logger.info("Model loaded successfully")

    # Adjust for resolutions other than the vision tower's default (384x384)
    vision_resolution_elevation(model, config)

    # TODO: Check cache usage - why is it set to False? -- seems it is to avoid any confusion which can happen with multimodal data
    model.llm.config.use_cache = False

    if need_to_modify_do_sample(model.llm.generation_config):
        model.llm.generation_config.do_sample = True

    model.get_llm().requires_grad_(False)
    model.get_vision_tower().requires_grad_(False)
    model.get_mm_projector().requires_grad_(False)
    model.eval()

    model = model.to("cuda")

    tokenizer = model.tokenizer

    if tokenizer.bos_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(bos_token="[BOS]"),
            tokenizer=tokenizer,
            model=model.llm,
        )

    tokenizer.pad_token = None

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"), tokenizer=tokenizer, model=model.llm
        )

    model.llm.pad_token_id = model.tokenizer.pad_token_id
    model.llm.config.tokenizer_padding_side = model.tokenizer.padding_side
    model.llm.config.tokenizer_model_max_length = model.tokenizer.model_max_length

    args.image_processor = model.get_vision_tower().image_processor
    args.is_multimodal = True

    model.config.fps = 0.0
    dataset = SLAKE(
        tokenizer,
        "/data/vlm/preprocessed/slake/slake_test_instruct.json",
        args,
        "/data/vlm/original/slake/Slake1.0/imgs",
        True
    )
    collate_fn = DataCollator(tokenizer=tokenizer)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, drop_last=False)

    media_config = defaultdict(dict)

    all_predicted_responses = []
    all_questions = []
    all_answers = []
    all_indexes = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            
            batch["input_ids"] = batch["input_ids"].to("cuda", non_blocking=True)
            batch["labels"] = batch["labels"].to("cuda", non_blocking=True)
            batch["attention_mask"] = batch["attention_mask"].to("cuda", non_blocking=True)
            batch["media"]["image"] = [img.to(compute_dtype).to("cuda", non_blocking=True).half() for img in batch["media"]["image"]]
            
            logger.info(f"Inputids shape: {batch['input_ids'].shape}")

            # breakpoint()    
            # GPU mem noted for batchsize 300 and 1 gpu
            # 8 Gb
            breakpoint()

            output_ids = model.generate(
                input_ids=batch["input_ids"],
                media=batch["media"],
                media_config=media_config,
                generation_config=model.default_generation_config,
            ).cpu()

            text_responses = [tokenizer.decode(op_id, skip_special_tokens=True).strip() for op_id in output_ids]
            # 8 Gb
            # breakpoint() 
            # ground truth
            batch_questions, batch_answers = get_question_answer(dataset, batch["indexes"])
            
            all_predicted_responses += text_responses
            all_questions += batch_questions
            all_answers += batch_answers
            all_indexes += batch["indexes"]

            # del batch, output_ids, text_responses, batch_questions, batch_answers
            torch.cuda.empty_cache()

    
    data_list = []
    for i in range(len(all_indexes)):
        entry = {
            "id": all_indexes[i],
            "question": all_questions[i],
            "ground_truth": all_answers[i],
            "predicted_response": all_predicted_responses[i]
        }
        data_list.append(entry)

    file_name = "slake_predictions_testset.json"

    with open(file_name, "w", encoding="utf-8") as json_file:
        json.dump(data_list, json_file, indent=4, ensure_ascii=False)

    logger.info(f"JSON file '{file_name}' has been saved successfully!")

if __name__ == "__main__":
    main()
