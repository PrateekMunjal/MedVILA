import torch, os
from transformers import AutoProcessor, AutoModelForVision2Seq
from datasets import load_dataset
# from transformers import SiglipImageProcessor
from matplotlib import pyplot as plt

device = "cuda"

def get_model(model_id):
    return AutoModelForVision2Seq.from_pretrained(model_id)

def collate_fn(examples):

    system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
    Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
    The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
    Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

    text_inputs  = []
    image_inputs = []

    for example in examples:
        formatted_example = {
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
                            "text": example["query"],
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example['label'][0]}],
                },
            ]
        }
        formatted_text_inputs = processor.apply_chat_template(formatted_example["messages"], tokenize=False).strip()
        
        image = example["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
    
        text_inputs.append(formatted_text_inputs)
        image_inputs.append([image])

    


if __name__ == "__main__":

    preprocessor_model_id = "HuggingFaceTB/SmolVLM-Instruct"

    model_id = "HuggingFaceTB/SmolVLM-Instruct"

    # model_id = "/home/prateek/projects/MedVILA/smolvlm_output/checkpoint-443"
    dataset_id = "HuggingFaceM4/ChartQA"

    # img_processor = SiglipImageProcessor.from_pretrained("/models_vlm/VILA1.5-3b/vision_tower")

    model = get_model(model_id)
    model = model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(preprocessor_model_id)

    dataset = load_dataset(dataset_id)['test']
    sample = dataset[0]
    prompt = [
        {
            "role":"user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": sample["query"]}
            ]
        }
    ]

    print(sample)

    formatted_query = processor.apply_chat_template(prompt, tokenize=False)
    print(formatted_query)
    breakpoint()

    image = sample["image"]
    if image.mode != 'RGB':
        image = image.convert('RGB')
    inputs = processor(images=[image], text=formatted_query, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=4096)
    
    # decode prediction
    prediction = processor.batch_decode(outputs, skip_special_tokens=True)

    print(f"Query: {sample['query']}")
    print(f"Expected Answer: {sample['label'][0]}")
    print(f"Model Prediction: {prediction[0]}")
    sample['image'].save('question.png')
