from datasets import load_dataset
from transformers import AutoTokenizer, AutoImageProcessor
import torch
from llava.model import LlavaLlamaModel
from torchvision import transforms
import torch

dataset_name = "mdwiratathya/SLAKE-vqa-english"
llm_tokenizer_name = "/models_vlm/VILA1.5-3b/llm"
model_path = "/models_vlm/VILA1.5-3b"

dataset = load_dataset(dataset_name)
tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

image_processor = AutoImageProcessor.from_pretrained(model_path + "/vision_tower")

model = LlavaLlamaModel.from_pretrained(
    "/models_vlm/VILA1.5-3b",
    device_map="cuda:0"
)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



model.to("cuda")
model.eval()

example = dataset["train"][10]
question = example["question"]
image = example["image"].convert('RGB')

image = transform(image).unsqueeze(0)

conversation = f"<system> Answer the question based on the image. <assistant> {question}"
tokenized_text = tokenizer(conversation, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

images = [image]

device = next(model.parameters()).device

breakpoint()

processed_images = image_processor(images=images, return_tensors="pt")["pixel_values"]

processed_images = processed_images.to(device, dtype=torch.float16)

# image_embeddings = model.vision_tower(processed_images)
# image_embeddings = model.mm_projector(image_embeddings) 

# image_embeddings = image_embeddings.repeat(3,1,1)

image_embeddings = model.encode_images(processed_images) #[None,...]

media = {"image": [image_embeddings]}

input_ids = tokenized_text["input_ids"].to(device)
attention_mask = tokenized_text["attention_mask"].to(device)

breakpoint()
with torch.no_grad():
    outputs = model(input_ids=input_ids,media=media,attention_mask=attention_mask,packing=False,)

breakpoint()
generated_ids = model.generate(input_ids=input_ids, media=media, attention_mask=attention_mask, max_new_tokens=512)
generated_answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"Generated Answer: {generated_answer}")