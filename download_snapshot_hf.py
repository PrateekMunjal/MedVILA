from huggingface_hub.hf_api import HfFolder
from huggingface_hub import snapshot_download
import os


HfFolder.save_token(os.environ["HF_API"])


model_id = "Efficient-Large-Model/Llama-3-VILA1.5-8B"

snapshot_download(repo_id=model_id, local_dir='/models_vlm/Llama-3-VILA1.5-8B')