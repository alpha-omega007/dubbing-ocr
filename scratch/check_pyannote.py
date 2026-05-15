import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import gc

def print_vram(label):
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = total - free
        print(f"[{label}] VRAM Used: {used / 1024**2:.2f} MB")

print_vram("Start")

from pyannote.audio import Pipeline
hf_token = "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" # Fake token won't work, I'll need a real one if I want to download. But wait, it's already downloaded in HF cache. 
# We need the user's token. 
