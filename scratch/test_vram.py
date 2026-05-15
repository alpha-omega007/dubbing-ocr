import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

def print_vram(label):
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = total - free
        print(f"[{label}] VRAM Used: {used / 1024**2:.2f} MB")

print_vram("Start")

import whisper
model = whisper.load_model("medium", device="cuda")
print_vram("After Whisper")
