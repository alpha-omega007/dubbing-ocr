import os
import re
import subprocess
import torch
import easyocr
from tqdm import tqdm
from core.interfaces.engines import IOCREngine
from core.entities.models import Segment

class EasyOCREngine(IOCREngine):
    def __init__(self, vsf_cli_path="./tools/VideoSubFinder/VideoSubFinderCli"):
        self.vsf_cli = vsf_cli_path
        self.reader = None

    def _get_reader(self, lang):
        if self.reader is None:
            self.reader = easyocr.Reader([lang], gpu=torch.cuda.is_available())
        return self.reader

    def extract_text(self, video_path: str, tmp_dir: str, lang: str) -> list:
        vsf_dir = os.path.join(tmp_dir, "vsf")
        os.makedirs(vsf_dir, exist_ok=True)
        
        # 1. VideoSubFinder
        subprocess.run([
            self.vsf_cli, "-c", "-r", "-i", video_path, "-o", vsf_dir,
            "-te", "0.25", "-be", "0.0", "-le", "0.1", "-re", "0.9"
        ], capture_output=True)
        
        rgb_dir = os.path.join(vsf_dir, "RGBImages")
        if not os.path.exists(rgb_dir): return []
        
        # 2. EasyOCR
        reader = self._get_reader(lang)
        images = sorted(os.listdir(rgb_dir))
        segments = []
        
        def parse_time(ts_str):
            h, m, s, ms = map(int, ts_str.split('_'))
            return h * 3600 + m * 60 + s + ms / 1000

        for img_name in tqdm(images, desc="🔬 EasyOCR"):
            if not img_name.endswith(".jpeg"): continue
            match = re.match(r"(\d+_\d+_\d+_\d+)__(\d+_\d+_\d+_\d+)", img_name)
            if not match: continue
            
            img_full_path = os.path.join(rgb_dir, img_name)
            ocr_result = reader.readtext(img_full_path)
            raw_text = " ".join([res[1] for res in ocr_result]).strip()
            
            # 3. Filtrer les caractères chinois (souvent présents dans les logos/watermarks)
            clean_text = re.sub(r'[\u4e00-\u9fff]+', '', raw_text).strip()
            
            if clean_text:
                segments.append(Segment(
                    start=parse_time(match.group(1)),
                    end=parse_time(match.group(2)),
                    text=clean_text
                ))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return segments
