import torch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
from core.interfaces.engines import ITranslationEngine
from core.entities.models import Segment

class HelsinkiTranslationEngine(ITranslationEngine):
    MODELS = {
        "fr": "Helsinki-NLP/opus-mt-en-fr",
        "es": "Helsinki-NLP/opus-mt-en-es",
        # ... others
    }

    def translate(self, segments: list, target_lang: str) -> list:
        model_name = self.MODELS.get(target_lang)
        if not model_name: return segments

        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).eval()

        for seg in tqdm(segments, desc=f"🌐 Translating to {target_lang}"):
            if not seg.text: continue
            inputs = tokenizer([seg.text], return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model.generate(**inputs)
            seg.text_original = seg.text
            seg.text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return segments
