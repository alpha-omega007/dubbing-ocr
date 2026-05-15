import os
import soundfile as sf
from tqdm import tqdm
from melotts import MeloTTS
from core.interfaces.engines import ITTSEngine
from core.entities.models import Segment

class MeloTTSEngine(ITTSEngine):
    def __init__(self, device="cpu"):
        self.device = device
        self.models = {}

    def _get_model(self, lang):
        if lang not in self.models:
            self.models[lang] = MeloTTS(language=lang, device=self.device)
        return self.models[lang]

    def synthesize(self, segments: list, output_dir: str, target_lang: str) -> list:
        audio_segments = []
        # MeloTTS supports 'FR', 'EN', etc.
        melo_lang = target_lang.upper()
        model = self._get_model(melo_lang)
        
        for i, seg in enumerate(tqdm(segments, desc="🔊 Melo TTS")):
            if not seg.text: continue
            out_file = os.path.join(output_dir, f"tts_{i:04d}.wav")
            
            # Simplified: Melo speaker ID 0=Male/Female depending on model
            # For FR, usually 0 is female, 1 is male? Needs checking.
            speaker_id = 0 if seg.gender == "F" else 1
            model.tts_to_file(seg.text, speaker_id, out_file, speed=1.0)
            
            audio_segments.append({"file": out_file, "start": seg.start, "end": seg.end})
        return audio_segments
