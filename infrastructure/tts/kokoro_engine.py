import os
import soundfile as sf
import torch
from tqdm import tqdm
from kokoro_onnx import Kokoro
from core.interfaces.engines import ITTSEngine
from core.entities.models import Segment

class KokoroTTSEngine(ITTSEngine):
    def __init__(self, onnx_path, voices_path):
        self.tts = Kokoro(onnx_path, voices_path)
        self.voices = {
            "fr": {"M": "ff_siwis", "F": "ff_siwis"},
            "en": {"M": "am_adam", "F": "af_sarah"}
        }

    def synthesize(self, segments: list, output_dir: str, target_lang: str) -> list:
        audio_segments = []
        voice_set = self.voices.get(target_lang, self.voices["en"])
        
        for i, seg in enumerate(tqdm(segments, desc="🔊 Kokoro TTS")):
            if not seg.text: continue
            voice = voice_set.get(seg.gender, voice_set["F"])
            out_file = os.path.join(output_dir, f"tts_{i:04d}.wav")
            
            # Simplified speed calculation
            samples, sr = self.tts.create(seg.text, voice=voice, speed=1.0)
            sf.write(out_file, samples, sr)
            
            audio_segments.append({"file": out_file, "start": seg.start, "end": seg.end})
        return audio_segments
