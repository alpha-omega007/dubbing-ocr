import os
import subprocess
import torch
import whisper
from tqdm import tqdm
from core.interfaces.engines import ITranscriptionEngine
from core.entities.models import Segment

class WhisperTranscriptionEngine(ITranscriptionEngine):
    def __init__(self, model_size="small", device="cuda"):
        self.model_size = model_size
        self.device = device
        self.model = None

    def _load_model(self):
        if self.model is None:
            self.model = whisper.load_model(self.model_size, device=self.device)

    def transcribe(self, audio_path: str, language: str, segments: list, chunk_duration: int = 20) -> list:
        self._load_model()
        
        # Determine total duration
        import librosa
        duration = librosa.get_duration(path=audio_path)
        
        all_words = []
        for start in tqdm(range(0, int(duration), chunk_duration), desc="📝 Whisper Chunks"):
            end = min(start + chunk_duration, duration)
            # Transcribe window
            res = self.model.transcribe(
                audio_path, 
                language=language, 
                word_timestamps=True,
                initial_prompt="Transcription de l'épisode.",
                clip_timestamps=[start, end]
            )
            for s in res.get("segments", []):
                for w in s.get("words", []):
                    all_words.append({"word": w["word"].strip(), "start": w["start"], "end": w["end"]})
        
        for seg in segments:
            # Map words to segments
            seg_words = [w["word"] for w in all_words if w["start"] >= seg.start and w["end"] <= seg.end + 0.5]
            seg.text = " ".join(seg_words).strip()
            
        return segments
