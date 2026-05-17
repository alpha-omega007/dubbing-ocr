import os
import soundfile as sf
import torch
from tqdm import tqdm
from kokoro_onnx import Kokoro
from core.interfaces.engines import ITTSEngine
from core.entities.models import Segment

def apply_emotion_to_audio(audio_path: str, emotion: str, rate: int = 24000):
    import os
    import subprocess
    import shutil
    
    profiles = {
        "happy": {"pitch": 1.12, "tempo": 1.08, "volume": 1.1},
        "sad": {"pitch": 0.88, "tempo": 0.85, "volume": 0.8},
        "angry": {"pitch": 0.95, "tempo": 1.15, "volume": 1.3},
        "excited": {"pitch": 1.18, "tempo": 1.20, "volume": 1.2}
    }
    
    if emotion not in profiles:
        return
        
    prof = profiles[emotion]
    pitch = prof["pitch"]
    tempo = prof["tempo"]
    volume = prof["volume"]
    
    temp_out = audio_path + ".emo.wav"
    atempo_factor = tempo / pitch
    
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-filter_complex", f"asetrate={rate}*{pitch},atempo={atempo_factor},volume={volume}",
        temp_out
    ]
    
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode == 0 and os.path.exists(temp_out):
        shutil.move(temp_out, audio_path)
    else:
        if os.path.exists(temp_out):
            os.remove(temp_out)

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
            
            # Extraction de l'émotion si présente
            text = seg.text
            emotion = "neutral"
            if text.startswith("["):
                end_bracket = text.find("]")
                if end_bracket != -1:
                    emotion_tag = text[1:end_bracket].strip().lower()
                    emotion_map = {
                        "joyeux": "happy", "happy": "happy", "content": "happy",
                        "triste": "sad", "sad": "sad",
                        "colere": "angry", "colérique": "angry", "angry": "angry",
                        "excite": "excited", "excited": "excited", "enthousiaste": "excited",
                        "neutre": "neutral", "neutral": "neutral"
                    }
                    if emotion_tag in emotion_map:
                        emotion = emotion_map[emotion_tag]
                        text = text[end_bracket+1:].strip()
            
            voice = voice_set.get(seg.gender, voice_set["F"])
            out_file = os.path.join(output_dir, f"tts_{i:04d}.wav")
            
            # Synthèse du texte nettoyé
            samples, sr = self.tts.create(text, voice=voice, speed=1.0)
            sf.write(out_file, samples, sr)
            
            # Post-processing émotionnel
            if emotion != "neutral":
                apply_emotion_to_audio(out_file, emotion, sr)
            
            audio_segments.append({"file": out_file, "start": seg.start, "end": seg.end})
        return audio_segments
