import os
import asyncio
import edge_tts
from tqdm import tqdm
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

class EdgeTTSEngine(ITTSEngine):
    VOICES = {
        "fr": {"M": "fr-FR-HenriNeural", "F": "fr-FR-DeniseNeural"},
        "en": {"M": "en-US-GuyNeural", "F": "en-US-AriaNeural"},
        "es": {"M": "es-ES-AlvaroNeural", "F": "es-ES-ElviraNeural"},
    }

    def synthesize(self, segments: list, output_dir: str, target_lang: str) -> list:
        return asyncio.run(self._synthesize_async(segments, output_dir, target_lang))

    async def _synthesize_async(self, segments: list, output_dir: str, target_lang: str) -> list:
        audio_segments = []
        voice_set = self.VOICES.get(target_lang, self.VOICES["en"])
        
        use_fallback = False
        fallback_engine = None

        for i, seg in enumerate(tqdm(segments, desc="🔊 Edge TTS")):
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
            
            if use_fallback:
                # Use Kokoro fallback
                out_file = os.path.join(output_dir, f"tts_{i:04d}.wav")
                voice_set_k = fallback_engine.voices.get(target_lang, fallback_engine.voices["en"])
                voice_k = voice_set_k.get(seg.gender, voice_set_k["F"])
                
                samples, sr = fallback_engine.tts.create(text, voice=voice_k, speed=1.0)
                import soundfile as sf
                sf.write(out_file, samples, sr)
                
                if emotion != "neutral":
                    apply_emotion_to_audio(out_file, emotion, sr)
                    
                audio_segments.append({"file": out_file, "start": seg.start, "end": seg.end})
                continue

            voice = voice_set.get(seg.gender, voice_set["F"])
            out_file = os.path.join(output_dir, f"tts_{i:04d}.mp3")
            
            # Retry mechanism
            max_retries = 3
            success = False
            for attempt in range(max_retries):
                try:
                    communicate = edge_tts.Communicate(text, voice)
                    await communicate.save(out_file)
                    success = True
                    break # Success
                except Exception as e:
                    print(f"\n⚠️ Erreur EdgeTTS (essai {attempt+1}/{max_retries}) : {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
            
            if not success:
                print(f"\n⚠️ [EdgeTTS] Échec de connexion ou pas d'audio reçu (hors ligne ?).")
                print(f"🔄 Basculement automatique sur le moteur local et offline Kokoro...")
                try:
                    from infrastructure.tts.kokoro_engine import KokoroTTSEngine
                    onnx_path = "./models/kokoro-v1.0.onnx"
                    voices_path = "./models/voices-v1.0.bin"
                    if not os.path.exists(onnx_path):
                        onnx_path = "../models/kokoro-v1.0.onnx"
                        voices_path = "../models/voices-v1.0.bin"
                     
                    fallback_engine = KokoroTTSEngine(onnx_path, voices_path)
                    use_fallback = True
                    
                    # Synthesize current segment with Kokoro
                    out_file = os.path.join(output_dir, f"tts_{i:04d}.wav")
                    voice_set_k = fallback_engine.voices.get(target_lang, fallback_engine.voices["en"])
                    voice_k = voice_set_k.get(seg.gender, voice_set_k["F"])
                    
                    samples, sr = fallback_engine.tts.create(text, voice=voice_k, speed=1.0)
                    import soundfile as sf
                    sf.write(out_file, samples, sr)
                    
                    if emotion != "neutral":
                        apply_emotion_to_audio(out_file, emotion, sr)
                        
                    audio_segments.append({"file": out_file, "start": seg.start, "end": seg.end})
                except Exception as fe:
                    print(f"❌ Échec critique : impossible d'initialiser le fallback Kokoro : {fe}")
                    raise e
            else:
                if emotion != "neutral":
                    apply_emotion_to_audio(out_file, emotion, 24000)
                audio_segments.append({"file": out_file, "start": seg.start, "end": seg.end})

        return audio_segments
