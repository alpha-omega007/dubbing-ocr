import os
import subprocess
import numpy as np
from tqdm import tqdm
from core.interfaces.engines import IAudioService
from core.entities.models import Segment

class FFmpegAudioService(IAudioService):
    _gender_classifier = None

    def get_duration(self, video_path: str) -> float:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        return float(res.stdout.strip())

    def extract_audio(self, video_path: str, output_dir: str) -> str:
        audio_path = os.path.join(output_dir, "audio_original.wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
        ], capture_output=True)
        return audio_path

    def detect_gender(self, audio_path: str, segments: list) -> list:
        import librosa
        import torch
        import warnings
        
        # Silence redundant warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        
        # 1. Load Audio (16kHz standard for models)
        audio, sr = librosa.load(audio_path, sr=16000, mono=True, dtype=np.float32)
        
        # 2. Try Model-based Gender Classification (Cached)
        if FFmpegAudioService._gender_classifier is None:
            try:
                from transformers import pipeline, logging
                logging.set_verbosity_error()
                FFmpegAudioService._gender_classifier = pipeline(
                    "audio-classification", 
                    model="norwoodsystems/norwood-maleVSfemale", 
                    device="cpu"
                )
                print("  🎤 Analyse vocale (IA) initialisée")
            except Exception as e:
                # Ne pas printer l'erreur à chaque chunk si c'est juste un problème de connexion
                if FFmpegAudioService._gender_classifier is None:
                    print(f"  ⚠️ Modèle IA indisponible, repli sur le pitch")
                FFmpegAudioService._gender_classifier = "FALLBACK"

        classifier = FFmpegAudioService._gender_classifier if FFmpegAudioService._gender_classifier != "FALLBACK" else None

        speaker_gender = {}
        
        for i, seg in enumerate(tqdm(segments, desc="🔬 Analyse Genre")):
            if (seg.end - seg.start) < 0.3: 
                seg.gender = "M"
                continue
                
            start_i = int(seg.start * sr)
            end_i = int(seg.end * sr)
            chunk = audio[start_i:end_i]
            if len(chunk) < 1000: 
                seg.gender = "M"
                continue

            # Key for caching (especially for diarized speakers)
            key = seg.speaker if seg.speaker else f"seg_{i}"
            if key in speaker_gender:
                seg.gender = speaker_gender[key]
                continue

            gender = "M"
            try:
                if classifier:
                    # IA Analysis
                    res = classifier(chunk)
                    best = max(res, key=lambda x: x['score'])
                    gender = "F" if best['label'].lower() == 'female' else "M"
                else:
                    # Fallback Pitch (Improved)
                    import parselmouth
                    snd = parselmouth.Sound(chunk, sampling_frequency=float(sr))
                    pv = snd.to_pitch().selected_array["frequency"]
                    pv = pv[pv > 60]
                    if len(pv) > 0:
                        pitch = float(np.median(pv))
                        gender = "F" if pitch >= 165.0 else "M"
            except:
                pass
            
            speaker_gender[key] = gender
            seg.gender = gender
            
        return segments

    def mux_final(self, video_path: str, audio_segments: list, tmp_dir: str, output_path: str):
        filter_script = os.path.join(tmp_dir, "filter_complex.txt")
        filter_parts = ["[0:a]volume=0.15[bg];"]
        for i, seg in enumerate(audio_segments):
            start_ms = int(seg["start"] * 1000)
            filter_parts.append(f"[{i+1}:a]adelay={start_ms}|{start_ms}[a{i}];")
        
        labels = "".join([f"[a{i}]" for i in range(len(audio_segments))])
        filter_parts.append(f"[bg]{labels}amix=inputs={len(audio_segments)+1}:duration=first:dropout_transition=2[aout]")
        
        with open(filter_script, "w") as f:
            f.write("".join(filter_parts))

        cmd = ["ffmpeg", "-y", "-i", video_path]
        for seg in audio_segments:
            cmd.extend(["-i", seg["file"]])
        cmd.extend([
            "-filter_complex_script", filter_script,
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", output_path
        ])
        subprocess.run(cmd, capture_output=True)
