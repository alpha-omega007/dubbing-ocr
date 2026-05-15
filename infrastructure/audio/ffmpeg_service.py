import os
import subprocess
import numpy as np
from tqdm import tqdm
from core.interfaces.engines import IAudioService
from core.entities.models import Segment

class FFmpegAudioService(IAudioService):
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
        import parselmouth
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000, mono=True, dtype=np.float32)
        
        # 1. Collect pitches
        speaker_pitches = {}
        for i, seg in enumerate(tqdm(segments, desc="🔬 Pitch F0")):
            if (seg.end - seg.start) < 0.3: continue
            start_i = int(seg.start * sr)
            end_i = int(seg.end * sr)
            chunk = audio[start_i:end_i]
            if len(chunk) < 512: continue
            
            try:
                snd = parselmouth.Sound(chunk, sampling_frequency=float(sr))
                pv = snd.to_pitch().selected_array["frequency"]
                pv = pv[pv > 50] # Ignorer les bruits sourds
                if len(pv) > 0:
                    pitch = float(np.median(pv))
                    # Si speaker est défini (Whisper), on groupe. Sinon (OCR), on traite à part.
                    key = seg.speaker if seg.speaker else f"seg_{i}"
                    speaker_pitches[key] = pitch
            except:
                continue
        
        # 2. Assign Gender
        for i, seg in enumerate(segments):
            key = seg.speaker if seg.speaker else f"seg_{i}"
            pitch = speaker_pitches.get(key, 120.0) # Défaut Homme si échec
            seg.gender = "F" if pitch >= 165.0 else "M"
            
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
