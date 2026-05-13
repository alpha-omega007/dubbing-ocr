#!/usr/bin/env python3
"""
Pipeline de doublage vidéo local - 100% offline
GTX 1650 Ti (4Go VRAM) optimisé — gestion mémoire stricte

Stratégie mémoire :
  - pyannote   → CPU (évite crash GPU)
  - Whisper    → CUDA limité à 90% VRAM, audio découpé en chunks de 16s
  - Kokoro TTS → CPU
  - Nettoyage VRAM agressif entre chaque étape

Usage :
  python3 video_dubbing_pipeline.py --input video.mp4 --hf_token TON_TOKEN_HF --lang fr
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import shutil
import gc
import warnings
import time
import torch
import numpy as np
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
WHISPER_MODEL    = "medium"    # ~2.5 Go VRAM
KOKORO_VOICE_M   = "am_adam"   # voix homme
KOKORO_VOICE_F   = "af_sarah"  # voix femme
PITCH_THRESHOLD  = 165.0       # Hz — dessous = homme, dessus = femme
AUDIO_CHUNK_S    = 16          # secondes par chunk Whisper (évite OOM)
GPU_MARGIN       = 0.90        # utiliser max 90% de la VRAM
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg: str, emoji: str = "→"):
    print(f"\n{emoji}  {msg}", flush=True)


def free_vram(label: str = ""):
    """Nettoyage agressif de la VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free  = torch.cuda.mem_get_info()[0] / 1024**2
        total = torch.cuda.mem_get_info()[1] / 1024**2
        used  = total - free
        tag   = f" [{label}]" if label else ""
        print(f"  🧠 VRAM{tag} : {used:.0f} Mo utilisés / {total:.0f} Mo total ({free:.0f} Mo libres)",
              flush=True)
    time.sleep(0.5)


def limit_gpu_memory():
    """Limite l'utilisation GPU à GPU_MARGIN pour garder une marge de sécurité."""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_MARGIN)
        total = torch.cuda.mem_get_info()[1] // 1024**2
        log(f"VRAM limitée à {GPU_MARGIN*100:.0f}% = {int(total * GPU_MARGIN)} Mo / {total} Mo", "⚙️")


# ─────────────────────────────────────────────
# ÉTAPE 0 : vérification dépendances
# ─────────────────────────────────────────────
def check_dependencies():
    log("Vérification des dépendances...", "🔍")
    missing = []
    try:
        import whisper
    except ImportError:
        missing.append("openai-whisper")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pyannote.audio import Pipeline
    except ImportError:
        missing.append("pyannote-audio>=3.1.0")
    try:
        import parselmouth
    except ImportError:
        missing.append("praat-parselmouth")
    try:
        import kokoro_onnx
    except ImportError:
        missing.append("kokoro-onnx")
    try:
        import soundfile
    except ImportError:
        missing.append("soundfile")
    try:
        import librosa
    except ImportError:
        missing.append("librosa")
    try:
        import torchaudio
    except ImportError:
        missing.append("torchaudio")

    if missing:
        print("\n❌  Dépendances manquantes. Lance :")
        print(f"python3 -m pip install {' '.join(missing)} --break-system-packages")
        sys.exit(1)
    log("Toutes les dépendances sont présentes ✓", "✅")


# ─────────────────────────────────────────────
# ÉTAPE 1 : extraction audio
# ─────────────────────────────────────────────
def extract_audio(video_path: str, output_dir: str) -> str:
    log("Extraction de l'audio (ffmpeg)...", "🎬")
    audio_path = os.path.join(output_dir, "audio_original.wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ ffmpeg erreur : {result.stderr[-1000:]}")
        sys.exit(1)
    size_mb = os.path.getsize(audio_path) / 1024**2
    log(f"Audio extrait → {audio_path} ({size_mb:.1f} Mo)", "✅")
    return audio_path


# ─────────────────────────────────────────────
# ÉTAPE 1b : découper l'audio en chunks de 16s
# ─────────────────────────────────────────────
def split_audio_chunks(audio_path: str, output_dir: str, chunk_s: int = AUDIO_CHUNK_S) -> list:
    """Découpe le WAV en chunks de chunk_s secondes via ffmpeg."""
    log(f"Découpage audio en chunks de {chunk_s}s...", "✂️")
    chunks_dir = os.path.join(output_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True
    )
    total_duration = float(probe.stdout.strip())
    n_chunks = int(np.ceil(total_duration / chunk_s))

    chunk_paths = []
    for i in range(n_chunks):
        start    = i * chunk_s
        out_path = os.path.join(chunks_dir, f"chunk_{i:04d}.wav")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start), "-t", str(chunk_s),
            "-i", audio_path,
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            out_path
        ]
        subprocess.run(cmd, capture_output=True)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
            chunk_paths.append((i, start, out_path))

    log(f"{len(chunk_paths)} chunks créés ({chunk_s}s chacun, durée totale {total_duration:.0f}s)", "✅")
    return chunk_paths


# ─────────────────────────────────────────────
# ÉTAPE 2 : diarization sur CPU
# ─────────────────────────────────────────────
def run_diarization(audio_path: str, hf_token: str) -> list:
    log("Diarization (pyannote) → forcé sur CPU...", "🎙️")

    import torchaudio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pyannote.audio import Pipeline as PyannotePipeline

    log("Chargement pyannote...", "⏳")
    pipeline = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )
    pipeline = pipeline.to(torch.device("cpu"))  # CPU obligatoire

    waveform, sample_rate = torchaudio.load(audio_path)
    audio_dict = {"waveform": waveform, "sample_rate": sample_rate}

    log("Analyse en cours (15-20 min sur CPU pour 1h de vidéo)...", "⏳")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diarization = pipeline(audio_dict)

    segments = []
    for turn, speaker in diarization.itertracks():
        segments.append({
            "start":   round(turn.start, 3),
            "end":     round(turn.end, 3),
            "speaker": speaker,
            "gender":  None
        })

    del pipeline, waveform, audio_dict
    gc.collect()

    n_speakers = len(set(s["speaker"] for s in segments))
    log(f"{len(segments)} segments, {n_speakers} locuteurs", "✅")
    return segments


# ─────────────────────────────────────────────
# ÉTAPE 2b : détection genre par pitch F0
# ─────────────────────────────────────────────
def detect_gender(audio_path: str, segments: list) -> list:
    log("Détection genre par pitch F0 (CPU)...", "🔬")
    import parselmouth
    import librosa

    audio, sr = librosa.load(audio_path, sr=16000, mono=True, dtype=np.float32)
    speaker_pitches: dict = {}

    for seg in segments:
        duration = seg["end"] - seg["start"]
        if duration < 0.5:
            continue
        start_i = int(seg["start"] * sr)
        end_i   = int(seg["end"]   * sr)
        chunk   = audio[start_i:end_i]

        snd          = parselmouth.Sound(chunk, sampling_frequency=float(sr))
        pitch        = snd.to_pitch()
        pitch_values = pitch.selected_array["frequency"]
        pitch_values = pitch_values[pitch_values > 0]

        if len(pitch_values) > 0:
            sp = seg["speaker"]
            speaker_pitches.setdefault(sp, []).append(float(np.median(pitch_values)))

    del audio
    gc.collect()

    speaker_gender: dict = {}
    for speaker, pitches in speaker_pitches.items():
        avg    = float(np.median(pitches))
        gender = "F" if avg >= PITCH_THRESHOLD else "M"
        speaker_gender[speaker] = gender
        icon = "♀ Femme" if gender == "F" else "♂ Homme"
        print(f"  👤 {speaker} → {avg:.1f} Hz → {icon}", flush=True)

    for seg in segments:
        seg["gender"] = speaker_gender.get(seg["speaker"], "M")

    return segments


# ─────────────────────────────────────────────
# ÉTAPE 3 : transcription Whisper par chunks 16s
# ─────────────────────────────────────────────
def transcribe(audio_path: str, language: Optional[str], segments: list, output_dir: str) -> list:
    log(f"Transcription Whisper ({WHISPER_MODEL}) — chunks {AUDIO_CHUNK_S}s...", "📝")

    import whisper

    chunk_list    = split_audio_chunks(audio_path, output_dir, AUDIO_CHUNK_S)
    total_chunks  = len(chunk_list)

    log(f"Chargement Whisper sur {DEVICE.upper()} (VRAM ≤ {GPU_MARGIN*100:.0f}%)...", "⏳")
    free_vram("avant Whisper")
    model = whisper.load_model(WHISPER_MODEL, device=DEVICE)
    free_vram("Whisper chargé")

    all_words = []
    for idx, (chunk_i, time_offset, chunk_path) in enumerate(chunk_list):
        print(f"  📝 [{idx+1}/{total_chunks}] offset={time_offset:.0f}s ...", end=" ", flush=True)
        try:
            result = model.transcribe(
                chunk_path,
                word_timestamps=True,
                language=language,
                fp16=(DEVICE == "cuda")
            )
            n_words = 0
            for seg in result.get("segments", []):
                for w in seg.get("words", []):
                    all_words.append({
                        "word":  w["word"].strip(),
                        "start": w["start"] + time_offset,
                        "end":   w["end"]   + time_offset,
                    })
                    n_words += 1
            print(f"{n_words} mots", flush=True)
        except Exception as e:
            print(f"⚠️  échec : {e}", flush=True)

        # purge VRAM toutes les 10 tranches
        if (idx + 1) % 10 == 0:
            free_vram(f"chunk {idx+1}")

    del model
    free_vram("Whisper déchargé")

    log(f"{len(all_words)} mots transcrits au total", "✅")

    # aligner mots → segments diarization
    for seg in segments:
        seg_words = [
            w["word"] for w in all_words
            if w["start"] >= seg["start"] and w["end"] <= seg["end"] + 0.5
        ]
        seg["text"] = " ".join(seg_words).strip()

    return segments


# ─────────────────────────────────────────────
# ÉTAPE 4 : TTS Kokoro sur CPU
# ─────────────────────────────────────────────
def synthesize_speech(segments: list, output_dir: str) -> list:
    log("Synthèse vocale Kokoro (CPU)...", "🔊")

    def find_model(filename):
        for p in [
            filename,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
            os.path.expanduser(f"~/{filename}"),
        ]:
            if os.path.exists(p):
                return p
        return None

    onnx_path   = find_model("kokoro-v1.9.onnx")
    voices_path = find_model("voices-v1.0.bin")

    if not onnx_path or not voices_path:
        print("❌ Modèles Kokoro introuvables (kokoro-v1.9.onnx / voices-v1.0.bin)")
        print("   Lance : ./install_dubbing.sh")
        sys.exit(1)

    from kokoro_onnx import Kokoro
    import soundfile as sf

    tts_m = Kokoro(onnx_path, voices_path)
    tts_f = Kokoro(onnx_path, voices_path)

    valid_segs     = [(i, s) for i, s in enumerate(segments) if s.get("text", "").strip()]
    total          = len(valid_segs)
    audio_segments = []

    for count, (i, seg) in enumerate(valid_segs, 1):
        text   = seg["text"].strip()
        gender = seg.get("gender", "M")
        tts    = tts_m if gender == "M" else tts_f
        tts.voice = KOKORO_VOICE_M if gender == "M" else KOKORO_VOICE_F
        label  = f"♂" if gender == "M" else f"♀"

        print(f"  [{count}/{total}] {seg['start']:.1f}s {label} {text[:55]}", flush=True)

        try:
            out_file = os.path.join(output_dir, f"tts_{i:04d}.wav")
            samples, sample_rate = tts.create(text, speed=1.0)
            sf.write(out_file, samples, sample_rate)
            audio_segments.append({
                "file":   out_file,
                "start":  seg["start"],
                "end":    seg["end"],
                "gender": gender,
            })
        except Exception as e:
            print(f"  ⚠️  Segment {i} ignoré : {e}", flush=True)

    log(f"{len(audio_segments)} segments TTS générés", "✅")
    return audio_segments


# ─────────────────────────────────────────────
# ÉTAPE 5 : muxing final
# ─────────────────────────────────────────────
def mux_final(video_path: str, audio_segments: list, output_dir: str, output_path: str):
    log(f"Assemblage final ({len(audio_segments)} segments)...", "🎞️")

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True
    )
    total_duration = float(probe.stdout.strip())

    # Pour les vidéos longues (beaucoup de segments) → assemblage numpy
    if len(audio_segments) > 150:
        log(f"{len(audio_segments)} segments → assemblage numpy (plus stable)...", "⚙️")
        _mux_numpy(audio_segments, total_duration, output_dir, video_path, output_path)
        return

    silence_path = os.path.join(output_dir, "silence.wav")
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "anullsrc=r=24000:cl=mono",
        "-t", str(total_duration), silence_path
    ], capture_output=True)

    inputs  = ["-i", silence_path]
    filters = []
    prev    = "[0:a]"

    for idx, seg in enumerate(audio_segments):
        inputs   += ["-i", seg["file"]]
        label_in  = f"[a{idx}]"
        label_out = f"[mix{idx}]"
        delay_ms  = int(seg["start"] * 1000)
        filters.append(f"[{idx+1}:a]adelay={delay_ms}|{delay_ms}{label_in}")
        filters.append(f"{prev}{label_in}amix=inputs=2:duration=longest:normalize=0{label_out}")
        prev = label_out

    cmd = (
        ["ffmpeg", "-y"]
        + inputs + ["-i", video_path]
        + ["-filter_complex", ";".join(filters)]
        + ["-map", f"{len(audio_segments)+1}:v"]
        + ["-map", prev]
        + ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
        + [output_path]
    )

    log("Encodage ffmpeg...", "⏳")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ ffmpeg erreur :\n{result.stderr[-2000:]}")
        sys.exit(1)

    log(f"Vidéo finale → {output_path} ({os.path.getsize(output_path)//1024**2} Mo)", "✅")


def _mux_numpy(audio_segments, total_duration, output_dir, video_path, output_path):
    """Assemble tous les segments TTS en une seule piste audio via numpy."""
    import soundfile as sf
    import librosa

    SR    = 24000
    track = np.zeros(int(total_duration * SR) + SR, dtype=np.float32)

    for seg in audio_segments:
        try:
            data, sr = sf.read(seg["file"], dtype="float32")
            if sr != SR:
                data = librosa.resample(data, orig_sr=sr, target_sr=SR)
            start_i = int(seg["start"] * SR)
            end_i   = min(start_i + len(data), len(track))
            track[start_i:end_i] += data[:end_i - start_i]
        except Exception as e:
            print(f"  ⚠️  {e}", flush=True)

    peak = np.max(np.abs(track))
    if peak > 0:
        track = track / peak * 0.95

    merged_wav = os.path.join(output_dir, "merged_audio.wav")
    sf.write(merged_wav, track, SR)
    del track
    gc.collect()

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path, "-i", merged_wav,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ ffmpeg erreur :\n{result.stderr[-2000:]}")
        sys.exit(1)

    log(f"Vidéo finale → {output_path} ({os.path.getsize(output_path)//1024**2} Mo)", "✅")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pipeline doublage vidéo local")
    parser.add_argument("--input",    required=True)
    parser.add_argument("--hf_token", required=True)
    parser.add_argument("--output",   default=None)
    parser.add_argument("--lang",     default=None)
    parser.add_argument("--keep_tmp", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Fichier introuvable : {args.input}")
        sys.exit(1)

    output_path = args.output or Path(args.input).stem + "_dubbed.mp4"

    print("\n" + "="*60)
    print("  🎬 PIPELINE DOUBLAGE VIDÉO LOCAL")
    print("="*60)
    print(f"  Entrée  : {args.input}")
    print(f"  Sortie  : {output_path}")
    print(f"  Device  : {DEVICE.upper()}")
    print(f"  Whisper : {WHISPER_MODEL} | chunks {AUDIO_CHUNK_S}s | VRAM ≤{GPU_MARGIN*100:.0f}%")
    print(f"  TTS ♂   : {KOKORO_VOICE_M}  |  TTS ♀ : {KOKORO_VOICE_F}")
    print("="*60)

    check_dependencies()
    limit_gpu_memory()
    free_vram("démarrage")

    tmp_dir = tempfile.mkdtemp(prefix="dubbing_")
    log(f"Dossier temporaire : {tmp_dir}", "📁")

    try:
        audio_path = extract_audio(args.input, tmp_dir)
        segments   = run_diarization(audio_path, args.hf_token)
        segments   = detect_gender(audio_path, segments)

        diar_json = os.path.join(tmp_dir, "diarization.json")
        with open(diar_json, "w") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        log(f"Diarization sauvegardée → {diar_json}", "💾")

        segments = transcribe(audio_path, args.lang, segments, tmp_dir)

        print("\n📋 Aperçu transcription :")
        shown = 0
        for seg in segments:
            if seg.get("text") and shown < 10:
                g = "♀" if seg["gender"] == "F" else "♂"
                print(f"  [{seg['start']:.1f}s] {g} {seg['speaker']}: {seg['text'][:80]}")
                shown += 1

        audio_segs = synthesize_speech(segments, tmp_dir)
        mux_final(args.input, audio_segs, tmp_dir, output_path)

        print("\n" + "="*60)
        print("  ✅ PIPELINE TERMINÉ")
        print(f"  📹 Résultat : {output_path}")
        print("="*60 + "\n")

    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            log("Fichiers temporaires supprimés", "🧹")
        else:
            log(f"Fichiers temporaires conservés : {tmp_dir}", "📁")


if __name__ == "__main__":
    main()