#!/usr/bin/env python3
"""
Pipeline de doublage vidéo local - 100% offline
GTX 1650 Ti (4Go VRAM) optimisé — gestion mémoire stricte + barres de progression

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
import threading
import torch
import numpy as np
import re
from pathlib import Path
from typing import Optional

# auto-install tqdm si absent
try:
    from tqdm import tqdm
except ImportError:
    os.system(f"{sys.executable} -m pip install tqdm --break-system-packages -q")
    from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
WHISPER_MODEL   = "small" # Passez à "medium" si vous avez des oublis
KOKORO_VOICE_M  = "am_adam"
KOKORO_VOICE_F  = "af_sarah"
PITCH_THRESHOLD = 165.0
AUDIO_CHUNK_S   = 16
GPU_MARGIN      = 0.50 # Par défaut 50% pour utiliser "un peu" de GPU
TTS_ENGINE      = "edge"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Masquer les warnings Triton/Python.h
os.environ["TRITON_INTERPRET"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"


def log(msg: str, emoji: str = "→"):
    print(f"\n{emoji}  {msg}", flush=True)


def free_vram(label: str = ""):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free  = torch.cuda.mem_get_info()[0] / 1024**2
        total = torch.cuda.mem_get_info()[1] / 1024**2
        used  = total - free
        tag   = f" [{label}]" if label else ""
        print(f"  🧠 VRAM{tag} : {used:.0f}/{total:.0f} Mo ({free:.0f} Mo libres)", flush=True)
    time.sleep(0.3)


def limit_gpu_memory(margin: float = None):
    global GPU_MARGIN
    if margin is not None:
        GPU_MARGIN = margin
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_MARGIN)
        total = torch.cuda.mem_get_info()[1] // 1024**2
        log(f"VRAM limitée à {GPU_MARGIN*100:.0f}% = {int(total * GPU_MARGIN)} Mo / {total} Mo", "⚙️")
        if GPU_MARGIN < 0.6:
            print(f"  💡 Tip: Pour limiter aussi la chauffe, lance : sudo nvidia-smi -pl 25")


# ─────────────────────────────────────────────
# SPINNER — pour les étapes sans hook de progression
# ─────────────────────────────────────────────
class Spinner:
    """Spinner animé avec timer pour les étapes longues sans progression connue."""
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, label: str):
        self.label    = label
        self.running  = False
        self._thread  = None
        self._start   = None

    def _spin(self):
        i = 0
        while self.running:
            elapsed = time.time() - self._start
            m, s    = divmod(int(elapsed), 60)
            timer   = f"{m:02d}:{s:02d}"
            frame   = self.FRAMES[i % len(self.FRAMES)]
            print(f"\r  {frame}  {self.label}  [{timer}]   ", end="", flush=True)
            i += 1
            time.sleep(0.1)

    def start(self):
        self.running = True
        self._start  = time.time()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self, success: bool = True):
        self.running = False
        if self._thread:
            self._thread.join()
        elapsed = time.time() - self._start
        m, s    = divmod(int(elapsed), 60)
        icon    = "✅" if success else "❌"
        print(f"\r  {icon}  {self.label}  [{m:02d}:{s:02d}]          ", flush=True)


# ─────────────────────────────────────────────
# ÉTAPE 0 : vérification dépendances
# ─────────────────────────────────────────────
def check_dependencies(check_ocr: bool = False):
    log("Vérification des dépendances...", "🔍")
    missing = []
    checks = [
        ("whisper",       "openai-whisper"),
        ("parselmouth",   "praat-parselmouth"),
        ("kokoro_onnx",   "kokoro-onnx"),
        ("soundfile",     "soundfile"),
        ("librosa",       "librosa"),
        ("torchaudio",    "torchaudio"),
    ]
    if check_ocr:
        checks.append(("easyocr", "easyocr"))

    for module, pkg in checks:
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pyannote.audio import Pipeline
    except ImportError:
        missing.append("pyannote-audio>=3.1.0")

    if missing:
        print("\n❌  Dépendances manquantes. Lance :")
        print(f"python3 -m pip install {' '.join(missing)} --break-system-packages")
        sys.exit(1)
    log("Toutes les dépendances sont présentes ✓", "✅")


# ─────────────────────────────────────────────
# ÉTAPE 1 : extraction audio
# ─────────────────────────────────────────────
def extract_audio(video_path: str, output_dir: str) -> str:
    log("Extraction de l'audio...", "🎬")
    audio_path = os.path.join(output_dir, "audio_original.wav")

    # récupérer durée pour la barre
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True
    )
    duration = float(probe.stdout.strip()) if probe.returncode == 0 else 0

    spinner = Spinner(f"ffmpeg extraction ({duration/60:.1f} min de vidéo)")
    spinner.start()
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path],
        capture_output=True, text=True
    )
    spinner.stop(result.returncode == 0)

    if result.returncode != 0:
        print(f"❌ ffmpeg erreur : {result.stderr[-500:]}")
        sys.exit(1)

    size_mb = os.path.getsize(audio_path) / 1024**2
    log(f"Audio extrait → {audio_path} ({size_mb:.1f} Mo)", "✅")
    return audio_path


# ─────────────────────────────────────────────
# ÉTAPE 1b : découper l'audio en chunks
# ─────────────────────────────────────────────
def split_audio_chunks(audio_path: str, output_dir: str, chunk_s: int = AUDIO_CHUNK_S) -> list:
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
    with tqdm(total=n_chunks, desc="  ✂️  Découpage", unit="chunk",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        for i in range(n_chunks):
            start    = i * chunk_s
            out_path = os.path.join(chunks_dir, f"chunk_{i:04d}.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(start), "-t", str(chunk_s),
                 "-i", audio_path, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out_path],
                capture_output=True
            )
            if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
                chunk_paths.append((i, start, out_path))
            pbar.update(1)

    log(f"{len(chunk_paths)} chunks créés ({total_duration/60:.1f} min total)", "✅")
    return chunk_paths


# ─────────────────────────────────────────────
# ÉTAPE 2 : diarization sur CPU
# ─────────────────────────────────────────────
def run_diarization(audio_path: str, hf_token: str) -> list:
    log("Diarization (pyannote) → CPU...", "🎙️")

    import torchaudio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pyannote.audio import Pipeline as PyannotePipeline

    spinner = Spinner("Chargement modèle pyannote")
    spinner.start()
    pipeline = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )
    pipeline = pipeline.to(torch.device("cpu"))
    spinner.stop()

    waveform, sample_rate = torchaudio.load(audio_path)
    audio_dict = {"waveform": waveform, "sample_rate": sample_rate}

    # pyannote n'expose pas de hook de progression → spinner avec timer
    print(f"\n  ℹ️  Durée estimée : 15-25 min pour 1h de vidéo sur CPU", flush=True)
    spinner2 = Spinner("Analyse diarization (qui parle quand)")
    spinner2.start()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diarization = pipeline(audio_dict).speaker_diarization
    spinner2.stop()

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start":   round(turn.start, 3),
            "end":     round(turn.end, 3),
            "speaker": speaker,
            "gender":  None
        })

    del pipeline, waveform, audio_dict
    torch.cuda.empty_cache()
    import time; time.sleep(2)
    gc.collect()

    n_speakers = len(set(s["speaker"] for s in segments))
    log(f"{len(segments)} segments détectés — {n_speakers} locuteurs", "✅")
    return segments


# ─────────────────────────────────────────────
# ÉTAPE 2b : détection genre par pitch F0
# ─────────────────────────────────────────────
def detect_gender(audio_path: str, segments: list) -> list:
    log("Détection genre par pitch F0...", "🔬")
    import parselmouth
    import librosa

    audio, sr     = librosa.load(audio_path, sr=16000, mono=True, dtype=np.float32)
    speaker_pitches: dict = {}

    valid_segs = [s for s in segments if (s["end"] - s["start"]) >= 0.5]
    with tqdm(total=len(valid_segs), desc="  🔬  Pitch F0",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        for seg in valid_segs:
            start_i = int(seg["start"] * sr)
            end_i   = int(seg["end"]   * sr)
            chunk   = audio[start_i:end_i]
            snd     = parselmouth.Sound(chunk, sampling_frequency=float(sr))
            pv      = snd.to_pitch().selected_array["frequency"]
            pv      = pv[pv > 0]
            if len(pv) > 0:
                speaker_pitches.setdefault(seg["speaker"], []).append(float(np.median(pv)))
            pbar.update(1)

    del audio
    gc.collect()

    speaker_gender: dict = {}
    print()
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
# ÉTAPE 3 : transcription Whisper par chunks
# ─────────────────────────────────────────────
def transcribe(audio_path: str, language: Optional[str], segments: list, output_dir: str, task: str = 'transcribe') -> list:
    log(f"Transcription Whisper ({WHISPER_MODEL}) — chunks {AUDIO_CHUNK_S}s...", "📝")

    import whisper

    chunk_list   = split_audio_chunks(audio_path, output_dir, AUDIO_CHUNK_S)
    total_chunks = len(chunk_list)

    log(f"Chargement Whisper sur {DEVICE.upper()} (VRAM ≤ {GPU_MARGIN*100:.0f}%)...", "⏳")
    free_vram("avant Whisper")
    model = whisper.load_model(WHISPER_MODEL, device=DEVICE)
    free_vram("Whisper chargé")

    all_words = []
    with tqdm(total=total_chunks, desc="  📝  Transcription",
              unit="chunk", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for idx, (chunk_i, time_offset, chunk_path) in enumerate(chunk_list):
            try:
                result = model.transcribe(
                    chunk_path,
                    word_timestamps=True,
                    language=language,
                    task=task,
                    fp16=(DEVICE == "cuda")
                )
                for seg in result.get("segments", []):
                    for w in seg.get("words", []):
                        all_words.append({
                            "word":  w["word"].strip(),
                            "start": w["start"] + time_offset,
                            "end":   w["end"]   + time_offset,
                        })
            except Exception as e:
                tqdm.write(f"  ⚠️  Chunk {idx+1} échoué : {e}")

            if (idx + 1) % 10 == 0:
                free_vram(f"chunk {idx+1}")

            pbar.update(1)
            pbar.set_postfix({"mots": len(all_words), "offset": f"{time_offset:.0f}s"})

    del model
    free_vram("Whisper déchargé")
    log(f"{len(all_words)} mots transcrits", "✅")

    for seg in segments:
        seg_words = [
            w["word"] for w in all_words
            if w["start"] >= seg["start"] and w["end"] <= seg["end"] + 0.5
        ]
        seg["text"] = " ".join(seg_words).strip()

    return segments


# ─────────────────────────────────────────────
# ÉTAPE 3 OCR : Extraction par OCR (VideoSubFinder + PaddleOCR)
# ─────────────────────────────────────────────
def extract_subtitles_ocr(video_path: str, tmp_dir: str, lang: str = "fr") -> list:
    log("Extraction des sous-titres via OCR (VideoSubFinder + PaddleOCR)...", "🖼️")
    
    vsf_dir = os.path.join(tmp_dir, "vsf")
    os.makedirs(vsf_dir, exist_ok=True)
    
    # 1. VideoSubFinder : extraire les frames
    vsf_cli = "./tools/VideoSubFinder/VideoSubFinderCli"
    if not os.path.exists(vsf_cli):
        log(f"VideoSubFinder introuvable à {vsf_cli}", "❌")
        sys.exit(1)
    
    log("Lancement de VideoSubFinder (recherche des textes)...", "🔍")
    spinner = Spinner("VideoSubFinder search")
    spinner.start()
    # On utilise -c (clear), -r (run search), -i (input), -o (output)
    # On restreint un peu la zone (te=0.2, be=0.0) pour accélérer et éviter les logos
    result = subprocess.run([
        vsf_cli, "-c", "-r", "-i", video_path, "-o", vsf_dir,
        "-te", "0.25", "-be", "0.0", "-le", "0.1", "-re", "0.9"
    ], capture_output=True, text=True)
    spinner.stop(result.returncode == 0)
    
    rgb_dir = os.path.join(vsf_dir, "RGBImages")
    if not os.path.exists(rgb_dir) or not os.listdir(rgb_dir):
        log("Aucune image de sous-titre détectée par VideoSubFinder.", "⚠️")
        return []
    
    # 2. EasyOCR : extraire le texte des images
    log("Lancement d'EasyOCR sur les frames extraites...", "🔬")
    try:
        import easyocr
    except ImportError:
        log("EasyOCR non installé. Installez-le avec : pip install easyocr", "❌")
        sys.exit(1)
    
    # Initialiser EasyOCR (langue cible)
    # gpu=True si CUDA dispo
    reader = easyocr.Reader([lang], gpu=torch.cuda.is_available())
    
    images = sorted(os.listdir(rgb_dir))
    segments = []
    
    def parse_time(ts_str):
        h, m, s, ms = map(int, ts_str.split('_'))
        return h * 3600 + m * 60 + s + ms / 1000

    with tqdm(total=len(images), desc="  🔬  EasyOCR", unit="img") as pbar:
        for img_name in images:
            if not img_name.endswith(".jpeg"):
                continue
                
            img_path = os.path.join(rgb_dir, img_name)
            
            # Nom format: H_M_S_MS__H_M_S_MS_...
            match = re.match(r"(\d+_\d+_\d+_\d+)__(\d+_\d+_\d+_\d+)", img_name)
            if not match:
                continue
                
            start_s = parse_time(match.group(1))
            end_s   = parse_time(match.group(2))
            
            # OCR
            ocr_result = reader.readtext(img_path)
            # ocr_result format: [([[x,y], [x,y], [x,y], [x,y]], text, confidence), ...]
            text_lines = [res[1] for res in ocr_result]
            
            text = " ".join(text_lines).strip()
            if text:
                segments.append({
                    "start": start_s,
                    "end": end_s,
                    "text": text,
                    "speaker": "SPEAKER_00",
                    "gender": "M"
                })
            pbar.update(1)
            
    log(f"{len(segments)} segments extraits par OCR", "✅")
    return segments

# ─────────────────────────────────────────────
# ÉTAPE 3b : traduction générique (anglais → langue cible)
# ─────────────────────────────────────────────

# Modèles Helsinki disponibles pour les langues courantes
HELSINKI_MODELS = {
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "it": "Helsinki-NLP/opus-mt-en-it",
    "pt": "Helsinki-NLP/opus-mt-en-ROMANCE",
    "nl": "Helsinki-NLP/opus-mt-en-nl",
    "pl": "Helsinki-NLP/opus-mt-en-pl",
    "ru": "Helsinki-NLP/opus-mt-en-ru",
    "ja": "Helsinki-NLP/opus-mt-en-jap",
    "zh": "Helsinki-NLP/opus-mt-en-zh",
    "ar": "Helsinki-NLP/opus-mt-en-ar",
}

def translate_segments(segments: list, tgt_lang: str) -> list:
    """Traduit le texte (anglais) de chaque segment vers tgt_lang via Helsinki-NLP."""

    model_name = HELSINKI_MODELS.get(tgt_lang)
    if not model_name:
        log(f"Langue cible '{tgt_lang}' non supportée. Langues dispo : {list(HELSINKI_MODELS.keys())}", "❌")
        return segments

    log(f"Traduction anglais → {tgt_lang} ({model_name})...", "🌐")

    try:
        from transformers import MarianMTModel, MarianTokenizer
        # Patch agressif pour contourner l'erreur de sécurité CVE-2025-32434 sur torch < 2.6
        import transformers.utils.import_utils as transformers_utils
        import transformers.modeling_utils as modeling_utils
        mock_check = lambda *args, **kwargs: None
        transformers_utils.check_torch_load_is_safe = mock_check
        modeling_utils.check_torch_load_is_safe = mock_check
    except ImportError:
        log("Installation de transformers...", "📦")
        os.system(f"{sys.executable} -m pip install transformers sentencepiece sacremoses --break-system-packages -q")
        from transformers import MarianMTModel, MarianTokenizer
        import transformers.utils.import_utils as transformers_utils
        import transformers.modeling_utils as modeling_utils
        mock_check = lambda *args, **kwargs: None
        transformers_utils.check_torch_load_is_safe = mock_check
        modeling_utils.check_torch_load_is_safe = mock_check

    spinner = Spinner(f"Chargement modèle traduction")
    spinner.start()
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model     = MarianMTModel.from_pretrained(model_name)
    model.eval()
    spinner.stop()

    valid = [(i, s) for i, s in enumerate(segments) if s.get("text", "").strip()]

    with tqdm(total=len(valid), desc=f"  🌐  {tgt_lang}",
              unit="seg", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        for i, seg in valid:
            text = seg["text"].strip()
            if not text:
                pbar.update(1)
                continue
            try:
                inputs  = tokenizer([text], return_tensors="pt", padding=True,
                                    truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model.generate(**inputs)
                translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                seg["text_original"] = text
                seg["text"]          = translated
            except Exception as e:
                tqdm.write(f"  ⚠️  Traduction seg {i} échouée : {e}")
            pbar.update(1)

    del model, tokenizer
    gc.collect()
    log(f"Traduction → {tgt_lang} terminée ✓", "✅")

    # Aperçu
    print(f"\n📋 Aperçu traduction (5 premiers segments) :")
    shown = 0
    for seg in segments:
        if seg.get("text") and seg.get("text_original") and shown < 5:
            print(f"  EN : {seg['text_original'][:70]}")
            print(f"  {tgt_lang.upper()} : {seg['text'][:70]}")
            print()
            shown += 1

    return segments


# ─────────────────────────────────────────────
# ÉTAPE 4 : Moteurs de synthèse vocale (TTS)
# ─────────────────────────────────────────────

def synthesize_speech(segments: list, output_dir: str, tgt_lang: str = "en", engine: str = "kokoro") -> list:
    """Point d'entrée modulaire pour le TTS."""
    if engine == "melo":
        return synthesize_melo(segments, output_dir, tgt_lang)
    elif engine == "edge":
        return synthesize_edge(segments, output_dir, tgt_lang)
    else:
        return synthesize_kokoro(segments, output_dir, tgt_lang)


def synthesize_kokoro(segments: list, output_dir: str, tgt_lang: str = "en") -> list:
    log(f"Synthèse Kokoro ({tgt_lang})...", "🔊")
    
    # Mappage des langues vers les voix Kokoro (M: Homme, F: Femme)
    lang_voices = {
        "fr": {"M": "ff_siwis", "F": "ff_siwis"}, 
        "en": {"M": "am_adam",  "F": "af_sarah"},
        "ja": {"M": "jf_alpha", "F": "jf_alpha"},
        "zh": {"M": "zf_alpha", "F": "zf_alpha"},
    }
    voice_set = lang_voices.get(tgt_lang, lang_voices["en"])

    def find_model(filename):
        for p in [filename,
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", filename),
                  os.path.expanduser(f"~/{filename}")]:
            if os.path.exists(p):
                return p
        return None

    onnx_path   = find_model("kokoro-v1.0.onnx")
    voices_path = find_model("voices-v1.0.bin")
    if not onnx_path or not voices_path:
        print("❌ Modèles Kokoro introuvables. Lance : ./install_dubbing.sh")
        sys.exit(1)

    try:
        from kokoro_onnx import Kokoro
        import soundfile as sf
    except ImportError:
        os.system(f"{sys.executable} -m pip install kokoro-onnx soundfile --break-system-packages -q")
        from kokoro_onnx import Kokoro
        import soundfile as sf

    tts_m = Kokoro(onnx_path, voices_path)
    tts_f = Kokoro(onnx_path, voices_path)

    valid_segs     = [(i, s) for i, s in enumerate(segments) if s.get("text", "").strip()]
    audio_segments = []

    with tqdm(total=len(valid_segs), desc="  🔊  Kokoro", unit="seg") as pbar:
        for count, (i, seg) in enumerate(valid_segs, 1):
            text   = seg["text"].strip()
            gender = seg.get("gender", "M")
            voice  = voice_set.get(gender, voice_set["F"])
            
            seg_duration = seg["end"] - seg["start"]
            estimated    = len(text) / 11.0
            speed        = max(0.8, min(1.25, estimated / seg_duration)) if seg_duration > 0 else 1.0

            try:
                out_file = os.path.join(output_dir, f"tts_{i:04d}.wav")
                samples, sample_rate = tts_m.create(text, voice=voice, speed=speed)
                
                if gender == "M" and voice.startswith("ff_"):
                    import librosa
                    samples = librosa.effects.pitch_shift(samples, sr=sample_rate, n_steps=-4)

                sf.write(out_file, samples, sample_rate)
                audio_segments.append({"file": out_file, "start": seg["start"], "end": seg["end"], "gender": gender})
            except Exception as e:
                tqdm.write(f"  ⚠️  Erreur Kokoro seg {i}: {e}")
            pbar.update(1)

    return audio_segments


def synthesize_melo(segments: list, output_dir: str, tgt_lang: str = "en") -> list:
    log(f"Synthèse MeloTTS ({tgt_lang})...", "🔊")
    
    try:
        from melo.api import TTS
        import soundfile as sf
    except ImportError:
        log("Installation MeloTTS (long)...", "📦")
        os.system(f"{sys.executable} -m pip install git+https://github.com/myshell-ai/MeloTTS.git --break-system-packages -q")
        # Melo a besoin de dépendances spécifiques et du dictionnaire unidic
        os.system(f"{sys.executable} -m pip install unidic-lite mecab-python3 unidic --break-system-packages -q")
        os.system(f"{sys.executable} -m unidic download")
        from melo.api import TTS
        import soundfile as sf

    # Melo supporte : EN, ES, FR, ZH, JP, KR
    melo_lang = tgt_lang.upper()
    if melo_lang not in ["EN", "ES", "FR", "ZH", "JP", "KR"]:
        melo_lang = "EN"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = TTS(language=melo_lang, device=device)
    speaker_ids = model.hps.data.spk2id
    # On prend le premier speaker dispo pour la langue (souvent un seul en FR)
    spk_id = speaker_ids.get(melo_lang, list(speaker_ids.values())[0])

    valid_segs = [(i, s) for i, s in enumerate(segments) if s.get("text", "").strip()]
    audio_segments = []

    with tqdm(total=len(valid_segs), desc="  🔊  MeloTTS", unit="seg") as pbar:
        for count, (i, seg) in enumerate(valid_segs, 1):
            text   = seg["text"].strip()
            gender = seg.get("gender", "M")
            
            seg_duration = seg["end"] - seg["start"]
            estimated    = len(text) / 12.0 # Melo est un peu plus rapide
            speed        = max(0.8, min(1.3, estimated / seg_duration)) if seg_duration > 0 else 1.0

            try:
                out_file = os.path.join(output_dir, f"tts_melo_{i:04d}.wav")
                # Melo sauve direct en fichier
                model.tts_to_file(text, spk_id, out_file, speed=speed)
                
                # Pitch shift pour l'homme si Melo n'a qu'une voix de femme (cas du FR)
                if gender == "M" and melo_lang == "FR":
                    import librosa
                    samples, sr = librosa.load(out_file, sr=None)
                    samples = librosa.effects.pitch_shift(samples, sr=sr, n_steps=-4)
                    sf.write(out_file, samples, sr)

                audio_segments.append({"file": out_file, "start": seg["start"], "end": seg["end"], "gender": gender})
            except Exception as e:
                tqdm.write(f"  ⚠️  Erreur Melo seg {i}: {e}")
            pbar.update(1)

    return audio_segments


def synthesize_edge(segments: list, output_dir: str, tgt_lang: str = "en") -> list:
    log(f"Synthèse Edge-TTS ({tgt_lang} - Internet)...", "🌐")
    
    try:
        import edge_tts
        import asyncio
    except ImportError:
        os.system(f"{sys.executable} -m pip install edge-tts --break-system-packages -q")
        import edge_tts
        import asyncio

    # Mappage voix Edge (Qualité Premium)
    # fr-FR-HenriNeural (M), fr-FR-DeniseNeural (F)
    voices = {
        "fr": {"M": "fr-FR-HenriNeural", "F": "fr-FR-DeniseNeural"},
        "en": {"M": "en-US-GuyNeural",   "F": "en-US-AriaNeural"},
        "zh": {"M": "zh-CN-YunxiNeural", "F": "zh-CN-XiaoxiaoNeural"},
    }
    voice_set = voices.get(tgt_lang, voices["en"])

    async def _gen(text, voice, file, rate):
        # rate format: "+0%", "-10%"
        r_str = f"{rate:+d}%" if isinstance(rate, int) else rate
        communicate = edge_tts.Communicate(text, voice, rate=r_str)
        await communicate.save(file)

    valid_segs = [(i, s) for i, s in enumerate(segments) if s.get("text", "").strip()]
    audio_segments = []

    with tqdm(total=len(valid_segs), desc="  🔊  Edge-TTS", unit="seg") as pbar:
        for count, (i, seg) in enumerate(valid_segs, 1):
            text   = seg["text"].strip()
            gender = seg.get("gender", "M")
            voice  = voice_set.get(gender, voice_set["F"])
            
            # Edge-TTS gère sa propre vitesse via 'rate'
            seg_duration = seg["end"] - seg["start"]
            estimated    = len(text) / 13.0
            rate_val     = int(( (estimated / seg_duration) - 1.0 ) * 100) if seg_duration > 0 else 0
            rate_val     = max(-20, min(30, rate_val))
            rate_str     = f"{rate_val:+d}%"

            try:
                out_file = os.path.join(output_dir, f"tts_edge_{i:04d}.mp3")
                asyncio.run(_gen(text, voice, out_file, rate_str))
                audio_segments.append({"file": out_file, "start": seg["start"], "end": seg["end"], "gender": gender})
            except Exception as e:
                tqdm.write(f"  ⚠️  Erreur Edge seg {i}: {e}")
            pbar.update(1)

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

    # Si aucun segment TTS : copier la vidéo originale sans audio modifié
    if len(audio_segments) == 0:
        log("Aucun segment TTS — copie de la vidéo originale", "⚠️")
        import shutil
        shutil.copy(video_path, output_path)
        return

    if len(audio_segments) > 150:
        log(f"{len(audio_segments)} segments → assemblage numpy...", "⚙️")
        _mux_numpy(audio_segments, total_duration, output_dir, video_path, output_path)
        return

    silence_path = os.path.join(output_dir, "silence.wav")
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "anullsrc=r=24000:cl=mono",
        "-t", str(total_duration), silence_path
    ], capture_output=True)
    result = subprocess.run(
        ["ffmpeg", "-y"] + inputs + ["-i", video_path]
        + ["-filter_complex", ";".join(filters)]
        + ["-map", f"{len(audio_segments)+1}:v"]
        + ["-map", prev]
        + ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
        + [output_path],
        capture_output=True, text=True
    )
    spinner.stop(result.returncode == 0)

    if result.returncode != 0:
        print(f"❌ ffmpeg erreur :\n{result.stderr[-2000:]}")
        sys.exit(1)

    log(f"Vidéo finale → {output_path} ({os.path.getsize(output_path)//1024**2} Mo)", "✅")


def _mux_numpy(audio_segments, total_duration, output_dir, video_path, output_path):
    import soundfile as sf
    import librosa

    SR    = 24000
    track = np.zeros(int(total_duration * SR) + SR, dtype=np.float32)

    with tqdm(total=len(audio_segments), desc="  🎛️  Assemblage audio",
              unit="seg", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        for seg in audio_segments:
            try:
                data, sr = sf.read(seg["file"], dtype="float32")
                if sr != SR:
                    data = librosa.resample(data, orig_sr=sr, target_sr=SR)
                start_i = int(seg["start"] * SR)
                end_i   = min(start_i + len(data), len(track))
                track[start_i:end_i] += data[:end_i - start_i]
            except Exception as e:
                tqdm.write(f"  ⚠️  {e}")
            pbar.update(1)

    peak = np.max(np.abs(track))
    if peak > 0:
        track = track / peak * 0.95

    merged_wav = os.path.join(output_dir, "merged_audio.wav")
    sf.write(merged_wav, track, SR)
    del track
    gc.collect()

    spinner = Spinner("Encodage ffmpeg final")
    spinner.start()
    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path, "-i", merged_wav,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", output_path
    ], capture_output=True, text=True)
    spinner.stop(result.returncode == 0)

    if result.returncode != 0:
        print(f"❌ ffmpeg erreur :\n{result.stderr[-2000:]}")
        sys.exit(1)

    log(f"Vidéo finale → {output_path} ({os.path.getsize(output_path)//1024**2} Mo)", "✅")


def mux_final(video_path: str, audio_segs: list, tmp_dir: str, output_path: str):
    """Mixe l'audio original (fond) avec les segments TTS (voix)."""
    log("Assemblage final (mixage voix + fond)...", "🎞️")
    if not audio_segs:
        log("Aucun segment audio à assembler.", "⚠️")
        return

    filter_script = os.path.join(tmp_dir, "filter_complex.txt")
    filter_parts  = []
    
    # [0:a] est l'audio de la vidéo d'entrée. On baisse son volume à 15% pour le fond.
    filter_parts.append("[0:a]volume=0.15[bg];")
    
    for i, seg in enumerate(audio_segs):
        # Les entrées audio commencent à l'index 1 (index 0 est la vidéo)
        start_ms = int(seg["start"] * 1000)
        filter_parts.append(f"[{i+1}:a]adelay={start_ms}|{start_ms}[a{i}];")
    
    labels = "".join([f"[a{i}]" for i in range(len(audio_segs))])
    filter_parts.append(f"[bg]{labels}amix=inputs={len(audio_segs)+1}:duration=first:dropout_transition=2[aout]")
    
    with open(filter_script, "w") as f:
        f.write("".join(filter_parts))

    cmd = ["ffmpeg", "-y", "-i", video_path]
    for seg in audio_segs:
        cmd.extend(["-i", seg["file"]])
        
    cmd.extend([
        "-filter_complex_script", filter_script,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", output_path
    ])
    
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        log(f"Erreur mixage : {result.stderr.decode()[-500:]}", "❌")
    else:
        log(f"Vidéo finale mixée → {output_path}", "✅")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    global AUDIO_CHUNK_S
    parser = argparse.ArgumentParser(description="Pipeline doublage vidéo local")
    parser.add_argument("--input",    required=True)
    parser.add_argument("--hf_token", required=False, default=None)
    parser.add_argument("--output",   default=None)
    parser.add_argument("--lang",     default=None)
    parser.add_argument("--keep_tmp",  action="store_true")
    parser.add_argument("--src-lang", default=None, dest="src_lang",
                        help="Langue source de la vidéo (ex: zh, ja, es, de). "
                             "Auto-détection si absent.")
    parser.add_argument("--tgt-lang", default=None, dest="tgt_lang",
                        help="Langue cible (ex: fr, en, de, ja, zh, ar...)")
    parser.add_argument("--tts-engine", default=TTS_ENGINE, choices=["kokoro", "melo", "edge"],
                        help="Moteur TTS : kokoro (local), melo (local), edge (internet, haute qualité)")
    parser.add_argument("--chunk", type=int, default=AUDIO_CHUNK_S,
                        help=f"Durée des chunks audio (défaut: {AUDIO_CHUNK_S}s). "
                             "Réduire si vous manquez de VRAM.")
    parser.add_argument("--sample", metavar="START:END", default=None,
                        help="Mode test : ex --sample 0:16 pour tester sur les 16 premières secondes")
    parser.add_argument("--ocr",       action="store_true",
                        help="Utiliser l'OCR (VideoSubFinder + PaddleOCR) au lieu de Whisper")
    parser.add_argument("--gpu_limit", type=float, default=0.5,
                        help="Limite de VRAM à utiliser (0.1 à 1.0). Défaut: 0.5 (50%)")
    args = parser.parse_args()
    AUDIO_CHUNK_S = args.chunk

    if not os.path.exists(args.input):
        print(f"❌ Fichier introuvable : {args.input}")
        sys.exit(1)

    # ── Mode échantillon ─────────────────────────────────
    if args.sample:
        try:
            start_s, end_s = [int(x) for x in args.sample.split(":")]
        except ValueError:
            print("❌ Format invalide. Utilise : --sample 0:16")
            import sys; sys.exit(1)
        sample_clip = f"_sample_{start_s}_{end_s}.mp4"
        print(f"\n🧪 Extraction échantillon {start_s}s → {end_s}s...")
        import subprocess
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(start_s), "-t", str(end_s - start_s),
            "-i", args.input, "-c", "copy", sample_clip
        ], capture_output=True)
        print(f"✅ Clip extrait → {sample_clip}")
        args.input  = sample_clip
        args.output = args.output or f"_sample_{start_s}_{end_s}_dubbed.mp4"
        print(f"👉 Sortie : {args.output}\n")

    output_path = args.output or os.path.join("output", Path(args.input).stem + "_dubbed.mp4")

    print("\n" + "="*60)
    print("  🎬 PIPELINE DOUBLAGE VIDÉO LOCAL")
    print("="*60)
    print(f"  Entrée  : {args.input}")
    print(f"  Sortie  : {output_path}")
    print(f"  Device  : {DEVICE.upper()}")
    print(f"  Whisper : {WHISPER_MODEL} | chunks {AUDIO_CHUNK_S}s | VRAM ≤{GPU_MARGIN*100:.0f}%")
    
    # Résoudre les langues tôt pour le header
    src_lang   = args.src_lang   # ex: "zh", "ja", None (auto)
    tgt_lang   = args.tgt_lang   # ex: "fr", "en", None (pas de traduction)
    
    v_m = "ff_siwis" if tgt_lang == "fr" else KOKORO_VOICE_M
    v_f = "ff_siwis" if tgt_lang == "fr" else KOKORO_VOICE_F
    
    print(f"  TTS ♂   : {v_m}  |  TTS ♀ : {v_f}")
    print(f"  Langue  : {src_lang or 'auto'} → {tgt_lang or '(aucune traduction)'}")
    print("="*60)

    # Résoudre les langues
    src_lang   = args.src_lang   # ex: "zh", "ja", None (auto)
    tgt_lang   = args.tgt_lang   # ex: "fr", "en", None (pas de traduction)
    do_translate = tgt_lang is not None and tgt_lang != src_lang

    if do_translate:
        log(f"Mode traduction : {src_lang or 'auto'} → {tgt_lang}", "🌐")
    elif src_lang:
        log(f"Transcription en langue source : {src_lang}", "📝")

    check_dependencies(check_ocr=args.ocr)
    limit_gpu_memory(args.gpu_limit)
    free_vram("démarrage")

    tmp_dir = tempfile.mkdtemp(prefix="dubbing_")
    log(f"Dossier temporaire : {tmp_dir}", "📁")

    try:
        # ── Étapes ──────────────────────────────
        audio_path = extract_audio(args.input, tmp_dir)
        
        # Whisper ou OCR
        if args.ocr:
            segments = extract_subtitles_ocr(args.input, tmp_dir, lang=args.lang or "fr")
            # Pour l'OCR, on peut optionnellement lancer une diarization si on veut 
            # mais ici on reste simple pour le moment.
        else:
            if not args.hf_token:
                log("Token HuggingFace requis pour la diarization (Whisper).", "❌")
                sys.exit(1)
            segments   = run_diarization(audio_path, args.hf_token)
            segments   = detect_gender(audio_path, segments)
            
            # Whisper : si traduction demandée, utiliser task=translate (→ anglais)
            # puis Helsinki traduit anglais → tgt_lang
            if do_translate:
                whisper_task = "translate"   # Whisper → anglais
                whisper_lang = src_lang      # langue source pour meilleure détection
            else:
                whisper_task = "transcribe"
                whisper_lang = src_lang or args.lang

            segments = transcribe(audio_path, whisper_lang, segments, tmp_dir,
                                  task=whisper_task)

        # Traduction anglais → langue cible (si tgt_lang != "en")
        if do_translate and tgt_lang != "en":
            segments = translate_segments(segments, tgt_lang=tgt_lang)
        elif do_translate and tgt_lang == "en":
            log("Cible = anglais, Whisper a déjà traduit ✓", "✅")

        print("\n📋 Aperçu transcription :")
        shown = 0
        for seg in segments:
            if seg.get("text") and shown < 10:
                g = "♀" if seg["gender"] == "F" else "♂"
                print(f"  [{seg['start']:.1f}s] {g} {seg['speaker']}: {seg['text'][:80]}")
                shown += 1

        audio_segs = synthesize_speech(segments, tmp_dir, 
                                       tgt_lang=tgt_lang or args.lang or "en",
                                       engine=args.tts_engine)
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