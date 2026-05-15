import os
import argparse
import tempfile
import shutil
import subprocess
from core.use_cases.dubbing_pipeline import DubbingPipeline
from infrastructure.audio.ffmpeg_service import FFmpegAudioService
from infrastructure.transcription.whisper_engine import WhisperTranscriptionEngine
from infrastructure.ocr.easyocr_engine import EasyOCREngine
from infrastructure.translation.helsinki_engine import HelsinkiTranslationEngine
from infrastructure.tts.kokoro_engine import KokoroTTSEngine
from infrastructure.diarization.pyannote_service import PyannoteDiarizationService

import torch
import sys

def limit_gpu_memory(fraction: float):
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)
        total = torch.cuda.mem_get_info()[1] // 1024**2
        print(f"⚙️ VRAM limitée à {fraction*100:.0f}% = {int(total * fraction)} Mo / {total} Mo")
        if fraction < 0.6:
            print(f"  💡 Tip: Pour limiter aussi la chauffe, lance : sudo nvidia-smi -pl 25")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--lang", default="fr")
    parser.add_argument("--tgt-lang", default=None)
    parser.add_argument("--output", default="output/output_dubbed.mp4")
    parser.add_argument("--chunk", type=int, default=20, help="Durée des chunks en secondes")
    parser.add_argument("--gpu_limit", type=float, default=0.5, help="Limite VRAM (0.1 à 1.0)")
    parser.add_argument("--sample", metavar="START:END", default=None, help="Mode test : ex --sample 0:16")
    parser.add_argument("--tts-engine", default="kokoro", choices=["kokoro", "edge", "melo"], help="Moteur TTS")
    parser.add_argument("--skip-chunks", "--skip_chunks", type=int, default=0, help="Nombre de blocs à sauter au début")
    parser.add_argument("--threads", type=int, default=1, help="Nombre de chunks à traiter en parallèle (recommandé: 1 ou 2)")
    args = parser.parse_args()
    limit_gpu_memory(args.gpu_limit)

    # Dependency Injection
    audio_service = FFmpegAudioService()
    transcription_engine = WhisperTranscriptionEngine()
    ocr_engine = EasyOCREngine()
    translation_engine = HelsinkiTranslationEngine()
    
    # TTS Selection
    if args.tts_engine == "edge":
        from infrastructure.tts.edge_engine import EdgeTTSEngine
        tts_engine = EdgeTTSEngine()
    elif args.tts_engine == "melo":
        from infrastructure.tts.melo_engine import MeloTTSEngine
        tts_engine = MeloTTSEngine()
    else:
        from infrastructure.tts.kokoro_engine import KokoroTTSEngine
        onnx_path = "./models/kokoro-v1.0.onnx"
        voices_path = "./models/voices-v1.0.bin"
        tts_engine = KokoroTTSEngine(onnx_path, voices_path)
    
    diarization_service = PyannoteDiarizationService()

    # Orchestrator
    pipeline = DubbingPipeline(
        audio_service=audio_service,
        transcription_engine=transcription_engine,
        ocr_engine=ocr_engine,
        translation_engine=translation_engine,
        tts_engine=tts_engine,
        diarization_service=diarization_service
    )

    tmp_dir = tempfile.mkdtemp(prefix="dub_clean_")
    video_to_process = args.input

    try:
        if args.sample:
            # Mode Échantillon Unique
            start_s, end_s = args.sample.split(':')
            duration = float(end_s) - float(start_s)
            sample_path = os.path.join(tmp_dir, f"sample_{start_s}_{end_s}.mp4")
            print(f"🧪 Extraction échantillon {start_s}s → {end_s}s...")
            subprocess.run([
                "ffmpeg", "-y", "-ss", start_s, "-i", args.input,
                "-t", str(duration), "-c", "copy", sample_path
            ], capture_output=True)
            
            pipeline.run(
                video_path=sample_path,
                tmp_dir=tmp_dir,
                hf_token=args.hf_token,
                src_lang=args.lang,
                tgt_lang=args.tgt_lang,
                use_ocr=args.ocr,
                output_path=args.output,
                chunk_duration=args.chunk
            )
        elif args.chunk:
            # Mode Séquentiel Parallèle (Découpe en morceaux + Fusion)
            from concurrent.futures import ThreadPoolExecutor
            total_duration = audio_service.get_duration(args.input)
            print(f"🎬 Traitement séquentiel ({args.threads} threads) par morceaux de {args.chunk}s")
            
            chunk_indices = [i for i in range(len(range(0, int(total_duration), args.chunk)))]
            chunk_outputs = [None] * len(chunk_indices)

            def process_chunk(i):
                start = i * args.chunk
                if i < args.skip_chunks:
                    print(f"⏭️ Saut du bloc {i+1}")
                    return None
                    
                end = min(start + args.chunk, total_duration)
                chunk_tmp = os.path.join(tmp_dir, f"chunk_{i}")
                os.makedirs(chunk_tmp, exist_ok=True)
                
                clip_in = os.path.join(chunk_tmp, f"in_{i}.mp4")
                clip_out = os.path.join(chunk_tmp, f"out_{i}.mp4")
                
                print(f"📦 Bloc {i+1} : {start}s → {end}s")
                # Extraire le clip
                subprocess.run([
                    "ffmpeg", "-y", "-ss", str(start), "-i", args.input,
                    "-t", str(end-start), "-c", "copy", clip_in
                ], capture_output=True)
                
                # Doubler le clip
                pipeline.run(
                    video_path=clip_in,
                    tmp_dir=chunk_tmp,
                    hf_token=args.hf_token,
                    src_lang=args.lang,
                    tgt_lang=args.tgt_lang,
                    use_ocr=args.ocr,
                    output_path=clip_out,
                    chunk_duration=args.chunk
                )
                return clip_out

            with ThreadPoolExecutor(max_workers=args.threads) as executor:
                results = list(executor.map(process_chunk, chunk_indices))
            
            chunk_outputs = [r for r in results if r is not None]
            
            # Fusion finale
            print("\n🔄 Fusion finale des morceaux...")
            concat_file = os.path.join(tmp_dir, "concat.txt")
            with open(concat_file, "w") as f:
                for path in chunk_outputs:
                    f.write(f"file '{os.path.abspath(path)}'\n")
            
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
                "-c", "copy", args.output
            ], capture_output=True)
            print(f"✅ Vidéo complète générée : {args.output}")
        else:
            # Mode Standard (Normalement déconseillé pour les longues vidéos)
            pipeline.run(
                video_path=args.input,
                tmp_dir=tmp_dir,
                hf_token=args.hf_token,
                src_lang=args.lang,
                tgt_lang=args.tgt_lang,
                use_ocr=args.ocr,
                output_path=args.output,
                chunk_duration=20
            )
    finally:
        shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    main()
