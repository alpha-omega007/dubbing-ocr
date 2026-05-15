#!/usr/bin/env python3
"""
Pipeline complet : Téléchargement HLS → Assemblage MP4 → Doublage
Usage :
  python3 full_pipeline.py \
    --url_template "https://gate-.../seg-{i}-v1-a1.ts?t=TOKEN&s=...&e=...&srv=...&i=0.4&sp=0&asn=12322" \
    --hf_token hf_XXXXXXXXX \
    --lang fr
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path


def run_step(cmd: list, step_name: str):
    print(f"\n{'='*60}")
    print(f"  🚀 {step_name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n❌ Échec à l'étape : {step_name}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Pipeline HLS → MP4 → Doublage")
    parser.add_argument("--url_template", required=True,
                        help='URL template avec {i}. Ex: "https://.../seg-{i}-v1-a1.ts?t=..."')
    parser.add_argument("--hf_token",  required=True,  help="Token Hugging Face pour pyannote")
    parser.add_argument("--output",    default=None,   help="Fichier final (défaut: video_dubbed.mp4)")
    parser.add_argument("--lang",      default=None,   help="Langue Whisper (ex: fr, en). Auto si absent")
    parser.add_argument("--start",     type=int, default=1, help="Index premier segment (défaut: 1)")
    parser.add_argument("--threads",   type=int, default=4, help="Téléchargements parallèles")
    parser.add_argument("--keep_tmp",  action="store_true", help="Garder les fichiers intermédiaires")
    args = parser.parse_args()

    # chemins
    script_dir   = Path(__file__).parent
    assembled    = "video_assembled.mp4"
    output_final = args.output or "video_dubbed.mp4"

    print("\n" + "="*60)
    print("  🎬 PIPELINE COMPLET : HLS → MP4 → DOUBLAGE")
    print("="*60)
    print(f"  Sortie finale : {output_final}")
    print(f"  Langue        : {args.lang or 'auto-détection'}")
    print("="*60)

    # ── Étape 1 : téléchargement + assemblage ──
    dl_cmd = [
        sys.executable, str(script_dir / "download_hls.py"),
        "--url_template", args.url_template,
        "--output",       assembled,
        "--start",        str(args.start),
        "--threads",      str(args.threads),
    ]
    if args.keep_tmp:
        dl_cmd.append("--keep_tmp")

    run_step(dl_cmd, "Téléchargement HLS + Assemblage MP4")

    if not os.path.exists(assembled):
        print(f"❌ Fichier assemblé introuvable : {assembled}")
        sys.exit(1)

    size_mb = os.path.getsize(assembled) / (1024 * 1024)
    print(f"\n✅ Vidéo assemblée : {assembled} ({size_mb:.1f} Mo)")

    # ── Étape 2 : doublage ──
    dub_cmd = [
        sys.executable, str(script_dir / "video_dubbing_pipeline.py"),
        "--input",     assembled,
        "--hf_token",  args.hf_token,
        "--output",    output_final,
    ]
    if args.lang:
        dub_cmd += ["--lang", args.lang]
    if args.keep_tmp:
        dub_cmd.append("--keep_tmp")

    run_step(dub_cmd, "Doublage (diarization + TTS genré)")

    print("\n" + "="*60)
    print("  ✅ PIPELINE COMPLET TERMINÉ")
    print(f"  📹 Résultat final : {output_final}")
    print("="*60 + "\n")

    # optionnel : supprimer le MP4 intermédiaire
    if not args.keep_tmp and os.path.exists(assembled):
        os.remove(assembled)
        print(f"🧹 Intermédiaire supprimé : {assembled}")


if __name__ == "__main__":
    main()
