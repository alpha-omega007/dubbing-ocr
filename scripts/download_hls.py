#!/usr/bin/env python3
"""
Téléchargeur de segments HLS (.ts) avec assemblage en MP4
Incrémente seg-<i> jusqu'à 404/erreur

Usage :
  python3 download_hls.py --url_template "https://gate-.../seg-{i}-v1-a1.ts?t=..." --output video.mp4
  python3 download_hls.py --url_template "..." --output video.mp4 --start 1 --threads 4
"""

import os
import sys
import time
import argparse
import subprocess
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import requests
except ImportError:
    print("❌ requests manquant : pip install requests --break-system-packages")
    sys.exit(1)


def log(msg, emoji="→"):
    print(f"{emoji}  {msg}", flush=True)


def download_segment(args):
    """Télécharge un segment, retourne (index, path) ou (index, None) si échec."""
    i, url_template, tmp_dir, timeout = args
    url  = url_template.replace("{i}", str(i))
    path = os.path.join(tmp_dir, f"seg_{i:06d}.ts")

    try:
        r = requests.get(url, timeout=timeout, stream=True)
        if r.status_code == 404:
            return (i, None, "404")
        if r.status_code != 200:
            return (i, None, f"HTTP {r.status_code}")

        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)

        size = os.path.getsize(path)
        if size < 100:  # fichier vide / invalide
            return (i, None, "empty")

        return (i, path, "ok")

    except requests.exceptions.Timeout:
        return (i, None, "timeout")
    except Exception as e:
        return (i, None, str(e))


def download_all_segments(url_template: str, tmp_dir: str, start: int = 1,
                          threads: int = 4, timeout: int = 30,
                          max_consecutive_fails: int = 5) -> list:
    """
    Télécharge tous les segments en incrémentant i.
    S'arrête après max_consecutive_fails échecs consécutifs.
    """
    log(f"Démarrage du téléchargement depuis seg-{start}...", "⬇️")
    log(f"Threads parallèles : {threads}", "⚙️")

    segments   = {}
    i          = start
    batch_size = threads * 3  # télécharger par batch
    consecutive_fails = 0
    total_downloaded  = 0

    while True:
        # préparer un batch
        batch = [(j, url_template, tmp_dir, timeout) for j in range(i, i + batch_size)]

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {executor.submit(download_segment, args): args[0] for args in batch}
            results = {}
            for future in as_completed(futures):
                idx, path, status = future.result()
                results[idx] = (path, status)

        # traiter les résultats dans l'ordre
        batch_had_success = False
        for j in range(i, i + batch_size):
            path, status = results.get(j, (None, "missing"))

            if path and status == "ok":
                segments[j] = path
                total_downloaded += 1
                batch_had_success = True
                consecutive_fails = 0
                print(f"  ✅ seg-{j} ({os.path.getsize(path) // 1024} Ko)", flush=True)
            else:
                consecutive_fails += 1
                print(f"  ⏭️  seg-{j} → {status} (échecs consécutifs: {consecutive_fails})", flush=True)
                if consecutive_fails >= max_consecutive_fails:
                    log(f"Arrêt : {max_consecutive_fails} échecs consécutifs — dernier segment valide : seg-{j - consecutive_fails}", "🛑")
                    # retourner les segments téléchargés jusqu'ici
                    return [segments[k] for k in sorted(segments.keys())]

        i += batch_size
        log(f"Progression : {total_downloaded} segments téléchargés...", "📊")

    return [segments[k] for k in sorted(segments.keys())]


def assemble_mp4(segment_paths: list, output_path: str, tmp_dir: str):
    """Assemble les segments .ts en un seul MP4 via ffmpeg concat."""
    log(f"Assemblage de {len(segment_paths)} segments en MP4...", "🎬")

    # créer le fichier de liste pour ffmpeg concat
    concat_list = os.path.join(tmp_dir, "concat.txt")
    with open(concat_list, "w") as f:
        for path in segment_paths:
            # ffmpeg concat demande des chemins absolus escapés
            escaped = path.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list,
        "-c", "copy",       # pas de re-encodage
        output_path
    ]

    log("Encodage ffmpeg en cours...", "⏳")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ ffmpeg erreur :\n{result.stderr[-2000:]}")
        sys.exit(1)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log(f"MP4 assemblé → {output_path} ({size_mb:.1f} Mo)", "✅")


def main():
    parser = argparse.ArgumentParser(description="Téléchargeur HLS segments")
    parser.add_argument("--url_template", required=True,
                        help='URL avec {i} comme placeholder. Ex: "https://.../seg-{i}-v1-a1.ts?t=..."')
    parser.add_argument("--output",   default="video_assembled.mp4", help="Fichier MP4 de sortie")
    parser.add_argument("--start",    type=int, default=1,  help="Index de départ (défaut: 1)")
    parser.add_argument("--threads",  type=int, default=4,  help="Téléchargements parallèles (défaut: 4)")
    parser.add_argument("--timeout",  type=int, default=30, help="Timeout par segment en secondes")
    parser.add_argument("--max_fails",type=int, default=5,  help="Échecs consécutifs avant arrêt")
    parser.add_argument("--keep_tmp", action="store_true",  help="Garder les segments .ts")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  ⬇️  TÉLÉCHARGEUR HLS SEGMENTS")
    print("="*60)
    print(f"  Template : {args.url_template[:80]}...")
    print(f"  Sortie   : {args.output}")
    print(f"  Threads  : {args.threads}")
    print("="*60 + "\n")

    tmp_dir = tempfile.mkdtemp(prefix="hls_download_")

    try:
        # Téléchargement
        segment_paths = download_all_segments(
            url_template       = args.url_template,
            tmp_dir            = tmp_dir,
            start              = args.start,
            threads            = args.threads,
            timeout            = args.timeout,
            max_consecutive_fails = args.max_fails,
        )

        if not segment_paths:
            print("❌ Aucun segment téléchargé.")
            sys.exit(1)

        log(f"Total : {len(segment_paths)} segments téléchargés", "📦")

        # Assemblage
        assemble_mp4(segment_paths, args.output, tmp_dir)

        print("\n" + "="*60)
        print("  ✅ TÉLÉCHARGEMENT TERMINÉ")
        print(f"  📹 Fichier : {args.output}")
        print("="*60 + "\n")

    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            log(f"Segments conservés dans : {tmp_dir}", "📁")


if __name__ == "__main__":
    main()
