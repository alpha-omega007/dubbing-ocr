#!/bin/bash
# ============================================
# Installation dépendances pipeline doublage
# Ubuntu 22.04 — GTX 1650 Ti — CUDA 12.2
# ============================================

echo "📦 Installation des dépendances..."

# 1. ffmpeg (extraction + muxing)
sudo apt-get install -y ffmpeg

# 2. Whisper (transcription)
pip install openai-whisper --break-system-packages

# 3. pyannote (diarization)
pip install pyannote-audio>=3.1.0 --break-system-packages

# 4. Analyse pitch
pip install praat-parselmouth librosa --break-system-packages

# 5. Audio I/O
pip install soundfile --break-system-packages

# 6. Kokoro TTS
pip install kokoro-onnx --break-system-packages

# 7. Télécharger les modèles Kokoro (ONNX)
echo ""
echo "📥 Téléchargement des modèles Kokoro..."
python3 -c "
import urllib.request, os

models = [
    ('https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v1.9.onnx', 'kokoro-v1.9.onnx'),
    ('https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices-v1.0.bin', 'voices-v1.0.bin'),
]

for url, filename in models:
    if os.path.exists(filename):
        print(f'  ✅ {filename} déjà présent')
        continue
    print(f'  ⬇️  Téléchargement {filename}...')
    urllib.request.urlretrieve(url, filename)
    print(f'  ✅ {filename} téléchargé')
"

echo ""
echo "✅ Installation terminée !"
echo ""
echo "👉 Usage :"
echo "   python3 video_dubbing_pipeline.py --input ta_video.mp4 --hf_token TON_TOKEN_HF"
echo ""
echo "⚠️  Pense à arrêter Ollama avant de lancer le pipeline :"
echo "   sudo systemctl stop ollama"
echo ""
echo "🔑 Token HF à récupérer sur : https://huggingface.co/settings/tokens"
echo "   Et accepter les conditions pyannote sur :"
echo "   https://huggingface.co/pyannote/speaker-diarization-3.1"
