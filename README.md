# 🎬 Pipeline Doublage Vidéo Local

Téléchargement HLS + transcription + TTS genré (homme/femme) 100% local.

**Stack :** pyannote · Whisper · Kokoro · ffmpeg  
**GPU :** GTX 1650 Ti 4 Go VRAM (CUDA 12.x)

---

## 📁 Fichiers

| Fichier | Rôle |
|---|---|
| `full_pipeline.py` | Script maître (téléchargement + doublage) |
| `download_hls.py` | Téléchargement segments HLS (.ts) → MP4 |
| `video_dubbing_pipeline.py` | Doublage (diarization + TTS genré) |
| `install_dubbing.sh` | Installation des dépendances |

---

## ⚙️ Installation

### 1. Si pip est cassé, le réinstaller

Symptôme :
```
ModuleNotFoundError: No module named 'pip'
```

Fix :
```bash
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py --break-system-packages
```

> Utiliser ensuite `python3 -m pip install ...` à la place de `pip install ...`

### 2. PyTorch (CUDA 12.1 — GTX 1650 Ti)

```bash
python3 -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121 \
  --break-system-packages
```

### 3. Dépendances du pipeline

```bash
chmod +x install_dubbing.sh
./install_dubbing.sh
```

### 4. requests (pour le téléchargement HLS)

```bash
python3 -m pip install requests --break-system-packages
```

---

## 🔑 Prérequis Hugging Face (pyannote)

1. Créer un token sur https://huggingface.co/settings/tokens
2. Accepter les conditions sur ces deux pages :
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

---

## 🚀 Usage

### Pipeline complet (téléchargement HLS + doublage)

```bash
# Arrêter Ollama pour libérer la VRAM
sudo systemctl stop ollama

python3 full_pipeline.py \
  --url_template "https://gate-1562-an.vmeas.cloud/hls2/02/02057/kbo0seuf2sxn_n/seg-{i}-v1-a1.ts?t=TOKEN&s=...&e=43200&v=&srv=box-1500-u&i=0.4&sp=0&asn=12322" \
  --hf_token hf_XXXXXXXXXXXXXXX \
  --lang fr

# Résultat : video_dubbed.mp4
```

### Doublage seul (si vidéo déjà téléchargée)

```bash
python3 video_dubbing_pipeline.py \
  --input video_assembled.mp4 \
  --hf_token hf_XXXXXXXXXXXXXXX \
  --lang fr

# Résultat : video_assembled_dubbed.mp4
```

### Téléchargement seul (sans doublage)

```bash
python3 download_hls.py \
  --url_template "https://.../seg-{i}-v1-a1.ts?t=..." \
  --output video.mp4 \
  --threads 4

# Résultat : video.mp4
```

---

## 🔧 Options disponibles

### `full_pipeline.py`

| Option | Défaut | Description |
|---|---|---|
| `--url_template` | *(requis)* | URL avec `{i}` comme placeholder du numéro de segment |
| `--hf_token` | *(requis)* | Token Hugging Face |
| `--output` | `video_dubbed.mp4` | Fichier de sortie final |
| `--lang` | auto | Langue pour Whisper (`fr`, `en`, `es`...) |
| `--start` | `1` | Index du premier segment |
| `--threads` | `4` | Téléchargements parallèles |
| `--keep_tmp` | off | Garder les fichiers intermédiaires |

### `video_dubbing_pipeline.py`

| Option | Défaut | Description |
|---|---|---|
| `--input` | *(requis)* | Vidéo source |
| `--hf_token` | *(requis)* | Token Hugging Face |
| `--output` | `*_dubbed.mp4` | Fichier de sortie |
| `--lang` | auto | Langue Whisper |
| `--keep_tmp` | off | Garder les fichiers temporaires (utile pour debug) |

---

## 🔄 Ce qui se passe automatiquement

```
seg-1 → seg-2 → ... → 404 (5 échecs consécutifs = arrêt auto)
        ↓
  video_assembled.mp4
        ↓
  pyannote  →  qui parle à quel moment
  pitch F0  →  détection homme (< 165 Hz) / femme (≥ 165 Hz)
  Whisper   →  transcription avec timestamps
  Kokoro    →  synthèse voix ♂ (am_adam) / ♀ (af_sarah)
  ffmpeg    →  retire voix originale + insère TTS synchronisé
        ↓
  video_dubbed.mp4 ✅
```

---

## ⏱️ Temps estimé pour 1h de vidéo (GTX 1650 Ti)

| Étape | Durée estimée |
|---|---|
| Téléchargement HLS | ~5-15 min (réseau) |
| Diarization pyannote | ~15-20 min |
| Transcription Whisper medium | ~25-35 min |
| TTS Kokoro (CPU) | ~20-30 min |
| Muxing ffmpeg | ~2-3 min |
| **Total** | **~1h à 1h30** |

---

## ⚠️ Notes importantes

- Les URLs HLS sont signées et **expirent** (`e=43200` = 12h) — lancer le téléchargement rapidement
- Arrêter Ollama avant de lancer (`sudo systemctl stop ollama`) pour libérer 2 Go de VRAM
- Whisper `large-v3` est trop lourd pour 4 Go VRAM → on utilise `medium`
- Les modèles sont chargés **séquentiellement** (jamais deux en même temps) pour rester dans les 4 Go

---

## 🐛 Dépannage

### `ModuleNotFoundError: No module named 'pip'`
```bash
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py --break-system-packages
# Puis toujours utiliser :
python3 -m pip install ... --break-system-packages
```

### `ModuleNotFoundError: No module named 'torch'`
```bash
python3 -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121 \
  --break-system-packages
```

### VRAM insuffisante / CUDA out of memory
```bash
# Vérifier ce qui occupe la VRAM
nvidia-smi

# Arrêter Ollama
sudo systemctl stop ollama

# Relancer
python3 video_dubbing_pipeline.py --input video_assembled.mp4 --hf_token hf_XXX
```

### Segments HLS qui échouent tous (403 Forbidden)
Les URLs ont expiré (`e=43200` = 12h). Générer de nouvelles URLs et relancer.

### pyannote refusé (401 Unauthorized)
Token HF invalide ou conditions pas acceptées. Vérifier :
- https://huggingface.co/settings/tokens (token actif ?)
- https://huggingface.co/pyannote/speaker-diarization-3.1 (conditions acceptées ?)
- https://huggingface.co/pyannote/segmentation-3.0 (conditions acceptées ?)
