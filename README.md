# 🎬 Pipeline Doublage Vidéo Local

Téléchargement HLS + transcription + TTS genré (homme/femme) 100% local.

**Stack :** pyannote · Whisper · Kokoro · ffmpeg  
**GPU :** GTX 1650 Ti 4 Go VRAM (CUDA 12.x)

---

## 📁 Fichiers

| Fichier | Rôle |
|---|---|
| `main.py` | Nouveau point d'entrée modulaire (Clean Architecture) |
| `video_dubbing_pipeline.py` | Script monolithique optimisé (recommandé pour usage direct) |
| `download_hls.py` | Téléchargement segments HLS (.ts) → MP4 |
| `install_dubbing.sh` | Installation des dépendances |
| `tools/VideoSubFinder/` | Outil requis pour l'OCR (sous-titres incrustés) |

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
# 100% local avec Whisper (par défaut)
python3 video_dubbing_pipeline.py \
  --input video.mp4 \
  --hf_token hf_XXXXXXXXXXXXXXX \
  --lang fr

# Via OCR (pour les vidéos avec sous-titres incrustés, sans diarization)
python3 video_dubbing_pipeline.py --input video.mp4 --ocr --lang fr

# Avec traduction automatique (ex: Japonais -> Français)
python3 video_dubbing_pipeline.py --input video.mp4 --src-lang ja --tgt-lang fr --hf_token hf_XXX

# Doublage par morceaux (Séquentiel) - Recommandé pour les vidéos > 10 min
# Découpe la vidéo en blocs de 20s, les traite, puis les fusionne (très robuste)
python3 main.py --input video.mp4 --ocr --lang fr --chunk 20

# Limiter l'usage GPU (ex: 25%) et utiliser Edge TTS (Cloud haute qualité)
python3 main.py --input video.mp4 --gpu_limit 0.25 --tts-engine edge
```
python3 main.py --input input/episode9.mp4 --ocr --lang fr --chunk 30 --gpu_limit 0.25 --tts-engine edge


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

### `main.py` (Architecture Modulaire)

| Option | Défaut | Description |
|---|---|---|
| `--input` | *(requis)* | Vidéo source |
| `--hf_token` | None | Requis uniquement pour Diarization (pyannote) |
| `--output` | `output/...` | Dossier/Fichier de sortie |
| `--lang` | `fr` | Langue Whisper / OCR |
| `--tgt-lang` | None | Langue cible (active la traduction si présent) |
| `--ocr` | off | Utilise VideoSubFinder + EasyOCR (plus stable) |
| `--chunk` | `20` | **Séquentiel** : découpe en blocs de X secondes (recommandé: 20) |
| `--skip-chunks` | `0` | Saute les X premiers blocs (utile pour reprendre après un crash) |
| `--tts-engine` | `kokoro` | `kokoro` (local léger), `melo` (local pro), `edge` (cloud premium) |
| `--gpu_limit` | `0.5` | Limite de VRAM (0.1 à 1.0) |
| `--sample` | None | Test sur une plage (ex: `120:150`) |

---

## 🔄 Ce qui se passe automatiquement

```
seg-1 → seg-2 → ... → 404 (5 échecs consécutifs = arrêt auto)
        ↓
  video_assembled.mp4
        ↓
  pyannote / OCR  →  qui parle ou quel texte est affiché
  pitch F0        →  détection genre (si Whisper)
  Whisper / OCR   →  extraction du texte (transcription ou lecture)
  Helsinki-NLP    →  traduction (si --tgt-lang est utilisé)
  Kokoro/Melo/Edge→  synthèse vocale synchronisée
  ffmpeg          →  mixage audio (ducking) + assemblage final
        ↓
  video_dubbed.mp4 ✅
```

---

## ⏱️ Temps estimé pour 1h de vidéo (GTX 1650 Ti)

| Étape | Durée estimée |
|---|---|
| Téléchargement HLS | ~5-15 min (réseau) |
| Diarization pyannote | ~15-20 min |
| Transcription Whisper small | ~10-15 min |
| Traduction Helsinki | ~2-5 min |
| TTS Kokoro (CPU) | ~15-20 min |
| Muxing ffmpeg | ~2-3 min |
| **Total** | **~45 min à 1h** |

---

## ⚠️ Notes importantes

- Les URLs HLS sont signées et **expirent** (`e=43200` = 12h) — lancer le téléchargement rapidement
- **Brider le GPU** : Pour éviter que le PC ne rame ou ne chauffe, lancez `sudo nvidia-smi -pl 25` (limite la puissance à 25W).
- **VRAM** : Le script limite par défaut l'usage à 50% de la VRAM totale.
- **Modèles** : Whisper `small` est utilisé par défaut pour la vitesse. Utilisez `medium` dans le code pour plus de précision.

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
