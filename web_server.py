# ==========================================================================
# 🎬 local-dubbing-ui - FastAPI Web Server Backend
# ==========================================================================

import os
import sys
import uuid
import json
import time
import shutil
import asyncio
import threading
import subprocess
import psutil
from datetime import datetime
from typing import Optional, Dict, List
from collections import defaultdict

from fastapi import FastAPI, Form, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cache active async event loop for thread-safe cross-thread calls
    app.state.loop = asyncio.get_running_loop()
    yield

# Initialize FastAPI
app = FastAPI(
    title="Studio de Doublage IA Local",
    description="Interface Web moderne de contrôle pour le pipeline de doublage vidéo",
    lifespan=lifespan
)

# Enable CORS for convenience
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Workspace directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
JOBS_FILE = os.path.join(BASE_DIR, "jobs.json")

# Ensure required folders exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Mount Static & Output Files (Serves range requests for videos out of the box)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

# In-memory status & active listeners
JOBS_STATE: Dict[str, dict] = {}
JOB_LISTENERS = defaultdict(list)  # {job_id: [asyncio.Queue, ...]}

# Load & Save Jobs JSON Helper
def load_jobs_history():
    global JOBS_STATE
    if os.path.exists(JOBS_FILE):
        try:
            with open(JOBS_FILE, "r", encoding="utf-8") as f:
                JOBS_STATE = json.load(f)
        except Exception as e:
            print(f"⚠️ Impossible de charger {JOBS_FILE} : {e}")
            JOBS_STATE = {}
    else:
        JOBS_STATE = {}

def save_jobs_history():
    try:
        with open(JOBS_FILE, "w", encoding="utf-8") as f:
            json.dump(JOBS_STATE, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Impossible d'écrire dans {JOBS_FILE} : {e}")

# Load history at startup
load_jobs_history()

# Thread-safe log & progress broadcaster
def broadcast_event(job_id: str, event_type: str, data):
    """
    Sends live logs or structured progress updates to all SSE listeners
    of a specific job ID.
    """
    listeners = JOB_LISTENERS.get(job_id, [])
    if not listeners:
        return
        
    # Get current active async loop or write to queues in loop thread-safe
    for queue in list(listeners):
        try:
            asyncio.run_coroutine_threadsafe(
                queue.put((event_type, data)),
                app.state.loop
            )
        except Exception:
            pass

# Log writing helper
def log_to_file(job_id: str, message: str, level="info"):
    log_path = os.path.join(OUTPUT_DIR, f"job_{job_id}.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    
    # Write to local file
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(formatted_msg + "\n")
        
    # Broadcast to web terminal UI
    broadcast_event(job_id, "log", formatted_msg)

# Update Job Status Helper
def update_job_progress(job_id: str, percentage: int, active_step_index: int, step_desc: str, status="running"):
    if job_id not in JOBS_STATE:
        return
        
    JOBS_STATE[job_id]["status"] = status
    JOBS_STATE[job_id]["progress"] = {
        "percentage": percentage,
        "active_step_index": active_step_index,
        "current_step_desc": step_desc
    }
    
    # Save to history file
    save_jobs_history()
    
    # Broadcast status event to frontend
    broadcast_event(job_id, "progress", JOBS_STATE[job_id]["progress"])

# System status query
def get_system_metrics():
    metrics = {
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent,
        "gpu_available": False,
        "gpu_active": False,
        "gpu_name": "",
        "gpu_load": 0,
        "gpu_vram_used": 0,
        "gpu_vram_total": 0
    }
    
    try:
        # Run nvidia-smi to query name, load, and vram details
        res = subprocess.run([
            "nvidia-smi", 
            "--query-gpu=name,utilization.gpu,memory.used,memory.total", 
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, timeout=1)
        
        if res.returncode == 0:
            parts = res.stdout.strip().split(",")
            if len(parts) >= 4:
                metrics["gpu_available"] = True
                metrics["gpu_name"] = parts[0].strip()
                metrics["gpu_load"] = int(parts[1].strip())
                metrics["gpu_vram_used"] = int(parts[2].strip())
                metrics["gpu_vram_total"] = int(parts[3].strip())
                if metrics["gpu_load"] > 0 or metrics["gpu_vram_used"] > (0.1 * metrics["gpu_vram_total"]):
                    metrics["gpu_active"] = True
    except Exception:
        pass
        
    return metrics

# --------------------------------------------------------------------------
# Background Worker Thread: Download, Assemble & Dub Video
# --------------------------------------------------------------------------
def run_job_pipeline_thread(job_id: str, params: dict, source_type: str, video_source_path: str):
    """
    Orchestrator running in a separate thread.
    Chains HLS downloads if necessary, then invokes main.py inside a subprocess,
    captures stdout logs, maps pipeline progression steps, and saves the outputs.
    """
    start_time = time.time()
    input_file = os.path.join(INPUT_DIR, f"{job_id}_input.mp4")
    output_file = os.path.join(OUTPUT_DIR, f"{job_id}_dubbed.mp4")
    
    log_to_file(job_id, "🚀 Initialisation du Job de Doublage...", "info")
    
    try:
        # 1. DOWNLOAD PHASE (If remote link supplied)
        if source_type == "link":
            if "{i}" in video_source_path:
                # HLS segment template URL
                update_job_progress(job_id, 5, 0, "Téléchargement & assemblage HLS...")
                log_to_file(job_id, f"📥 URL HLS détectée. Lancement du script de téléchargement segmenté...", "info")
                
                hls_script = os.path.join(BASE_DIR, "scripts", "download_hls.py")
                cmd_dl = [
                    sys.executable, hls_script,
                    "--url_template", video_source_path,
                    "--output", input_file,
                    "--threads", "4"
                ]
                
                log_to_file(job_id, f"CMD: {' '.join(cmd_dl)}", "command")
                
                process_dl = subprocess.Popen(
                    cmd_dl,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                for line in iter(process_dl.stdout.readline, ''):
                    log_to_file(job_id, line.rstrip())
                    
                process_dl.wait()
                if process_dl.returncode != 0:
                    raise Exception(f"Le téléchargement HLS a échoué avec le code retour {process_dl.returncode}")
            else:
                # Direct HTTP Video Link
                update_job_progress(job_id, 5, 0, "Téléchargement direct du fichier vidéo...")
                log_to_file(job_id, f"📥 Téléchargement direct depuis l'URL : {video_source_path}", "info")
                
                import requests
                response = requests.get(video_source_path, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(input_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = int(5 + (downloaded / total_size) * 15) # maps to 5% - 20%
                                update_job_progress(job_id, percent, 0, f"Téléchargement : {percent}% ({downloaded // (1024*1024)} Mo)")
                                
                log_to_file(job_id, "✅ Fichier vidéo téléchargé avec succès !", "success")
        else:
            # File was already uploaded and placed in input_file
            log_to_file(job_id, "📁 Fichier importé localement, prêt à être traité.", "info")
            
        # Verify input file exists
        if not os.path.exists(input_file):
            raise Exception("Fichier vidéo source introuvable.")
            
        input_size_mb = os.path.getsize(input_file) / (1024*1024)
        log_to_file(job_id, f"🎬 Fichier Source : {input_file} ({input_size_mb:.1f} Mo)")
        
        # 2. MAIN DUBBING PIPELINE PHASE
        update_job_progress(job_id, 25, 1, "Démarrage du doublage (Clean Arch)...")
        
        main_script = os.path.join(BASE_DIR, "main.py")
        cmd_dub = [
            sys.executable, main_script,
            "--input", input_file,
            "--output", output_file,
            "--tts-engine", params.get("tts_engine", "kokoro"),
            "--chunk", str(params.get("chunk", 20)),
            "--gpu_limit", str(params.get("gpu_limit", 0.5)),
            "--threads", str(params.get("threads", 1)),
            "--skip-chunks", str(params.get("skip_chunks", 0))
        ]
        
        # Optional arguments
        if params.get("hf_token"):
            cmd_dub += ["--hf_token", params["hf_token"]]
        if params.get("lang"):
            cmd_dub += ["--lang", params["lang"]]
        if params.get("tgt_lang"):
            cmd_dub += ["--tgt-lang", params["tgt_lang"]]
        if params.get("ocr"):
            cmd_dub.append("--ocr")
        if params.get("sample"):
            cmd_dub += ["--sample", params["sample"]]
        if params.get("clone_voice"):
            cmd_dub.append("--clone-voice")
            
        # Log command
        # Obfuscate HF Token for safety
        cmd_logged = [x if not (params.get("hf_token") and x == params["hf_token"]) else "hf_xxxxxx" for x in cmd_dub]
        log_to_file(job_id, f"🚀 Commande du Pipeline : {' '.join(cmd_logged)}", "command")
        
        # Spawn execution process
        env = os.environ.copy()
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        
        process_dub = subprocess.Popen(
            cmd_dub,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        
        # Stream stdout line-by-line & dynamically estimate sub-step progression
        for line in iter(process_dub.stdout.readline, ''):
            stripped_line = line.rstrip()
            log_to_file(job_id, stripped_line)
            
            # Smart log-parsing for visual progress updates
            if "diarization" in stripped_line.lower() or "pyannote" in stripped_line.lower():
                update_job_progress(job_id, 35, 1, "Diarization pyannote (Identification voix)...")
            elif "transcription" in stripped_line.lower() or "whisper" in stripped_line.lower():
                update_job_progress(job_id, 50, 2, "Transcription Whisper (Extraction texte)...")
            elif "easyocr" in stripped_line.lower() or "videosubfinder" in stripped_line.lower():
                update_job_progress(job_id, 50, 2, "Extraction OCR (Lecture sous-titres incrustés)...")
            elif "traduction" in stripped_line.lower() or "helsinki" in stripped_line.lower():
                update_job_progress(job_id, 60, 2, "Traduction IA du texte...")
            elif "tts" in stripped_line.lower() or "synthèse" in stripped_line.lower() or "edge" in stripped_line.lower() or "kokoro" in stripped_line.lower() or "melo" in stripped_line.lower():
                update_job_progress(job_id, 75, 3, "Synthèse vocale synchronisée (TTS)...")
            elif "bloc" in stripped_line.lower() or "chunk" in stripped_line.lower():
                # Extract block number if present to show progression
                update_job_progress(job_id, 80, 3, f"Synthèse Vocale : {stripped_line}")
            elif "fusion" in stripped_line.lower() or "concat" in stripped_line.lower() or "ffmpeg" in stripped_line.lower() or "mixage" in stripped_line.lower():
                update_job_progress(job_id, 92, 4, "Mixage audio final & assemblage vidéo...")
                
        process_dub.wait()
        
        if process_dub.returncode != 0:
            raise Exception(f"Le pipeline de doublage a crashé avec le code retour {process_dub.returncode}")
            
        # Verify output exists
        if not os.path.exists(output_file):
            raise Exception("Le pipeline s'est terminé sans générer de vidéo de sortie.")
            
        # Complete
        duration = time.time() - start_time
        out_size = os.path.getsize(output_file)
        
        log_to_file(job_id, f"🎉 DOUBLAGE COMPLÉTÉ EN {duration:.1f} SECONDES !", "success")
        log_to_file(job_id, f"📹 Vidéo finale générée : {output_file} ({(out_size / (1024*1024)):.1f} Mo)")
        
        # Clean temporary input file
        if os.path.exists(input_file):
            os.remove(input_file)
            log_to_file(job_id, "🧹 Fichier intermédiaire d'entrée supprimé.", "info")
            
        # Update metadata state
        JOBS_STATE[job_id]["status"] = "completed"
        JOBS_STATE[job_id]["processing_duration"] = duration
        JOBS_STATE[job_id]["output_filename"] = f"{job_id}_dubbed.mp4"
        JOBS_STATE[job_id]["output_size_bytes"] = out_size
        save_jobs_history()
        
        # Broadcast structured completion event
        broadcast_event(job_id, "complete", JOBS_STATE[job_id])
        
    except Exception as e:
        # Error handling
        duration = time.time() - start_time
        err_msg = str(e)
        log_to_file(job_id, f"❌ ERREUR CRITIQUE : {err_msg}", "error")
        
        # Attempt cleanup of intermediate file
        if os.path.exists(input_file):
            try: os.remove(input_file)
            except Exception: pass
            
        # Update metadata state
        JOBS_STATE[job_id]["status"] = "failed"
        JOBS_STATE[job_id]["processing_duration"] = duration
        JOBS_STATE[job_id]["error_message"] = err_msg
        save_jobs_history()
        
        # Broadcast failure event
        broadcast_event(job_id, "failed", JOBS_STATE[job_id])

# --------------------------------------------------------------------------
# REST API Endpoints
# --------------------------------------------------------------------------

# Serve Web UI SPA
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Index template introuvable")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# Query live system resources (CPU, GPU details)
@app.get("/api/system-status")
async def system_status():
    return get_system_metrics()

# List past job history
@app.get("/api/jobs")
async def get_jobs_list():
    load_jobs_history()
    return list(JOBS_STATE.values())

# Retrieve single job metadata
@app.get("/api/jobs/{job_id}")
async def get_job_details(job_id: str):
    if job_id not in JOBS_STATE:
        raise HTTPException(status_code=404, detail="Job introuvable")
    return JOBS_STATE[job_id]

# Delete a single job from history
@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id not in JOBS_STATE:
        raise HTTPException(status_code=404, detail="Job introuvable")
        
    # Remove metadata
    del JOBS_STATE[job_id]
    save_jobs_history()
    
    # Delete associated log file
    log_path = os.path.join(OUTPUT_DIR, f"job_{job_id}.log")
    if os.path.exists(log_path):
        try: os.remove(log_path)
        except Exception: pass
        
    return {"status": "success", "message": "Tâche supprimée de l'historique"}

# Fetch archive text logs of a completed/failed job
@app.get("/api/jobs/{job_id}/logs")
async def get_job_logs_text(job_id: str):
    log_path = os.path.join(OUTPUT_DIR, f"job_{job_id}.log")
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Logs indisponibles")
    return FileResponse(log_path, media_type="text/plain")

# Create and execute a new dubbing job
@app.post("/api/jobs")
async def create_new_job(
    source_type: str = Form(...), # "upload" or "link"
    video_file: Optional[UploadFile] = File(None),
    video_url: Optional[str] = Form(None),
    hf_token: Optional[str] = Form(None),
    lang: Optional[str] = Form("fr"),
    tgt_lang: Optional[str] = Form(None),
    ocr: bool = Form(False),
    tts_engine: str = Form("kokoro"),
    chunk: int = Form(20),
    gpu_limit: float = Form(0.5),
    threads: int = Form(1),
    skip_chunks: int = Form(0),
    sample: Optional[str] = Form(None),
    clone_voice: bool = Form(False)
):
    # Basic Validations
    if source_type == "upload":
        if not video_file:
            raise HTTPException(status_code=400, detail="Fichier vidéo d'upload manquant")
        video_name = video_file.filename
    else:
        if not video_url or not video_url.strip():
            raise HTTPException(status_code=400, detail="Lien URL vidéo manquant")
        # Extract filename from URL
        video_name = video_url.strip().split("/")[-1].split("?")[0]
        if not video_name:
            video_name = "flux_video.mp4"
            
    # Generate unique ID
    job_id = str(uuid.uuid4())[:8] + "_" + datetime.now().strftime("%H%M%S")
    
    # Save uploaded file immediately if upload
    if source_type == "upload" and video_file:
        input_target = os.path.join(INPUT_DIR, f"{job_id}_input.mp4")
        with open(input_target, "wb") as f:
            # Stream in chunks of 1MB to avoid locking memory
            while chunk_bytes := await video_file.read(1024*1024):
                f.write(chunk_bytes)
                
    # Prepare parameters map
    params = {
        "hf_token": hf_token if hf_token and hf_token.strip() else None,
        "lang": lang if lang and lang.strip() else None,
        "tgt_lang": tgt_lang if tgt_lang and tgt_lang.strip() else None,
        "ocr": ocr,
        "tts_engine": tts_engine,
        "chunk": chunk,
        "gpu_limit": gpu_limit,
        "threads": threads,
        "skip_chunks": skip_chunks,
        "sample": sample if sample and sample.strip() else None,
        "clone_voice": clone_voice
    }
    
    # Store initial state
    JOBS_STATE[job_id] = {
        "id": job_id,
        "input_name": video_name,
        "source_type": source_type,
        "status": "running",
        "created_at": datetime.now().isoformat(),
        "params": params,
        "progress": {
            "percentage": 0,
            "active_step_index": 0,
            "current_step_desc": "Initialisation..."
        }
    }
    save_jobs_history()
    
    # Spawn pipeline in background thread
    video_src = video_url if source_type == "link" else ""
    thread = threading.Thread(
        target=run_job_pipeline_thread,
        args=(job_id, params, source_type, video_src)
    )
    thread.daemon = True
    thread.start()
    
    return JOBS_STATE[job_id]

# --------------------------------------------------------------------------
# Server-Sent Events (SSE) Live log-streaming Endpoint
# --------------------------------------------------------------------------
@app.get("/api/jobs/{job_id}/stream")
async def stream_job_live_logs(job_id: str):
    """
    Server-Sent Events streaming pipeline.
    Connects client browser, pushes retroactively generated log files instantly,
    and listens on an active async queue for real-time subprocess pipe logs
    and progress changes.
    """
    if job_id not in JOBS_STATE:
        raise HTTPException(status_code=404, detail="Job non trouvé")
        
    async def sse_event_generator():
        # Register a new queue for this client connection
        queue = asyncio.Queue()
        JOB_LISTENERS[job_id].append(queue)
        
        try:
            # 1. PUSH LOG BACKLOG: If log file already contains text, push it immediately
            log_path = os.path.join(OUTPUT_DIR, f"job_{job_id}.log")
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        yield f"data: {line.rstrip()}\n\n"
                        
            # 2. PUSH LIVE SSE EVENTS: Loop and stream new items from background worker
            while True:
                try:
                    # Wait for items on queue (Non-blocking with short timeout for keepalive)
                    item = await asyncio.wait_for(queue.get(), timeout=1.5)
                    event_type, payload = item
                    
                    if event_type == "log":
                        yield f"data: {payload}\n\n"
                    elif event_type == "progress":
                        yield f"event: progress\ndata: {json.dumps(payload)}\n\n"
                    elif event_type == "complete":
                        yield f"event: job_complete\ndata: {json.dumps(payload)}\n\n"
                        break
                    elif event_type == "failed":
                        yield f"event: job_failed\ndata: {json.dumps(payload)}\n\n"
                        break
                except asyncio.TimeoutError:
                    # Connection Keep-Alive comment
                    yield ": keep-alive\n\n"
                    # Safe check: if job terminated and queue is empty, close connection
                    if JOBS_STATE[job_id]["status"] in ["completed", "failed"] and queue.empty():
                        break
        finally:
            # Clean up queue when client connection closes
            if job_id in JOB_LISTENERS:
                if queue in JOB_LISTENERS[job_id]:
                    JOB_LISTENERS[job_id].remove(queue)
                    
    return StreamingResponse(sse_event_generator(), media_type="text/event-stream")

# --------------------------------------------------------------------------
# FastAPI Server Started Successfully
# --------------------------------------------------------------------------
