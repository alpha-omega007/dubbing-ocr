# ==========================================================================
# 🎬 local-dubbing-ui - E2E Integration API Test
# ==========================================================================

import os
import sys
import time
import requests
import json

BASE_URL = "http://127.0.0.1:8000"
VIDEO_PATH = "input/test_5s.mp4"

def run_test():
    print("🎬 Démarrage du test d'intégration API...")
    
    # 1. Check if server is reachable
    try:
        res = requests.get(f"{BASE_URL}/api/system-status")
        res.raise_for_status()
        print(f"✅ Serveur en ligne ! Métriques système: {res.json()}")
    except Exception as e:
        print(f"❌ Impossible de se connecter au serveur sur {BASE_URL} : {e}")
        sys.exit(1)
        
    # 2. Verify test video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Fichier vidéo de test introuvable : {VIDEO_PATH}")
        sys.exit(1)
        
    print(f"📁 Vidéo de test trouvée : {VIDEO_PATH} ({(os.path.getsize(VIDEO_PATH) / 1024):.1f} Ko)")
    
    # 3. Submit Dubbing Job (Upload Local File)
    print("\n🚀 Envoi de la requête de création de job (Upload + OCR + Edge TTS)...")
    
    form_data = {
        "source_type": "upload",
        "ocr": "true",
        "tts_engine": "edge",
        "chunk": "20",
        "gpu_limit": "0.5",
        "threads": "1",
        "skip_chunks": "0"
    }
    
    with open(VIDEO_PATH, "rb") as video_file:
        files = {
            "video_file": (os.path.basename(VIDEO_PATH), video_file, "video/mp4")
        }
        
        try:
            res = requests.post(f"{BASE_URL}/api/jobs", data=form_data, files=files)
            res.raise_for_status()
            job = res.json()
            job_id = job["id"]
            print(f"✅ Job créé avec succès ! ID: {job_id} | Status: {job['status']}")
        except Exception as e:
            print(f"❌ Échec de la création du job : {e}")
            if 'res' in locals():
                print(f"Réponse du serveur: {res.text}")
            sys.exit(1)
            
    # 4. Stream Logs and Progress via SSE
    print(f"\n🔌 Connexion au flux SSE pour suivre les logs du Job {job_id} en direct...")
    
    try:
        # requests.get with stream=True allows us to read line by line
        response = requests.get(f"{BASE_URL}/api/jobs/{job_id}/stream", stream=True)
        response.raise_for_status()
        
        current_event = None
        
        for line in response.iter_lines():
            if not line:
                continue
                
            line_str = line.decode('utf-8')
            
            # SSE parsing
            if line_str.startswith("event:"):
                current_event = line_str.split(":", 1)[1].strip()
            elif line_str.startswith("data:"):
                data_content = line_str.split(":", 1)[1].strip()
                
                if current_event == "progress":
                    progress = json.loads(data_content)
                    print(f"📊 [PROGRÈS] {progress['percentage']}% | Étape: {progress['current_step_desc']}")
                    current_event = None
                elif current_event == "job_complete":
                    job_result = json.loads(data_content)
                    print(f"\n🎉 [SUCCÈS] Le job s'est terminé avec succès en {job_result['processing_duration']:.1f}s !")
                    print(f"📹 Fichier généré: {job_result['output_filename']} ({(job_result['output_size_bytes']/1024/1024):.1f} Mo)")
                    break
                elif current_event == "job_failed":
                    job_result = json.loads(data_content)
                    print(f"\n❌ [ÉCHEC] Le job a échoué ! Message: {job_result.get('error_message')}")
                    sys.exit(1)
                else:
                    # Raw log line
                    # Strip timestamps to make it clean
                    print(f"📟 [CONSOLE] {data_content}")
                    
    except Exception as e:
        print(f"❌ Erreur durant le streaming des logs SSE : {e}")
        sys.exit(1)
        
    # 5. Verify Output File Exists
    output_path = f"output/{job_id}_dubbed.mp4"
    if os.path.exists(output_path):
        print(f"\n✅ VERIFICATION: Fichier de sortie réel trouvé à {output_path} ({(os.path.getsize(output_path)/1024/1024):.1f} Mo)")
    else:
        print(f"\n❌ VERIFICATION: Fichier de sortie introuvable à {output_path}")
        sys.exit(1)
        
    # 6. Verify Job History
    print("\n🔍 Interrogation de l'historique général...")
    try:
        history_res = requests.get(f"{BASE_URL}/api/jobs")
        history_res.raise_for_status()
        jobs = history_res.json()
        
        found = any(j["id"] == job_id for j in jobs)
        if found:
            print("✅ VERIFICATION: Le job apparaît bien dans l'historique persistent jobs.json !")
        else:
            print("❌ VERIFICATION: Job introuvable dans l'historique.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Échec de la récupération de l'historique : {e}")
        sys.exit(1)
        
    print("\n🌟 TEST D'INTÉGRATION END-TO-END VALIDÉ AVEC SUCCÈS ! L'APPLICATION EST 100% OPÉRATIONNELLE.")

if __name__ == "__main__":
    run_test()
