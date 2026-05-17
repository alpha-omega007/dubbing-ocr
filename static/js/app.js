/* ==========================================================================
   🎬 local-dubbing-ui - Interactive JS Controller
   ========================================================================== */

// Global State
let selectedFile = null;
let currentSourceTab = 'upload';
let activeSSE = null;
let activeJobId = null;

// Initialize when DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
  // Load HF Token if stored
  const savedToken = localStorage.getItem('hf_token');
  if (savedToken) {
    document.getElementById('hf-token').value = savedToken;
    document.getElementById('remember-token').checked = true;
  }

  // Setup Drag & Drop Listeners
  const dropzone = document.getElementById('dropzone');
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, preventDefaults, false);
  });
  
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, () => dropzone.classList.add('dragover'), false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, () => dropzone.classList.remove('dragover'), false);
  });

  dropzone.addEventListener('drop', handleFileDrop, false);

  // Load Job History & System Status immediately
  loadJobHistory();
  pollSystemStatus();
  setInterval(pollSystemStatus, 5000); // Poll every 5s
});

// Prevent browser defaults for Drag & Drop
function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

// Switch Source Tabs (Upload vs Link)
function switchSourceTab(tab) {
  currentSourceTab = tab;
  
  // Update Buttons
  const buttons = document.querySelectorAll('.tab-btn');
  buttons.forEach(btn => btn.classList.remove('active'));
  
  if (tab === 'upload') {
    buttons[0].classList.add('active');
    document.getElementById('source-upload-content').classList.add('active');
    document.getElementById('source-link-content').classList.remove('active');
  } else {
    buttons[1].classList.add('active');
    document.getElementById('source-upload-content').classList.remove('active');
    document.getElementById('source-link-content').classList.add('active');
  }
}

// Drag & Drop File Handlers
function handleFileDrop(e) {
  const dt = e.dataTransfer;
  const files = dt.files;
  if (files.length > 0) {
    processFileSelection(files[0]);
  }
}

function handleFileSelected(e) {
  const files = e.target.files;
  if (files.length > 0) {
    processFileSelection(files[0]);
  }
}

function processFileSelection(file) {
  // Simple validation for MP4/MKV
  selectedFile = file;
  
  const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
  document.getElementById('selected-file-name').textContent = file.name;
  document.getElementById('selected-file-size').textContent = `${sizeMB} Mo`;
  
  // Show file indicator, hide dropzone
  document.getElementById('dropzone').style.display = 'none';
  document.getElementById('file-indicator').style.display = 'flex';
}

function clearSelectedFile() {
  selectedFile = null;
  document.getElementById('video-file-input').value = '';
  document.getElementById('dropzone').style.display = 'flex';
  document.getElementById('file-indicator').style.display = 'none';
}

// Password toggle helper
function togglePasswordVisibility(inputId, icon) {
  const input = document.getElementById(inputId);
  if (input.type === 'password') {
    input.type = 'text';
    icon.classList.remove('fa-eye-slash');
    icon.classList.add('fa-eye');
  } else {
    input.type = 'password';
    icon.classList.remove('fa-eye');
    icon.classList.add('fa-eye-slash');
  }
}

// Toggle OCR vs Diarization UX
function toggleOCRMode(isOcr) {
  const hfTokenGroup = document.getElementById('hf-token').closest('.form-group');
  if (isOcr) {
    hfTokenGroup.style.opacity = '0.5';
    appendTerminalLine('💡 Info: Mode OCR activé. La diarization vocale et le token Hugging Face sont ignorés.', 'info');
  } else {
    hfTokenGroup.style.opacity = '1';
  }
}

// TTS Engine selector cards
function selectTtsEngine(engine) {
  document.querySelectorAll('.tts-card').forEach(card => card.classList.remove('active'));
  document.getElementById(`card-${engine}`).classList.add('active');
  document.getElementById('tts-engine').value = engine;
  appendTerminalLine(`🤖 Moteur TTS sélectionné : ${engine.toUpperCase()}`, 'info');
}

// Advanced accordion toggle
function toggleAccordion(header) {
  header.classList.toggle('active');
}

// Terminal Actions
function appendTerminalLine(text, type = '') {
  const body = document.getElementById('terminal-body');
  
  // Clean up initial message if present
  if (body.children.length === 1 && body.firstChild.textContent.includes('attente de soumission')) {
    body.innerHTML = '';
  }

  const line = document.createElement('div');
  line.className = `terminal-line ${type}`;
  
  // ANSI code conversion to plain text with style
  let cleanText = text;
  if (typeof text === 'string') {
    // Regex out ANSI escape sequences
    cleanText = text.replace(/[\u001b\u009b][[()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><]/g, '');
  }
  
  line.textContent = cleanText;
  body.appendChild(line);
  body.scrollTop = body.scrollHeight;
}

function clearTerminal() {
  const body = document.getElementById('terminal-body');
  body.innerHTML = '<div class="terminal-line info">📟 Console effacée. En attente...</div>';
}

// Polling System Status
async function pollSystemStatus() {
  try {
    const res = await fetch('/api/system-status');
    if (!res.ok) throw new Error();
    const data = await res.json();
    
    // CPU
    document.getElementById('cpu-val').textContent = `${data.cpu}%`;
    
    // GPU
    const gpuVal = document.getElementById('gpu-val');
    const gpuDot = document.getElementById('gpu-dot');
    
    if (data.gpu_active) {
      gpuVal.textContent = `${data.gpu_load}% (${data.gpu_vram_used} / ${data.gpu_vram_total} Mo)`;
      gpuDot.classList.add('active');
      gpuDot.style.backgroundColor = 'var(--secondary)';
      gpuDot.style.boxShadow = '0 0 8px var(--secondary)';
    } else if (data.gpu_available) {
      gpuVal.textContent = `Inactif (${data.gpu_name})`;
      gpuDot.classList.remove('active');
      gpuDot.style.backgroundColor = 'var(--info)';
      gpuDot.style.boxShadow = '0 0 8px var(--info)';
    } else {
      gpuVal.textContent = 'Indisponible';
      gpuDot.classList.remove('active');
      gpuDot.style.backgroundColor = 'var(--text-muted)';
      gpuDot.style.boxShadow = 'none';
    }
  } catch (err) {
    // Graceful fail
    document.getElementById('server-status').textContent = 'Hors ligne';
    document.getElementById('server-status').style.color = 'var(--danger)';
  }
}

// Get Form Data
function compileFormData() {
  const formData = new FormData();
  
  // Hugging Face Token handling
  const hfToken = document.getElementById('hf-token').value.trim();
  formData.append('hf_token', hfToken);
  
  // Save token locally if checked
  if (document.getElementById('remember-token').checked) {
    localStorage.setItem('hf_token', hfToken);
  } else {
    localStorage.removeItem('hf_token');
  }

  // Core settings
  formData.append('lang', document.getElementById('src-lang').value);
  formData.append('tgt_lang', document.getElementById('tgt-lang').value);
  formData.append('ocr', document.getElementById('use-ocr').checked);
  formData.append('clone_voice', document.getElementById('clone-voice').checked);
  formData.append('tts_engine', document.getElementById('tts-engine').value);
  
  // Advanced options
  formData.append('chunk', document.getElementById('chunk-duration').value);
  
  // Convert 10-100% to 0.1-1.0
  const gpuLimitPercent = document.getElementById('gpu-limit').value;
  formData.append('gpu_limit', (gpuLimitPercent / 100).toFixed(2));
  
  formData.append('threads', document.getElementById('threads').value);
  formData.append('skip_chunks', document.getElementById('skip-chunks').value);
  
  const sample = document.getElementById('sample').value.trim();
  if (sample) formData.append('sample', sample);

  // Video Source (File upload vs Link download)
  formData.append('source_type', currentSourceTab);
  if (currentSourceTab === 'upload') {
    if (!selectedFile) {
      alert('Veuillez sélectionner un fichier vidéo MP4 local.');
      return null;
    }
    formData.append('video_file', selectedFile);
  } else {
    const videoUrl = document.getElementById('video-url').value.trim();
    if (!videoUrl) {
      alert('Veuillez spécifier une URL valide de vidéo.');
      return null;
    }
    formData.append('video_url', videoUrl);
  }

  return formData;
}

// Submit Job Creation
async function submitJob(event) {
  event.preventDefault();
  
  const formData = compileFormData();
  if (!formData) return; // Validation failed
  
  const submitBtn = document.getElementById('submit-job-btn');
  submitBtn.disabled = true;
  submitBtn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Démarrage en cours...';
  
  appendTerminalLine('🚀 Soumission de la tâche de doublage au serveur...', 'command');
  
  try {
    const response = await fetch('/api/jobs', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.detail || 'Erreur inconnue du serveur.');
    }
    
    const job = await response.json();
    appendTerminalLine(`✅ Tâche créée avec succès ! ID: ${job.id}`, 'success');
    
    // Switch UI to active tracking
    startJobTracking(job.id, job.input_name);
    loadJobHistory();
  } catch (error) {
    appendTerminalLine(`❌ Échec de la création du job : ${error.message}`, 'error');
    alert(`Échec : ${error.message}`);
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<i class="fa-solid fa-circle-play"></i> Démarrer le Doublage';
  }
}

// Track running job
function startJobTracking(jobId, videoName) {
  // If we were already tracking, close active connection
  if (activeSSE) {
    activeSSE.close();
  }
  
  activeJobId = jobId;
  
  // Hide video player box, clear result source
  document.getElementById('video-result-box').style.display = 'none';
  document.getElementById('result-video-player').pause();
  document.getElementById('result-video-source').src = '';
  
  // Show progress tracker
  document.getElementById('status-tracker').style.display = 'flex';
  document.getElementById('running-job-title').textContent = `Traitement : ${videoName}`;
  document.getElementById('running-status-badge').className = 'status-badge running';
  document.getElementById('running-status-badge').textContent = 'Doublage en cours';
  
  // Reset steps indicators
  resetStepIndicators();
  
  // Connect Server-Sent Events (SSE)
  activeSSE = new EventSource(`/api/jobs/${jobId}/stream`);
  clearTerminal();
  appendTerminalLine(`🔌 Connexion au flux en direct pour le Job #${jobId}...`, 'info');
  
  activeSSE.onmessage = (event) => {
    appendTerminalLine(event.data);
  };
  
  // Listen for specific structured progress event
  activeSSE.addEventListener('progress', (event) => {
    const progressData = JSON.parse(event.data);
    updateProgressUI(progressData);
  });
  
  // Listen for job completion / failure directly
  activeSSE.addEventListener('job_complete', (event) => {
    const jobData = JSON.parse(event.data);
    handleJobFinished(jobData, 'success');
  });

  activeSSE.addEventListener('job_failed', (event) => {
    const jobData = JSON.parse(event.data);
    handleJobFinished(jobData, 'failed');
  });
  
  activeSSE.onerror = (e) => {
    // Keep trying or handle close
    console.log('SSE connection status changed:', e);
  };
}

// Reset Pipeline steps UI
function resetStepIndicators() {
  document.querySelectorAll('.step-indicator').forEach(el => {
    el.className = 'step-indicator';
  });
  document.getElementById('progress-bar-fill').style.width = '0%';
  document.getElementById('progress-percentage-label').textContent = '0%';
}

// Update progress dials
function updateProgressUI(data) {
  // Update Percentage
  const percent = data.percentage || 0;
  document.getElementById('progress-bar-fill').style.width = `${percent}%`;
  document.getElementById('progress-percentage-label').textContent = `${percent}%`;
  
  // Update Current step description label
  if (data.current_step_desc) {
    document.getElementById('current-step-label').textContent = data.current_step_desc;
  }
  
  // Handle steps highlight
  // Steps: step-dl, step-diar, step-trans, step-tts, step-mux
  const stepIds = ['step-dl', 'step-diar', 'step-trans', 'step-tts', 'step-mux'];
  const activeIndex = data.active_step_index; // index from 0 to 4
  
  for (let i = 0; i < stepIds.length; i++) {
    const stepEl = document.getElementById(stepIds[i]);
    if (i < activeIndex) {
      stepEl.className = 'step-indicator completed';
    } else if (i === activeIndex) {
      stepEl.className = 'step-indicator active';
    } else {
      stepEl.className = 'step-indicator';
    }
  }
}

// Finish job
function handleJobFinished(jobData, status) {
  if (activeSSE) {
    activeSSE.close();
    activeSSE = null;
  }
  
  const submitBtn = document.getElementById('submit-job-btn');
  submitBtn.disabled = false;
  submitBtn.innerHTML = '<i class="fa-solid fa-circle-play"></i> Démarrer le Doublage';
  
  // Re-enable file input just in case
  clearSelectedFile();
  
  // Load History
  loadJobHistory();
  
  if (status === 'success') {
    document.getElementById('running-status-badge').className = 'status-badge completed';
    document.getElementById('running-status-badge').textContent = 'Terminé avec succès';
    document.getElementById('progress-bar-fill').style.width = '100%';
    document.getElementById('progress-percentage-label').textContent = '100%';
    document.getElementById('current-step-label').textContent = 'Le doublage est terminé !';
    
    // Highlight all steps as completed
    document.querySelectorAll('.step-indicator').forEach(el => {
      el.className = 'step-indicator completed';
    });
    
    appendTerminalLine('🎉 FÉLICITATIONS ! Pipeline terminé avec succès.', 'success');
    
    // Load result video
    displayDubbedVideo(jobData);
  } else {
    document.getElementById('running-status-badge').className = 'status-badge failed';
    document.getElementById('running-status-badge').textContent = 'Échec';
    document.getElementById('current-step-label').textContent = 'Une erreur est survenue durant le traitement.';
    
    appendTerminalLine('❌ ERREUR CRITIQUE : Le traitement a échoué. Consulter les logs ci-dessus.', 'error');
    alert(`Échec du doublage. Erreur : ${jobData.error_message || 'Inconnue'}`);
  }
}

// Display finished video inside the browser video player
function displayDubbedVideo(job) {
  const resultBox = document.getElementById('video-result-box');
  const player = document.getElementById('result-video-player');
  const source = document.getElementById('result-video-source');
  
  // Setup file paths
  source.src = `/output/${job.output_filename}?t=${new Date().getTime()}`; // cache bust
  player.load();
  
  document.getElementById('result-video-title').textContent = job.input_name;
  
  const durationText = job.processing_duration ? formatDuration(job.processing_duration) : 'N/A';
  const sizeText = job.output_size_bytes ? `${(job.output_size_bytes / (1024*1024)).toFixed(1)} Mo` : 'N/A';
  document.getElementById('result-video-specs').textContent = `Durée de calcul: ${durationText} | Taille: ${sizeText}`;
  
  document.getElementById('result-download-link').href = `/output/${job.output_filename}`;
  
  // Reveal video block, scroll smoothly to it
  resultBox.style.display = 'block';
  resultBox.scrollIntoView({ behavior: 'smooth' });
}

// Copy URL link to clipboard
function shareLocalDubLink() {
  const url = document.getElementById('result-download-link').href;
  navigator.clipboard.writeText(url).then(() => {
    alert('Lien de téléchargement de la vidéo copié dans le presse-papiers !');
  });
}

// Load Job History
async function loadJobHistory() {
  const historyList = document.getElementById('history-list');
  
  try {
    const res = await fetch('/api/jobs');
    if (!res.ok) throw new Error();
    const jobs = await res.json();
    
    if (jobs.length === 0) {
      historyList.innerHTML = `
        <div class="history-empty-state">
          <i class="fa-solid fa-folder-open"></i>
          <span>Aucune tâche enregistrée pour le moment.</span>
        </div>`;
      return;
    }
    
    // Sort reverse chronological
    jobs.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    
    historyList.innerHTML = '';
    jobs.forEach(job => {
      const item = document.createElement('div');
      item.className = `history-item ${activeJobId === job.id ? 'active' : ''}`;
      
      let statusIconClass = 'fa-solid fa-circle-question';
      let statusIconColor = '';
      let badgeClass = '';
      let badgeText = job.status;
      
      if (job.status === 'completed') {
        statusIconClass = 'fa-solid fa-circle-check success';
        badgeClass = 'success';
        badgeText = 'Terminé';
      } else if (job.status === 'failed') {
        statusIconClass = 'fa-solid fa-circle-xmark failed';
        badgeClass = 'failed';
        badgeText = 'Échec';
      } else if (job.status === 'running') {
        statusIconClass = 'fa-solid fa-arrows-spin running';
        badgeClass = 'running';
        badgeText = 'En cours';
      }
      
      const createdDate = new Date(job.created_at).toLocaleString('fr-FR', {
        day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit'
      });

      item.innerHTML = `
        <div class="history-item-left" onclick="viewHistoricalJob('${job.id}')">
          <i class="${statusIconClass} history-item-icon"></i>
          <div class="history-item-details">
            <div class="history-item-name" title="${job.input_name}">${job.input_name}</div>
            <div class="history-item-time">${createdDate} | TTS: ${job.params.tts_engine.toUpperCase()}</div>
          </div>
        </div>
        <div class="history-item-right">
          <span class="history-item-badge ${badgeClass}">${badgeText}</span>
          <div class="history-actions">
            ${job.status === 'completed' ? `<button class="history-action-btn" onclick="playHistoricalVideo('${job.id}')" title="Lire la vidéo"><i class="fa-solid fa-circle-play"></i></button>` : ''}
            <button class="history-action-btn delete" onclick="deleteJobHistory('${job.id}', event)" title="Supprimer la tâche"><i class="fa-solid fa-trash-can"></i></button>
          </div>
        </div>
      `;
      historyList.appendChild(item);
    });
    
  } catch (err) {
    historyList.innerHTML = `<div class="history-empty-state"><i class="fa-solid fa-triangle-exclamation" style="color: var(--danger);"></i><span>Erreur lors du chargement de l'historique</span></div>`;
  }
}

// Clicking a historical job shows its logs in the console
async function viewHistoricalJob(jobId) {
  try {
    const res = await fetch(`/api/jobs/${jobId}`);
    if (!res.ok) throw new Error();
    const job = await res.json();
    
    // Connect SSE to stream logs if it is currently running!
    if (job.status === 'running') {
      startJobTracking(jobId, job.input_name);
      return;
    }
    
    activeJobId = jobId;
    document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
    
    // Highlight item
    loadJobHistory();
    
    clearTerminal();
    appendTerminalLine(`📦 Chargement des logs archivés pour le Job #${jobId}...`, 'info');
    appendTerminalLine(`Paramètres: Source=${job.source_type} | Langue=${job.params.lang || 'Auto'} | Target=${job.params.tgt_lang || 'None'} | TTS=${job.params.tts_engine}`, 'info');
    
    // Load historical log file
    const logRes = await fetch(`/api/jobs/${jobId}/logs`);
    if (logRes.ok) {
      const logs = await logRes.text();
      const lines = logs.split('\n');
      lines.forEach(line => {
        if (line.trim()) {
          // Detect errors or success patterns
          let type = '';
          if (line.includes('❌') || line.toLowerCase().includes('failed') || line.toLowerCase().includes('erreur') || line.toLowerCase().includes('error')) type = 'error';
          else if (line.includes('✅') || line.includes('SUCCESS') || line.includes('terminé')) type = 'success';
          else if (line.includes('🚀') || line.startsWith('===')) type = 'command';
          
          appendTerminalLine(line, type);
        }
      });
    } else {
      appendTerminalLine('⚠️ Impossible de charger les détails du log pour cette tâche. Fichier manquant ou expiré.', 'warning');
    }
    
    // If complete, reveal the video card
    if (job.status === 'completed') {
      displayDubbedVideo(job);
    } else {
      document.getElementById('video-result-box').style.display = 'none';
      document.getElementById('result-video-player').pause();
    }
    
  } catch (err) {
    alert('Impossible de charger les détails du job.');
  }
}

// Directly play the video of a finished job
async function playHistoricalVideo(jobId) {
  try {
    const res = await fetch(`/api/jobs/${jobId}`);
    if (!res.ok) throw new Error();
    const job = await res.json();
    if (job.status === 'completed') {
      displayDubbedVideo(job);
    }
  } catch (err) {
    alert('Impossible de charger la vidéo.');
  }
}

// Delete job from list
async function deleteJobHistory(jobId, event) {
  event.preventDefault();
  event.stopPropagation();
  
  if (!confirm('Êtes-vous sûr de vouloir supprimer cette tâche de l\'historique ? Cela n\'effacera pas les vidéos générées de votre disque.')) {
    return;
  }
  
  try {
    const res = await fetch(`/api/jobs/${jobId}`, {
      method: 'DELETE'
    });
    
    if (res.ok) {
      if (activeJobId === jobId) {
        activeJobId = null;
        document.getElementById('video-result-box').style.display = 'none';
        document.getElementById('result-video-player').pause();
        document.getElementById('status-tracker').style.display = 'none';
        clearTerminal();
      }
      loadJobHistory();
    } else {
      throw new Error();
    }
  } catch (err) {
    alert('Échec de la suppression de la tâche.');
  }
}

// Helpers
function formatDuration(seconds) {
  if (!seconds) return '0s';
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m === 0) return `${s}s`;
  return `${m}m ${s}s`;
}
