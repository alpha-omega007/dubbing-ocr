import os
import json
from typing import List, Optional
from core.interfaces.engines import (
    ITranscriptionEngine, IOCREngine, ITranslationEngine, 
    ITTSEngine, IAudioService, IDiarizationService
)
from core.entities.models import Segment

class DubbingPipeline:
    def __init__(
        self,
        audio_service: IAudioService,
        transcription_engine: ITranscriptionEngine,
        ocr_engine: IOCREngine,
        translation_engine: ITranslationEngine,
        tts_engine: ITTSEngine,
        diarization_service: IDiarizationService
    ):
        self.audio_service = audio_service
        self.transcription_engine = transcription_engine
        self.ocr_engine = ocr_engine
        self.translation_engine = translation_engine
        self.tts_engine = tts_engine
        self.diarization_service = diarization_service

    def run(self, video_path: str, tmp_dir: str, hf_token: str, 
            src_lang: Optional[str], tgt_lang: Optional[str], 
            use_ocr: bool = False, output_path: str = "output.mp4",
            chunk_duration: int = 20, filter_range: Optional[tuple] = None,
            clone_voice: bool = False):
        
        # 1. Extract Audio
        audio_path = self.audio_service.extract_audio(video_path, tmp_dir)
        
        # 2. Get Segments (via OCR or Diarization + Transcription)
        if use_ocr:
            segments = self.ocr_engine.extract_text(video_path, tmp_dir, lang=src_lang or "fr")
            segments = self.audio_service.detect_gender(audio_path, segments)
        else:
            segments = self.diarization_service.analyze(audio_path, hf_token)
            segments = self.audio_service.detect_gender(audio_path, segments)
            segments = self.transcription_engine.transcribe(audio_path, src_lang, segments, chunk_duration=chunk_duration)
            
        # 2.5 Filter segments by range (avoid duplicates at boundaries)
        if filter_range:
            start_f, end_f = filter_range
            # Utiliser le milieu du segment pour décider à quel bloc il appartient
            # C'est beaucoup plus stable que le 'start' car Whisper peut décaler légèrement les bords
            segments = [s for s in segments if start_f <= ((s.start + s.end) / 2) < end_f]
            
        # 2.6 Deduplicate adjacent segments with identical text (anti-repetition)
        cleaned = []
        for s in segments:
            if cleaned and cleaned[-1].text == s.text and (s.start - cleaned[-1].end) < 1.0:
                # Si le texte est identique et très proche, on ignore le doublon
                continue
            cleaned.append(s)
        segments = cleaned
            
        # 3. Translation
        if tgt_lang and tgt_lang != src_lang:
            segments = self.translation_engine.translate(segments, tgt_lang)
        # 3.5 Auto-Detect Emotion for ALL languages
        def auto_detect_emotion(text: str) -> str:
            text_lower = text.lower()
            
            # Mots-clés Anglais & Français
            happy_words = ["joie", "génial", "super", "incroyable", "magnifique", "heureux", "plaisir", "amour", "aimer", "gagné", "fête", "rire", "glorieux", "merveilleux", "excellent", "parfait", "happy", "great", "awesome", "love", "perfect", "good", "nice", "wonderful", "yay"]
            sad_words = ["triste", "malheur", "pleurer", "pleure", "mort", "décès", "désolé", "regret", "souffrir", "douleur", "perdu", "malheureux", "sombre", "larme", "pleurs", "seul", "sad", "sorry", "cry", "death", "die", "pain", "lose", "lost", "bad", "unhappy"]
            angry_words = ["colère", "haine", "déteste", "tuer", "frapper", "arrête", "idiot", "imbécile", "mensonge", "mentir", "trahir", "fou", "folle", "énerve", "agace", "angry", "hate", "kill", "stop", "lie", "liar", "crazy", "mad"]
            
            if any(w in text_lower for w in sad_words): return "triste"
            if any(w in text_lower for w in angry_words): return "colere"
            if any(w in text_lower for w in happy_words): return "joyeux"
                
            # Vérification universelle de ponctuation (y compris caractères chinois)
            if "!" in text or "！" in text:
                return "excite"
                
            return "neutre"

        for seg in segments:
            if seg.text and not seg.text.startswith("["):
                emotion = auto_detect_emotion(seg.text)
                if emotion != "neutre":
                    seg.text = f"[{emotion}] {seg.text}"
            
        # 4. TTS
        audio_segs = self.tts_engine.synthesize(segments, tmp_dir, target_lang=tgt_lang or src_lang or "en")
        
        # 4.5 Clonage Vocal (Transfert de timbre original)
        if clone_voice and audio_segs:
            print("🎙️ Initialisation du Clonage Vocal (Transfert de Timbre d'Origine)...")
            try:
                from infrastructure.audio.tone_converter import OpenVoiceToneConverter
                tone_converter = OpenVoiceToneConverter()
                
                # Validation préalable des modèles locaux pour afficher un message descriptif si nécessaire
                tone_converter._load_converter()
                
                import subprocess
                for j, seg_audio in enumerate(audio_segs):
                    orig_seg = segments[j]
                    if not orig_seg.text: continue
                    
                    ref_wav = os.path.join(tmp_dir, f"ref_{j:04d}.wav")
                    
                    # Ignorer les segments extrêmement courts pour éviter les erreurs de caractéristiques audio
                    duration = orig_seg.end - orig_seg.start
                    if duration < 0.25: continue
                    
                    split_cmd = [
                        "ffmpeg", "-y",
                        "-ss", f"{orig_seg.start:.3f}",
                        "-to", f"{orig_seg.end:.3f}",
                        "-i", audio_path,
                        "-ac", "1", "-ar", "16000",
                        ref_wav
                    ]
                    subprocess.run(split_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    if os.path.exists(ref_wav) and os.path.getsize(ref_wav) > 0:
                        cloned_wav = os.path.join(tmp_dir, f"cloned_{j:04d}.wav")
                        try:
                            tone_converter.convert_timbre(
                                synthesized_wav=seg_audio["file"],
                                reference_wav=ref_wav,
                                output_wav=cloned_wav
                            )
                            if os.path.exists(cloned_wav) and os.path.getsize(cloned_wav) > 0:
                                seg_audio["file"] = cloned_wav
                        except Exception as e:
                            print(f"⚠️ Échec du transfert de timbre pour le segment {j} : {e}")
            except Exception as e:
                print(f"⚠️ Clonage Vocal indisponible : {e}. Poursuite avec les voix de synthèse d'origine.")
        
        # 5. Muxing
        self.audio_service.mux_final(video_path, audio_segs, tmp_dir, output_path)
        
        return output_path
