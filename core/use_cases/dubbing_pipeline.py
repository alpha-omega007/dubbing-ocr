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
            chunk_duration: int = 20):
        
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
            
        # 3. Translation
        if tgt_lang and tgt_lang != src_lang:
            segments = self.translation_engine.translate(segments, tgt_lang)
            
        # 4. TTS
        audio_segs = self.tts_engine.synthesize(segments, tmp_dir, target_lang=tgt_lang or src_lang or "en")
        
        # 5. Muxing
        self.audio_service.mux_final(video_path, audio_segs, tmp_dir, output_path)
        
        return output_path
