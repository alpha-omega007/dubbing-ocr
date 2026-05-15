from abc import ABC, abstractmethod
from typing import List, Optional
from core.entities.models import Segment

class ITranscriptionEngine(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str, language: Optional[str], segments: List[Segment]) -> List[Segment]:
        pass

class IOCREngine(ABC):
    @abstractmethod
    def extract_text(self, video_path: str, tmp_dir: str, lang: str) -> List[Segment]:
        pass

class ITranslationEngine(ABC):
    @abstractmethod
    def translate(self, segments: List[Segment], target_lang: str) -> List[Segment]:
        pass

class ITTSEngine(ABC):
    @abstractmethod
    def synthesize(self, segments: List[Segment], output_dir: str, target_lang: str) -> List[dict]:
        pass

class IDiarizationService(ABC):
    @abstractmethod
    def analyze(self, audio_path: str, hf_token: Optional[str]) -> List[Segment]:
        pass

class IAudioService(ABC):
    @abstractmethod
    def get_duration(self, video_path: str) -> float:
        pass

    @abstractmethod
    def extract_audio(self, video_path: str, output_dir: str) -> str:
        pass
    
    @abstractmethod
    def detect_gender(self, audio_path: str, segments: List[Segment]) -> List[Segment]:
        pass
    
    @abstractmethod
    def mux_final(self, video_path: str, audio_segments: List[dict], tmp_dir: str, output_path: str):
        pass
