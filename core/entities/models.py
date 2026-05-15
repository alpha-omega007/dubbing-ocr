from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class Segment:
    start: float
    end: float
    text: str = ""
    text_original: str = ""
    speaker: str = "SPEAKER_00"
    gender: str = "M"  # "M" or "F"
    audio_file: Optional[str] = None

@dataclass
class AudioMetadata:
    duration: float
    sample_rate: int
    channels: int
    path: str
