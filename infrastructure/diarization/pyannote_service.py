import warnings
import torch
from tqdm import tqdm
from core.interfaces.engines import IDiarizationService
from core.entities.models import Segment

class PyannoteDiarizationService(IDiarizationService):
    def analyze(self, audio_path: str, hf_token: str) -> list:
        from pyannote.audio import Pipeline
        import torchaudio

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        ).to(torch.device("cpu"))

        waveform, sample_rate = torchaudio.load(audio_path)
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}).speaker_diarization

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(Segment(
                start=round(turn.start, 3),
                end=round(turn.end, 3),
                speaker=speaker
            ))
        return segments
