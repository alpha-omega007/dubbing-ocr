import os
import asyncio
import edge_tts
from tqdm import tqdm
from core.interfaces.engines import ITTSEngine
from core.entities.models import Segment

class EdgeTTSEngine(ITTSEngine):
    VOICES = {
        "fr": {"M": "fr-FR-HenriNeural", "F": "fr-FR-DeniseNeural"},
        "en": {"M": "en-US-GuyNeural", "F": "en-US-AriaNeural"},
        "es": {"M": "es-ES-AlvaroNeural", "F": "es-ES-ElviraNeural"},
    }

    def synthesize(self, segments: list, output_dir: str, target_lang: str) -> list:
        return asyncio.run(self._synthesize_async(segments, output_dir, target_lang))

    async def _synthesize_async(self, segments: list, output_dir: str, target_lang: str) -> list:
        audio_segments = []
        voice_set = self.VOICES.get(target_lang, self.VOICES["en"])
        
        for i, seg in enumerate(tqdm(segments, desc="🔊 Edge TTS")):
            if not seg.text: continue
            voice = voice_set.get(seg.gender, voice_set["F"])
            out_file = os.path.join(output_dir, f"tts_{i:04d}.mp3")
            
            # Retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    communicate = edge_tts.Communicate(seg.text, voice)
                    await communicate.save(out_file)
                    break # Success
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"❌ Échec définitif pour le segment {i} : {e}")
                        raise
                    print(f"⚠️ Erreur EdgeTTS (essai {attempt+1}/{max_retries}), nouvel essai dans 2s...")
                    await asyncio.sleep(2)
            
            audio_segments.append({"file": out_file, "start": seg.start, "end": seg.end})
        return audio_segments
