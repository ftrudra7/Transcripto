import logging
import io
import tempfile
import os
import torch
from app.core.config import settings

logger = logging.getLogger(__name__)

class ASRService:
    def __init__(self):
        self.provider = settings.ASR_PROVIDER
        self.model = None
        self._model_loaded = False
        
    def _load_model(self):
        if self._model_loaded:
            return
            
        if self.provider == "faster_whisper":
            try:
                from faster_whisper import WhisperModel
                logger.info("Initializing faster_whisper model...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                logger.info(f"Using device: {device}, compute_type: {compute_type}")
                self.model = WhisperModel("base", device=device, compute_type=compute_type)
                self._model_loaded = True
            except ImportError:
                logger.error("faster_whisper not installed.")
                self.provider = "mock"
        else:
            self._model_loaded = True

    async def transcribe(self, audio_bytes: bytes) -> dict:
        """
        Transcribes audio bytes to text using the configured ASR provider.
        Returns a dict with transcription, language, probability, and segments.
        """
        self._load_model()
        
        if self.provider == "faster_whisper" and self.model:
            # We use a temporary file to support large files efficiently
            fd, temp_path = tempfile.mkstemp(suffix=".tmp")
            try:
                with os.fdopen(fd, 'wb') as f:
                    f.write(audio_bytes)
                    
                segments, info = self.model.transcribe(temp_path, beam_size=5)
                
                segment_list = []
                full_text = []
                for segment in segments:
                    segment_list.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip()
                    })
                    full_text.append(segment.text.strip())
                
                return {
                    "transcription": " ".join(full_text),
                    "language": info.language,
                    "language_probability": info.language_probability,
                    "segments": segment_list
                }
            except Exception as e:
                logger.error(f"Error during faster_whisper transcription: {str(e)}")
                raise
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as cleanup_error:
                        logger.error(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")
        
        # Fallback to mock
        logger.info("Using mock ASR provider.")
        return {
            "transcription": "This is a mock transcription. The ASR provider is either not set up correctly or set to 'mock'.",
            "language": "en",
            "language_probability": 1.0,
            "segments": [{"start": 0.0, "end": 1.0, "text": "This is a mock transcription."}]
        }

asr_service = ASRService()
