from pydantic import BaseModel, Field
from typing import Optional, List

class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str

class AudioProcessResponse(BaseModel):
    transcription: str = Field(..., description="The transcribed text from the audio")
    language: Optional[str] = Field(None, description="Detected language")
    language_probability: Optional[float] = Field(None, description="Confidence of the detected language")
    segments: Optional[List[TranscriptionSegment]] = Field(None, description="Transcription segments with timestamps")
    summary: str = Field(..., description="A summary of the transcribed text")
