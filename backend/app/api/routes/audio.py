import logging
from fastapi import APIRouter, UploadFile, File
from app.schemas.audio import AudioProcessResponse, TranscriptionSegment
from app.services.asr_service import asr_service
from app.services.llm_service import llm_service
from app.core.exceptions import AppError
import filetype

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=AudioProcessResponse)
async def process_audio(file: UploadFile = File(...)):
    if not file:
        raise AppError("No file uploaded", status_code=400)
    
    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        raise AppError("Failed to read file", status_code=400)

    # Validate file type using magic numbers (filetype library)
    kind = filetype.guess(content)
    if kind is None or not (kind.mime.startswith("audio/") or kind.mime.startswith("video/")):
        if not (file.content_type.startswith("audio/") or file.content_type.startswith("video/") or file.content_type in ["application/octet-stream"]):
            raise AppError("Invalid file type. Only audio or video files are allowed.", status_code=415)

    try:
        # Transcribe
        transcription_result = await asr_service.transcribe(content)
        
        # Summarize (using default short summary to not slow down the initial response too much)
        summary_result = {"summary": ""}
        if transcription_result.get("transcription"):
            summary_result = await llm_service.summarize(transcription_result["transcription"], summary_type="short")

        segments_data = transcription_result.get("segments", [])
        segments = [TranscriptionSegment(**seg) for seg in segments_data]

        return AudioProcessResponse(
            transcription=transcription_result.get("transcription", ""),
            language=transcription_result.get("language"),
            language_probability=transcription_result.get("language_probability"),
            segments=segments,
            summary=summary_result.get("summary", "")
        )
    except Exception as e:
        logger.exception("Error processing audio")
        raise AppError(f"Failed to process audio: {str(e)}", status_code=500)
