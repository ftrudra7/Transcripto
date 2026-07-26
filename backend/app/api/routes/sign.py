import logging
from fastapi import APIRouter, UploadFile, File
from app.schemas.sign import SignPredictionRequest, SignPredictionResponse
from app.services.sign_service import sign_service
from app.core.exceptions import AppError

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=SignPredictionResponse)
async def predict_sign_language(request: SignPredictionRequest):
    try:
        result = await sign_service.predict_sign_from_landmarks([l.dict() for l in request.landmarks])
        return SignPredictionResponse(
            text=result.get("text", ""),
            score=result.get("score", 0.0)
        )
    except Exception as e:
        logger.exception("Error predicting sign language from landmarks")
        raise AppError(f"Failed to predict sign: {str(e)}", status_code=500)

@router.post("/image", response_model=SignPredictionResponse)
async def predict_sign_from_image(file: UploadFile = File(...)):
    if not file:
        raise AppError("No file uploaded", status_code=400)
        
    try:
        content = await file.read()
        result = await sign_service.predict_sign_from_image(content)
        return SignPredictionResponse(
            text=result.get("text", ""),
            score=result.get("score", 0.0),
            landmarks_extracted=result.get("landmarks_extracted", [])
        )
    except Exception as e:
        logger.exception("Error predicting sign language from image")
        raise AppError(f"Failed to predict sign from image: {str(e)}", status_code=500)
