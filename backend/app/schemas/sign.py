from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class Landmark(BaseModel):
    x: float
    y: float
    z: float

class SignPredictionRequest(BaseModel):
    landmarks: List[Landmark] = Field(..., min_items=1, description="List of hand/body landmarks")

class SignPredictionResponse(BaseModel):
    text: str = Field(..., description="The predicted sign language text")
    score: float = Field(..., description="Confidence score of the prediction")
    landmarks_extracted: Optional[List[Any]] = Field(None, description="Extracted landmarks for images/videos")
