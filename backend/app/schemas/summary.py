from pydantic import BaseModel, Field
from typing import Optional, List

class SummaryRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The text to summarize")
    summary_type: Optional[str] = Field("short", description="Type of summary: short, detailed, key_points, meeting_notes, action_items")

class SummaryResponse(BaseModel):
    title: Optional[str] = Field(None, description="Generated title")
    summary: str = Field(..., description="The generated summary or output")
    key_points: Optional[List[str]] = Field(None, description="Extracted key points")
    action_items: Optional[List[str]] = Field(None, description="Action items")
    keywords: Optional[List[str]] = Field(None, description="Keywords extracted from text")
