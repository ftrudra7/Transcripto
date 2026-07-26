import logging
from fastapi import APIRouter
from app.schemas.summary import SummaryRequest, SummaryResponse
from app.services.llm_service import llm_service
from app.core.exceptions import AppError

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=SummaryResponse)
async def summarize_text(request: SummaryRequest):
    try:
        result = await llm_service.summarize(request.text, request.summary_type)
        return SummaryResponse(
            title=result.get("title"),
            summary=result.get("summary", ""),
            key_points=result.get("key_points"),
            action_items=result.get("action_items"),
            keywords=result.get("keywords")
        )
    except Exception as e:
        logger.exception("Error generating summary")
        raise AppError(f"Failed to generate summary: {str(e)}", status_code=500)
