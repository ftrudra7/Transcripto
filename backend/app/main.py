import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.exceptions import add_exception_handlers
from app.api.routes import audio, summary, sign
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise-grade AI SaaS for transcription, summarization, and sign language detection.",
    docs_url=f"{settings.API_V1_STR}/docs",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add centralized exception handlers
add_exception_handlers(app)

# Include routers
app.include_router(audio.router, prefix=f"{settings.API_V1_STR}/audio", tags=["audio"])
app.include_router(summary.router, prefix=f"{settings.API_V1_STR}/summary", tags=["summary"])
app.include_router(sign.router, prefix=f"{settings.API_V1_STR}/sign", tags=["sign"])

@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok", "environment": settings.APP_ENV, "version": settings.APP_VERSION}

@app.get("/")
async def root():
    return {"message": "Welcome to Transcripto API. Access /api/v1/docs for Swagger UI."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)

