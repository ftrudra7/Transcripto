# Architecture Overview

Transcripto has been re-architected from a fragmented prototype into a unified, enterprise-grade AI SaaS application using **FastAPI** and modern engineering practices.

## System Design

The system follows a classic client-server model, utilizing a monolithic modular backend tailored for AI inference tasks.

### 1. Frontend (Client)
- **Tech Stack**: Vanilla JavaScript (ES6), HTML5, CSS3.
- **Role**: Serves as the presentation layer. Uses the Web Speech API as a fallback and MediaPipe for in-browser video inference (Sign Language). Uploads heavy audio blobs to the backend via REST endpoints.
- **Hosting**: Served statically via FastAPI's `StaticFiles` middleware.

### 2. Backend (FastAPI)
- **Tech Stack**: Python 3.11+, FastAPI, Pydantic, Uvicorn.
- **Role**: Handles heavy asynchronous IO, authentication (optional), validation, and orchestration of AI model calls.
- **Why FastAPI?**: AI tasks (like contacting OpenAI or running Whisper) are I/O bound or CPU bound. FastAPI handles I/O concurrently without blocking the main event loop, significantly outperforming Node.js or basic Flask for these specific workloads.

### 3. Service Layer
The business logic is encapsulated in the `app/services/` directory:
- **`asr_service.py`**: Handles Automatic Speech Recognition. Supports plug-and-play providers (Local `faster-whisper` or remote `OpenAI Whisper API`).
- **`llm_service.py`**: Handles Summarization. Connects to OpenAI's Chat API with heuristic fallbacks.
- **`sign_service.py`**: Evaluates coordinate landmarks passed from the client to predict sign language.

## Data Flow

1. User records audio via browser `MediaRecorder` or uploads a file.
2. The blob is sent to `/api/v1/audio`.
3. The router reads the bytes into memory (avoiding slow disk writes).
4. `filetype` validates the magic bytes to ensure it's a valid media file.
5. `asr_service` processes the bytes and returns a transcript.
6. The router passes the transcript to `llm_service` for summarization.
7. The structured JSON response is returned to the client and displayed.

## Security Practices
- **Secret Management**: All secrets are injected via `.env` and parsed strictly by `pydantic-settings`.
- **Validation**: Incoming payloads (like JSON landmarks) are strictly validated by Pydantic schemas.
- **Centralized Exceptions**: A custom exception handler ensures stack traces are never leaked to the client.
