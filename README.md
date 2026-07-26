# Transcripto AI SaaS

Transcripto is an enterprise-grade AI language platform designed to seamlessly transcribe audio, generate intelligent summaries, and detect sign language.

## Features

- **High-Accuracy Transcription**: Powered by `faster-whisper` and OpenAI Whisper.
- **Intelligent Summarization**: Powered by OpenAI's GPT models.
- **Sign Language Detection**: Real-time browser-based MediaPipe integration.
- **Modern UI**: Clean, glassmorphism design with responsive elements.
- **Production-Ready Backend**: Built on FastAPI for asynchronous, high-throughput processing.

## Architecture

Please see [Architecture.md](Architecture.md) for a detailed technical overview.

## Quickstart

### Prerequisites

- Python 3.11+
- [FFmpeg](https://ffmpeg.org/) (for audio processing)
- (Optional) Docker

### Local Installation

1. **Clone the repository**
2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Environment Variables**
   Copy the example config and add your API keys:
   ```bash
   cp .env.example .env
   ```
5. **Run the Application**
   ```bash
   uvicorn app.main:app --reload
   ```
   Visit `http://localhost:8000` to view the app!

### Docker Deployment

```bash
docker-compose up --build -d
```

## Security Notice

Never commit your `.env` file or hardcode your `OPENAI_API_KEY`. The repository includes pre-commit hooks and Git ignores to prevent accidental secret leakage.

## License
MIT License.
