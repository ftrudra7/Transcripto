import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Type
from app.core.config import settings

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Strategy interface for LLM Providers."""
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        pass

class GroqProvider(LLMProvider):
    def __init__(self):
        from groq import AsyncGroq
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set")
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)

    async def generate(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()

class OpenAIProvider(LLMProvider):
    def __init__(self):
        from openai import AsyncOpenAI
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set")
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def generate(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()

class MockProvider(LLMProvider):
    async def generate(self, prompt: str) -> str:
        logger.warning("Using MockProvider. This should not be used in production.")
        return json.dumps({
            "title": "Mock Summary",
            "summary": "This is a mock summary because no valid provider was found.",
            "key_points": ["Point 1", "Point 2"],
            "action_items": ["Action 1"],
            "keywords": ["mock", "test"]
        })

class LLMService:
    def __init__(self):
        self.provider_name = settings.SUMMARY_PROVIDER.lower()
        self.provider: LLMProvider = self._initialize_provider()

    def _initialize_provider(self) -> LLMProvider:
        try:
            if self.provider_name == "groq":
                return GroqProvider()
            elif self.provider_name == "openai":
                return OpenAIProvider()
            else:
                logger.warning(f"Unsupported provider '{self.provider_name}', falling back to Mock.")
                return MockProvider()
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider_name} provider: {e}")
            return MockProvider()

    async def summarize(self, text: str, summary_type: str = "short") -> Dict[str, Any]:
        """
        Summarizes text using the configured LLM provider strategy.
        """
        if not text or len(text.strip()) == 0:
            return {"summary": ""}

        instructions = {
            "detailed": "Provide a detailed summary.",
            "key_points": "Extract key points as concise bullet points.",
            "meeting_notes": "Provide structured meeting notes.",
            "action_items": "Extract actionable items.",
            "short": "Provide a short, concise summary."
        }
        
        prompt_instruction = instructions.get(summary_type, instructions["short"])
        
        prompt = (
            f"You are a professional AI assistant. Analyze the transcript below.\n"
            f"Output MUST be valid JSON with these keys exactly:\n"
            f"- \"title\": A suitable title.\n"
            f"- \"summary\": {prompt_instruction}\n"
            f"- \"key_points\": A list of key points.\n"
            f"- \"action_items\": A list of action items.\n"
            f"- \"keywords\": A list of 5-10 keywords.\n\n"
            f"Return ONLY raw JSON. No markdown ticks.\n\n"
            f"Transcript:\n{text}"
        )

        try:
            response_text = await self.provider.generate(prompt)
            
            # Clean possible markdown
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            elif response_text.startswith("```"):
                response_text = response_text[3:-3]
                
            data = json.loads(response_text)
            return {
                "title": data.get("title", "Untitled"),
                "summary": data.get("summary", ""),
                "key_points": data.get("key_points", []),
                "action_items": data.get("action_items", []),
                "keywords": data.get("keywords", [])
            }
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}")
            # Heuristic fallback
            sentences = text.replace('!', '.').replace('?', '.').split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            summary = ". ".join(sentences[:2]) + "." if len(sentences) >= 2 else text[:200]
            return {"summary": summary, "fallback": True}

llm_service = LLMService()
