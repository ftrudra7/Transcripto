import logging
import json
from app.core.config import settings
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.provider = settings.SUMMARY_PROVIDER
        self.client = None
        if self.provider == "groq" and settings.GROQ_API_KEY:
            try:
                from groq import AsyncGroq
                self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
            except ImportError:
                logger.error("groq not installed.")
                self.provider = "mock"

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    async def _call_groq(self, prompt: str) -> str:
        if not self.client:
            raise Exception("Groq client not initialized")
        
        response = await self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()

    async def summarize(self, text: str, summary_type: str = "short") -> dict:
        """
        Summarizes text using the configured LLM provider.
        Returns a dictionary suitable for SummaryResponse schema.
        """
        if not text or len(text.strip()) == 0:
            return {"summary": ""}

        if self.provider == "groq" and self.client:
            try:
                if summary_type == "detailed":
                    prompt_instruction = "Provide a detailed summary of the following transcript."
                elif summary_type == "key_points":
                    prompt_instruction = "Extract the key points from the following transcript, format as a concise paragraph or short bullet points."
                elif summary_type == "meeting_notes":
                    prompt_instruction = "Provide meeting notes from the following transcript."
                elif summary_type == "action_items":
                    prompt_instruction = "Extract the action items from the following transcript."
                else:
                    prompt_instruction = "Provide a short and concise summary of the following transcript."

                prompt_json = (
                    f"You are a professional AI assistant. Analyze the following transcript.\n"
                    f"Output MUST be valid JSON with the following keys:\n"
                    f"- \"title\": A suitable title for the text.\n"
                    f"- \"summary\": {prompt_instruction}\n"
                    f"- \"key_points\": A list of strings containing key points.\n"
                    f"- \"action_items\": A list of strings containing action items.\n"
                    f"- \"keywords\": A list of 5-10 strings containing keywords.\n\n"
                    f"Do not include markdown blocks like ```json, just output the raw JSON.\n\n"
                    f"Transcript:\n{text}"
                )
                
                response_text = await self._call_groq(prompt_json)
                
                try:
                    if response_text.startswith("```json"):
                        response_text = response_text[7:-3]
                    elif response_text.startswith("```"):
                        response_text = response_text[3:-3]
                    
                    data = json.loads(response_text)
                    return {
                        "title": data.get("title"),
                        "summary": data.get("summary", ""),
                        "key_points": data.get("key_points", []),
                        "action_items": data.get("action_items", []),
                        "keywords": data.get("keywords", [])
                    }
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON from Groq response. Raw: {response_text}")
                    # Fallback if json fails
                    summary = await self._call_groq(
                        f"You are a professional assistant. {prompt_instruction}\n\nTranscript:\n{text}"
                    )
                    return {"summary": summary}

            except Exception as e:
                logger.error(f"Error during Groq summarization: {str(e)}")
                # Fallback to heuristic on error
                pass

        # Heuristic fallback
        logger.info("Using heuristic summarization fallback.")
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 2:
            summary = ". ".join(sentences[:2]) + "."
        else:
            summary = text[:200] + ("..." if len(text) > 200 else "")
            
        return {"summary": summary}

llm_service = LLMService()
