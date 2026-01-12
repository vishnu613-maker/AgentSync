"""
LLM service wrapper for Ollama integration
"""
import logging
from typing import Optional, Dict
import httpx
import json
from app.config import get_settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for LLM operations via Ollama
    """
    
    def __init__(self, ollama_url: str):
        self.ollama_url = ollama_url
        self.settings = get_settings()
        self.models = {}
    
    def get_model(self, model_name: str):
        """Get model (placeholder for future OpenAI-compatible setup)"""
        return model_name
    
    async def call_ollama(
        self,
        model_name: str,
        prompt: str,
        timeout: int=500.0,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Call Ollama model
        
        Args:
            model_name: Model name (e.g., "smollm3:3b")
            prompt: User prompt
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Max tokens in response
            system_prompt: System prompt
        
        Returns:
            Model response
        """
        try:
            # Build full prompt with system context if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            logger.info(f"[LLM] Calling {model_name}")
            
            async with httpx.AsyncClient(timeout = timeout) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "top_p": top_p,
                        }
                    }
                )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "").strip()
                
                logger.info(f"[LLM] {model_name} responded ({len(generated_text)} chars)")
                return generated_text
            else:
                logger.error(f"[LLM] Error calling {model_name}: {response.status_code}")
                return ""
            
        except Exception as e:
            logger.error(f"[LLM] Error: {e}")
            return ""
    
    async def health_check(self) -> bool:
        """Check Ollama server health"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
            
            if response.status_code == 200:
                logger.info("[LLM] Ollama health check passed")
                return True
            
            logger.error(f"[LLM] Ollama health check failed: {response.status_code}")
            return False
            
        except Exception as e:
            logger.error(f"[LLM] Ollama connection error: {e}")
            return False
