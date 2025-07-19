import httpx
import json
import os
import random
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class OpenRouterClient:
    def __init__(self):
        self.api_keys = [
            os.environ.get("OPENROUTER_API_KEY_1"),
            os.environ.get("OPENROUTER_API_KEY_2"),
            os.environ.get("OPENROUTER_API_KEY_3")
        ]
        self.api_keys = [key for key in self.api_keys if key]  # Remove None values
        self.base_url = "https://openrouter.ai/api/v1"
        self.default_model = os.environ.get("DEFAULT_MODEL", "mistralai/mistral-7b-instruct:free")
        
    def _get_api_key(self) -> str:
        """Get a random API key for load balancing."""
        if not self.api_keys:
            raise ValueError("No OpenRouter API keys configured")
        return random.choice(self.api_keys)
    
    async def get_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from OpenRouter."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self._get_api_key()}"}
                )
                response.raise_for_status()
                data = response.json()
                return data.get("data", [])
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return []
    
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 1.0
    ) -> Dict[str, Any]:
        """Send chat completion request to OpenRouter."""
        if not model:
            model = self.default_model
            
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stream": False
                }
                
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._get_api_key()}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost:3000",
                        "X-Title": "AI Multi-Agent Platform"
                    },
                    json=payload
                )
                
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error in chat completion: {e.response.status_code} - {e.response.text}")
            raise Exception(f"OpenRouter API error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise Exception(f"Chat completion failed: {str(e)}")
    
    def calculate_tokens(self, text: str) -> int:
        """Rough token calculation (4 chars ≈ 1 token)."""
        return max(1, len(text) // 4)
    
    def get_model_cost(self, model: str) -> Dict[str, float]:
        """Get estimated cost per token for a model."""
        # This is a simplified version - in production you'd get this from the models API
        free_models = [
            "mistralai/mistral-7b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free",
            "google/gemma-7b-it:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "meta-llama/llama-3.2-1b-instruct:free"
        ]
        
        if model in free_models:
            return {"prompt": 0.0, "completion": 0.0}
        else:
            # Default pricing for paid models (this would come from API in production)
            return {"prompt": 0.0001, "completion": 0.0002}

# Global client instance
openrouter_client = OpenRouterClient()