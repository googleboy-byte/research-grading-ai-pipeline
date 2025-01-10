import json
import os
from pathlib import Path
import hashlib
import logging

logger = logging.getLogger(__name__)

class GeminiCache:
    def __init__(self, cache_dir: str = "master/cache/gemini", use_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized Gemini cache at {cache_dir}")
    
    def _get_cache_key(self, text: str, prompt_template: str) -> str:
        """Generate a unique cache key based on input text and prompt"""
        combined = f"{text}{prompt_template}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key"""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, text: str, prompt_template: str) -> str | None:
        """Try to get a cached response"""
        if not self.use_cache:
            return None
            
        cache_key = self._get_cache_key(text, prompt_template)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                logger.info(f"Cache hit for key {cache_key[:10]}...")
                return cached_data['response']
            except Exception as e:
                logger.warning(f"Error reading cache: {str(e)}")
                return None
        return None
    
    def set(self, text: str, prompt_template: str, response: str) -> None:
        """Cache a response"""
        if not self.use_cache:
            return
            
        cache_key = self._get_cache_key(text, prompt_template)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    'text': text[:200] + '...' if len(text) > 200 else text,  # Store preview for debugging
                    'prompt': prompt_template[:200] + '...' if len(prompt_template) > 200 else prompt_template,
                    'response': response
                }, f, indent=2)
            logger.info(f"Cached response for key {cache_key[:10]}...")
        except Exception as e:
            logger.warning(f"Error writing to cache: {str(e)}") 