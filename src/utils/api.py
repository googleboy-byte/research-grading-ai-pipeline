import os
import time
from itertools import cycle
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup API keys rotation
GOOGLE_API_KEYS = [
    os.getenv('GOOGLE_API_KEY'),
    os.getenv('GOOGLE_API_KEY_1'),
    os.getenv('GOOGLE_API_KEY_2'),
    os.getenv('GOOGLE_API_KEY_3'),
    os.getenv('GOOGLE_API_KEY_4')
]
# Filter out None or empty keys
GOOGLE_API_KEYS = [key for key in GOOGLE_API_KEYS if key]
if not GOOGLE_API_KEYS:
    raise ValueError("No Google API keys found in environment variables")
api_key_cycle = cycle(GOOGLE_API_KEYS)

# Rate limiting parameters
REQUESTS_PER_MINUTE = 60  # Adjust based on your API tier
last_request_time = time.time()

def rate_limit() -> None:
    """Implement rate limiting for API calls."""
    global last_request_time
    current_time = time.time()
    elapsed = current_time - last_request_time
    min_interval = 60.0 / REQUESTS_PER_MINUTE
    
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    
    last_request_time = time.time()

def get_next_api_key() -> str:
    """Get next API key with error handling and rotation."""
    key = next(api_key_cycle)
    genai.configure(api_key=key)
    return key 