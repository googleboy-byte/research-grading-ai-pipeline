import hashlib
import diskcache

# Initialize cache
cache = diskcache.Cache('./cache')

def get_cache_key(data: str, prefix: str = '') -> str:
    """Generate a cache key for the given data."""
    return f"{prefix}_{hashlib.md5(data.encode()).hexdigest()}" 