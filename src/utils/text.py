import PyPDF2
from typing import List
import logging
from .cache import get_cache_key, cache

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from PDF file with caching."""
    cache_key = get_cache_key(pdf_path, 'pdf_text')
    
    if cache_key in cache:
        return cache[cache_key]
    
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {str(e)}")
        return None
    
    cache[cache_key] = text
    return text

def segment_text(text: str, max_length: int = 8000) -> List[str]:
    """Segment text into smaller chunks while preserving sentence boundaries."""
    # Split into sentences (using multiple delimiters)
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in ['.', '!', '?'] and len(current.strip()) > 0:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) > max_length:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks 