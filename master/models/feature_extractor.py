import pathway as pw
from typing import Dict, Any, List
import google.generativeai as genai
from google.api_core import retry
import time
from tenacity import retry, wait_exponential, stop_after_attempt, before_sleep_log
import logging
import json
from itertools import cycle
from datetime import datetime, timedelta
import argparse
from utils.cache import GeminiCache
import re
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define prompts for different feature extractions
ABSTRACT_PROMPT = """You are a research paper analyzer. Analyze this research paper and extract key information. If you can't find certain information, focus on what you can find in the text.

Return your analysis in this JSON format:
{{
    "main_topic": "The main topic/field of research",
    "objective": "The main objective/goal of the paper",
    "methodology": "Brief description of methodology used",
    "key_findings": "Key findings or results"
}}

Paper content:
{text}

Remember to focus on the information that is actually present in the text. If certain aspects are not clear, provide analysis based on what you can find.
"""

METHODOLOGY_PROMPT = """You are a research paper analyzer. Analyze the research methodology in this paper. If the methodology section is not clearly defined, look for methodological elements throughout the text.

Return your analysis in this JSON format:
{{
    "research_type": "empirical/theoretical/survey/etc (based on available information)",
    "methods_used": "List of methods/techniques mentioned in the text",
    "data_collection": "How data was collected/generated (if mentioned)",
    "analysis_techniques": "Analysis approaches found in the text"
}}

Paper content:
{text}

Focus on any methodological elements you can find in the text, even if they're not in a dedicated methodology section.
"""

QUALITY_PROMPT = """You are a research paper analyzer. Evaluate the quality aspects of this research paper based on the available content. Focus on what you can determine from the text provided.

Return your analysis in this JSON format:
{
    "methodology_score": {
        "score": "Assessment of methodology robustness (1-10)",
        "justification": "Detailed explanation of methodology score",
        "strengths": ["List of methodological strengths"],
        "weaknesses": ["List of methodological weaknesses"]
    },
    "results_score": {
        "score": "Assessment of results quality and validation (1-10)",
        "justification": "Detailed explanation of results score",
        "strengths": ["List of result strengths"],
        "weaknesses": ["List of result weaknesses"]
    },
    "novelty_score": {
        "score": "Assessment of research novelty and contribution (1-10)",
        "justification": "Detailed explanation of novelty score",
        "key_innovations": ["List of key innovative aspects"]
    },
    "presentation_score": {
        "score": "Assessment of paper organization and clarity (1-10)",
        "justification": "Detailed explanation of presentation score",
        "improvements_needed": ["List of suggested improvements"]
    },
    "citation_analysis": {
        "citation_quality": "Assessment of citation quality and relevance (1-10)",
        "citation_coverage": "Assessment of literature coverage (1-10)",
        "key_citations": ["List of most important citations"],
        "missing_citations": ["Areas where citations should be added"]
    },
    "statistical_rigor": {
        "score": "Assessment of statistical analysis quality (1-10)",
        "justification": "Explanation of statistical analysis score",
        "improvements_needed": ["Suggested statistical improvements"]
    },
    "overall_recommendation": {
        "score": "Overall paper quality score (1-10)",
        "confidence": "Confidence in assessment (1-10)",
        "key_strengths": ["List of major strengths"],
        "critical_weaknesses": ["List of critical weaknesses"],
        "improvement_priority": ["Prioritized list of suggested improvements"]
    }
}

Paper content:
{text}

Base your evaluation on the content that is actually present in the text. Be specific and detailed in your assessments.
"""

# Global variables
_models = {}  # Dictionary to store models for each API key
_current_model = None
_current_key = None
_api_keys_cycle = None
_last_call_times = {}  # Dictionary to store last call time for each key
_rate_limit_hits = {}  # Dictionary to track rate limit hits
_cache = None  # Global cache instance
MIN_DELAY = 2  # Reduced delay since we're using multiple keys
RATE_LIMIT_COOLDOWN = 60  # Cooldown period in seconds after hitting rate limit

class RateLimitExhausted(Exception):
    """Raised when all API keys are rate limited."""
    pass

def init_gemini(api_keys: List[str], use_cache: bool = True) -> None:
    """Initialize Gemini with multiple API keys"""
    global _models, _api_keys_cycle, _current_model, _current_key, _last_call_times, _rate_limit_hits, _cache
    
    # Create a model for each API key
    for key in api_keys:
        genai.configure(api_key=key)
        _models[key] = genai.GenerativeModel('gemini-pro')
        _last_call_times[key] = 0
        _rate_limit_hits[key] = None  # None means no rate limit hit
    
    # Create a cycler for the keys
    _api_keys_cycle = cycle(api_keys)
    _current_key = next(_api_keys_cycle)
    _current_model = _models[_current_key]
    
    # Initialize cache
    _cache = GeminiCache(use_cache=use_cache)
    
    logger.info(f"Initialized {len(api_keys)} Gemini models for API key rotation")
    if use_cache:
        logger.info("Gemini response caching is enabled")
    else:
        logger.info("Gemini response caching is disabled")

def is_key_available(key: str) -> bool:
    """Check if a key is available (not rate limited)"""
    hit_time = _rate_limit_hits.get(key)
    if hit_time is None:
        return True
    
    # Check if cooldown period has passed
    elapsed = time.time() - hit_time
    if elapsed >= RATE_LIMIT_COOLDOWN:
        _rate_limit_hits[key] = None  # Reset rate limit status
        return True
    
    return False

def find_available_key() -> str:
    """Find the next available API key"""
    global _current_key, _current_model, _api_keys_cycle
    
    # Try all keys once
    start_key = _current_key
    while True:
        key = next(_api_keys_cycle)
        if is_key_available(key):
            _current_key = key
            _current_model = _models[key]
            logger.info(f"Switched to API key: {key[:10]}...")
            return key
        
        # If we've tried all keys and come back to start
        if key == start_key:
            available_in = min(
                RATE_LIMIT_COOLDOWN - (time.time() - hit_time)
                for hit_time in _rate_limit_hits.values()
                if hit_time is not None
            )
            logger.warning(f"All keys rate limited. Next key available in {available_in:.1f} seconds")
            time.sleep(min(available_in + 1, RATE_LIMIT_COOLDOWN))
            # Reset rate limits after waiting
            for k in _rate_limit_hits:
                if _rate_limit_hits[k] is not None and time.time() - _rate_limit_hits[k] >= RATE_LIMIT_COOLDOWN:
                    _rate_limit_hits[k] = None

def handle_rate_limit(key: str) -> None:
    """Handle rate limit hit for a key"""
    global _rate_limit_hits
    _rate_limit_hits[key] = time.time()
    logger.warning(f"Rate limit hit for key {key[:10]}... Marking as limited for {RATE_LIMIT_COOLDOWN} seconds")

def wait_for_rate_limit():
    """Ensure minimum delay between API calls"""
    global _last_call_times
    current_time = time.time()
    last_call = _last_call_times.get(_current_key, 0)
    time_since_last_call = current_time - last_call
    
    if time_since_last_call < MIN_DELAY:
        sleep_time = MIN_DELAY - time_since_last_call
        logger.info(f"Rate limiting: Waiting {sleep_time:.2f} seconds")
        time.sleep(sleep_time)
    
    _last_call_times[_current_key] = time.time()

def chunk_text(text: str, max_chars: int = 60000) -> List[str]:
    """Split text into chunks that won't exceed token limits."""
    # Rough estimate: 1 token ≈ 4 characters
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        # If a single paragraph is too long, split it by sentences
        if len(paragraph) > max_chars:
            sentences = paragraph.split('. ')
            for sentence in sentences:
                if current_length + len(sentence) > max_chars:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = len(sentence)
                else:
                    current_chunk.append(sentence)
                    current_length += len(sentence)
        else:
            if current_length + len(paragraph) > max_chars:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_length = len(paragraph)
            else:
                current_chunk.append(paragraph)
                current_length += len(paragraph)
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def extract_key_sections(text: str) -> str:
    """Extract key sections from the paper text using heuristics."""
    key_sections = []
    
    # Try to find abstract
    abstract_match = re.search(r'(?i)abstract\s*(.*?)(?=\n\n|\n[A-Z][^\n]*\n\n|$)', text, re.DOTALL)
    if abstract_match:
        key_sections.append(abstract_match.group(1).strip())
    
    # Try to find introduction
    intro_match = re.search(r'(?i)(?:introduction|1\.?\s+introduction)\s*(.*?)(?=\n\n|\n[A-Z12][^\n]*\n\n|$)', text, re.DOTALL)
    if intro_match:
        key_sections.append(intro_match.group(1).strip())
    
    # Try to find methodology/methods section
    method_match = re.search(r'(?i)(?:methodology|methods|approach|proposed method)\s*(.*?)(?=\n\n|\n[A-Z12][^\n]*\n\n|$)', text, re.DOTALL)
    if method_match:
        key_sections.append(method_match.group(1).strip())
    
    # Try to find results/conclusion
    results_match = re.search(r'(?i)(?:results|conclusion|discussion)\s*(.*?)(?=\n\n|\n[A-Z12][^\n]*\n\n|$)', text, re.DOTALL)
    if results_match:
        key_sections.append(results_match.group(1).strip())
    
    # If we couldn't find any sections, take first and last few paragraphs
    if not key_sections:
        paragraphs = text.split('\n\n')
        key_sections.extend(paragraphs[:2])  # First two paragraphs
        if len(paragraphs) > 2:
            key_sections.extend(paragraphs[-2:])  # Last two paragraphs
    
    # Combine sections and ensure we don't exceed length
    combined = '\n\n'.join(key_sections)
    if len(combined) > 30000:  # Safe limit for tokens
        # Take first 15000 and last 15000 characters
        combined = combined[:15000] + "\n...[middle content omitted]...\n" + combined[-15000:]
    
    return combined

def select_relevant_content(text: str, query_type: str) -> str:
    """Use RAG to select the most relevant content based on analysis type."""
    # Define smaller chunk size (approximately 1000 tokens)
    MAX_CHUNK_SIZE = 4000  # characters
    FINAL_SIZE_LIMIT = 12000  # Significantly reduced from 30000
    
    # Split text into smaller semantic chunks
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        # If paragraph is too long, split it
        if len(para) > MAX_CHUNK_SIZE:
            sentences = para.split('. ')
            for sent in sentences:
                if current_size + len(sent) > MAX_CHUNK_SIZE:
                    if current_chunk:
                        chunks.append('\n'.join(current_chunk))
                    current_chunk = [sent]
                    current_size = len(sent)
                else:
                    current_chunk.append(sent)
                    current_size += len(sent)
        else:
            if current_size + len(para) > MAX_CHUNK_SIZE:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [para]
                current_size = len(para)
            else:
                current_chunk.append(para)
                current_size += len(para)
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    # Define scoring criteria based on analysis type
    keywords = {
        "abstract": {
            "high": ["abstract", "overview", "summary", "contribution"],
            "medium": ["objective", "goal", "propose"],
            "low": ["introduction", "background"]
        },
        "methodology": {
            "high": ["method", "algorithm", "approach", "implementation"],
            "medium": ["technique", "procedure", "experiment"],
            "low": ["data", "analysis", "tool"]
        },
        "quality": {
            "high": ["result", "evaluation", "performance", "accuracy"],
            "medium": ["comparison", "improvement", "novel"],
            "low": ["experiment", "test", "measure"]
        }
    }
    
    # Score chunks with weighted keyword matching
    chunk_scores = []
    selected_keywords = keywords.get(query_type, keywords["abstract"])
    
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = 0
        
        # Weight scores by keyword importance
        score += sum(3 for word in selected_keywords["high"] if word in chunk_lower)
        score += sum(2 for word in selected_keywords["medium"] if word in chunk_lower)
        score += sum(1 for word in selected_keywords["low"] if word in chunk_lower)
        
        # Boost score for section headers
        if re.search(r'^(?:abstract|introduction|method|approach|result|conclusion)', chunk_lower):
            score += 5
        
        # Boost score for first few chunks (likely abstract/intro)
        if chunks.index(chunk) < 3:
            score += 2
        
        # Penalize very short chunks
        if len(chunk) < 100:
            score -= 1
        
        chunk_scores.append((chunk, score))
    
    # Sort and select chunks
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Take top scoring chunks that fit within limit
    selected_text = []
    total_length = 0
    
    for chunk, score in chunk_scores:
        if score <= 0:  # Skip irrelevant chunks
            continue
        if total_length + len(chunk) <= FINAL_SIZE_LIMIT:
            selected_text.append(chunk)
            total_length += len(chunk)
        else:
            break
    
    # If we have no chunks with positive scores, take the first chunk
    if not selected_text and chunk_scores:
        selected_text = [chunk_scores[0][0]]
    
    relevant_text = '\n\n'.join(selected_text)
    
    logger.info(f"Selected {len(selected_text)} chunks totaling {len(relevant_text)} characters")
    
    return relevant_text

def extract_citations(text: str) -> Dict[str, Any]:
    """Extract and analyze citations from the paper text."""
    citations = []
    citation_pattern = r'\[([^\]]+)\]|\(([^)]+(?:\d{4})[^)]*)\)'
    
    matches = re.finditer(citation_pattern, text)
    for match in matches:
        citation = match.group(1) or match.group(2)
        citations.append(citation.strip())
    
    # Analyze citation patterns
    total_citations = len(citations)
    unique_citations = len(set(citations))
    recent_citations = sum(1 for c in citations if re.search(r'20[1-2]\d', c))
    
    return {
        "total_citations": total_citations,
        "unique_citations": unique_citations,
        "recent_citations": recent_citations,
        "citation_list": list(set(citations))
    }

@retry(
    wait=wait_exponential(multiplier=2, min=10, max=120),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.INFO),
    reraise=True
)
def analyze_with_gemini(text: str, prompt_template: str, include_citations: bool = True) -> str:
    """Analyze text using Gemini with enhanced citation analysis."""
    global _current_model, _cache
    
    # Check cache first
    cache_key = f"{text[:100]}_{prompt_template[:100]}"
    cached_response = _cache.get(cache_key) if _cache else None
    if cached_response:
        return cached_response
    
    try:
        # Extract citations if needed
        citation_info = extract_citations(text) if include_citations else None
        
        # Add citation information to text if available
        if citation_info:
            text += f"\n\nCitation Analysis:\n"
            text += f"Total Citations: {citation_info['total_citations']}\n"
            text += f"Unique Citations: {citation_info['unique_citations']}\n"
            text += f"Recent Citations (2010+): {citation_info['recent_citations']}\n"
        
        # Format prompt
        prompt = prompt_template.format(text=text)
        
        # Ensure we have an available key
        find_available_key()
        wait_for_rate_limit()
        
        try:
            response = _current_model.generate_content(prompt)
            result = response.text
            
            # Cache successful response
            if _cache:
                _cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            if "rate limit" in str(e).lower():
                handle_rate_limit(_current_key)
                raise RateLimitExhausted("Rate limit exceeded for all keys")
            raise
            
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

def extract_features(table: pw.Table, api_keys: List[str], use_cache: bool = True) -> pw.Table:
    """
    Extract features from papers using Gemini analysis with API key rotation.
    """
    # Initialize Gemini with multiple keys
    init_gemini(api_keys, use_cache=use_cache)
    
    # Create a progress tracker that updates based on completions
    class ProgressTracker:
        def __init__(self):
            self.pbar = None
            self.completed = 0
            self.current_type = ""
            
        def start(self):
            """Start progress bar without total (will show as unknown)"""
            # position=0 keeps it at bottom, leave=True keeps last bar, dynamic_ncols=True adjusts to terminal width
            self.pbar = tqdm(
                desc="Analyzing papers",
                unit="analysis",
                position=0,
                leave=True,
                dynamic_ncols=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
            )
            
        def update(self, result: Any, analysis_type: str) -> Any:
            """Update progress and return result"""
            if self.pbar is None:
                self.start()
            self.completed += 1
            self.current_type = analysis_type
            self.pbar.update(1)
            # Clear previous postfix and set new one to avoid overlapping
            self.pbar.set_postfix_str(f"Completed: {self.completed} | Current: {analysis_type}", refresh=True)
            # Force refresh to ensure bar stays at bottom
            self.pbar.refresh()
            return result
        
        def __del__(self):
            """Ensure progress bar is closed properly"""
            if self.pbar is not None:
                self.pbar.close()
    
    progress = ProgressTracker()
    
    # Redirect logging to avoid interfering with progress bar
    class TqdmLoggingHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except Exception:
                self.handleError(record)
    
    # Replace the default logger handler to use tqdm.write
    logger.handlers = [TqdmLoggingHandler()]
    
    # Create prompts and analyze
    logger.info(f"Processing papers using {len(api_keys)} API keys")
    return table.select(
        title=pw.this.title,
        num_pages=pw.this.num_pages,
        text=pw.this.text,
        abstract_analysis=pw.apply(
            lambda text: progress.update(
                analyze_with_gemini(text, ABSTRACT_PROMPT),
                "Abstract Analysis"
            ),
            pw.this.text
        ),
        methodology_analysis=pw.apply(
            lambda text: progress.update(
                analyze_with_gemini(text, METHODOLOGY_PROMPT),
                "Methodology Analysis"
            ),
            pw.this.text
        ),
        quality_analysis=pw.apply(
            lambda text: progress.update(
                analyze_with_gemini(text, QUALITY_PROMPT),
                "Quality Analysis"
            ),
            pw.this.text
        )
    ) 