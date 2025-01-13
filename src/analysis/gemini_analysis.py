import json
import logging
import time
import numpy as np
import google.generativeai as genai
from typing import List, Dict
from ..utils.cache import get_cache_key, cache
from ..utils.api import rate_limit, get_next_api_key

def get_gemini_analysis(text: str, max_retries: int = 3) -> str:
    """Get paper analysis from Gemini with API key rotation, caching, and retry logic."""
    cache_key = get_cache_key(text, 'gemini_analysis')
    
    if cache_key in cache:
        return cache[cache_key]
    
    prompt = f"""You are a research paper analysis system. Your task is to analyze the given paper segment and return ONLY a JSON object with no additional text, markdown, or formatting.

    Required JSON structure:
    {{
        "methodology_quality": {{
            "score": 8,
            "justification": "The methodology is well-defined and follows standard practices"
        }},
        "argument_coherence": {{
            "score": 7,
            "justification": "Arguments are presented logically with clear transitions"
        }},
        "technical_depth": {{
            "score": 9,
            "justification": "Demonstrates thorough understanding of the subject matter"
        }},
        "innovation_level": {{
            "score": 8,
            "justification": "Presents novel approaches to existing problems"
        }},
        "result_validation": {{
            "score": 7,
            "justification": "Results are validated through appropriate statistical methods"
        }},
        "writing_quality": {{
            "score": 8,
            "justification": "Clear and professional writing style"
        }}
    }}

    Rules:
    1. Return ONLY the JSON object, nothing else
    2. All scores must be integers between 1 and 10
    3. All justifications must be clear, concise strings
    4. Do not include any comments in the JSON
    5. Ensure all field names exactly match the template
    6. Use proper JSON formatting with double quotes

    Paper segment to analyze:
    {text}
    """
    
    for attempt in range(max_retries):
        try:
            # Get next API key and configure
            get_next_api_key()
            
            # Apply rate limiting
            rate_limit()
            
            # Make API call
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            result = response.text.strip()
            
            # Remove any potential markdown formatting
            if result.startswith('```json'):
                result = result[7:]
            if result.startswith('```'):
                result = result[3:]
            if result.endswith('```'):
                result = result[:-3]
            result = result.strip()
            
            # Validate JSON
            try:
                analysis_data = json.loads(result)
                
                # Validate structure and types
                required_fields = ['methodology_quality', 'argument_coherence', 'technical_depth',
                                 'innovation_level', 'result_validation', 'writing_quality']
                
                for field in required_fields:
                    if field not in analysis_data:
                        raise ValueError(f"Missing required field: {field}")
                    if 'score' not in analysis_data[field]:
                        raise ValueError(f"Missing score in {field}")
                    if 'justification' not in analysis_data[field]:
                        raise ValueError(f"Missing justification in {field}")
                    
                    score = analysis_data[field]['score']
                    if not isinstance(score, int) or score < 1 or score > 10:
                        raise ValueError(f"Invalid score in {field}: {score}")
                    
                    justification = analysis_data[field]['justification']
                    if not isinstance(justification, str) or not justification.strip():
                        raise ValueError(f"Invalid justification in {field}")
                
                cache[cache_key] = result
                return result
                
            except json.JSONDecodeError as e:
                logging.warning(f"JSON parsing error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    return None
                continue
            except ValueError as e:
                logging.warning(f"Validation error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    return None
                continue
            
        except Exception as e:
            logging.error(f"API error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)  # Exponential backoff

def combine_segment_analyses(analyses: List[str], max_retries: int = 3) -> str:
    """Combine analyses from different segments into a single analysis with retry logic."""
    cache_key = get_cache_key(str(analyses), 'combined_analysis')
    
    if cache_key in cache:
        return cache[cache_key]
    
    prompt = f"""You are a research paper analysis system. Your task is to combine multiple segment analyses into a single coherent analysis.
    Return ONLY a JSON object with no additional text, markdown, or formatting.

    Required JSON structure:
    {{
        "methodology_quality": {{
            "score": 8,
            "justification": "The methodology is well-defined and follows standard practices"
        }},
        "argument_coherence": {{
            "score": 7,
            "justification": "Arguments are presented logically with clear transitions"
        }},
        "technical_depth": {{
            "score": 9,
            "justification": "Demonstrates thorough understanding of the subject matter"
        }},
        "innovation_level": {{
            "score": 8,
            "justification": "Presents novel approaches to existing problems"
        }},
        "result_validation": {{
            "score": 7,
            "justification": "Results are validated through appropriate statistical methods"
        }},
        "writing_quality": {{
            "score": 8,
            "justification": "Clear and professional writing style"
        }}
    }}

    Rules:
    1. Return ONLY the JSON object, nothing else
    2. All scores must be integers between 1 and 10
    3. All justifications must be clear, concise strings
    4. Do not include any comments in the JSON
    5. Ensure all field names exactly match the template
    6. Use proper JSON formatting with double quotes
    7. Consider all segment analyses when creating the final scores

    Segment analyses to combine:
    {json.dumps(analyses, indent=2)}
    """
    
    for attempt in range(max_retries):
        try:
            # Get next API key and configure
            get_next_api_key()
            
            # Apply rate limiting
            rate_limit()
            
            # Make API call
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            result = response.text.strip()
            
            # Remove any potential markdown formatting
            if result.startswith('```json'):
                result = result[7:]
            if result.startswith('```'):
                result = result[3:]
            if result.endswith('```'):
                result = result[:-3]
            result = result.strip()
            
            # Validate JSON
            try:
                analysis_data = json.loads(result)
                
                # Validate structure and types
                required_fields = ['methodology_quality', 'argument_coherence', 'technical_depth',
                                 'innovation_level', 'result_validation', 'writing_quality']
                
                for field in required_fields:
                    if field not in analysis_data:
                        raise ValueError(f"Missing required field: {field}")
                    if 'score' not in analysis_data[field]:
                        raise ValueError(f"Missing score in {field}")
                    if 'justification' not in analysis_data[field]:
                        raise ValueError(f"Missing justification in {field}")
                    
                    score = analysis_data[field]['score']
                    if not isinstance(score, int) or score < 1 or score > 10:
                        raise ValueError(f"Invalid score in {field}: {score}")
                    
                    justification = analysis_data[field]['justification']
                    if not isinstance(justification, str) or not justification.strip():
                        raise ValueError(f"Invalid justification in {field}")
                
                cache[cache_key] = result
                return result
                
            except json.JSONDecodeError as e:
                logging.warning(f"JSON parsing error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    return None
                continue
            except ValueError as e:
                logging.warning(f"Validation error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    return None
                continue
            
        except Exception as e:
            logging.error(f"API error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)  # Exponential backoff

def get_publishability_decision(embedding: np.ndarray, analysis: str, max_retries: int = 3) -> str:
    """Get publishability decision using both BERT embedding and Gemini analysis with retry logic."""
    cache_key = get_cache_key(f"{str(embedding)}_{analysis}", 'decision')
    
    if cache_key in cache:
        return cache[cache_key]
    
    prompt = f"""You are a research paper evaluation system. Based on the following paper analysis, determine if the paper is publishable or not.

    Return ONLY a JSON object with exactly this structure, and nothing else before or after:
    {{
        "decision": "PUBLISHABLE",  // or "NON-PUBLISHABLE"
        "confidence_score": 0.85,   // number between 0.0 and 1.0
        "main_reasons": [
            "Clear methodology with proper validation",
            "Strong technical depth and innovation",
            "Well-structured arguments"
        ]
    }}

    Rules:
    1. The response must be a valid JSON object
    2. The "decision" must be exactly "PUBLISHABLE" or "NON-PUBLISHABLE"
    3. The "confidence_score" must be a number between 0.0 and 1.0
    4. The "main_reasons" must be a list of strings
    5. Do not include any comments in the actual response
    6. Do not include any text outside the JSON object

    Consider these evaluation criteria:
    - Methodology should be appropriate and well-justified
    - Arguments should be coherent and well-structured
    - Technical content should be sound
    - Results should be properly validated
    - Writing should be clear and professional

    Paper analysis:
    {analysis}
    """
    
    for attempt in range(max_retries):
        try:
            # Get next API key and configure
            get_next_api_key()
            
            # Apply rate limiting
            rate_limit()
            
            # Make API call
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            result = response.text.strip()
            
            # Remove any potential markdown formatting
            if result.startswith('```json'):
                result = result[7:]
            if result.startswith('```'):
                result = result[3:]
            if result.endswith('```'):
                result = result[:-3]
            result = result.strip()
            
            # Validate JSON
            try:
                decision_data = json.loads(result)
                
                # Validate decision format
                if decision_data.get('decision') not in ['PUBLISHABLE', 'NON-PUBLISHABLE']:
                    raise ValueError(f"Invalid decision value: {decision_data.get('decision')}")
                
                # Validate confidence score
                confidence_score = decision_data.get('confidence_score')
                if not isinstance(confidence_score, (int, float)) or confidence_score < 0 or confidence_score > 1:
                    raise ValueError(f"Invalid confidence score: {confidence_score}")
                
                # Validate main reasons
                main_reasons = decision_data.get('main_reasons')
                if not isinstance(main_reasons, list) or not all(isinstance(r, str) for r in main_reasons):
                    raise ValueError(f"Invalid main reasons format: {main_reasons}")
                
                cache[cache_key] = result
                return result
                
            except json.JSONDecodeError as e:
                logging.warning(f"Invalid JSON response on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    return None
                continue
            except (KeyError, ValueError) as e:
                logging.warning(f"Invalid response format on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    return None
                continue
            
        except Exception as e:
            logging.error(f"API error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)  # Exponential backoff

def extract_features_from_analysis(analysis_json: str) -> Dict:
    """Extract numerical features from Gemini analysis JSON."""
    try:
        analysis = json.loads(analysis_json)
        features = {}
        
        # Extract scores from each category
        for category in ['methodology_quality', 'argument_coherence', 'technical_depth',
                        'innovation_level', 'result_validation', 'writing_quality']:
            features[f"{category}_score"] = analysis[category]['score']
        
        return features
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Error extracting features from analysis: {str(e)}")
        return None 