import os
import json
from typing import Dict, Any, List
import google.generativeai as genai
from dotenv import load_dotenv
import time
from collections import deque

class GeminiAnalyzer:
    def __init__(self):
        """Initialize Gemini analyzer with API keys from .env"""
        load_dotenv()
        
        # Get all available Google API keys dynamically
        self.api_keys = []
        
        # Get all environment variables
        for key_name, value in os.environ.items():
            # Match any key that contains 'GOOGLE' and 'API_KEY'
            if 'GOOGLE' in key_name and 'API_KEY' in key_name and value:
                self.api_keys.append(value)
            
        if not self.api_keys:
            raise ValueError("No Google API keys found in .env file")
            
        print(f"Found {len(self.api_keys)} Google API keys")
            
        # Create a rotating queue of API keys
        self.key_queue = deque(self.api_keys)
        
        # Initialize models for each key
        self.models = {}
        self.active_keys = []  # Track only working keys
        
        for key in self.api_keys:
            try:
                genai.configure(api_key=key)
                model = genai.GenerativeModel('gemini-pro')
                # Test the model with a simple prompt
                test_response = model.generate_content("Test connection")
                if test_response:
                    self.models[key] = model
                    self.active_keys.append(key)
                    print(f"Successfully initialized model for key ending in ...{key[-4:]}")
            except Exception as e:
                print(f"Failed to initialize model for key ending in ...{key[-4:]}: {str(e)}")
        
        if not self.active_keys:
            raise ValueError("No working Google API keys found")
        
        # Update queue to only use working keys
        self.key_queue = deque(self.active_keys)
            
        # Conference descriptions for context
        self.conference_descriptions = {
            'CVPR': 'Premier computer vision conference focusing on visual understanding, image processing, and pattern recognition.',
            'NeurIPS': 'Top machine learning conference covering neural networks, AI, optimization, and theoretical advances.',
            'EMNLP': 'Leading natural language processing conference focusing on empirical methods and computational linguistics.',
            'TMLR': 'Transactions on Machine Learning Research, covering theoretical and practical ML advances.',
            'KDD': 'Premier data mining conference focusing on knowledge discovery and data science.'
        }
        
        # Add rate limiting
        self.last_call_time = {}
        self.min_delay = 1.0  # Minimum delay between calls to same API key
        self.max_retries_per_key = 3  # Maximum retries per key before moving to next
        
    def _rotate_key(self) -> str:
        """Rotate to next API key in queue with rate limiting and error tracking."""
        current_time = time.time()
        attempts = 0
        max_attempts = len(self.active_keys) * 2  # Allow two full rotations before giving up
        
        while attempts < max_attempts:
            current_key = self.key_queue[0]
            self.key_queue.rotate(-1)
            
            # Check if key is still in active keys (might have been removed due to errors)
            if current_key not in self.active_keys:
                continue
                
            # Check rate limiting
            last_used = self.last_call_time.get(current_key, 0)
            if current_time - last_used >= self.min_delay:
                self.last_call_time[current_key] = current_time
                return current_key
            
            attempts += 1
            time.sleep(0.1)  # Small delay before trying next key
            
        # If we're here, all keys are rate limited
        # Use the least recently used key
        current_key = min(
            ((k, v) for k, v in self.last_call_time.items() if k in self.active_keys),
            key=lambda x: x[1]
        )[0]
        self.last_call_time[current_key] = current_time
        return current_key
        
    def _handle_key_error(self, key: str, error: Exception) -> None:
        """Handle API key errors and potentially remove problematic keys."""
        print(f"Error with key ending in ...{key[-4:]}: {str(error)}")
        
        # Check if error indicates invalid or revoked key
        error_str = str(error).lower()
        if any(msg in error_str for msg in ['invalid', 'revoked', 'expired', 'unauthorized']):
            if key in self.active_keys:
                self.active_keys.remove(key)
                self.models.pop(key, None)
                print(f"Removed invalid key ending in ...{key[-4:]}")
                
                # Recreate queue with remaining active keys
                self.key_queue = deque(self.active_keys)
                
                if not self.active_keys:
                    raise ValueError("No working API keys remaining")
        
    def _create_paper_summary(self, paper_text: str) -> str:
        """Create a concise summary of the paper for Gemini."""
        if isinstance(paper_text, str):
            # Handle plain text input
            # Split the text into sections based on common headers
            text_lower = paper_text.lower()
            
            # Extract sections
            sections = {}
            
            # Try to find title (usually at the start, before abstract)
            title_end = text_lower.find('abstract')
            if title_end > 0:
                sections['title'] = paper_text[:title_end].strip()
            
            # Find abstract
            abstract_start = text_lower.find('abstract')
            if abstract_start >= 0:
                abstract_end = text_lower.find('introduction', abstract_start)
                if abstract_end > abstract_start:
                    sections['abstract'] = paper_text[abstract_start:abstract_end].strip()
            
            # Find methodology/methods section
            method_keywords = ['methodology', 'methods', 'approach', 'proposed method']
            for keyword in method_keywords:
                method_start = text_lower.find(keyword)
                if method_start >= 0:
                    next_section = float('inf')
                    for next_header in ['results', 'evaluation', 'experiments', 'conclusion']:
                        pos = text_lower.find(next_header, method_start)
                        if pos > method_start:
                            next_section = min(next_section, pos)
                    if next_section < float('inf'):
                        sections['methodology'] = paper_text[method_start:next_section].strip()
                    break
            
            # Find results/experiments section
            result_keywords = ['results', 'evaluation', 'experiments', 'experimental results']
            for keyword in result_keywords:
                result_start = text_lower.find(keyword)
                if result_start >= 0:
                    next_section = text_lower.find('conclusion', result_start)
                    if next_section > result_start:
                        sections['results'] = paper_text[result_start:next_section].strip()
                    break
            
            # Create summary
            summary_parts = []
            
            if 'title' in sections:
                summary_parts.append(f"Title: {sections['title'][:200]}")
            
            if 'abstract' in sections:
                summary_parts.append(f"Abstract: {sections['abstract'][:500]}")
                
            if 'methodology' in sections:
                summary_parts.append(f"Methodology Highlights: {sections['methodology'][:300]}")
                
            if 'results' in sections:
                summary_parts.append(f"Key Results: {sections['results'][:300]}")
            
            # If no sections were found, use the first part of the text
            if not summary_parts:
                summary_parts.append(f"Paper Content: {paper_text[:1000]}")
            
            return "\n\n".join(summary_parts)
            
        else:
            # Handle object with attributes (original behavior)
            summary_parts = []
            
            if hasattr(paper_text, 'title') and paper_text.title:
                summary_parts.append(f"Title: {paper_text.title[:200]}")
            
            if hasattr(paper_text, 'abstract') and paper_text.abstract:
                summary_parts.append(f"Abstract: {paper_text.abstract[:500]}")
                
            if hasattr(paper_text, 'methodology') and paper_text.methodology:
                summary_parts.append(f"Methodology Highlights: {paper_text.methodology[:300]}")
                
            if hasattr(paper_text, 'results') and paper_text.results:
                summary_parts.append(f"Key Results: {paper_text.results[:300]}")
                
            return "\n\n".join(summary_parts)
        
    def analyze_paper(self, paper_section: Any) -> Dict[str, Any]:
        """Analyze paper using Gemini and get conference recommendations."""
        print("\n=== Starting Gemini Analysis ===")
        
        if not self.active_keys:
            print("No active API keys available")
            return {
                "recommended_conference": None,
                "confidence_score": 0.0,
                "justification": "No working API keys available",
                "topic_alignment": None
            }

        # Create paper summary
        print("Creating paper summary...")
        try:
            paper_summary = self._create_paper_summary(paper_section)
            print(f"Summary length: {len(paper_summary)} characters")
        except Exception as e:
            print(f"Error creating paper summary: {str(e)}")
            print(f"Paper section type: {type(paper_section)}")
            print(f"Paper section attributes: {dir(paper_section)}")
            raise
        
        # Create prompt
        print("Creating Gemini prompt...")
        prompt = f'''As an expert in academic research, analyze this paper summary and recommend the most suitable conference among CVPR, NeurIPS, EMNLP, TMLR, and KDD.

Paper Summary:
{paper_summary}

Conference Descriptions:
{json.dumps(self.conference_descriptions, indent=2)}

Provide your analysis in the following JSON format (do not include any comments in the JSON):
{{
    "recommended_conference": "CONFERENCE_NAME",
    "confidence_score": 0.XX,
    "justification": "Clear explanation of why this conference is most suitable",
    "topic_alignment": "Main research topics/areas identified in the paper"
}}

Important: Return only valid JSON without any comments or additional text.'''

        errors = {}  # Track errors per key
        print(f"Number of active keys: {len(self.active_keys)}")
        
        while self.active_keys:  # Continue as long as we have working keys
            try:
                # Get current API key and model
                print("\nRotating to next API key...")
                current_key = self._rotate_key()
                print(f"Using key ending in ...{current_key[-4:]}")
                
                print("Getting model instance...")
                model = self.models[current_key]
                print(f"Model type: {type(model)}")
                
                # Generate response
                print("Generating content with Gemini...")
                response = model.generate_content(prompt)
                print(f"Response type: {type(response)}")
                print(f"Response attributes: {dir(response)}")
                
                # Parse JSON response
                try:
                    print("\nExtracting text from response...")
                    # Get text content from response
                    response_text = None
                    
                    try:
                        if hasattr(response, 'text'):
                            print("Using response.text")
                            response_text = response.text
                        elif hasattr(response, 'parts') and response.parts:
                            print("Using response.parts[0].text")
                            response_text = response.parts[0].text
                        elif hasattr(response, 'candidates') and response.candidates:
                            print("Using response.candidates[0].content.parts[0].text")
                            response_text = response.candidates[0].content.parts[0].text
                        else:
                            print("Using str(response)")
                            response_text = str(response)
                            
                        if not response_text:
                            raise ValueError("Empty response text")
                            
                    except Exception as e:
                        print(f"Error extracting text from response: {str(e)}")
                        print(f"Response structure: {response}")
                        if hasattr(response, '_result'):
                            print(f"Response _result: {response._result}")
                        raise ValueError(f"Failed to extract text from response: {str(e)}")
                    
                    print(f"\nRaw response text:\n{response_text}")
                    
                    # Clean the response text
                    # Remove any comments (text after # on any line)
                    cleaned_text = '\n'.join(
                        line.split('#')[0].rstrip()
                        for line in response_text.splitlines()
                    )
                    
                    # Try to parse JSON directly
                    try:
                        print("\nAttempting direct JSON parsing...")
                        result = json.loads(cleaned_text)
                        print("Successfully parsed JSON!")
                        print(f"Result: {json.dumps(result, indent=2)}")
                        return result  # Success! Return the result
                    except json.JSONDecodeError as je:
                        print(f"Direct JSON parsing failed: {str(je)}")
                        # If direct parsing fails, try to extract JSON-like content
                        print("Attempting to extract JSON-like content...")
                        start = cleaned_text.find('{')
                        end = cleaned_text.rfind('}') + 1
                        print(f"Found JSON markers at positions: {start} to {end}")
                        
                        if start >= 0 and end > start:
                            try:
                                extracted_json = cleaned_text[start:end]
                                print(f"\nExtracted JSON-like content:\n{extracted_json}")
                                result = json.loads(extracted_json)
                                print("Successfully parsed extracted JSON!")
                                print(f"Result: {json.dumps(result, indent=2)}")
                                return result
                            except json.JSONDecodeError as je2:
                                print(f"Failed to parse extracted JSON: {str(je2)}")
                                raise ValueError("Could not parse extracted JSON content")
                        else:
                            print("No JSON-like content found in response")
                            raise ValueError("No JSON-like content found in response")
                    
                except Exception as e:
                    print(f"\nError parsing Gemini response: {str(e)}")
                    print(f"Response type: {type(response)}")
                    if response_text:
                        print(f"Raw response: {response_text}")
                    raise
                
            except Exception as e:
                print(f"\nError in main analysis loop: {str(e)}")
                # Handle the error and potentially remove the key
                self._handle_key_error(current_key, e)
                
                # Track error for this key
                errors[current_key] = str(e)
                
                # If this was our last key, break
                if not self.active_keys:
                    print("No more active keys available")
                    break
                    
                print("Waiting before trying next key...")
                time.sleep(1)
        
        # If we're here, all keys have failed
        error_details = "; ".join(f"key ...{k[-4:]}: {e}" for k, e in errors.items())
        print(f"\nAll keys failed. Error details: {error_details}")
        return {
            "recommended_conference": None,
            "confidence_score": 0.0,
            "justification": f"All API keys failed. Errors: {error_details}",
            "topic_alignment": None
        }
            
    def combine_recommendations(self, 
                              gemini_result: Dict[str, Any],
                              model_result: Dict[str, Any],
                              weights: Dict[str, float] = None) -> Dict[str, Any]:
        """Combine Gemini and model recommendations."""
        print("\nCombining recommendations...")
        print(f"Gemini result: {json.dumps(gemini_result, indent=2)}")
        print(f"Model result: {json.dumps(model_result, indent=2)}")
        
        if weights is None:
            weights = {'gemini': 0.4, 'model': 0.6}
            
        # Validate and set default values for required fields
        def validate_result(result: Dict[str, Any], source: str) -> Dict[str, Any]:
            validated = result.copy()
            if not validated.get('recommended_conference'):
                print(f"Warning: Missing recommended_conference in {source} result")
                validated['recommended_conference'] = None
            if 'confidence_score' not in validated:
                print(f"Warning: Missing confidence_score in {source} result")
                validated['confidence_score'] = 0.0
            if 'justification' not in validated:
                print(f"Warning: Missing justification in {source} result")
                validated['justification'] = f"No justification provided by {source}"
            return validated
            
        gemini_result = validate_result(gemini_result, 'Gemini')
        model_result = validate_result(model_result, 'Model')
            
        # If Gemini result is invalid, return model result
        if not gemini_result['recommended_conference']:
            print("Using model result only (invalid Gemini result)")
            return model_result
            
        # If both recommend the same conference
        if gemini_result['recommended_conference'] == model_result['recommended_conference']:
            print("Both models recommend the same conference")
            combined_score = (
                weights['gemini'] * gemini_result['confidence_score'] +
                weights['model'] * model_result['confidence_score']
            )
            
            combined_result = {
                'recommended_conference': model_result['recommended_conference'],
                'confidence_score': combined_score,
                'justification': f"{model_result['justification']} Additionally, {gemini_result['justification']}",
                'model_score': model_result['confidence_score'],
                'gemini_score': gemini_result['confidence_score'],
                'topic_alignment': gemini_result.get('topic_alignment'),
                'conference_distribution': model_result.get('conference_distribution', {}),
                'similar_papers': model_result.get('similar_papers', []),
                'vector_recommendation': model_result['recommended_conference'],
                'gemini_recommendation': gemini_result['recommended_conference']
            }
            print(f"Combined result: {json.dumps(combined_result, indent=2)}")
            return combined_result
            
        # If different recommendations, go with the higher confidence one
        print("Models recommend different conferences, comparing confidence scores")
        gemini_weighted = gemini_result['confidence_score'] * weights['gemini']
        model_weighted = model_result['confidence_score'] * weights['model']
        
        if gemini_weighted > model_weighted:
            print("Using Gemini recommendation (higher weighted confidence)")
            primary, secondary = gemini_result, model_result
            source = 'Gemini'
        else:
            print("Using Model recommendation (higher weighted confidence)")
            primary, secondary = model_result, gemini_result
            source = 'Model'
            
        combined_result = {
            'recommended_conference': primary['recommended_conference'],
            'confidence_score': max(gemini_weighted, model_weighted),
            'justification': f"{primary['justification']} However, {source} analysis suggests {secondary['recommended_conference']} as an alternative with justification: {secondary['justification']}",
            'model_score': model_result['confidence_score'],
            'gemini_score': gemini_result['confidence_score'],
            'topic_alignment': gemini_result.get('topic_alignment'),
            'conference_distribution': model_result.get('conference_distribution', {}),
            'similar_papers': model_result.get('similar_papers', []),
            'vector_recommendation': model_result['recommended_conference'],
            'gemini_recommendation': gemini_result['recommended_conference']
        }
        print(f"Combined result: {json.dumps(combined_result, indent=2)}")
        return combined_result 