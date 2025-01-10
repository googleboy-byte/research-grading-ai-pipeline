import pathway as pw
import anthropic
from typing import Optional, List, Dict, Any, Union
import json
import logging
import pandas as pd
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tenacity import retry, stop_after_attempt, wait_exponential
import traceback
import os
from dotenv import load_dotenv
from pprint import pformat
from itertools import cycle
from threading import Lock
import hashlib
from sentence_transformers import CrossEncoder

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create a separate logger for RAG operations
rag_logger = logging.getLogger("rag_operations")
rag_logger.setLevel(logging.DEBUG)

# Add file handler for RAG operations
rag_handler = logging.FileHandler("rag_operations.log")
rag_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
rag_logger.addHandler(rag_handler)

# Load environment variables
load_dotenv()

# Configure API keys
ANTHROPIC_API_KEYS = [
    os.getenv("ANTHROPIC_API_KEY"),
    os.getenv("ANTHROPIC_API_KEY_1"),
    os.getenv("ANTHROPIC_API_KEY_2"),
    os.getenv("ANTHROPIC_API_KEY_3"),
    os.getenv("ANTHROPIC_API_KEY_4"),
    os.getenv("ANTHROPIC_API_KEY_5")
]

# Filter out None values and validate
ANTHROPIC_API_KEYS = [key for key in ANTHROPIC_API_KEYS if key]
if not ANTHROPIC_API_KEYS:
    raise ValueError("No valid Anthropic API keys found in environment variables")

class APIKeyRotator:
    """Handles API key rotation with thread safety"""
    def __init__(self, api_keys: List[str]):
        self.api_keys = cycle(api_keys)
        self.lock = Lock()
        self.current_key = next(self.api_keys)
    
    def get_next_key(self) -> str:
        """Get the next API key in the rotation"""
        with self.lock:
            self.current_key = next(self.api_keys)
            return self.current_key

class APICache:
    """Cache for API responses"""
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir / "api_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, prompt: str, model: str) -> str:
        """Generate a unique cache key for the request"""
        # Create a unique hash of the prompt and model
        content = f"{prompt}:{model}".encode('utf-8')
        return hashlib.sha256(content).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key"""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, prompt: str, model: str) -> Optional[str]:
        """Get cached response if it exists"""
        try:
            cache_key = self._get_cache_key(prompt, model)
            cache_path = self._get_cache_path(cache_key)
            
            if cache_path.exists():
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    rag_logger.debug(f"Cache hit for key: {cache_key[:8]}...")
                    return cached_data['response']
            
            return None
        except Exception as e:
            rag_logger.warning(f"Cache read failed: {str(e)}")
            return None
    
    def set(self, prompt: str, model: str, response: str):
        """Cache an API response"""
        try:
            cache_key = self._get_cache_key(prompt, model)
            cache_path = self._get_cache_path(cache_key)
            
            cache_data = {
                'prompt': prompt,
                'model': model,
                'response': response,
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            rag_logger.debug(f"Cached response for key: {cache_key[:8]}...")
        except Exception as e:
            rag_logger.warning(f"Cache write failed: {str(e)}")

class RAGError(Exception):
    """Base exception for RAG-related errors"""
    pass

# Define Pathway schema for documents
class DocumentSchema(pw.Schema):
    id: str
    content: str
    metadata: Dict
    embedding: List[float]

class ResearchPaperRAG:
    """RAG system for research paper analysis using Pathway"""
    
    def __init__(
        self,
        model_name: str = "claude-3-opus-20240229",  # Claude's latest model
        cache_dir: str = "master/cache/rag",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",  # Upgraded embedding model
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Added cross-encoder
        device: str = None,  # Will auto-detect
        min_content_length: int = 50  # Reduced minimum content length
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_content_length = min_content_length
        
        # Initialize API cache
        self.api_cache = APICache(self.cache_dir)
        
        # Initialize API key rotator
        self.key_rotator = APIKeyRotator(ANTHROPIC_API_KEYS)
        
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        rag_logger.info(f"Using device: {self.device}")
        
        try:
            # Initialize components
            self.model_name = model_name
            self._initialize_model()  # Initialize Claude client with first API key
            self.embedding_model = SentenceTransformer(embedding_model).to(self.device)
            self.cross_encoder = CrossEncoder(cross_encoder_model).to(self.device)  # Initialize cross-encoder
            self.docs = None
            
            # Store metadata about indexed papers
            self.paper_metadata = {}
            self.conference_metadata = {}
            
            rag_logger.info("Successfully initialized RAG components")
        except Exception as e:
            rag_logger.error(f"Failed to initialize RAG components: {str(e)}")
            rag_logger.debug(traceback.format_exc())
            raise RAGError(f"RAG initialization failed: {str(e)}")
    
    def _initialize_model(self):
        """Initialize or reinitialize the Claude client with current API key"""
        self.model = anthropic.Client(api_key=self.key_rotator.current_key)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_claude(self, prompt: str) -> str:
        """Call Claude API with retry logic, key rotation, and caching"""
        # Check cache first
        cached_response = self.api_cache.get(prompt, self.model_name)
        if cached_response is not None:
            return cached_response
        
        try:
            system_prompt = "You are a helpful research paper analysis assistant. Analyze the following academic content professionally and objectively."
            
            message = self.model.messages.create(
                model=self.model_name,
                max_tokens=4096,
                temperature=0.3,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            response_text = message.content[0].text
            
            # Cache the successful response
            self.api_cache.set(prompt, self.model_name, response_text)
            
            return response_text
            
        except Exception as e:
            # Log the specific error for debugging
            rag_logger.error(f"Claude API call failed: {str(e)}")
            
            # Rotate to next API key on failure
            rag_logger.warning(f"API call failed with key {self.key_rotator.current_key[:10]}..., rotating to next key")
            self.key_rotator.get_next_key()
            self._initialize_model()
            raise RAGError(f"Failed to call Claude API: {str(e)}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts with retry logic"""
        try:
            # Basic input validation
            if not texts:
                raise ValueError("No texts provided for embedding")
            if not all(isinstance(t, str) for t in texts):
                raise ValueError("All inputs must be strings")
            
            # Process in batches to avoid OOM
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                with torch.no_grad():
                    embeddings = self.embedding_model.encode(
                        batch,
                        convert_to_tensor=True,
                        device=self.device
                    )
                    all_embeddings.append(embeddings.cpu().numpy())
            
            return np.vstack(all_embeddings)
            
        except Exception as e:
            rag_logger.error(f"Embedding creation failed: {str(e)}")
            rag_logger.debug(traceback.format_exc())
            raise RAGError(f"Failed to create embeddings: {str(e)}")
    
    def _create_document_index(self, documents: List[Dict[str, Any]]) -> pw.Table:
        """Create a document index from the provided papers"""
        try:
            # Input validation
            if not documents:
                raise ValueError("No documents provided for indexing")
            
            # Create embeddings for all documents
            contents = [doc["content"] for doc in documents]
            embeddings = self._create_embeddings(contents)
            
            # Add embeddings to documents
            for doc, embedding in zip(documents, embeddings):
                doc["embedding"] = embedding.tolist()
            
            # Create a temporary JSONL file
            temp_file = self.cache_dir / "temp_docs.jsonl"
            with open(temp_file, 'w', encoding='utf-8') as f:
                for doc in documents:
                    f.write(json.dumps(doc) + '\n')
            
            # Create input table with documents
            docs = pw.io.jsonlines.read(
                temp_file,
                schema=DocumentSchema,
                mode="static"
            )
            
            return docs
            
        except Exception as e:
            rag_logger.error(f"Document indexing failed: {str(e)}")
            rag_logger.debug(traceback.format_exc())
            raise RAGError(f"Failed to create document index: {str(e)}")
        finally:
            # Clean up temporary file
            if 'temp_file' in locals():
                try:
                    temp_file.unlink(missing_ok=True)
                except Exception as e:
                    rag_logger.warning(f"Failed to delete temporary file: {str(e)}")
    
    def index_papers(
        self,
        papers: List[Dict[str, Any]],
        conference_papers: Optional[Dict[str, List[Dict[str, Any]]]] = None
    ):
        """Index research papers and conference papers for retrieval"""
        try:
            # Process main papers
            paper_docs = []
            for paper in papers:
                doc_id = paper.get("id", str(len(paper_docs)))
                content = self._prepare_paper_content(paper)
                metadata = {
                    "type": "research_paper",
                    "title": paper.get("title", "Unknown"),
                    "original_id": paper.get("id", "")
                }
                paper_docs.append({
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata
                })
                self.paper_metadata[doc_id] = metadata
            
            # Process conference papers if provided
            if conference_papers:
                for conf_name, conf_papers in conference_papers.items():
                    for paper in conf_papers:
                        doc_id = f"{conf_name}_{paper.get('id', str(len(paper_docs)))}"
                        content = self._prepare_paper_content(paper)
                        metadata = {
                            "type": "conference_paper",
                            "conference": conf_name,
                            "title": paper.get("title", "Unknown"),
                            "original_id": paper.get("id", "")
                        }
                        paper_docs.append({
                            "id": doc_id,
                            "content": content,
                            "metadata": metadata
                        })
                        self.conference_metadata[doc_id] = metadata
            
            # Create document index
            self.docs = self._create_document_index(paper_docs)
            
            rag_logger.info(f"Indexed {len(paper_docs)} documents")
            
        except Exception as e:
            rag_logger.error(f"Paper indexing failed: {str(e)}")
            rag_logger.debug(traceback.format_exc())
            raise RAGError(f"Failed to index papers: {str(e)}")
    
    def _validate_paper_content(self, content: str, paper_id: str) -> None:
        """Validate paper content meets minimum requirements"""
        if len(content) < self.min_content_length:
            rag_logger.warning(f"Paper {paper_id} content length ({len(content)}) is below minimum ({self.min_content_length})")
            raise RAGError(f"Paper content too short: {len(content)} chars")
        
        required_sections = ["Title", "Abstract", "Methodology"]
        missing_sections = [section for section in required_sections if section not in content]
        if missing_sections:
            rag_logger.warning(f"Paper {paper_id} missing required sections: {missing_sections}")
            raise RAGError(f"Missing required sections: {missing_sections}")

    def _prepare_paper_content(self, paper: Dict[str, Any]) -> str:
        """Prepare paper content for indexing"""
        try:
            # Input validation
            if not isinstance(paper, dict):
                raise ValueError("Paper must be a dictionary")
            
            paper_id = paper.get("id", "unknown")
            rag_logger.debug(f"Preparing content for paper: {paper_id}")
            
            # Combine all paper sections with proper formatting
            sections = []
            
            # Add each section with its full content
            if "title" in paper and paper["title"]:
                sections.append(f"Title:\n{paper['title']}")
            
            if "abstract" in paper and paper["abstract"]:
                sections.append(f"Abstract:\n{paper['abstract']}")
                
            if "methodology" in paper and paper["methodology"]:
                sections.append(f"Methodology:\n{paper['methodology']}")
                
            if "results" in paper and paper["results"]:
                sections.append(f"Results:\n{paper['results']}")
                
            if "conclusion" in paper and paper["conclusion"]:
                sections.append(f"Conclusion:\n{paper['conclusion']}")
            
            if not sections:
                raise ValueError("Paper has no valid content sections")
            
            # Join sections with clear separation
            content = "\n\n" + "="*50 + "\n\n".join(sections) + "\n\n" + "="*50 + "\n"
            
            # Log content details
            rag_logger.debug(f"Paper {paper_id} content length: {len(content)} chars")
            rag_logger.debug(f"Paper {paper_id} sections: {[s.split(':')[0] for s in sections]}")
            
            return content
            
        except Exception as e:
            rag_logger.error(f"Paper content preparation failed: {str(e)}")
            rag_logger.debug(traceback.format_exc())
            raise RAGError(f"Failed to prepare paper content: {str(e)}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def assess_publishability(self, paper_content: str, retrieved_context: str = "") -> Dict[str, Any]:
        """Assess if a paper is likely to be publishable"""
        try:
            prompt = f"""
            Analyze the following research paper content and assess its publishability.
            Consider factors like methodology, novelty, and technical depth.
            
            Paper content:
            {paper_content}  # Removed truncation to use full content
            
            Retrieved Context:
            {retrieved_context}

            Provide a structured assessment with these aspects:
            1. Overall publishability score (0-1)
            2. Key strengths
            3. Areas for improvement
            4. Specific recommendations
            
            Format your response as a JSON object with these exact keys:
            {{
                "publishable": bool,
                "confidence": float,
                "reasons": list[str],
                "improvements_needed": list[str]
            }}
            """
            
            response = self._call_claude(prompt)
            
            # Parse response as JSON
            try:
                result = json.loads(response)
                required_keys = ["publishable", "confidence", "reasons", "improvements_needed"]
                if not all(key in result for key in required_keys):
                    raise RAGError("Missing required keys in response")
                return result
            except json.JSONDecodeError:
                raise RAGError("Failed to parse Claude response as JSON")
                
        except Exception as e:
            rag_logger.error(f"Publishability assessment failed: {str(e)}")
            rag_logger.debug(traceback.format_exc())
            raise RAGError(f"Failed to assess publishability: {str(e)}")
    
    def recommend_conference(self, paper_content: str, retrieved_context: str = "", top_k: int = 3) -> List[Dict[str, Any]]:
        """Recommend conferences for paper submission based on content"""
        try:
            prompt = f"""
            Analyze this research paper and recommend the top {top_k} most suitable conferences for submission.
            Consider the paper's topic, methodology, and technical depth.
            
            Paper content:
            {paper_content}
            
            Retrieved Context:
            {retrieved_context}

            For each conference, provide:
            1. Conference name
            2. Relevance score (0-1)
            3. Reasoning for recommendation
            4. Key requirements to address
            
            Format your response as a JSON array of objects with these exact keys:
            [{{
                "name": str,
                "confidence": float,
                "reasoning": str,
                "requirements": list[str]
            }}]
            """
            
            response = self._call_claude(prompt)
            
            # Parse response as JSON
            try:
                results = json.loads(response)
                if not isinstance(results, list) or len(results) == 0:
                    raise RAGError("Invalid response format")
                return results[:top_k]  # Ensure we return at most top_k results
            except json.JSONDecodeError:
                raise RAGError("Failed to parse Claude response as JSON")
                
        except Exception as e:
            rag_logger.error(f"Conference recommendation failed: {str(e)}")
            rag_logger.debug(traceback.format_exc())
            raise RAGError(f"Failed to recommend conferences: {str(e)}")
    
    def _get_similar_documents(
        self,
        query: str,
        doc_type: str,
        num_similar: int,
        similarity_threshold: float = 0.5,
        rerank_top_k: int = 20  # Number of candidates for reranking
    ) -> pd.DataFrame:
        """Get similar documents using hybrid search and cross-encoder reranking"""
        try:
            # Get query embedding
            query_embedding = self._create_embeddings([query])[0]
            
            # Get all document embeddings
            doc_embeddings = np.array([doc.embedding for doc in self.docs])
            
            # Calculate semantic similarity scores
            semantic_scores = np.dot(doc_embeddings, query_embedding) / (
                np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top candidates based on semantic similarity
            top_indices = np.argsort(semantic_scores)[-rerank_top_k:][::-1]
            candidates = [self.docs[i] for i in top_indices]
            
            # Prepare pairs for cross-encoder
            pairs = [(query, doc.content) for doc in candidates]
            
            # Get cross-encoder scores
            with torch.no_grad():
                cross_encoder_scores = self.cross_encoder.predict(pairs)
            
            # Create DataFrame with results
            results = pd.DataFrame({
                'content': [doc.content for doc in candidates],
                'metadata': [doc.metadata for doc in candidates],
                'semantic_score': semantic_scores[top_indices],
                'cross_encoder_score': cross_encoder_scores,
                'final_score': cross_encoder_scores * 0.7 + semantic_scores[top_indices] * 0.3  # Weighted combination
            })
            
            # Filter by similarity threshold and sort by final score
            results = results[results['final_score'] >= similarity_threshold]
            results = results.sort_values('final_score', ascending=False)
            
            return results.head(num_similar)
            
        except Exception as e:
            rag_logger.error(f"Failed to get similar documents: {str(e)}")
            rag_logger.debug(traceback.format_exc())
            return pd.DataFrame()
    
    def get_publishability_assessment(
        self,
        paper: Dict[str, Any],
        num_similar: int = 3,
        similarity_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Get publishability assessment for a paper using RAG"""
        try:
            if not self.docs:
                raise ValueError("RAG system not initialized. Call index_papers first.")
            
            # Prepare paper content
            paper_content = self._prepare_paper_content(paper)
            rag_logger.info(f"Processing paper for publishability assessment: {paper.get('id', 'unknown')}")
            
            # Get similar papers
            similar_papers = self._get_similar_documents(
                paper_content,
                "research_paper",
                num_similar,
                similarity_threshold
            )
            
            if len(similar_papers) == 0:
                rag_logger.warning("No similar papers found for comparison")
                context = "No similar papers found for comparison."
            else:
                # Prepare context from similar papers
                context_parts = []
                for _, row in similar_papers.iterrows():
                    similarity = row['similarity']
                    content = row['content']
                    paper_id = row['id']
                    context_parts.append(
                        f"Similar Paper (similarity: {similarity:.3f}, id: {paper_id}):\n{content}"
                    )
                context = "\n\n".join(context_parts)
                
                rag_logger.debug(f"Using {len(similar_papers)} similar papers as context")
                rag_logger.debug(f"Total context length: {len(context)} chars")
            
            # Get assessment
            assessment = self.assess_publishability(
                paper_content=paper_content,
                retrieved_context=context
            )
            
            try:
                result = json.loads(assessment)
                rag_logger.info(
                    f"Assessment complete - "
                    f"Publishable: {result['publishable']}, "
                    f"Confidence: {result['confidence']:.3f}"
                )
                return result
            except json.JSONDecodeError:
                rag_logger.error("Failed to parse publishability assessment")
                return {
                    "publishable": False,
                    "confidence": 0.0,
                    "reasons": ["Error in assessment"],
                    "improvements_needed": ["Assessment failed"]
                }
                
        except Exception as e:
            rag_logger.error(f"Publishability assessment pipeline failed: {str(e)}")
            rag_logger.debug(traceback.format_exc())
            return {
                "publishable": False,
                "confidence": 0.0,
                "reasons": [f"Error: {str(e)}"],
                "improvements_needed": ["System error occurred"]
            }
    
    def get_conference_recommendations(
        self,
        paper: Dict[str, Any],
        num_similar: int = 5,
        similarity_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Get conference recommendations for a paper using RAG"""
        try:
            if not self.docs:
                raise ValueError("RAG system not initialized. Call index_papers first.")
            
            # Prepare paper content
            paper_content = self._prepare_paper_content(paper)
            rag_logger.info(f"Processing paper for conference recommendations: {paper.get('id', 'unknown')}")
            
            # Get similar conference papers
            similar_papers = self._get_similar_documents(
                paper_content,
                "conference_paper",
                num_similar,
                similarity_threshold
            )
            
            if len(similar_papers) == 0:
                rag_logger.warning("No similar conference papers found for comparison")
                context = "No similar conference papers found for comparison."
            else:
                # Prepare context from similar papers
                context_parts = []
                for _, row in similar_papers.iterrows():
                    similarity = row['similarity']
                    content = row['content']
                    paper_id = row['id']
                    conference = row['metadata'].get('conference', 'Unknown')
                    context_parts.append(
                        f"Similar Paper (conference: {conference}, similarity: {similarity:.3f}, id: {paper_id}):\n{content}"
                    )
                context = "\n\n".join(context_parts)
                
                rag_logger.debug(f"Using {len(similar_papers)} similar conference papers as context")
                rag_logger.debug(f"Total context length: {len(context)} chars")
            
            # Get recommendations
            recommendations = self.recommend_conference(
                paper_content=paper_content,
                retrieved_context=context
            )
            
            try:
                result = json.loads(recommendations)
                rag_logger.info(
                    f"Recommendations complete - "
                    f"Found {len(result.get('recommended_conferences', []))} recommendations"
                )
                return result
            except json.JSONDecodeError:
                rag_logger.error("Failed to parse conference recommendations")
                return {
                    "recommended_conferences": []
                }
                
        except Exception as e:
            rag_logger.error(f"Conference recommendation pipeline failed: {str(e)}")
            rag_logger.debug(traceback.format_exc())
            return {
                "recommended_conferences": []
            }
    
    def predict(self, table: pw.Table) -> List[Dict[str, Any]]:
        """Predict publishability for papers"""
        try:
            # Convert pathway table to pandas DataFrame
            pw.run()
            df = pw.debug.table_to_pandas(table)
            
            predictions = []
            for idx, row in df.iterrows():
                paper = {
                    "id": f"test_{idx}",
                    "title": f"Test Paper {str(idx)}",
                    "abstract": str(row.get("abstract", "")),
                    "methodology": str(row.get("methodology", "")),
                    "results": str(row.get("results", "")),
                    "conclusion": str(row.get("conclusion", ""))
                }
                
                # Get publishability assessment
                assessment = self.get_publishability_assessment(paper)
                predictions.append({
                    "publishable": assessment.get("publishable", False),
                    "confidence": assessment.get("confidence", 0.0)
                })
            
            return predictions
            
        except Exception as e:
            rag_logger.error(f"Prediction failed: {str(e)}")
            rag_logger.debug(traceback.format_exc())
            return [{"publishable": False, "confidence": 0.0}] 