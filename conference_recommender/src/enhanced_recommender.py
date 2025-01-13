from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from .gemini_analyzer import GeminiAnalyzer
from .paper_analyzer import PaperAnalyzer

class EnhancedRecommender:
    def __init__(self, publishability_threshold: float = 0.7):
        """Initialize the enhanced recommender system."""
        self.general_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.domain_models = {
            'CVPR': SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1"),
            'NeurIPS': SentenceTransformer("sentence-transformers/all-mpnet-base-v2"),
            'EMNLP': SentenceTransformer("sentence-transformers/all-distilroberta-v1")
        }
        self.publishability_threshold = publishability_threshold
        self.section_weights = {
            'abstract': 0.15,
            'methodology': 0.3,
            'experiments': 0.25,
            'results': 0.2,
            'conclusion': 0.1
        }
        
        # Initialize analyzers
        self.paper_analyzer = PaperAnalyzer()
        self.gemini_analyzer = GeminiAnalyzer()
        
    def analyze_paper_structure(self, paper_text: str) -> Any:
        """Extract structured information from paper."""
        return self.paper_analyzer.extract_sections(paper_text)
        
    def compute_technical_depth(self, paper_section: Any) -> float:
        """Evaluate technical depth of the paper."""
        return self.paper_analyzer.compute_technical_depth(paper_section)
        
    def analyze_citations(self, references: List[str]) -> Dict[str, int]:
        """Analyze citation patterns and conference distribution."""
        # TODO: Implement citation analysis
        pass
        
    def compute_section_similarities(self, 
                                   paper_section: Any,
                                   reference_papers: Dict[str, List[Any]]) -> Dict[str, float]:
        """Compute section-wise similarities with reference papers."""
        similarities = defaultdict(dict)
        
        for conf, papers in reference_papers.items():
            model = self.domain_models.get(conf, self.general_model)
            
            for section_name, weight in self.section_weights.items():
                section_text = getattr(paper_section, section_name)
                section_embeddings = model.encode([section_text])[0]
                
                ref_sections = [getattr(p, section_name) for p in papers]
                ref_embeddings = model.encode(ref_sections)
                
                section_similarities = np.dot(section_embeddings, ref_embeddings.T)
                similarities[conf][section_name] = np.mean(section_similarities) * weight
                
        return similarities
        
    def recommend_conference(self, 
                           paper_text: str,
                           publishability_score: float) -> Dict[str, Any]:
        """Generate comprehensive conference recommendation."""
        if publishability_score < self.publishability_threshold:
            return {
                "status": "rejected",
                "reason": "Paper does not meet minimum publishability threshold"
            }
            
        # Analyze paper structure
        paper_section = self.analyze_paper_structure(paper_text)
        
        # Get model-based recommendation
        model_recommendation = self._get_model_recommendation(paper_section)
        
        # Get Gemini-based recommendation
        gemini_recommendation = self.gemini_analyzer.analyze_paper(paper_section)
        
        # Combine recommendations
        final_recommendation = self.gemini_analyzer.combine_recommendations(
            gemini_recommendation,
            model_recommendation
        )
        
        return final_recommendation
        
    def _get_model_recommendation(self, paper_section: Any) -> Dict[str, Any]:
        """Get recommendation using the traditional model approach."""
        # Compute technical depth
        technical_depth = self.compute_technical_depth(paper_section)
        
        # Analyze citations
        citation_analysis = self.analyze_citations(paper_section.references)
        
        # Compute similarities
        similarities = self.compute_section_similarities(paper_section, self.reference_papers)
        
        # Determine best conference
        conference_scores = {
            conf: sum(section_scores.values())
            for conf, section_scores in similarities.items()
        }
        best_conference = max(conference_scores.items(), key=lambda x: x[1])[0]
        
        return {
            'recommended_conference': best_conference,
            'confidence_score': conference_scores[best_conference],
            'justification': self._generate_model_justification(
                paper_section,
                best_conference,
                similarities,
                technical_depth,
                citation_analysis
            ),
            'technical_depth': technical_depth,
            'section_scores': similarities[best_conference],
            'citation_analysis': citation_analysis,
            'similar_papers': []  # TODO: Add similar papers
        }
        
    def _generate_model_justification(self,
                                    paper_section: Any,
                                    conference: str,
                                    similarities: Dict[str, float],
                                    technical_depth: float,
                                    citation_analysis: Dict[str, int]) -> str:
        """Generate justification based on model analysis."""
        justification_points = []
        
        # Add methodology alignment
        method_similarity = similarities[conference]['methodology']
        justification_points.append(
            f"The methodology shows {method_similarity:.1%} alignment with {conference} papers."
        )
        
        # Add technical depth
        justification_points.append(
            f"Technical depth analysis indicates a {technical_depth:.1%} match with conference standards."
        )
        
        # Add experimental validation
        exp_similarity = similarities[conference]['experiments']
        justification_points.append(
            f"The experimental methodology shows {exp_similarity:.1%} alignment."
        )
        
        return " ".join(justification_points) 