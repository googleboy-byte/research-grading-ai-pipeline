from typing import List, Dict, Any
from .vector_store import ConferenceVectorStore
from .data_loader import ConferenceDataLoader
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter

class ConferenceRecommender:
    def __init__(self, vector_store: ConferenceVectorStore, data_loader: ConferenceDataLoader):
        """Initialize the conference recommender."""
        self.vector_store = vector_store
        self.data_loader = data_loader
        self.conference_metadata = data_loader.get_conference_metadata()
        
    def analyze_paper(self, paper_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Analyze a paper and recommend suitable conferences with justification.
        """
        # Find similar papers
        similar_papers = self.vector_store.find_similar_papers(paper_text, top_k=top_k)
        
        # Count conference occurrences among similar papers
        conference_counts = Counter([p['conference'] for p in similar_papers])
        
        # Get the most relevant conference
        recommended_conference = conference_counts.most_common(1)[0][0]
        
        # Generate justification
        justification = self._generate_justification(
            paper_text=paper_text,
            recommended_conference=recommended_conference,
            similar_papers=similar_papers
        )
        
        return {
            'recommended_conference': recommended_conference,
            'justification': justification,
            'similar_papers': similar_papers,
            'conference_distribution': dict(conference_counts)
        }
    
    def _generate_justification(self, paper_text: str, recommended_conference: str, 
                              similar_papers: List[Dict]) -> str:
        """Generate a justification for the conference recommendation."""
        conference_info = self.conference_metadata[recommended_conference]
        
        # Calculate similarity scores
        avg_similarity = np.mean([p['similarity_score'] for p in similar_papers 
                                if p['conference'] == recommended_conference])
        
        justification = (
            f"This paper aligns strongly with {recommended_conference}'s focus on "
            f"{', '.join(conference_info['focus_areas'][:3])}. "
            f"It shows {avg_similarity:.2%} similarity with existing {recommended_conference} papers. "
            f"The research methodology and findings are consistent with "
            f"{conference_info['description']}."
        )
        
        return justification[:500]  # Limit to 100 words approximately
    
    def get_conference_details(self, conference_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific conference."""
        return self.conference_metadata.get(conference_name, {}) 