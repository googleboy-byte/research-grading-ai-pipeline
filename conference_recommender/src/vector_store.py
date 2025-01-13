from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Any

class ConferenceVectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the vector store with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.papers = []
        self.embeddings = None
        
    def add_papers(self, conference_papers: Dict[str, List[Dict]]):
        """Add papers to the vector store."""
        for conference, papers in conference_papers.items():
            for paper in papers:
                self.papers.append({
                    'id': len(self.papers),
                    'conference': conference,
                    'text': paper['text'],
                    'filename': paper['filename']
                })
        
        # Create embeddings for all papers
        texts = [paper['text'] for paper in self.papers]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create nearest neighbors index
        self.index = NearestNeighbors(
            n_neighbors=min(10, len(self.papers)),  # Default to 10 neighbors or less if fewer papers
            metric='cosine'
        )
        self.index.fit(self.embeddings)
    
    def find_similar_papers(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar papers to the query text."""
        if self.index is None:
            raise ValueError("No papers have been added to the vector store")
        
        # Get query embedding
        query_embedding = self.model.encode([query_text])[0].reshape(1, -1)
        
        # Find nearest neighbors
        distances, indices = self.index.kneighbors(
            query_embedding,
            n_neighbors=min(top_k, len(self.papers))
        )
        
        # Process results
        similar_papers = []
        for idx, distance in zip(indices[0], distances[0]):
            paper = self.papers[idx]
            similar_papers.append({
                'conference': paper['conference'],
                'filename': paper['filename'],
                'similarity_score': 1 - distance  # Convert distance to similarity
            })
        
        return similar_papers 