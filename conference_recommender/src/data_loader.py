import os
import json
from typing import Dict, List
from PyPDF2 import PdfReader
from tqdm import tqdm

class ConferenceDataLoader:
    def __init__(self, base_path: str):
        """Initialize the data loader with the path to conference data."""
        self.base_path = base_path
        self.conferences = ['CVPR', 'NeurIPS', 'EMNLP', 'TMLR', 'KDD']
        
    def load_papers(self) -> Dict[str, List[Dict]]:
        """Load all papers from all conferences."""
        conference_papers = {}
        
        for conference in self.conferences:
            conference_path = os.path.join(self.base_path, conference)
            papers = []
            
            if not os.path.exists(conference_path):
                continue
                
            for filename in tqdm(os.listdir(conference_path), desc=f"Loading {conference} papers"):
                if filename.endswith('.pdf'):
                    paper_path = os.path.join(conference_path, filename)
                    try:
                        text = self._extract_text_from_pdf(paper_path)
                        papers.append({
                            'conference': conference,
                            'filename': filename,
                            'text': text
                        })
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
            
            conference_papers[conference] = papers
        
        return conference_papers
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()

    def get_conference_metadata(self) -> Dict[str, Dict]:
        """Return metadata about each conference's focus areas."""
        return {
            'CVPR': {
                'focus_areas': ['computer vision', 'pattern recognition', 'image processing', 
                              'deep learning for vision', 'visual understanding'],
                'description': 'Premier annual computer vision event'
            },
            'NeurIPS': {
                'focus_areas': ['machine learning', 'neural networks', 'artificial intelligence', 
                              'deep learning', 'optimization', 'statistical learning'],
                'description': 'Leading conference in machine learning and computational neuroscience'
            },
            'EMNLP': {
                'focus_areas': ['natural language processing', 'computational linguistics', 
                              'text mining', 'language understanding', 'speech processing'],
                'description': 'Top conference in empirical methods in natural language processing'
            },
            'TMLR': {
                'focus_areas': ['machine learning', 'theoretical advances', 'algorithms', 
                              'foundations of ML', 'learning theory'],
                'description': 'Transactions on Machine Learning Research'
            },
            'KDD': {
                'focus_areas': ['data mining', 'knowledge discovery', 'big data analytics', 
                              'data science', 'applied machine learning'],
                'description': 'Premier conference on data science and knowledge discovery'
            }
        } 