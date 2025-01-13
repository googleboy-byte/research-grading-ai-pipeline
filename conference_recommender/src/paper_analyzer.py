import re
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
from transformers import pipeline

@dataclass
class PaperSection:
    title: str
    abstract: str
    introduction: str
    methodology: str
    experiments: str
    results: str
    conclusion: str
    references: List[str]

class PaperAnalyzer:
    def __init__(self):
        """Initialize the paper analyzer with necessary models."""
        self.section_classifier = pipeline(
            "text-classification",
            model="bert-base-uncased",
            return_all_scores=True
        )
        
    def extract_sections(self, text: str) -> PaperSection:
        """Extract different sections from the paper text."""
        # Split text into potential sections
        sections = self._split_into_sections(text)
        
        # Initialize section contents
        section_contents = {
            'title': '',
            'abstract': '',
            'introduction': '',
            'methodology': '',
            'experiments': '',
            'results': '',
            'conclusion': '',
            'references': []
        }
        
        current_section = None
        for section in sections:
            section_type = self._classify_section(section)
            if section_type in section_contents:
                if section_type == 'references':
                    section_contents[section_type].extend(self._extract_references(section))
                else:
                    section_contents[section_type] = section
                    
        return PaperSection(**section_contents)
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split paper text into sections based on headers."""
        # Common section header patterns
        section_patterns = [
            r'\n[0-9]+\.\s+[A-Z][A-Za-z\s]+\n',  # Numbered sections
            r'\n[A-Z][A-Za-z\s]+\n',             # Capitalized headers
            r'\n[A-Z][A-Z\s]+\n'                 # All caps headers
        ]
        
        # Find all potential section boundaries
        boundaries = []
        for pattern in section_patterns:
            for match in re.finditer(pattern, text):
                boundaries.append((match.start(), match.end()))
        
        # Sort boundaries by position
        boundaries.sort()
        
        # Extract sections
        sections = []
        start = 0
        for boundary_start, boundary_end in boundaries:
            if start < boundary_start:
                sections.append(text[start:boundary_start].strip())
            sections.append(text[boundary_start:boundary_end].strip())
            start = boundary_end
            
        if start < len(text):
            sections.append(text[start:].strip())
            
        return [s for s in sections if s]
    
    def _classify_section(self, text: str) -> Optional[str]:
        """Classify the type of section based on its content."""
        # First check for common section headers
        header_patterns = {
            'abstract': r'abstract|summary',
            'introduction': r'introduction|background',
            'methodology': r'method|approach|methodology',
            'experiments': r'experiment|evaluation|implementation',
            'results': r'result|finding|discussion',
            'conclusion': r'conclusion|future work',
            'references': r'reference|bibliography|citation'
        }
        
        # Check first few lines for headers
        first_lines = '\n'.join(text.split('\n')[:3]).lower()
        for section_type, pattern in header_patterns.items():
            if re.search(pattern, first_lines):
                return section_type
                
        # If no clear header, use content classification
        scores = self.section_classifier(text[:512])[0]  # Use first 512 chars
        return max(scores, key=lambda x: x['score'])['label']
    
    def _extract_references(self, text: str) -> List[str]:
        """Extract individual references from the references section."""
        # Common reference patterns
        patterns = [
            r'\[[0-9]+\].*?\n',  # [1] Author et al...
            r'[0-9]+\.\s+.*?\n',  # 1. Author et al...
            r'\n[A-Z][a-z]+,.*?\n'  # Surname, Initial...
        ]
        
        references = []
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            references.extend(match.group().strip() for match in matches)
            
        return references
    
    def compute_technical_depth(self, paper: PaperSection) -> float:
        """Compute technical depth score based on various metrics."""
        metrics = {
            'equation_density': self._compute_equation_density(paper),
            'citation_density': self._compute_citation_density(paper),
            'methodology_complexity': self._compute_methodology_complexity(paper),
            'experimental_rigor': self._compute_experimental_rigor(paper)
        }
        
        # Weighted average of metrics
        weights = {
            'equation_density': 0.3,
            'citation_density': 0.2,
            'methodology_complexity': 0.3,
            'experimental_rigor': 0.2
        }
        
        return sum(score * weights[metric] for metric, score in metrics.items())
    
    def _compute_equation_density(self, paper: PaperSection) -> float:
        """Compute density of mathematical equations."""
        equation_patterns = [
            r'\$.*?\$',  # Inline equations
            r'\\\[.*?\\\]',  # Display equations
            r'\\begin\{equation\}.*?\\end\{equation\}'  # Numbered equations
        ]
        
        total_equations = 0
        total_text = len(paper.methodology) + len(paper.experiments) + len(paper.results)
        
        for pattern in equation_patterns:
            total_equations += len(re.findall(pattern, paper.methodology))
            total_equations += len(re.findall(pattern, paper.experiments))
            total_equations += len(re.findall(pattern, paper.results))
            
        return min(1.0, total_equations / (total_text / 500))  # Normalize to [0,1]
    
    def _compute_citation_density(self, paper: PaperSection) -> float:
        """Compute density and relevance of citations."""
        return min(1.0, len(paper.references) / 30)  # Normalize assuming 30 refs is good
    
    def _compute_methodology_complexity(self, paper: PaperSection) -> float:
        """Analyze complexity of methodology section."""
        # Look for key technical terms and concepts
        technical_terms = [
            'algorithm', 'theorem', 'proof', 'optimization',
            'framework', 'architecture', 'implementation'
        ]
        
        term_count = sum(
            paper.methodology.lower().count(term)
            for term in technical_terms
        )
        
        return min(1.0, term_count / 20)  # Normalize
    
    def _compute_experimental_rigor(self, paper: PaperSection) -> float:
        """Analyze rigor of experimental evaluation."""
        rigor_indicators = [
            'baseline', 'comparison', 'ablation', 'statistical',
            'significance', 'evaluation metric', 'dataset'
        ]
        
        indicator_count = sum(
            paper.experiments.lower().count(indicator)
            for indicator in rigor_indicators
        )
        
        return min(1.0, indicator_count / 15)  # Normalize 