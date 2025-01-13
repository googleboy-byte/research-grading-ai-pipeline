from typing import Dict, List, Any
import numpy as np
from transformers import pipeline
from collections import defaultdict
import spacy
from dataclasses import dataclass

@dataclass
class JustificationContext:
    methodology_keywords: List[str]
    experimental_keywords: List[str]
    technical_terms: List[str]
    novelty_phrases: List[str]
    impact_phrases: List[str]

class JustificationGenerator:
    def __init__(self):
        """Initialize the justification generator with necessary models."""
        # Load spaCy for text analysis
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load summarizer for extracting key points
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            max_length=100,
            min_length=30
        )
        
        # Load zero-shot classifier for aspect analysis
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        # Initialize context templates
        self.context = JustificationContext(
            methodology_keywords=[
                "novel", "innovative", "state-of-the-art", "advanced",
                "comprehensive", "rigorous", "systematic"
            ],
            experimental_keywords=[
                "extensive", "thorough", "comparative", "ablation",
                "benchmark", "evaluation", "analysis"
            ],
            technical_terms=[
                "algorithm", "framework", "architecture", "optimization",
                "model", "system", "implementation"
            ],
            novelty_phrases=[
                "introduces a novel", "proposes an innovative",
                "advances the field through", "contributes significantly to"
            ],
            impact_phrases=[
                "has potential impact on", "addresses key challenges in",
                "provides valuable insights for", "advances understanding of"
            ]
        )

    def extract_key_aspects(self, paper_section: Any) -> Dict[str, List[str]]:
        """Extract key aspects from different paper sections."""
        aspects = defaultdict(list)
        
        # Analyze methodology
        method_doc = self.nlp(paper_section.methodology)
        aspects['methodology'] = [
            chunk.text for chunk in method_doc.noun_chunks
            if any(term in chunk.text.lower() for term in self.context.technical_terms)
        ]
        
        # Analyze experiments
        exp_doc = self.nlp(paper_section.experiments)
        aspects['experiments'] = [
            chunk.text for chunk in exp_doc.noun_chunks
            if any(term in chunk.text.lower() for term in self.context.experimental_keywords)
        ]
        
        # Get key results
        results_summary = self.summarizer(paper_section.results)[0]['summary_text']
        aspects['results'] = [results_summary]
        
        return aspects

    def analyze_technical_contribution(self, paper_section: Any) -> Dict[str, float]:
        """Analyze the technical contribution of the paper."""
        # Define aspects to analyze
        aspects = [
            "technical novelty",
            "theoretical contribution",
            "practical application",
            "experimental validation"
        ]
        
        # Combine relevant sections
        text = f"{paper_section.methodology} {paper_section.experiments} {paper_section.results}"
        
        # Classify text against aspects
        result = self.classifier(
            text,
            candidate_labels=aspects,
            multi_label=True
        )
        
        return dict(zip(result['labels'], result['scores']))

    def generate_conference_specific_points(self, 
                                         conference: str,
                                         similarities: Dict[str, float],
                                         aspects: Dict[str, List[str]]) -> List[str]:
        """Generate conference-specific justification points."""
        points = []
        
        # Conference-specific emphasis
        if conference == "CVPR":
            visual_result = self.classifier(
                " ".join(aspects['methodology']),
                candidate_labels=["computer vision", "image processing", "visual understanding"],
                multi_label=True
            )
            if max(visual_result['scores']) > 0.7:
                points.append(f"Strong focus on {visual_result['labels'][0]} with novel approaches")
                
        elif conference == "NeurIPS":
            theory_result = self.classifier(
                " ".join(aspects['methodology']),
                candidate_labels=["theoretical analysis", "mathematical framework", "algorithmic innovation"],
                multi_label=True
            )
            if max(theory_result['scores']) > 0.7:
                points.append(f"Strong theoretical foundation in {theory_result['labels'][0]}")
                
        elif conference == "EMNLP":
            nlp_result = self.classifier(
                " ".join(aspects['methodology']),
                candidate_labels=["natural language processing", "computational linguistics", "text analysis"],
                multi_label=True
            )
            if max(nlp_result['scores']) > 0.7:
                points.append(f"Significant contribution to {nlp_result['labels'][0]}")
        
        return points

    def generate_justification(self,
                             paper_section: Any,
                             conference: str,
                             similarities: Dict[str, float],
                             technical_depth: float,
                             citation_analysis: Dict[str, int]) -> str:
        """Generate a detailed justification for the conference recommendation."""
        # Extract key aspects from the paper
        aspects = self.extract_key_aspects(paper_section)
        
        # Analyze technical contribution
        contribution_scores = self.analyze_technical_contribution(paper_section)
        
        # Generate conference-specific points
        conf_points = self.generate_conference_specific_points(
            conference, similarities, aspects
        )
        
        # Build justification
        justification_points = []
        
        # Add methodology alignment
        if aspects['methodology']:
            method_highlight = np.random.choice(aspects['methodology'])
            method_phrase = np.random.choice(self.context.methodology_keywords)
            justification_points.append(
                f"The paper presents a {method_phrase} approach using {method_highlight}, "
                f"showing {similarities['methodology']:.1%} alignment with {conference} methodologies."
            )
        
        # Add technical contribution
        top_contribution = max(contribution_scores.items(), key=lambda x: x[1])
        justification_points.append(
            f"Strong {top_contribution[0]} (confidence: {top_contribution[1]:.1%}) "
            f"aligns well with {conference}'s focus areas."
        )
        
        # Add experimental validation
        if aspects['experiments']:
            exp_highlight = np.random.choice(aspects['experiments'])
            exp_phrase = np.random.choice(self.context.experimental_keywords)
            justification_points.append(
                f"The {exp_phrase} evaluation using {exp_highlight} demonstrates "
                f"rigorous experimental validation ({similarities['experiments']:.1%} match)."
            )
        
        # Add conference-specific points
        justification_points.extend(conf_points)
        
        # Add citation impact
        conf_citations = citation_analysis.get(conference, 0)
        if conf_citations > 0:
            justification_points.append(
                f"The paper builds upon {conf_citations} works from {conference}, "
                f"showing strong connection to the conference's research community."
            )
        
        # Combine all points
        return " ".join(justification_points) 