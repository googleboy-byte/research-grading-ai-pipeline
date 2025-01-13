import time
from typing import Dict, List, Any
from collections import defaultdict
import json
import os
from datetime import datetime

class StatsCollector:
    def __init__(self):
        """Initialize statistics collector."""
        self.reset_stats()

    def reset_stats(self):
        """Reset all statistics."""
        self.start_time = time.time()
        self.stats = {
            'total_papers': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'processing_times': [],
            'conference_distribution': defaultdict(int),
            'average_confidence': defaultdict(list),
            'gemini_success_rate': 0,
            'vector_success_rate': 0,
            'errors': defaultdict(int),
            'model_agreement_rate': 0,  # Rate at which Gemini and vector model agree
        }
        self.paper_times = {}
        self.model_agreements = []

    def start_paper_analysis(self, paper_path: str):
        """Start timing for a paper analysis."""
        self.paper_times[paper_path] = time.time()
        self.stats['total_papers'] += 1

    def end_paper_analysis(self, paper_path: str, success: bool, error: str = None):
        """End timing for a paper analysis."""
        if paper_path in self.paper_times:
            processing_time = time.time() - self.paper_times[paper_path]
            self.stats['processing_times'].append(processing_time)
            
            if success:
                self.stats['successful_analyses'] += 1
            else:
                self.stats['failed_analyses'] += 1
                if error:
                    self.stats['errors'][error] += 1

    def record_recommendation(self, analysis: Dict[str, Any]):
        """Record statistics from a recommendation."""
        if not analysis.get('recommended_conference'):
            return

        conf = analysis['recommended_conference']
        self.stats['conference_distribution'][conf] += 1
        self.stats['average_confidence'][conf].append(analysis.get('confidence_score', 0))

        # Record model agreement
        if 'model_score' in analysis and 'gemini_score' in analysis:
            vector_conf = analysis.get('vector_recommendation')
            gemini_conf = analysis.get('gemini_recommendation')
            self.model_agreements.append(vector_conf == gemini_conf)

    def calculate_final_stats(self) -> Dict[str, Any]:
        """Calculate final statistics."""
        total_time = time.time() - self.start_time
        processing_times = self.stats['processing_times']

        final_stats = {
            'total_papers': self.stats['total_papers'],
            'successful_analyses': self.stats['successful_analyses'],
            'failed_analyses': self.stats['failed_analyses'],
            'success_rate': self.stats['successful_analyses'] / max(self.stats['total_papers'], 1),
            'total_time': total_time,
            'average_time_per_paper': sum(processing_times) / max(len(processing_times), 1),
            'min_processing_time': min(processing_times) if processing_times else 0,
            'max_processing_time': max(processing_times) if processing_times else 0,
            'conference_distribution': dict(self.stats['conference_distribution']),
            'average_confidence_by_conference': {
                conf: sum(scores) / len(scores)
                for conf, scores in self.stats['average_confidence'].items()
            },
            'model_agreement_rate': sum(self.model_agreements) / max(len(self.model_agreements), 1),
            'common_errors': dict(self.stats['errors']),
        }

        return final_stats

    def save_stats(self, output_dir: str = "results"):
        """Save statistics to a JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"performance_stats_{timestamp}.json")
        
        stats = self.calculate_final_stats()
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return output_file

    def print_stats(self):
        """Print performance statistics."""
        stats = self.calculate_final_stats()
        
        print("\n=== Performance Statistics ===")
        print(f"Total Papers Analyzed: {stats['total_papers']}")
        print(f"Successful Analyses: {stats['successful_analyses']}")
        print(f"Failed Analyses: {stats['failed_analyses']}")
        print(f"Success Rate: {stats['success_rate']:.2%}")
        
        print(f"\nProcessing Times:")
        print(f"Total Time: {stats['total_time']:.2f} seconds")
        print(f"Average Time per Paper: {stats['average_time_per_paper']:.2f} seconds")
        print(f"Min Processing Time: {stats['min_processing_time']:.2f} seconds")
        print(f"Max Processing Time: {stats['max_processing_time']:.2f} seconds")
        
        print(f"\nConference Distribution:")
        for conf, count in stats['conference_distribution'].items():
            print(f"- {conf}: {count} papers")
            
        print(f"\nAverage Confidence by Conference:")
        for conf, confidence in stats['average_confidence_by_conference'].items():
            print(f"- {conf}: {confidence:.2%}")
            
        print(f"\nModel Agreement Rate: {stats['model_agreement_rate']:.2%}")
        
        if stats['common_errors']:
            print("\nCommon Errors:")
            for error, count in stats['common_errors'].items():
                print(f"- {error}: {count} occurrences") 