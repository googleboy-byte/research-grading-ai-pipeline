import os
from src.data_loader import ConferenceDataLoader
from src.vector_store import ConferenceVectorStore
from src.recommender import ConferenceRecommender
from src.gemini_analyzer import GeminiAnalyzer
from src.stats_collector import StatsCollector
from PyPDF2 import PdfReader
import argparse
import json
from datetime import datetime

def load_paper(paper_path: str) -> str:
    """Load and extract text from a paper PDF."""
    reader = PdfReader(paper_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def analyze_papers(input_path: str, recommender: ConferenceRecommender, 
                  gemini_analyzer: GeminiAnalyzer, stats: StatsCollector) -> list:
    """Analyze all papers in a directory or a single paper."""
    results = []
    
    if os.path.isfile(input_path):
        # Single file analysis
        if not input_path.lower().endswith('.pdf'):
            print(f"Skipping non-PDF file: {input_path}")
            return results
            
        print(f"\nAnalyzing paper: {input_path}")
        stats.start_paper_analysis(input_path)
        try:
            print("Loading paper text...")
            paper_text = load_paper(input_path)
            print(f"Paper text length: {len(paper_text)} characters")
            
            # Get vector-based analysis
            print("\nPerforming vector-based analysis...")
            vector_analysis = recommender.analyze_paper(paper_text)
            print("Vector analysis complete")
            print(f"Vector recommendation: {vector_analysis.get('recommended_conference')}")
            
            # Get Gemini analysis if available
            gemini_analysis = None
            if gemini_analyzer is not None:
                print("\nAttempting Gemini analysis...")
                try:
                    gemini_analysis = gemini_analyzer.analyze_paper(paper_text)
                    print("Gemini analysis complete")
                    print(f"Gemini recommendation: {gemini_analysis.get('recommended_conference') if gemini_analysis else 'None'}")
                except Exception as e:
                    print(f"Gemini analysis failed with error: {str(e)}")
                    print(f"Error type: {type(e)}")
                    print(f"Error details: {dir(e)}")
                    gemini_analysis = None
            else:
                print("Gemini analyzer not available, skipping Gemini analysis")
            
            # Combine recommendations if Gemini analysis is available
            print("\nCombining recommendations...")
            if gemini_analysis is not None and gemini_analyzer is not None:
                print("Using combined analysis from both models")
                analysis = gemini_analyzer.combine_recommendations(
                    gemini_result=gemini_analysis,
                    model_result=vector_analysis
                )
            else:
                print("Using vector-based analysis only")
                analysis = vector_analysis
                
            analysis['file_path'] = input_path
            results.append(analysis)
            
            # Record successful analysis
            print("Recording analysis statistics...")
            stats.end_paper_analysis(input_path, success=True)
            stats.record_recommendation(analysis)
            print("Analysis complete")
            
        except Exception as e:
            print(f"Error analyzing {input_path}: {str(e)}")
            print(f"Error type: {type(e)}")
            print(f"Error details: {dir(e)}")
            stats.end_paper_analysis(input_path, success=False, error=str(e))
            
    elif os.path.isdir(input_path):
        # Directory analysis
        for filename in os.listdir(input_path):
            if not filename.lower().endswith('.pdf'):
                print(f"Skipping non-PDF file: {filename}")
                continue
                
            file_path = os.path.join(input_path, filename)
            print(f"\nAnalyzing paper: {file_path}")
            stats.start_paper_analysis(file_path)
            try:
                print("Loading paper text...")
                paper_text = load_paper(file_path)
                print(f"Paper text length: {len(paper_text)} characters")
                
                # Get vector-based analysis
                print("\nPerforming vector-based analysis...")
                vector_analysis = recommender.analyze_paper(paper_text)
                print("Vector analysis complete")
                print(f"Vector recommendation: {vector_analysis.get('recommended_conference')}")
                
                # Get Gemini analysis if available
                gemini_analysis = None
                if gemini_analyzer is not None:
                    print("\nAttempting Gemini analysis...")
                    try:
                        gemini_analysis = gemini_analyzer.analyze_paper(paper_text)
                        print("Gemini analysis complete")
                        print(f"Gemini recommendation: {gemini_analysis.get('recommended_conference') if gemini_analysis else 'None'}")
                    except Exception as e:
                        print(f"Gemini analysis failed with error: {str(e)}")
                        print(f"Error type: {type(e)}")
                        print(f"Error details: {dir(e)}")
                        gemini_analysis = None
                else:
                    print("Gemini analyzer not available, skipping Gemini analysis")
                
                # Combine recommendations if Gemini analysis is available
                print("\nCombining recommendations...")
                if gemini_analysis is not None and gemini_analyzer is not None:
                    print("Using combined analysis from both models")
                    analysis = gemini_analyzer.combine_recommendations(
                        gemini_result=gemini_analysis,
                        model_result=vector_analysis
                    )
                else:
                    print("Using vector-based analysis only")
                    analysis = vector_analysis
                    
                analysis['file_path'] = file_path
                results.append(analysis)
                
                # Record successful analysis
                print("Recording analysis statistics...")
                stats.end_paper_analysis(file_path, success=True)
                stats.record_recommendation(analysis)
                print("Analysis complete")
                
            except Exception as e:
                print(f"Error analyzing {filename}: {str(e)}")
                print(f"Error type: {type(e)}")
                print(f"Error details: {dir(e)}")
                stats.end_paper_analysis(file_path, success=False, error=str(e))
    
    return results

def save_results(results: list, output_dir: str = "results"):
    """Save analysis results to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"conference_recommendations_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

def print_analysis(analysis: dict):
    """Print analysis results for a single paper."""
    print("\n=== Conference Recommendation Results ===")
    print(f"File: {analysis['file_path']}")
    print(f"\nRecommended Conference: {analysis['recommended_conference']}")
    print(f"Confidence Score: {analysis['confidence_score']:.2%}")
    
    if 'model_score' in analysis and 'gemini_score' in analysis:
        print(f"Vector Model Score: {analysis['model_score']:.2%}")
        print(f"Gemini Model Score: {analysis['gemini_score']:.2%}")
    
    print(f"\nJustification:\n{analysis['justification']}")
    
    if 'topic_alignment' in analysis and analysis['topic_alignment']:
        print(f"\nTopic Alignment:\n{analysis['topic_alignment']}")
    
    print("\nConference Distribution:")
    for conf, count in analysis['conference_distribution'].items():
        print(f"- {conf}: {count} similar papers")
    
    print("\nMost Similar Papers:")
    for i, paper in enumerate(analysis['similar_papers'][:3], 1):
        print(f"\n{i}. Conference: {paper['conference']}")
        print(f"   File: {paper['filename']}")
        print(f"   Similarity Score: {paper['similarity_score']:.2%}")

def main():
    parser = argparse.ArgumentParser(description='Conference Recommender System')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to a PDF file or directory containing PDF files to analyze')
    parser.add_argument('--training_data', type=str, default='conference_training',
                      help='Path to the conference training data directory')
    parser.add_argument('--save', action='store_true',
                      help='Save results to a JSON file')
    parser.add_argument('--no-gemini', action='store_true',
                      help='Disable Gemini analysis and use only vector-based recommendations')
    args = parser.parse_args()

    # Initialize components
    data_loader = ConferenceDataLoader(args.training_data)
    vector_store = ConferenceVectorStore()
    stats_collector = StatsCollector()
    
    print("Loading reference papers...")
    conference_papers = data_loader.load_papers()
    
    print("Building vector store...")
    vector_store.add_papers(conference_papers)
    
    # Initialize recommender and Gemini analyzer
    recommender = ConferenceRecommender(vector_store, data_loader)
    gemini_analyzer = None if args.no_gemini else GeminiAnalyzer()
    
    # Analyze papers
    results = analyze_papers(args.input, recommender, gemini_analyzer, stats_collector)
    
    # Print and save results
    for analysis in results:
        print_analysis(analysis)
    
    if args.save:
        save_results(results)
        stats_collector.save_stats()
    
    # Print performance statistics
    stats_collector.print_stats()
    
    print(f"\nAnalyzed {len(results)} papers successfully.")

if __name__ == "__main__":
    main() 