import os
import time
import requests
import arxiv
import logging
from typing import List, Dict
from urllib.parse import urlparse
from pathlib import Path
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestPaperDownloader:
    def __init__(self):
        self.arxiv_client = arxiv.Client()
        
        # Conference-specific search queries - using different years/criteria than training
        self.conference_queries = {
            'CVPR': 'cat:cs.CV AND (CVPR OR "Computer Vision and Pattern Recognition") AND submittedDate:[20220101 TO 20231231] AND ("camera ready" OR "accepted" OR "proceedings")',
            'NeurIPS': 'cat:cs.LG AND (NeurIPS OR "Neural Information Processing Systems") AND submittedDate:[20220101 TO 20231231] AND ("proceedings" OR "conference" OR "accepted")',
            'EMNLP': 'cat:cs.CL AND (EMNLP OR "Empirical Methods in Natural Language Processing") AND submittedDate:[20220101 TO 20231231] AND ("proceedings" OR "conference")',
            'KDD': 'cat:cs.DB AND (KDD OR "Knowledge Discovery and Data Mining") AND submittedDate:[20220101 TO 20231231] AND ("proceedings" OR "conference")',
            'TMLR': 'cat:cs.LG AND (TMLR OR "Transactions on Machine Learning Research") AND submittedDate:[20220101 TO 20231231]'
        }
        
        # Keep track of downloaded papers to avoid duplicates
        self.downloaded_papers = set()
        self.download_history_file = "test_papers_history.json"
        self.load_download_history()
        
    def load_download_history(self):
        """Load previously downloaded paper IDs."""
        try:
            if os.path.exists(self.download_history_file):
                with open(self.download_history_file, 'r') as f:
                    self.downloaded_papers = set(json.load(f))
        except Exception as e:
            logger.error(f"Error loading download history: {str(e)}")
            
    def save_download_history(self):
        """Save downloaded paper IDs."""
        try:
            with open(self.download_history_file, 'w') as f:
                json.dump(list(self.downloaded_papers), f)
        except Exception as e:
            logger.error(f"Error saving download history: {str(e)}")
        
    def download_paper(self, url: str, save_path: str, paper_id: str, title: str) -> bool:
        """Download a paper from a URL and save it to the specified path."""
        if paper_id in self.downloaded_papers:
            logger.info(f"Skipping already downloaded paper: {title}")
            return False
            
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            # Create progress bar for this paper
            desc = f"Downloading: {title[:50]}..." if len(title) > 50 else f"Downloading: {title}"
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        pbar.update(size)
            
            self.downloaded_papers.add(paper_id)
            self.save_download_history()
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {title}: {str(e)}")
            return False
    
    def download_papers_for_conference(self, conference: str, output_dir: str, num_papers: int = 5) -> List[str]:
        """Download papers for a specific conference."""
        conference_dir = os.path.join(output_dir, conference)
        os.makedirs(conference_dir, exist_ok=True)
        
        logger.info(f"\nSearching for {conference} test papers...")
        
        # Create arxiv search
        search = arxiv.Search(
            query=self.conference_queries[conference],
            max_results=num_papers * 3,  # Get more results to account for filtering
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        downloaded_papers = []
        downloaded_count = 0
        
        try:
            results = list(self.arxiv_client.results(search))
            
            for paper in results:
                if downloaded_count >= num_papers:
                    break
                    
                # Skip if we've downloaded this paper before
                if paper.entry_id in self.downloaded_papers:
                    continue
                    
                # Create filename from paper title
                safe_title = "".join(c for c in paper.title if c.isalnum() or c.isspace()).rstrip()
                safe_title = safe_title.replace(" ", "_")[:100]  # Limit length
                pdf_path = os.path.join(conference_dir, f"{safe_title}.pdf")
                
                # Skip if already downloaded
                if os.path.exists(pdf_path):
                    continue
                
                if self.download_paper(paper.pdf_url, pdf_path, paper.entry_id, paper.title):
                    downloaded_papers.append(pdf_path)
                    downloaded_count += 1
                    time.sleep(3)  # Be nice to arxiv servers
        
        except Exception as e:
            logger.error(f"Error searching papers for {conference}: {str(e)}")
        
        logger.info(f"Downloaded {len(downloaded_papers)} test papers for {conference}")
        return downloaded_papers
    
    def download_all_conferences(self, output_dir: str, papers_per_conference: int = 5) -> Dict[str, List[str]]:
        """Download papers for all conferences."""
        results = {}
        
        for conference in self.conference_queries.keys():
            papers = self.download_papers_for_conference(
                conference, 
                output_dir, 
                papers_per_conference
            )
            results[conference] = papers
            time.sleep(5)  # Add delay between conferences
        
        return results

def main():
    downloader = TestPaperDownloader()
    output_dir = "conference_training"  # Change to training directory
    papers_per_conference = 15  # Increase number of papers per conference
    
    try:
        logger.info("Starting paper downloads for training...")
        results = downloader.download_all_conferences(output_dir, papers_per_conference)
        
        # Print summary
        logger.info("\nDownload Summary:")
        total_papers = 0
        for conference, papers in results.items():
            logger.info(f"{conference}: {len(papers)} papers downloaded")
            total_papers += len(papers)
        logger.info(f"\nTotal papers downloaded: {total_papers}")
        
        # Download test papers
        test_output_dir = "test_papers"
        test_papers_per_conference = 10
        
        logger.info("\nStarting paper downloads for testing...")
        test_results = downloader.download_all_conferences(test_output_dir, test_papers_per_conference)
        
        # Print test summary
        logger.info("\nTest Set Download Summary:")
        total_test_papers = 0
        for conference, papers in test_results.items():
            logger.info(f"{conference}: {len(papers)} papers downloaded")
            total_test_papers += len(papers)
        logger.info(f"\nTotal test papers downloaded: {total_test_papers}")
        
    except Exception as e:
        logger.error(f"Error during download: {str(e)}")
        raise

if __name__ == "__main__":
    main() 