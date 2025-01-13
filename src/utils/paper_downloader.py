import os
import time
import requests
import arxiv
import logging
from typing import List, Dict
from urllib.parse import urlparse
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperDownloader:
    def __init__(self):
        self.arxiv_client = arxiv.Client()
        
        # Conference-specific search queries
        self.conference_queries = {
            'CVPR': 'cat:cs.CV AND (CVPR OR "Computer Vision and Pattern Recognition")',
            'NeurIPS': 'cat:cs.LG AND (NeurIPS OR "Neural Information Processing Systems")',
            'EMNLP': 'cat:cs.CL AND (EMNLP OR "Empirical Methods in Natural Language Processing")',
            'KDD': 'cat:cs.DB AND (KDD OR "Knowledge Discovery and Data Mining")',
            'TMLR': 'cat:cs.LG AND (TMLR OR "Transactions on Machine Learning Research")'
        }
        
    def download_paper(self, url: str, save_path: str, title: str) -> bool:
        """Download a paper from a URL and save it to the specified path."""
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
            
            return True
        except Exception as e:
            logger.error(f"Error downloading {title}: {str(e)}")
            return False
    
    def download_papers_for_conference(self, conference: str, output_dir: str, num_papers: int = 5) -> List[str]:
        """Download papers for a specific conference."""
        conference_dir = os.path.join(output_dir, conference)
        os.makedirs(conference_dir, exist_ok=True)
        
        logger.info(f"\nSearching for {conference} papers...")
        
        # Create arxiv search
        search = arxiv.Search(
            query=self.conference_queries[conference],
            max_results=num_papers * 2,  # Get more results in case some downloads fail
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
                    
                # Create filename from paper title
                safe_title = "".join(c for c in paper.title if c.isalnum() or c.isspace()).rstrip()
                safe_title = safe_title.replace(" ", "_")[:100]  # Limit length
                pdf_path = os.path.join(conference_dir, f"{safe_title}.pdf")
                
                # Skip if already downloaded
                if os.path.exists(pdf_path):
                    continue
                
                if self.download_paper(paper.pdf_url, pdf_path, paper.title):
                    downloaded_papers.append(pdf_path)
                    downloaded_count += 1
                    time.sleep(3)  # Be nice to arxiv servers
        
        except Exception as e:
            logger.error(f"Error searching papers for {conference}: {str(e)}")
        
        logger.info(f"Downloaded {len(downloaded_papers)} papers for {conference}")
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
    downloader = PaperDownloader()
    output_dir = "conference_training"
    papers_per_conference = 5
    
    try:
        logger.info("Starting paper downloads...")
        results = downloader.download_all_conferences(output_dir, papers_per_conference)
        
        # Print summary
        logger.info("\nDownload Summary:")
        total_papers = 0
        for conference, papers in results.items():
            logger.info(f"{conference}: {len(papers)} papers downloaded")
            total_papers += len(papers)
        logger.info(f"\nTotal papers downloaded: {total_papers}")
        
    except Exception as e:
        logger.error(f"Error during download: {str(e)}")
        raise

if __name__ == "__main__":
    main() 