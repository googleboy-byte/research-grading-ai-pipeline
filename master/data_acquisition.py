import feedparser
import requests
import os
from pathlib import Path
import logging
import random
from tqdm import tqdm
import time
from typing import List, Tuple
import PyPDF2
import io
import urllib.parse

# Set up logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = Path("master/data/Reference")
PUBLISHABLE_PATH = BASE_DIR / "Publishable"
NON_PUBLISHABLE_PATH = BASE_DIR / "Non-Publishable"

# ArXiv API base URL
ARXIV_API_URL = "http://export.arxiv.org/api/query"

# Conference directories with specific keywords and categories
CONFERENCE_CONFIGS = {
    "NeurIPS": {
        "dir": PUBLISHABLE_PATH / "NeurIPS",
        "keywords": ["neural", "deep learning", "machine learning"],
        "categories": ["cs.LG", "cs.AI", "stat.ML"],
        "count": 2
    },
    "CVPR": {
        "dir": PUBLISHABLE_PATH / "CVPR",
        "keywords": ["computer vision", "image", "visual"],
        "categories": ["cs.CV"],
        "count": 2
    },
    "ICLR": {
        "dir": PUBLISHABLE_PATH / "ICLR",
        "keywords": ["representation learning", "deep learning"],
        "categories": ["cs.LG", "cs.AI"],
        "count": 2
    },
    "TMLR": {
        "dir": PUBLISHABLE_PATH / "TMLR",
        "keywords": ["machine learning", "statistical learning"],
        "categories": ["cs.LG", "stat.ML"],
        "count": 2
    }
}

def ensure_directories():
    """Ensure all necessary directories exist"""
    for conf_config in CONFERENCE_CONFIGS.values():
        conf_config["dir"].mkdir(parents=True, exist_ok=True)
    NON_PUBLISHABLE_PATH.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directories created/verified")

def get_next_paper_id(directory: Path) -> int:
    """Get the next available paper ID in a directory"""
    existing_files = list(directory.glob("R*.pdf"))
    if not existing_files:
        return 1
    ids = [int(f.stem[1:]) for f in existing_files if f.stem[1:].isdigit()]
    return max(ids, default=0) + 1

def download_paper(url: str, max_retries: int = 3) -> bytes:
    """Download a paper from a URL with retries"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            logger.info(f"Successfully downloaded paper from {url}")
            return response.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
            time.sleep(2 ** attempt)

def is_valid_paper(pdf_content: bytes, min_pages: int = 4) -> bool:
    """Check if the paper meets basic quality criteria"""
    try:
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        is_valid = len(pdf_reader.pages) >= min_pages
        logger.info(f"Paper validation: {is_valid} (pages: {len(pdf_reader.pages)})")
        return is_valid
    except Exception as e:
        logger.error(f"Error validating PDF: {str(e)}")
        return False

def search_arxiv(query: str, max_results: int = 10) -> List[dict]:
    """Search arXiv using the API"""
    params = {
        'search_query': query,
        'start': 0,
        'max_results': max_results * 2,  # Get more to filter
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }
    
    try:
        response = requests.get(ARXIV_API_URL, params=params)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        logger.info(f"Found {len(feed.entries)} results for query: {query}")
        return feed.entries
    except Exception as e:
        logger.error(f"Error searching arXiv: {str(e)}")
        return []

def get_arxiv_papers(
    conference_name: str,
    keywords: List[str],
    categories: List[str],
    max_results: int = 2,
    published_year: int = 2023
) -> List[Tuple[str, bytes]]:
    """Get papers from arXiv based on conference criteria"""
    # Build simpler query first
    category_query = " OR ".join(f"cat:{cat}" for cat in categories)
    query = f"({category_query})"
    
    logger.info(f"Searching arXiv with query: {query}")
    
    papers = []
    entries = search_arxiv(query, max_results=max_results * 2)  # Get more results to filter
    
    for entry in entries:
        try:
            # Check if any keyword matches (case-insensitive)
            title_abstract = (entry.title + " " + entry.summary).lower()
            if not any(kw.lower() in title_abstract for kw in keywords):
                continue
                
            # Get PDF URL
            pdf_links = [link.href for link in entry.links if link.type == 'application/pdf']
            if not pdf_links:
                logger.warning(f"No PDF link found for paper: {entry.title}")
                continue
            pdf_url = pdf_links[0]
            
            logger.info(f"Downloading paper: {entry.title}")
            pdf_content = download_paper(pdf_url)
            
            if is_valid_paper(pdf_content):
                papers.append((entry.title, pdf_content))
                logger.info(f"Successfully processed paper: {entry.title}")
                if len(papers) >= max_results:
                    break
        except Exception as e:
            logger.warning(f"Failed to process paper {entry.title}: {str(e)}")
            continue
        
        time.sleep(5)  # Increased delay between papers
    
    return papers

def get_non_publishable_papers(max_results: int = 4) -> List[Tuple[str, bytes]]:
    """Get papers that are likely non-publishable based on specific criteria"""
    queries = [
        'title:preliminary',
        'title:"work in progress"',
        'title:"technical report"'
    ]
    
    papers = []
    for query in queries:
        if len(papers) >= max_results:
            break
            
        logger.info(f"Searching for non-publishable papers with query: {query}")
        entries = search_arxiv(query, max_results=max_results)
        
        for entry in entries:
            if len(papers) >= max_results:
                break
                
            try:
                # Get PDF URL
                pdf_links = [link.href for link in entry.links if link.type == 'application/pdf']
                if not pdf_links:
                    logger.warning(f"No PDF link found for paper: {entry.title}")
                    continue
                pdf_url = pdf_links[0]
                
                logger.info(f"Downloading non-publishable paper: {entry.title}")
                pdf_content = download_paper(pdf_url)
                
                if is_valid_paper(pdf_content):
                    papers.append((entry.title, pdf_content))
                    logger.info(f"Successfully processed non-publishable paper: {entry.title}")
            except Exception as e:
                logger.warning(f"Failed to process paper {entry.title}: {str(e)}")
                continue
            
            time.sleep(5)  # Increased delay between papers
    
    return papers

def save_paper(content: bytes, directory: Path, paper_id: int):
    """Save a paper to the specified directory"""
    file_path = directory / f"R{paper_id:03d}.pdf"
    with open(file_path, "wb") as f:
        f.write(content)
    logger.info(f"Saved paper as {file_path}")
    return file_path

def main():
    """Main function to acquire and save papers"""
    logger.info("Starting paper acquisition process")
    ensure_directories()
    
    # Download publishable papers for each conference
    for conf_name, config in tqdm(CONFERENCE_CONFIGS.items(), desc="Processing conferences"):
        logger.info(f"Processing conference: {conf_name}")
        conf_dir = config["dir"]
        
        try:
            papers = get_arxiv_papers(
                conference_name=conf_name,
                keywords=config["keywords"],
                categories=config["categories"],
                max_results=config["count"]
            )
            
            for title, content in papers:
                paper_id = get_next_paper_id(conf_dir)
                save_paper(content, conf_dir, paper_id)
                logger.info(f"Saved {conf_name} paper: {title}")
                time.sleep(5)  # Added delay between saves
                
        except Exception as e:
            logger.error(f"Error processing {conf_name}: {str(e)}")
            time.sleep(30)  # Added longer delay on error
            continue
    
    # Download non-publishable papers
    logger.info("Processing non-publishable papers")
    try:
        non_pub_papers = get_non_publishable_papers(max_results=4)
        for title, content in non_pub_papers:
            paper_id = get_next_paper_id(NON_PUBLISHABLE_PATH)
            save_paper(content, conf_dir, paper_id)
            logger.info(f"Saved non-publishable paper: {title}")
            time.sleep(5)  # Added delay between saves
    except Exception as e:
        logger.error(f"Error downloading non-publishable papers: {str(e)}")
    
    logger.info("Paper acquisition process completed")

if __name__ == "__main__":
    main() 