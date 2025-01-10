import os
from pathlib import Path
import fitz  # PyMuPDF
from typing import Dict, Any

def extract_pdf_content(pdf_path: str) -> Dict[str, Any]:
    """
    Extract content from a PDF file including text and metadata.
    Returns a dictionary containing the extracted information.
    """
    try:
        doc = fitz.open(pdf_path)
        text_content = []
        
        # Extract text from each page
        for page in doc:
            text_content.append(page.get_text())
        
        # Basic metadata
        metadata = {
            "title": os.path.basename(pdf_path),
            "num_pages": len(doc),
            "text": "\n".join(text_content)
        }
        
        doc.close()
        return metadata
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return {"error": str(e), "title": os.path.basename(pdf_path), "num_pages": 0, "text": ""} 