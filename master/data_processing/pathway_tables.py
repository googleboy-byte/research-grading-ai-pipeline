import pathway as pw
from pathlib import Path
from typing import List, Dict, Any
from .pdf_extractor import extract_pdf_content

def create_pathway_table(data: List[Dict[str, Any]]) -> pw.Table:
    """
    Create a Pathway table with proper schema from the extracted data.
    """
    # Define schema
    schema = pw.schema_from_types(
        title=str,
        num_pages=int,
        text=str
    )
    
    # Convert dictionaries to tuples in the correct order
    tuple_data = [
        (item["title"], item["num_pages"], item["text"])
        for item in data
    ]
    
    # Create table with schema
    return pw.debug.table_from_rows(
        rows=tuple_data,
        schema=schema
    )

def process_directory(paths: List[str]) -> pw.Table:
    """
    Create a Pathway table from PDF files in the specified paths.
    """
    # Process each PDF file
    results = []
    for pdf_path in paths:
        content = extract_pdf_content(pdf_path)
        results.append(content)
    
    # Create Pathway table with schema
    return create_pathway_table(results) 