"""
Ingestion Agent - PDF Processing and Metadata Extraction
Handles PDF text extraction, well name detection, and initial document processing
"""

import fitz  # PyMuPDF
import re
from typing import Dict, List, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IngestionAgent:
    """
    Processes PDF documents and extracts text with metadata
    
    Key responsibilities:
    - Extract text from PDF using PyMuPDF
    - Preserve page numbers for citation
    - Detect well names using regex patterns
    - Extract document metadata (title, author, dates)
    """
    
    def __init__(self):
        # Well name pattern for Dutch geothermal wells
        self.well_name_pattern = re.compile(r'\b([A-Z]{2,10}-GT-\d{2}(?:-S\d+)?)\b')
    
    def process(self, pdf_paths: List[str]) -> List[Dict]:
        """
        Process one or more PDF files
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            List of document dictionaries with structure:
            {
                'filename': str,
                'content': str,
                'pages': int,
                'wells': List[str],
                'metadata': Dict,
                'page_contents': List[Dict]  # Per-page content with page numbers
            }
        """
        documents = []
        
        for pdf_path in pdf_paths:
            try:
                doc_data = self._process_single_pdf(pdf_path)
                documents.append(doc_data)
                logger.info(f"✓ Processed {pdf_path}: {doc_data['pages']} pages, "
                          f"{len(doc_data['wells'])} well(s) detected")
            except Exception as e:
                logger.error(f"✗ Failed to process {pdf_path}: {str(e)}")
                continue
        
        return documents
    
    def _process_single_pdf(self, pdf_path: str) -> Dict:
        """Process a single PDF file"""
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Open PDF
        doc = fitz.open(str(pdf_path))
        
        # Extract metadata
        metadata = {
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'creator': doc.metadata.get('creator', ''),
            'producer': doc.metadata.get('producer', ''),
            'creation_date': doc.metadata.get('creationDate', ''),
            'mod_date': doc.metadata.get('modDate', '')
        }
        
        # Extract text page by page
        page_contents = []
        all_text = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            page_contents.append({
                'page_number': page_num + 1,  # 1-indexed for human readability
                'text': text
            })
            
            all_text.append(text)
        
        doc.close()
        
        # Combine all text
        full_text = '\n\n'.join(all_text)
        
        # Extract well names
        well_names = self._extract_well_names(full_text)
        
        return {
            'filename': pdf_path.name,
            'filepath': str(pdf_path),
            'content': full_text,
            'pages': len(page_contents),
            'wells': well_names,
            'metadata': metadata,
            'page_contents': page_contents
        }
    
    def _extract_well_names(self, text: str) -> List[str]:
        """
        Extract well names from text using regex
        
        Handles formats like:
        - ADK-GT-01
        - ADK-GT-01-S1 (sidetrack)
        - RNAU-GT-02
        """
        matches = self.well_name_pattern.findall(text)
        
        # Remove duplicates while preserving order
        unique_wells = []
        seen = set()
        for well in matches:
            if well not in seen:
                unique_wells.append(well)
                seen.add(well)
        
        return unique_wells
    
    def get_page_text(self, document: Dict, page_number: int) -> Optional[str]:
        """
        Get text from a specific page
        
        Args:
            document: Document dict from process()
            page_number: Page number (1-indexed)
            
        Returns:
            Text of the page, or None if page doesn't exist
        """
        for page in document['page_contents']:
            if page['page_number'] == page_number:
                return page['text']
        return None
    
    def search_pages(self, document: Dict, query: str, case_sensitive: bool = False) -> List[int]:
        """
        Search for text across all pages
        
        Args:
            document: Document dict from process()
            query: Search string
            case_sensitive: Whether to match case
            
        Returns:
            List of page numbers containing the query
        """
        matching_pages = []
        
        for page in document['page_contents']:
            text = page['text']
            if not case_sensitive:
                text = text.lower()
                query = query.lower()
            
            if query in text:
                matching_pages.append(page['page_number'])
        
        return matching_pages
