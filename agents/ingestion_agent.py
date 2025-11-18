"""PDF ingestion agent for extracting text from geothermal well reports."""

import fitz  # PyMuPDF
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DocumentMetadata:
    """Metadata extracted from document."""
    filename: str
    num_pages: int
    well_name: Optional[str] = None
    report_type: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None


@dataclass
class PageContent:
    """Content extracted from a single page."""
    page_num: int
    text: str
    metadata: Dict


class IngestionAgent:
    """
    Agent responsible for ingesting PDF documents and extracting text.
    
    This agent:
    1. Opens PDF files
    2. Extracts text from each page
    3. Preserves page boundaries and metadata
    4. Handles tables and structured content
    """
    
    def __init__(self):
        """Initialize the ingestion agent."""
        pass
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict:
        """
        Extract text and metadata from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with 'pages' (list of PageContent) and 'metadata' (DocumentMetadata)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        pages = []
        
        try:
            doc = fitz.open(pdf_path)
            
            # Extract metadata
            metadata = DocumentMetadata(
                filename=pdf_path.name,
                num_pages=len(doc),
                author=doc.metadata.get('author'),
                date=doc.metadata.get('creationDate')
            )
            
            # Extract text from each page
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                
                page_content = PageContent(
                    page_num=page_num,
                    text=text,
                    metadata={
                        'width': page.rect.width,
                        'height': page.rect.height,
                        'rotation': page.rotation
                    }
                )
                pages.append(page_content)
            
            doc.close()
            
            return {
                'pages': pages,
                'metadata': metadata,
                'full_text': '\n\n'.join([p.text for p in pages])
            }
            
        except Exception as e:
            raise RuntimeError(f"Error processing PDF {pdf_path}: {str(e)}")
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract tables from PDF (advanced feature).
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of extracted tables with page numbers
        """
        # Placeholder for table extraction logic
        # In a full implementation, this would use specialized table detection
        # libraries like camelot-py or tabula-py
        tables = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc, start=1):
                # Simple heuristic: look for structured text
                text = page.get_text()
                
                # Check if page contains table-like content
                lines = text.split('\n')
                numeric_lines = sum(1 for line in lines if any(char.isdigit() for char in line))
                
                if numeric_lines / len(lines) > 0.3 if lines else False:
                    tables.append({
                        'page_num': page_num,
                        'content': text,
                        'confidence': 'low'  # Placeholder
                    })
            
            doc.close()
            
        except Exception as e:
            print(f"Warning: Error extracting tables from {pdf_path}: {str(e)}")
        
        return tables
    
    def detect_well_name(self, text: str) -> Optional[str]:
        """
        Attempt to detect well name from document text.
        
        Args:
            text: Document text
            
        Returns:
            Detected well name or None
        """
        import re
        
        # Common patterns for well names
        patterns = [
            r'Well\s+Name[:\s]+([A-Z0-9-]+)',
            r'Well[:\s]+([A-Z]{3,5}-[A-Z]{2,4}-\d{2})',
            r'([A-Z]{3,5}-[A-Z]{2,4}-\d{2})',  # e.g., ADK-GT-01
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def process_document(self, pdf_path: str) -> Dict:
        """
        Complete document processing pipeline.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Processed document with text, metadata, and tables
        """
        # Extract text
        doc_data = self.extract_text_from_pdf(pdf_path)
        
        # Try to detect well name
        well_name = self.detect_well_name(doc_data['full_text'])
        if well_name:
            doc_data['metadata'].well_name = well_name
        
        # Extract tables
        tables = self.extract_tables_from_pdf(pdf_path)
        doc_data['tables'] = tables
        
        return doc_data
