"""
Ingestion Agent - PDF Processing and Metadata Extraction
Handles PDF text extraction, well name detection, and initial document processing
Supports both text-based and image-based (scanned) PDFs with OCR fallback
"""

import fitz  # PyMuPDF
import re
from typing import Dict, List, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import pdfplumber for table extraction
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
    logger.info("pdfplumber available for table extraction")
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available - install: pip install pdfplumber")


class IngestionAgent:
    """
    Processes PDF documents and extracts text with metadata
    
    Key responsibilities:
    - Extract text from PDF using PyMuPDF
    - Preserve page numbers for citation
    - Detect well names using regex patterns
    - Extract document metadata (title, author, dates)
    """
    
    def __init__(self, database_manager=None, table_parser=None):
        # Well name pattern for Dutch geothermal wells
        self.well_name_pattern = re.compile(r'\b([A-Z]{2,10}-GT-\d{2}(?:-S\d+)?)\b')
        
        # Database and table parser (optional - can be set later)
        self.db = database_manager
        self.table_parser = table_parser
    
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
        
        # Extract full text
        all_text = []
        page_count = len(doc)  # Store page count before closing
        
        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text()
            all_text.append(text)
        
        doc.close()
        
        # Combine all text
        full_text = '\n\n'.join(all_text)
        
        # Check if we got any meaningful text
        if len(full_text.strip()) < 100:
            logger.warning(f"  ⚠️ Very little text extracted ({len(full_text)} chars)")
        
        # Extract well names
        well_names = self._extract_well_names(full_text)
        
        return {
            'filename': pdf_path.name,
            'filepath': str(pdf_path),
            'content': full_text,
            'pages': page_count,
            'wells': well_names,
            'metadata': metadata
        }
    
    def _extract_well_names(self, text: str) -> List[str]:
        """
        Extract well names from text using regex
        
        Handles formats like:
        - [CODE]-##
        - [CODE]-##-S# (sidetrack)
        - [CODE]-##
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
    
    def _extract_section_headers(self, text: str) -> List[str]:
        """
        Extract major section headers from text
        
        Args:
            text: Full document text
            
        Returns:
            List of section headers
        """
        section_headers = []
        
        # Extract section headers (numbered sections like "4. GEOLOGY" or "4.1 Formation Tops")
        section_pattern = r'^\s*([0-9]+(?:\.[0-9]+)*)\s+([A-Z][A-Z\s,&-]+)$'
        for line in text.split('\n'):
            match = re.match(section_pattern, line.strip())
            if match:
                section_num = match.group(1)
                section_name = match.group(2).strip()
                section_headers.append(f"{section_num} {section_name}")
        
        return section_headers
    
    def extract_tables(self, pdf_path: str) -> List[Dict]:
        """
        Extract tables from PDF using pdfplumber
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of table dictionaries with structure:
            {
                'page': int,
                'table_num': int (on that page),
                'headers': List[str],
                'rows': List[List[str]],
                'metadata': Dict (section, table ref, etc.)
            }
        """
        if not PDFPLUMBER_AVAILABLE:
            logger.warning("pdfplumber not available - cannot extract tables")
            return []
        
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract tables with vertical_strategy="text" for invisible grids
                    page_tables = page.extract_tables(table_settings={
                        "vertical_strategy": "text",
                        "horizontal_strategy": "lines",
                        "intersection_tolerance": 5
                    })
                    
                    for table_idx, table_data in enumerate(page_tables, 1):
                        if not table_data or len(table_data) < 2:  # Need headers + at least 1 row
                            continue
                        
                        # First row is usually headers
                        headers = [str(cell).strip() if cell else "" for cell in table_data[0]]
                        
                        # Remaining rows are data
                        rows = []
                        for row in table_data[1:]:
                            row_data = [str(cell).strip() if cell else "" for cell in row]
                            # Skip empty rows
                            if any(cell for cell in row_data):
                                rows.append(row_data)
                        
                        if not rows:
                            continue
                        
                        # Get page text for context
                        page_text = page.extract_text() or ""
                        
                        # Try to find table reference in page text
                        table_ref = self._find_table_reference(page_text, table_idx)
                        
                        tables.append({
                            'page': page_num,
                            'table_num': table_idx,
                            'headers': headers,
                            'rows': rows,
                            'metadata': {
                                'table_ref': table_ref,
                                'num_rows': len(rows),
                                'num_cols': len(headers)
                            }
                        })
                        
                        logger.debug(f"  Extracted table {table_idx} from page {page_num}: {len(rows)} rows x {len(headers)} cols")
            
            logger.info(f"Extracted {len(tables)} tables from {pdf_path}")
            
        except Exception as e:
            logger.error(f"Table extraction failed for {pdf_path}: {str(e)}")
        
        return tables
    
    def _find_table_reference(self, page_text: str, table_num: int) -> str:
        """Find table reference (like 'Table 4-1: Casing Details') in page text"""
        # Look for "Table X" patterns
        pattern = r'Table\\s+([0-9]+(?:-[0-9]+)?|[IVX]+)\\s*[:\\-]?\\s*([^\\n]*)'
        matches = list(re.finditer(pattern, page_text, re.IGNORECASE))
        
        if matches and table_num <= len(matches):
            match = matches[table_num - 1]
            table_id = match.group(1)
            table_title = match.group(2).strip()[:100]
            return f"Table {table_id}: {table_title}"
        
        return f"Table {table_num}"
    
    def process_and_store_complete_tables(self, pdf_path: str, well_names: List[str]) -> int:
        """
        Extract complete tables from PDF and store in database
        Stores entire table structure, not individual rows
        
        Args:
            pdf_path: Path to PDF file
            well_names: List of well names found in document
            
        Returns:
            Number of tables successfully stored
        """
        if not self.db:
            logger.warning("Database not initialized - skipping table storage")
            return 0
        
        if not PDFPLUMBER_AVAILABLE:
            logger.warning("pdfplumber not available - cannot extract tables")
            return 0
        
        # Default to first well name if multiple found
        primary_well = well_names[0] if well_names else "UNKNOWN"
        
        tables = self.extract_tables(pdf_path)
        stored_count = 0
        
        for table in tables:
            try:
                # Store complete table with all data
                table_id = self.db.store_complete_table(
                    well_name=primary_well,
                    source_document=Path(pdf_path).name,
                    page=table['page'],
                    table_type='auto_detected',
                    table_reference=table['metadata'].get('table_ref', f"Table on page {table['page']}"),
                    headers=table['headers'],
                    rows=table['rows']
                )
                stored_count += 1
                logger.debug(f"  Stored complete table {table_id} from page {table['page']}")
                
            except Exception as e:
                logger.error(f"Failed to store table on page {table['page']}: {str(e)}")
                continue
        
        logger.info(f"Stored {stored_count} complete tables for {primary_well}")
        return stored_count

