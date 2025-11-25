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
        
        # Extract text page by page
        page_contents = []
        all_text = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Extract metadata from page
            page_metadata = self._extract_page_metadata(text, page_num + 1)
            
            page_contents.append({
                'page_number': page_num + 1,  # 1-indexed for human readability
                'text': text,
                'section_headers': page_metadata['section_headers'],
                'table_refs': page_metadata['table_refs'],
                'figure_refs': page_metadata['figure_refs']
            })
            
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
            'pages': len(page_contents),
            'wells': well_names,
            'metadata': metadata,
            'page_contents': page_contents
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
    
    def _extract_page_metadata(self, text: str, page_num: int) -> Dict:
        """
        Extract metadata from page text: section headers, table/figure references
        
        Args:
            text: Page text content
            page_num: Page number (1-indexed)
            
        Returns:
            Dict with section_headers, table_refs, figure_refs
        """
        metadata = {
            'section_headers': [],
            'table_refs': [],
            'figure_refs': []
        }
        
        # Extract section headers (numbered sections like "4. GEOLOGY" or "4.1 Formation Tops")
        section_pattern = r'^\\s*([0-9]+(?:\\.[0-9]+)*)\\s+([A-Z][A-Z\\s,&-]+)$'
        for line in text.split('\\n'):
            match = re.match(section_pattern, line.strip())
            if match:
                section_num = match.group(1)
                section_name = match.group(2).strip()
                metadata['section_headers'].append(f"{section_num} {section_name}")
        
        # Extract table references ("Table 4-1", "Table 1:", etc.)
        table_pattern = r'Table\\s+([0-9]+(?:-[0-9]+)?|[IVX]+)\\s*[:\\-]?\\s*([^\\n]*?)(?:\\n|$)'
        for match in re.finditer(table_pattern, text, re.IGNORECASE):
            table_num = match.group(1)
            table_title = match.group(2).strip()[:50]  # First 50 chars of title
            metadata['table_refs'].append(f"Table {table_num}: {table_title}")
        
        # Extract figure references ("Figure 3-1", "Fig. 5:", etc.)
        figure_pattern = r'(?:Figure|Fig\\.)\\s+([0-9]+(?:-[0-9]+)?|[IVX]+)\\s*[:\\-]?\\s*([^\\n]*?)(?:\\n|$)'
        for match in re.finditer(figure_pattern, text, re.IGNORECASE):
            fig_num = match.group(1)
            fig_title = match.group(2).strip()[:50]
            metadata['figure_refs'].append(f"Figure {fig_num}: {fig_title}")
        
        return metadata
    
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
    
    def process_and_store_tables(self, pdf_path: str, well_names: List[str]) -> int:
        """
        Extract tables from PDF, parse them, and store in database
        Uses EnhancedTableParser for comprehensive data extraction
        
        Args:
            pdf_path: Path to PDF file
            well_names: List of well names found in document
            
        Returns:
            Number of tables successfully stored
        """
        if not self.db or not self.table_parser:
            logger.warning("Database or table parser not initialized - skipping table storage")
            return 0
        
        if not PDFPLUMBER_AVAILABLE:
            logger.warning("pdfplumber not available - cannot extract tables")
            return 0
        
        # Default to first well name if multiple found
        primary_well = well_names[0] if well_names else "UNKNOWN"
        
        tables = self.extract_tables(pdf_path)
        stored_count = 0
        general_data_stored = False
        
        for table in tables:
            try:
                # Get surrounding context from page text
                with pdfplumber.open(pdf_path) as pdf:
                    page = pdf.pages[table['page'] - 1]
                    page_text = page.extract_text() or ""
                
                # Identify table type
                table_type = self.table_parser.identify_table_type(
                    table['headers'], 
                    table['rows'],
                    page_text
                )
                
                logger.debug(f"Table on page {table['page']}: type={table_type}")
                
                # Skip excluded tables (geological logs, etc.)
                if table_type == 'excluded':
                    logger.debug(f"  Skipped excluded table on page {table['page']}")
                    continue
                
                # Parse and store based on type
                if table_type == 'general_data' and not general_data_stored:
                    general_data = self.table_parser.parse_general_data_table(
                        table['headers'],
                        table['rows'],
                        table['page']
                    )
                    if general_data:
                        # Remove source_page before passing to add_or_get_well (not a wells table column)
                        general_data.pop('source_page', None)
                        # Update well with general data
                        self.db.add_or_get_well(primary_well, **general_data)
                        stored_count += 1
                        general_data_stored = True
                        logger.info(f"  Stored general data for {primary_well}")
                
                elif table_type == 'casing':
                    casing_data = self.table_parser.parse_casing_table(
                        table['headers'], 
                        table['rows'],
                        table['page']
                    )
                    for casing in casing_data:
                        self.db.add_casing_string(primary_well, casing)
                    stored_count += len(casing_data)
                    logger.info(f"  Stored {len(casing_data)} casing strings for {primary_well}")
                
                elif table_type == 'cementing':
                    cementing_data = self.table_parser.parse_cementing_table(
                        table['headers'],
                        table['rows'],
                        table['page']
                    )
                    for cement_job in cementing_data:
                        self.db.add_cementing_job(primary_well, cement_job)
                    stored_count += len(cementing_data)
                    logger.info(f"  Stored {len(cementing_data)} cementing jobs for {primary_well}")
                
                elif table_type == 'fluids':
                    fluids_data = self.table_parser.parse_fluids_table(
                        table['headers'],
                        table['rows'],
                        table['page']
                    )
                    for fluid in fluids_data:
                        self.db.add_drilling_fluid(primary_well, fluid)
                    stored_count += len(fluids_data)
                    logger.info(f"  Stored {len(fluids_data)} fluid records for {primary_well}")
                
                elif table_type == 'formations':
                    formation_data = self.table_parser.parse_formation_table(
                        table['headers'],
                        table['rows'],
                        table['page']
                    )
                    for formation in formation_data:
                        self.db.add_formation(primary_well, formation)
                    stored_count += len(formation_data)
                    logger.info(f"  Stored {len(formation_data)} formations for {primary_well}")
                
                elif table_type == 'incidents':
                    incident_data = self.table_parser.parse_incidents_table(
                        table['headers'],
                        table['rows'],
                        table['page']
                    )
                    for incident in incident_data:
                        self.db.add_incident(primary_well, incident)
                    stored_count += len(incident_data)
                    logger.info(f"  Stored {len(incident_data)} incidents for {primary_well}")
                
            except Exception as e:
                    incident_data = self.table_parser.parse_incidents_table(
                        table['headers'],
                        table['rows'],
                        table['page']
                    )
                    for incident in incident_data:
                        self.db.add_incident(primary_well, incident)
                    stored_count += len(incident_data)
                    logger.info(f"  Stored {len(incident_data)} incidents for {primary_well}")
                
            except Exception as e:
                logger.error(f"Failed to process table on page {table['page']}: {str(e)}")
                logger.exception(e)  # Full stack trace for debugging
                continue
        
        logger.info(f"Stored {stored_count} table records for {primary_well}")
        return stored_count
    
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
