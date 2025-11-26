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

# Try to import camelot for high-accuracy table extraction
try:
    import camelot
    CAMELOT_AVAILABLE = True
    logger.info("✓ camelot-py available for high-accuracy table extraction")
except ImportError:
    CAMELOT_AVAILABLE = False
    logger.warning("⚠️ camelot-py not available - install: pip install camelot-py[cv]")
    logger.warning("   Also requires: sudo apt-get install ghostscript python3-tk")


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
        Extract tables from PDF using camelot with accuracy-based quality control
        
        Uses hybrid approach:
        1. Lattice mode (for tables with visible borders/lines)
        2. Stream mode (for tables without borders, whitespace-based)
        3. Quality filtering (accuracy threshold to reject headers/footers)
        4. Overlap detection (prevent duplicates from both modes)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of table dictionaries with structure:
            {
                'page': int,
                'table_num': int,
                'headers': List[str],
                'rows': List[List[str]],
                'page_text': str,
                'metadata': Dict (accuracy, method, bbox, table_ref)
            }
        """
        if not CAMELOT_AVAILABLE:
            logger.warning("⚠️ camelot-py not available - cannot extract tables")
            logger.warning("   Install: pip install camelot-py[cv]")
            logger.warning("   System deps: sudo apt-get install ghostscript python3-tk")
            return []
        
        all_tables = []
        
        try:
            # PHASE 1: Extract with Lattice mode (line-based detection)
            logger.info(f"🔍 Phase 1: Lattice mode (line-based tables)...")
            try:
                tables_lattice = camelot.read_pdf(
                    str(pdf_path),
                    pages='all',
                    flavor='lattice',
                    line_scale=40,  # Sensitivity to line detection
                    suppress_stdout=True
                )
                
                for table in tables_lattice:
                    # Quality filter
                    if table.accuracy >= 75:  # Keep high-quality tables
                        all_tables.append({
                            'page': table.page,
                            'camelot_obj': table,
                            'data': table.data,  # 2D list
                            'accuracy': table.accuracy,
                            'method': 'lattice',
                            'bbox': table._bbox
                        })
                        logger.debug(f"  ✓ Table on page {table.page}: accuracy={table.accuracy:.1f}% (lattice)")
                    else:
                        logger.debug(f"  ✗ Rejected page {table.page}: accuracy={table.accuracy:.1f}% too low")
                
                logger.info(f"  → Found {len([t for t in all_tables if t['method']=='lattice'])} high-quality lattice tables")
                
            except Exception as e:
                logger.warning(f"  Lattice mode failed: {e}")
            
            # PHASE 2: Extract with Stream mode (whitespace-based detection)
            logger.info(f"🔍 Phase 2: Stream mode (borderless tables)...")
            try:
                tables_stream = camelot.read_pdf(
                    str(pdf_path),
                    pages='all',
                    flavor='stream',
                    edge_tol=50,  # Tolerance for column edges
                    row_tol=2,    # Tolerance for row detection
                    suppress_stdout=True
                )
                
                for table in tables_stream:
                    # Slightly lower threshold for stream mode
                    if table.accuracy >= 70:
                        # Check if overlaps with existing lattice table
                        if not self._overlaps_existing_table(table, all_tables):
                            all_tables.append({
                                'page': table.page,
                                'camelot_obj': table,
                                'data': table.data,
                                'accuracy': table.accuracy,
                                'method': 'stream',
                                'bbox': table._bbox
                            })
                            logger.debug(f"  ✓ Table on page {table.page}: accuracy={table.accuracy:.1f}% (stream)")
                        else:
                            logger.debug(f"  ⊗ Skipped page {table.page}: overlaps with lattice table")
                    else:
                        logger.debug(f"  ✗ Rejected page {table.page}: accuracy={table.accuracy:.1f}% too low")
                
                logger.info(f"  → Found {len([t for t in all_tables if t['method']=='stream'])} additional stream tables")
                
            except Exception as e:
                logger.warning(f"  Stream mode failed: {e}")
            
            # PHASE 3: Format tables into standard structure
            logger.info(f"📊 Phase 3: Formatting {len(all_tables)} tables...")
            formatted_tables = []
            
            # Get page text for context (using PyMuPDF)
            page_texts = self._get_page_texts(pdf_path)
            
            for idx, table_info in enumerate(sorted(all_tables, key=lambda x: x['page']), 1):
                table_data = table_info['data']
                
                if not table_data or len(table_data) < 2:
                    continue
                
                # First row as headers
                headers = [str(cell).strip() if cell else "" for cell in table_data[0]]
                
                # Remaining rows as data
                rows = []
                for row in table_data[1:]:
                    row_data = [str(cell).strip() if cell else "" for cell in row]
                    # Keep non-empty rows
                    if any(cell for cell in row_data):
                        rows.append(row_data)
                
                if not rows:
                    continue
                
                # Get page text for well detection
                page_num = table_info['page']
                page_text = page_texts.get(page_num, "")
                
                # Find table reference in page text
                table_ref = self._find_table_reference(page_text, idx)
                
                formatted_tables.append({
                    'page': page_num,
                    'table_num': idx,
                    'headers': headers,
                    'rows': rows,
                    'page_text': page_text,
                    'metadata': {
                        'table_ref': table_ref,
                        'num_rows': len(rows),
                        'num_cols': len(headers),
                        'accuracy': table_info['accuracy'],
                        'method': table_info['method'],
                        'bbox': table_info['bbox']
                    }
                })
                
                logger.debug(f"  ✓ Formatted table {idx} (page {page_num}): {len(rows)}×{len(headers)}, {table_info['accuracy']:.1f}% ({table_info['method']})")
            
            logger.info(f"✅ Extracted {len(formatted_tables)} tables total from {pdf_path}")
            return formatted_tables
            
        except Exception as e:
            logger.error(f"❌ Camelot extraction failed for {pdf_path}: {str(e)}")
            return []
    
    def _overlaps_existing_table(self, new_table, existing_tables: List[Dict]) -> bool:
        """Check if new table overlaps significantly with existing tables on same page"""
        new_page = new_table.page
        new_bbox = new_table._bbox  # (x0, top, x1, bottom)
        
        for existing in existing_tables:
            if existing['page'] == new_page:
                existing_bbox = existing['bbox']
                
                # Calculate overlap percentage
                overlap = self._calculate_bbox_overlap(new_bbox, existing_bbox)
                
                # If more than 50% overlap, consider it duplicate
                if overlap > 0.5:
                    return True
        
        return False
    
    def _calculate_bbox_overlap(self, bbox1, bbox2) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x0_1, top_1, x1_1, bottom_1 = bbox1
        x0_2, top_2, x1_2, bottom_2 = bbox2
        
        # Calculate intersection
        x0_inter = max(x0_1, x0_2)
        x1_inter = min(x1_1, x1_2)
        top_inter = max(top_1, top_2)
        bottom_inter = min(bottom_1, bottom_2)
        
        if x0_inter >= x1_inter or top_inter >= bottom_inter:
            return 0.0  # No overlap
        
        # Areas
        inter_area = (x1_inter - x0_inter) * (bottom_inter - top_inter)
        bbox1_area = (x1_1 - x0_1) * (bottom_1 - top_1)
        bbox2_area = (x1_2 - x0_2) * (bottom_2 - top_2)
        
        # Overlap ratio (intersection / smaller bbox)
        smaller_area = min(bbox1_area, bbox2_area)
        return inter_area / smaller_area if smaller_area > 0 else 0.0
    
    def _get_page_texts(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from all pages using PyMuPDF (for context)"""
        page_texts = {}
        
        try:
            doc = fitz.open(str(pdf_path))
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_texts[page_num + 1] = page.get_text()  # 1-indexed
            doc.close()
        except Exception as e:
            logger.warning(f"Could not extract page texts: {e}")
        
        return page_texts
    
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
        Extract complete tables from PDF and store in database with intelligent well name detection
        
        Multi-well detection strategy (priority order):
        1. Check table header/caption (above/below table) for well names
        2. Check table content (headers + rows) for well names
        3. If no wells found, assign ALL document well names (table may be relevant to all)
        4. Tables with multiple wells are stored once per well
        
        Args:
            pdf_path: Path to PDF file
            well_names: List of ALL well names found in document
            
        Returns:
            Number of table-well associations stored
        """
        if not self.db:
            logger.warning("Database not initialized - skipping table storage")
            return 0
        
        if not CAMELOT_AVAILABLE:
            logger.warning("⚠️ camelot-py not available - cannot extract tables")
            return 0
        
        if not well_names:
            logger.warning("No well names detected - cannot store tables")
            return 0
        
        tables = self.extract_tables(pdf_path)
        stored_count = 0
        
        # Import LLM helper for classification (lazy import to avoid circular dependency)
        try:
            from agents.llm_helper import OllamaHelper
            llm = OllamaHelper()
            llm_available = llm.is_available()
            if llm_available:
                logger.info("Using LLM for intelligent table classification")
        except Exception as e:
            logger.warning(f"LLM not available for table classification: {e}")
            llm_available = False
        
        for table in tables:
            try:
                # STEP 1: Detect which well(s) this table belongs to
                table_wells = self._detect_table_wells(
                    table=table,
                    document_wells=well_names
                )
                
                # STEP 2: Classify table type using LLM if available
                if llm_available and llm:
                    try:
                        table_type = llm.classify_table(
                            headers=table['headers'],
                            rows=table['rows'],
                            page=table['page']
                        )
                    except Exception as e:
                        logger.warning(f"LLM classification failed for table on page {table['page']}: {e}")
                        table_type = 'auto_detected'
                else:
                    table_type = 'auto_detected'
                
                # STEP 3: Store table for EACH detected well
                for well_name in table_wells:
                    table_id = self.db.store_complete_table(
                        well_name=well_name,
                        source_document=Path(pdf_path).name,
                        page=table['page'],
                        table_type=table_type,
                        table_reference=table['metadata'].get('table_ref', f"Table on page {table['page']}"),
                        headers=table['headers'],
                        rows=table['rows']
                    )
                    stored_count += 1
                    logger.debug(f"  Stored table {table_id} (type: {table_type}, well: {well_name}) from page {table['page']}")
                
            except Exception as e:
                logger.error(f"Failed to store table on page {table['page']}: {str(e)}")
                continue
        
        logger.info(f"Stored {stored_count} table-well associations across {len(well_names)} wells")
        return stored_count
    
    def _detect_table_wells(self, table: Dict, document_wells: List[str]) -> List[str]:
        """
        Detect which well(s) a table belongs to using 4-step priority system
        
        Priority:
        1. Table caption/header (text above/below table with well name)
        2. Table content (headers + cell values)
        3. Fallback: ALL document wells (table may be relevant to all)
        
        Args:
            table: Table dict with 'headers', 'rows', 'page_text', 'metadata'
            document_wells: List of all well names in document
            
        Returns:
            List of well names this table belongs to (1+ wells)
        """
        page_text = table.get('page_text', '')
        table_ref = table['metadata'].get('table_ref', '')
        
        # STEP 1: Check table caption/header (text around table reference)
        caption_wells = self._find_wells_in_caption(page_text, table_ref, document_wells)
        if caption_wells:
            logger.debug(f"  Found {len(caption_wells)} well(s) in table caption: {caption_wells}")
            return caption_wells
        
        # STEP 2: Check table content (headers + rows)
        content_wells = self._find_wells_in_table_content(
            table['headers'],
            table['rows'],
            document_wells
        )
        if content_wells:
            logger.debug(f"  Found {len(content_wells)} well(s) in table content: {content_wells}")
            return content_wells
        
        # STEP 3: No wells found - assign to ALL document wells
        logger.debug(f"  No specific wells detected - assigning to all {len(document_wells)} wells")
        return document_wells
    
    def _find_wells_in_caption(self, page_text: str, table_ref: str, document_wells: List[str]) -> List[str]:
        """
        Find well names in table caption (1 line before + same line + 1 line after)
        
        Strategy:
        - Extract the line containing table reference
        - Extract 1 line BEFORE (header)
        - Extract 1 line AFTER (caption)
        - BUT: Stop at empty lines (line breaks) to avoid pollution
        
        Examples:
          "ABC-GT-01 Casing Data\nTable 5-1: Details\n[table]" → finds ABC-GT-01
          "Table 5-1: ABC-GT-01 Details\n\nNext section..." → finds ABC-GT-01, stops at empty line
          "Table 5-1: Comparison\nNote: ABC-GT-01 and XYZ-GT-02" → finds both
        
        Args:
            page_text: Full text of the page
            table_ref: Table reference string (e.g., "Table 4-1")
            document_wells: List of well names to search for
            
        Returns:
            List of well names found in caption (empty if none)
        """
        if not table_ref or not page_text:
            return []
        
        # Find table reference position in page text
        ref_pos = page_text.find(table_ref)
        if ref_pos == -1:
            return []
        
        # Find the line containing table reference
        line_start = page_text.rfind('\n', 0, ref_pos)
        if line_start == -1:
            line_start = 0
        else:
            line_start += 1  # Skip the \n
        
        line_end = page_text.find('\n', ref_pos)
        if line_end == -1:
            line_end = len(page_text)
        
        current_line = page_text[line_start:line_end]
        
        # Extract 1 line BEFORE (if exists and not empty)
        prev_line = ""
        if line_start > 0:
            prev_line_start = page_text.rfind('\n', 0, line_start - 1)
            if prev_line_start == -1:
                prev_line_start = 0
            else:
                prev_line_start += 1
            prev_line = page_text[prev_line_start:line_start - 1].strip()
            # If empty line, don't include it
            if not prev_line:
                prev_line = ""
        
        # Extract 1 line AFTER (if exists and not empty)
        next_line = ""
        if line_end < len(page_text):
            next_line_end = page_text.find('\n', line_end + 1)
            if next_line_end == -1:
                next_line_end = len(page_text)
            next_line = page_text[line_end + 1:next_line_end].strip()
            # If empty line, don't include it
            if not next_line:
                next_line = ""
        
        # Combine: prev_line + current_line + next_line
        caption_text = ' '.join(filter(None, [prev_line, current_line, next_line]))
        
        # Find well names in caption
        found_wells = []
        for well in document_wells:
            if well in caption_text:
                found_wells.append(well)
        
        return found_wells
    
    def _find_wells_in_table_content(self, headers: List[str], rows: List[List[str]], 
                                      document_wells: List[str]) -> List[str]:
        """
        Find well names in table content (headers + cell values)
        
        Args:
            headers: List of column headers
            rows: List of row data
            document_wells: List of well names to search for
            
        Returns:
            List of well names found in table content
        """
        # Combine all table text
        table_text = ' '.join(headers) + ' '
        for row in rows:
            table_text += ' '.join(str(cell) for cell in row) + ' '
        
        # Find all well names in table text
        found_wells = []
        for well in document_wells:
            if well in table_text:
                found_wells.append(well)
        
        return found_wells

