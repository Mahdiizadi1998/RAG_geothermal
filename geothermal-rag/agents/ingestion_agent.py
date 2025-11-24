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
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import OCR libraries (fallback chain: EasyOCR -> pytesseract)
OCR_AVAILABLE = False
OCR_ENGINE = None

try:
    import easyocr
    OCR_AVAILABLE = True
    OCR_ENGINE = 'easyocr'
    logger.info("Using EasyOCR (GPU/CPU, no external dependencies)")
except ImportError:
    try:
        import pytesseract
        OCR_AVAILABLE = True
        OCR_ENGINE = 'tesseract'
        logger.info("Using Tesseract OCR")
    except ImportError:
        logger.warning("No OCR available - scanned PDFs will not be processed. Install: pip install easyocr")


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
        
        # Initialize EasyOCR reader if available (lazy loading with memory optimization)
        self.ocr_reader = None
        if OCR_ENGINE == 'easyocr':
            try:
                # Force CPU-only mode for PyTorch
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable CUDA
                
                # Disable memory pinning warnings
                import warnings
                warnings.filterwarnings('ignore', message='.*pin_memory.*')
                
                # Use quantized model for lower memory usage
                self.ocr_reader = easyocr.Reader(
                    ['en'], 
                    gpu=False,  # CPU only
                    verbose=False,
                    quantize=True,  # Use quantized model (less memory)
                    download_enabled=True
                )
                logger.info("EasyOCR reader initialized (CPU-only, quantized, memory-optimized)")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {str(e)}")
    
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
        
        # Extract text page by page with OCR fallback
        page_contents = []
        all_text = []
        ocr_used = False
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # If no text extracted and OCR is available, try OCR
            if len(text.strip()) < 50 and OCR_AVAILABLE:
                try:
                    ocr_text = self._ocr_page(page)
                    if len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text
                        ocr_used = True
                        logger.debug(f"  OCR used for page {page_num + 1}")
                except Exception as e:
                    logger.warning(f"  OCR failed for page {page_num + 1}: {str(e)}")
            
            page_contents.append({
                'page_number': page_num + 1,  # 1-indexed for human readability
                'text': text
            })
            
            all_text.append(text)
        
        if ocr_used:
            logger.info(f"  ✓ OCR applied to extract text from images")
        
        doc.close()
        
        # Combine all text
        full_text = '\n\n'.join(all_text)
        
        # Check if we got any meaningful text
        if len(full_text.strip()) < 100:
            logger.warning(f"  ⚠️ Very little text extracted ({len(full_text)} chars)")
            if not OCR_AVAILABLE:
                logger.warning(f"  💡 Install pytesseract for OCR support: pip install pytesseract")
                logger.warning(f"  💡 Also install Tesseract: https://github.com/tesseract-ocr/tesseract")
        
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
    
    def _ocr_page(self, page) -> str:
        """
        Perform OCR on a PDF page that has images/scanned content
        Memory-optimized: uses lower resolution to avoid OOM errors
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Extracted text from OCR
        """
        # Render page at LOWER resolution to reduce memory (150 DPI instead of 300)
        # This reduces memory usage by 4x while maintaining readability
        pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        # Resize to max width of 1024px if larger (further memory reduction)
        max_width = 1024
        if img.width > max_width:
            ratio = max_width / img.width
            new_size = (max_width, int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Perform OCR based on available engine
        if OCR_ENGINE == 'easyocr' and self.ocr_reader:
            # EasyOCR: convert PIL to numpy array
            import numpy as np
            img_array = np.array(img)
            
            # Process with lower batch size and width_ths to reduce memory
            try:
                results = self.ocr_reader.readtext(
                    img_array, 
                    detail=0,  # Only return text, no boxes
                    paragraph=True,  # Group into paragraphs
                    width_ths=0.7,  # Threshold for grouping
                    batch_size=1  # Process one at a time to reduce memory
                )
                text = '\n'.join(results)
            except Exception as e:
                logger.warning(f"  OCR processing failed, trying fallback: {str(e)}")
                # Fallback: even smaller image
                img_small = img.resize((img.width // 2, img.height // 2), Image.Resampling.LANCZOS)
                img_array_small = np.array(img_small)
                results = self.ocr_reader.readtext(img_array_small, detail=0, batch_size=1)
                text = '\n'.join(results)
        elif OCR_ENGINE == 'tesseract':
            # Tesseract
            import pytesseract
            text = pytesseract.image_to_string(img, lang='eng')
        else:
            text = ""
        
        return text
