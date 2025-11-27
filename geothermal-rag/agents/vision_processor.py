"""
Vision Processor - Image/Plot captioning using Vision-Language Models
Uses llava:7b (Ollama) to caption technical diagrams, plots, and charts
"""

import logging
import requests
from typing import List, Dict, Optional
import base64
from pathlib import Path
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisionProcessor:
    """
    Process images and plots using Vision-Language Models (VLMs)
    
    Workflow:
    1. Extract images from PDFs
    2. Classify image type (plot, schematic, table, photo)
    3. Generate detailed text descriptions using llava:7b
    4. Embed descriptions for semantic search
    
    Use cases:
    - Well trajectory plots
    - Pressure/temperature graphs
    - Schematic diagrams
    - Geological cross-sections
    """
    
    def __init__(self, ollama_host: str = "http://localhost:11434",
                 vision_model: str = "llava:7b"):
        """
        Initialize vision processor
        
        Args:
            ollama_host: Ollama API endpoint
            vision_model: Vision-language model name
        """
        self.ollama_host = ollama_host
        self.vision_model = vision_model
        
        # Check if model is available
        self.model_available = self._check_model_available()
    
    def _check_model_available(self) -> bool:
        """Check if vision model is available in Ollama"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available = any(m['name'].startswith(self.vision_model.split(':')[0]) 
                              for m in models)
                if available:
                    logger.info(f"✓ Vision model '{self.vision_model}' available")
                else:
                    logger.warning(f"⚠️ Vision model '{self.vision_model}' not found. "
                                 f"Pull with: ollama pull {self.vision_model}")
                return available
            return False
        except Exception as e:
            logger.warning(f"Could not check Ollama models: {e}")
            return False
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract images from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of image dicts with: page, image_data, bbox, type
        """
        images = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    
                    # Extract image
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Get image bbox (approximate location on page)
                    # This is complex in PyMuPDF - simplified here
                    bbox = None
                    
                    images.append({
                        'page': page_num + 1,
                        'image_index': img_index,
                        'image_bytes': image_bytes,
                        'image_ext': image_ext,
                        'bbox': bbox,
                        'source_pdf': pdf_path
                    })
            
            doc.close()
            
            logger.info(f"Extracted {len(images)} images from {pdf_path}")
            
        except Exception as e:
            logger.error(f"Failed to extract images from {pdf_path}: {e}")
        
        return images
    
    def classify_image(self, image_bytes: bytes) -> str:
        """
        Classify image type using vision model
        
        Args:
            image_bytes: Image data
            
        Returns:
            Image type: 'plot', 'schematic', 'table', 'photo', 'diagram', 'other'
        """
        if not self.model_available:
            return 'unknown'
        
        # Encode image to base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Classification prompt
        prompt = """Classify this image into ONE category:
- plot: Graph, chart, or data visualization
- schematic: Technical diagram or schematic drawing
- table: Data table or tabular information
- photo: Photograph
- diagram: Geological cross-section or well diagram
- other: None of the above

Reply with ONLY the category name."""
        
        try:
            response = self._call_vision_model(prompt, image_b64, max_tokens=20)
            
            # Parse response
            image_type = response.strip().lower().split()[0]
            
            valid_types = ['plot', 'schematic', 'table', 'photo', 'diagram', 'other']
            if image_type in valid_types:
                return image_type
            else:
                return 'other'
                
        except Exception as e:
            logger.warning(f"Image classification failed: {e}")
            return 'unknown'
    
    def caption_image(self, image_bytes: bytes, image_type: str = None,
                     page_num: int = None) -> str:
        """
        Generate detailed caption for image using vision model
        
        Args:
            image_bytes: Image data
            image_type: Optional pre-classified image type
            page_num: Optional page number for context
            
        Returns:
            Text caption/description
        """
        if not self.model_available:
            return "Vision model not available - image not processed"
        
        # Encode image to base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Create context-aware prompt based on image type
        if image_type == 'plot':
            prompt = """Describe this plot/graph in detail. Include:
- Type of plot (line, scatter, bar, etc.)
- Axis labels and units
- Data trends or patterns
- Any annotations or legends
- Numerical values if visible

Be technical and precise."""

        elif image_type == 'schematic':
            prompt = """Describe this technical schematic in detail. Include:
- Components shown
- Labels and annotations
- Connections or flow paths
- Technical specifications if visible
- Purpose or function

Be technical and precise."""

        elif image_type == 'diagram':
            prompt = """Describe this diagram in detail. Include:
- Type of diagram (well bore, geological, etc.)
- Layers, zones, or sections
- Depth markers or scales
- Labels and annotations
- Technical details

Be technical and precise."""

        else:
            # Generic prompt
            prompt = """Describe this technical image in detail. Focus on:
- What is shown
- Any text, labels, or measurements
- Technical details and specifications
- Data or information conveyed

Be precise and comprehensive."""
        
        try:
            caption = self._call_vision_model(prompt, image_b64, max_tokens=300)
            
            # Add page context if available
            if page_num:
                caption = f"[Image from page {page_num}] {caption}"
            
            return caption
            
        except Exception as e:
            logger.error(f"Image captioning failed: {e}")
            return f"Failed to caption image: {str(e)}"
    
    def _call_vision_model(self, prompt: str, image_b64: str, 
                          max_tokens: int = 300) -> str:
        """
        Call Ollama vision API
        
        Args:
            prompt: Text prompt
            image_b64: Base64-encoded image
            max_tokens: Maximum response tokens
            
        Returns:
            Model response text
        """
        url = f"{self.ollama_host}/api/generate"
        
        payload = {
            "model": self.vision_model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.3,  # Lower temperature for factual descriptions
                "num_predict": max_tokens
            }
        }
        
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        return result['response'].strip()
    
    def process_pdf_images(self, pdf_path: str, well_names: List[str] = None) -> List[Dict]:
        """
        Extract and process all images from PDF
        
        Args:
            pdf_path: Path to PDF file
            well_names: Optional well names for metadata
            
        Returns:
            List of processed image dicts with captions
        """
        # Extract images
        images = self.extract_images_from_pdf(pdf_path)
        
        if not images:
            logger.info(f"No images found in {pdf_path}")
            return []
        
        logger.info(f"Processing {len(images)} images from {pdf_path}...")
        
        processed_images = []
        
        for img in images:
            try:
                # Classify image
                img_type = self.classify_image(img['image_bytes'])
                
                # Generate caption
                caption = self.caption_image(
                    img['image_bytes'],
                    image_type=img_type,
                    page_num=img['page']
                )
                
                processed_images.append({
                    'page': img['page'],
                    'image_type': img_type,
                    'caption': caption,
                    'well_names': well_names or [],
                    'source_pdf': pdf_path,
                    'metadata': {
                        'is_image': True,
                        'image_index': img['image_index']
                    }
                })
                
                logger.info(f"  ✓ Processed image {img['image_index']+1} on page {img['page']} ({img_type})")
                
            except Exception as e:
                logger.error(f"Failed to process image {img['image_index']} on page {img['page']}: {e}")
                continue
        
        logger.info(f"✓ Processed {len(processed_images)}/{len(images)} images successfully")
        
        return processed_images
    
    def get_image_chunks(self, processed_images: List[Dict]) -> List[Dict]:
        """
        Convert processed images to chunk format for indexing
        
        Args:
            processed_images: List of processed image dicts
            
        Returns:
            List of chunk dicts with image captions as text
        """
        chunks = []
        
        for img in processed_images:
            chunk = {
                'text': img['caption'],
                'chunk_id': f"{Path(img['source_pdf']).stem}_image_p{img['page']}_i{img['metadata']['image_index']}",
                'doc_id': Path(img['source_pdf']).name,
                'well_names': img.get('well_names', []),
                'metadata': {
                    'is_image': True,
                    'image_type': img['image_type'],
                    'page': img['page'],
                    'source_file': Path(img['source_pdf']).name
                }
            }
            chunks.append(chunk)
        
        return chunks


def create_vision_processor(config: Dict = None) -> VisionProcessor:
    """Factory function to create vision processor"""
    if config is None:
        config = {}
    
    return VisionProcessor(
        ollama_host=config.get('ollama_host', 'http://localhost:11434'),
        vision_model=config.get('vision_model', 'llava:7b')
    )
