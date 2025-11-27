"""
Preprocessing Agent - Advanced Semantic Chunking with Metadata Extraction
Uses Ultimate Semantic Chunker with Late Chunking and Contextual Enrichment
"""

import spacy
import re
from typing import Dict, List, Tuple
import logging
import yaml
from pathlib import Path

# Import advanced components
from agents.ultimate_semantic_chunker import create_chunker
from agents.universal_metadata_extractor import create_metadata_extractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessingAgent:
    """
    Multi-strategy text chunking agent with hybrid granularity support
    
    Base chunking strategies:
    1. factual_qa: 1000 words, 250 overlap - for precise Q&A
    2. technical_extraction: 3500 words, 600 overlap - keeps tables intact
    3. summary: 3000 words, 500 overlap - context for summaries
    
    Hybrid strategies (when enabled):
    4. fine_grained: 500 words, 150 overlap - for precise detail retrieval
    5. coarse_grained: 5000 words, 800 overlap - for broad context
    
    Hybrid chunking provides multiple granularities for better retrieval:
    - Fine-grained for specific facts and numbers
    - Medium-grained for Q&A and extraction
    - Coarse-grained for summaries and context
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize preprocessing agent
        
        Args:
            config_path: Path to config.yaml file
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.chunking_config = self.config['chunking']
        
        # Initialize Ultimate Semantic Chunker (replaces old RecursiveCharacterTextSplitter)
        semantic_chunking_enabled = self.config.get('semantic_chunking', {}).get('enabled', True)
        if semantic_chunking_enabled:
            logger.info("🚀 Initializing Ultimate Semantic Chunker (Late Chunking + Contextual Enrichment)")
            self.semantic_chunker = create_chunker(self.config.get('semantic_chunking', {}))
        else:
            logger.warning("⚠️  Semantic chunking disabled, using basic chunking")
            self.semantic_chunker = None
        
        # Initialize Universal Metadata Extractor (replaces basic regex)
        logger.info("🚀 Initializing Universal Metadata Extractor (spaCy NER + Regex)")
        self.metadata_extractor = create_metadata_extractor({'use_spacy': True})
        
        # Load spaCy for sentence segmentation (fallback only)
        try:
            self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
            self.nlp.add_pipe('sentencizer')
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def process(self, documents: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Create semantically-bounded chunks using Ultimate Semantic Chunker
        Includes Late Chunking, Contextual Enrichment, and Universal Metadata Extraction
        
        Args:
            documents: List of document dicts from IngestionAgent
            
        Returns:
            Dict with key 'fine_grained' containing list of chunk dicts with:
            {
                'text': str,
                'doc_id': str,
                'chunk_id': str,
                'well_names': List[str],
                'metadata': Dict (wells, formations, depths, temperatures, pressures, etc.)
            }
        """
        all_chunks = {'fine_grained': []}
        
        for doc in documents:
            logger.info(f"Processing document: {doc['filename']}")
            
            # Use Ultimate Semantic Chunker if enabled
            if self.semantic_chunker:
                logger.info("  Using Ultimate Semantic Chunker (Late Chunking + Contextual Enrichment)")
                doc_dict = {
                    'content': doc['content'],
                    'filename': doc['filename'],
                    'wells': doc['wells']
                }
                
                # Chunk with semantic boundaries
                chunks = self.semantic_chunker.chunk_document(doc_dict)
                
                # Enrich each chunk with universal metadata
                logger.info(f"  Extracting metadata for {len(chunks)} chunks...")
                # First extract document-level metadata
                doc_metadata = self.metadata_extractor.extract_metadata(doc['content'])
                # Then enrich chunks with it
                enriched_chunks = self.metadata_extractor.enrich_chunks_with_metadata(
                    chunks, 
                    document_metadata=doc_metadata
                )
                
                # Format for indexing
                for i, chunk in enumerate(enriched_chunks):
                    chunk['chunk_id'] = f"{doc['filename']}_semantic_{i}"
                    chunk['doc_id'] = doc['filename']
                    # Ensure well_names from document are preserved
                    if 'well_names' not in chunk or not chunk['well_names']:
                        chunk['well_names'] = doc['wells']
                
                all_chunks['fine_grained'].extend(enriched_chunks)
                logger.info(f"  ✓ Created {len(enriched_chunks)} semantic chunks with metadata")
            else:
                # Fallback to old method
                logger.warning("  Using fallback basic chunking (semantic chunking disabled)")
                section_headers = self._extract_section_headers(doc['content'])
                fine_config = self.chunking_config.get('fine_grained', {
                    'chunk_size': 500,
                    'chunk_overlap': 150
                })
                chunks = self._create_chunks(
                    doc=doc,
                    section_headers=section_headers,
                    chunk_size=fine_config['chunk_size'],
                    chunk_overlap=fine_config['chunk_overlap']
                )
                all_chunks['fine_grained'].extend(chunks)
                logger.info(f"  ✓ Created {len(chunks)} basic chunks")
        
        return all_chunks
    
    def _create_chunks(self, doc: Dict, section_headers: List[str], 
                       chunk_size: int, chunk_overlap: int) -> List[Dict]:
        """
        Create chunks from document content
        
        Args:
            doc: Document dict from IngestionAgent
            section_headers: List of section headers found in document
            chunk_size: Target chunk size in words
            chunk_overlap: Overlap size in words
        """
        chunks = []
        
        # Use spaCy for sentence segmentation if available
        if self.nlp:
            sentences = self._segment_sentences_spacy(doc['content'])
        else:
            sentences = self._segment_sentences_simple(doc['content'])
        
        # Group sentences into chunks
        current_chunk = []
        current_word_count = 0
        chunk_id = 0
        
        for sent in sentences:
            sent_word_count = len(sent.split())
            
            # Check if adding this sentence would exceed chunk size
            if current_word_count + sent_word_count > chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunk_dict = {
                    'text': chunk_text,
                    'doc_id': doc['filename'],
                    'chunk_id': f"{doc['filename']}_fine_{chunk_id}",
                    'well_names': doc['wells'],
                    'section_headers': section_headers
                }
                chunks.append(chunk_dict)
                chunk_id += 1
                
                # Calculate overlap: keep last N words
                overlap_words = chunk_overlap
                overlap_sentences = []
                overlap_word_count = 0
                
                # Take sentences from the end until we reach overlap size
                for s in reversed(current_chunk):
                    s_words = len(s.split())
                    if overlap_word_count + s_words <= overlap_words:
                        overlap_sentences.insert(0, s)
                        overlap_word_count += s_words
                    else:
                        break
                
                # Start new chunk with overlap
                current_chunk = overlap_sentences
                current_word_count = overlap_word_count
            
            current_chunk.append(sent)
            current_word_count += sent_word_count
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_dict = {
                'text': chunk_text,
                'doc_id': doc['filename'],
                'chunk_id': f"{doc['filename']}_fine_{chunk_id}",
                'well_names': doc['wells'],
                'section_headers': section_headers
            }
            chunks.append(chunk_dict)
        
        return chunks
    
    def _segment_sentences_spacy(self, text: str) -> List[str]:
        """Segment text into sentences using spaCy with technical term awareness
        Processes in batches to avoid memory issues"""
        sentences = []
        
        # Process in smaller batches to avoid memory issues (max 100KB per batch)
        max_batch_size = 100000  # 100KB
        if len(text) > max_batch_size:
            # Split text into smaller chunks at paragraph boundaries
            paragraphs = text.split('\n\n')
            current_batch = ""
            
            for para in paragraphs:
                if len(current_batch) + len(para) > max_batch_size and current_batch:
                    # Process current batch
                    try:
                        doc = self.nlp(current_batch)
                        for sent in doc.sents:
                            sent_text = sent.text.strip()
                            if sent_text:
                                sentences.append(sent_text)
                    except Exception as e:
                        logger.warning(f"SpaCy batch processing failed, using simple segmentation: {str(e)}")
                        sentences.extend(self._segment_sentences_simple(current_batch))
                    current_batch = para
                else:
                    current_batch += "\n\n" + para if current_batch else para
            
            # Process remaining batch
            if current_batch:
                try:
                    doc = self.nlp(current_batch)
                    for sent in doc.sents:
                        sent_text = sent.text.strip()
                        if sent_text:
                            sentences.append(sent_text)
                except Exception as e:
                    logger.warning(f"SpaCy batch processing failed, using simple segmentation: {str(e)}")
                    sentences.extend(self._segment_sentences_simple(current_batch))
        else:
            # Text is small enough, process directly
            try:
                doc = self.nlp(text)
                for sent in doc.sents:
                    sent_text = sent.text.strip()
                    if sent_text:
                        sentences.append(sent_text)
            except Exception as e:
                logger.warning(f"SpaCy processing failed, using simple segmentation: {str(e)}")
                sentences = self._segment_sentences_simple(text)
        
        return sentences
    
    def _segment_sentences_simple(self, text: str) -> List[str]:
        """Fallback: simple sentence segmentation with technical term awareness"""
        import re
        # Split on periods, question marks, exclamation marks followed by space and capital letter
        # But preserve technical abbreviations like "U.S.", "p.s.i.", etc.
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
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
    
    def get_chunk_statistics(self, chunks: Dict[str, List[Dict]]) -> Dict:
        """
        Get statistics about chunks
        
        Returns:
            Dict with stats for each strategy
        """
        stats = {}
        
        for strategy, chunk_list in chunks.items():
            if chunk_list:
                word_counts = [len(c['text'].split()) for c in chunk_list]
                stats[strategy] = {
                    'count': len(chunk_list),
                    'avg_words': sum(word_counts) / len(word_counts),
                    'min_words': min(word_counts),
                    'max_words': max(word_counts)
                }
            else:
                stats[strategy] = {
                    'count': 0,
                    'avg_words': 0,
                    'min_words': 0,
                    'max_words': 0
                }
        
        return stats
