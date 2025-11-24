"""
Preprocessing Agent - Multi-Strategy Text Chunking
Implements different chunking strategies for different query types
"""

import spacy
import re
from typing import Dict, List, Tuple
import logging
import yaml
from pathlib import Path

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
        
        # Load spaCy for sentence segmentation
        try:
            self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
            self.nlp.add_pipe('sentencizer')
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def process(self, documents: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Create multi-strategy chunks from documents with hybrid chunking
        
        Args:
            documents: List of document dicts from IngestionAgent
            
        Returns:
            Dict with keys: 'factual_qa', 'technical_extraction', 'summary', 
            and optionally 'fine_grained', 'coarse_grained' if hybrid enabled
            Each contains list of chunk dicts with:
            {
                'text': str,
                'doc_id': str,
                'chunk_id': str,
                'strategy': str,
                'page_numbers': List[int],
                'well_names': List[str],
                'metadata': Dict
            }
        """
        # Initialize with base strategies
        all_chunks = {
            'factual_qa': [],
            'technical_extraction': [],
            'summary': []
        }
        
        # Add hybrid strategies if enabled
        enable_hybrid = self.chunking_config.get('enable_hybrid', False)
        if enable_hybrid:
            all_chunks['fine_grained'] = []
            all_chunks['coarse_grained'] = []
            logger.info("Hybrid chunking enabled - using multiple granularities")
        
        for doc in documents:
            logger.info(f"Chunking document: {doc['filename']}")
            
            # Create chunks for each strategy
            for strategy_name, strategy_config in self.chunking_config.items():
                # Skip non-strategy config keys
                if strategy_name == 'enable_hybrid' or not isinstance(strategy_config, dict):
                    continue
                
                # Skip 'tables' strategy - handled separately in hybrid pipeline
                # Tables are NOT chunked - they're extracted row-by-row with metadata
                if strategy_name == 'tables':
                    logger.info(f"  {strategy_name}: Skipping (handled by hybrid table extraction)")
                    continue
                
                # Only process hybrid strategies if enabled
                if strategy_name in ['fine_grained', 'coarse_grained'] and not enable_hybrid:
                    continue
                
                # Check if strategy has chunk_size (narrative strategies do, tables don't)
                if 'chunk_size' not in strategy_config:
                    logger.warning(f"  {strategy_name}: Missing chunk_size, skipping")
                    continue
                
                chunks = self._create_chunks(
                    doc=doc,
                    strategy=strategy_name,
                    chunk_size=strategy_config['chunk_size'],
                    chunk_overlap=strategy_config['chunk_overlap']
                )
                all_chunks[strategy_name].extend(chunks)
                logger.info(f"  {strategy_name}: {len(chunks)} chunks")
        
        return all_chunks
    
    def _create_chunks(self, doc: Dict, strategy: str, chunk_size: int, chunk_overlap: int) -> List[Dict]:
        """
        Create chunks using specific strategy
        
        Args:
            doc: Document dict from IngestionAgent
            strategy: Strategy name
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
                chunk_dict = self._create_chunk_dict(
                    text=chunk_text,
                    doc=doc,
                    strategy=strategy,
                    chunk_id=chunk_id
                )
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
            chunk_dict = self._create_chunk_dict(
                text=chunk_text,
                doc=doc,
                strategy=strategy,
                chunk_id=chunk_id
            )
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
    
    def _create_chunk_dict(self, text: str, doc: Dict, strategy: str, chunk_id: int) -> Dict:
        """Create chunk dictionary with enhanced metadata including section headers and table/figure refs"""
        # Estimate which pages this chunk might be from
        # This is approximate since we don't track exact positions
        page_numbers, page_metadata = self._estimate_pages_with_metadata(text, doc)
        
        return {
            'text': text,
            'doc_id': doc['filename'],
            'chunk_id': f"{doc['filename']}_{strategy}_{chunk_id}",
            'strategy': strategy,
            'page_numbers': page_numbers,
            'well_names': doc['wells'],
            'metadata': {
                'total_pages': doc['pages'],
                'source_file': doc['filename'],
                'section_headers': page_metadata.get('section_headers', []),
                'table_refs': page_metadata.get('table_refs', []),
                'figure_refs': page_metadata.get('figure_refs', [])
            }
        }
    
    def _estimate_pages_with_metadata(self, chunk_text: str, doc: Dict) -> tuple:
        """
        Estimate which pages contain this chunk's text and extract metadata
        
        Returns:
            Tuple of (page_numbers: List[int], metadata: Dict)
        """
        sample = chunk_text[:100].lower()
        matching_pages = []
        combined_metadata = {
            'section_headers': [],
            'table_refs': [],
            'figure_refs': []
        }
        
        for page in doc['page_contents']:
            if sample in page['text'].lower():
                matching_pages.append(page['page_number'])
                
                # Collect metadata from matching pages
                if 'section_headers' in page:
                    combined_metadata['section_headers'].extend(page['section_headers'])
                if 'table_refs' in page:
                    combined_metadata['table_refs'].extend(page['table_refs'])
                if 'figure_refs' in page:
                    combined_metadata['figure_refs'].extend(page['figure_refs'])
        
        # Remove duplicates while preserving order
        combined_metadata['section_headers'] = list(dict.fromkeys(combined_metadata['section_headers']))
        combined_metadata['table_refs'] = list(dict.fromkeys(combined_metadata['table_refs']))
        combined_metadata['figure_refs'] = list(dict.fromkeys(combined_metadata['figure_refs']))
        
        # If no exact match, return approximate range
        if not matching_pages:
            matching_pages = [1]  # Default to page 1
        
        return matching_pages, combined_metadata
        
        return matching_pages
    
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
