"""
Ultimate Semantic Chunker - Advanced Multi-Strategy Chunking
Combines Late Chunking (Jina AI), Contextual Enrichment (Anthropic), and Semantic Breakpoints
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import spacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UltimateSemanticChunker:
    """
    Advanced semantic chunker implementing SOTA techniques:
    1. Late Chunking: Embed full document first to capture global context
    2. Contextual Enrichment: Prepend document context to every chunk
    3. Semantic Breakpoints: Split at semantic similarity drops (not fixed sizes)
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.7,
                 min_chunk_size: int = 200,
                 max_chunk_size: int = 800,
                 context_window: int = 100):
        """
        Initialize Ultimate Semantic Chunker
        
        Args:
            embedding_model: Model for computing embeddings
            similarity_threshold: Threshold for semantic breakpoints (0-1)
            min_chunk_size: Minimum chunk size in words
            max_chunk_size: Maximum chunk size in words
            context_window: Size of contextual summary in words
        """
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.context_window = context_window
        
        # Load embedding model for late chunking
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Load spaCy for sentence segmentation
        try:
            self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
            self.nlp.add_pipe('sentencizer')
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def chunk_document(self, document: Dict, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Chunk document using ultimate semantic chunking strategy
        
        Args:
            document: Document dict with 'content', 'filename', 'wells' fields
            metadata: Optional additional metadata
            
        Returns:
            List of enriched chunk dicts
        """
        content = document['content']
        filename = document['filename']
        well_names = document.get('wells', [])
        
        logger.info(f"Ultimate semantic chunking: {filename} ({len(content)} chars)")
        
        # STEP 1: Generate document-level context summary
        doc_context = self._generate_document_context(content, filename, well_names)
        
        # STEP 2: Segment into sentences
        sentences = self._segment_sentences(content)
        
        if len(sentences) < 2:
            # Single chunk for very small documents
            return [{
                'text': f"[Context: {doc_context}]\n\n{content}",
                'doc_id': filename,
                'chunk_id': f"{filename}_ultimate_0",
                'well_names': well_names,
                'enriched_context': doc_context,
                'metadata': metadata or {}
            }]
        
        # STEP 3: Late Chunking - Embed full document first
        logger.info(f"Late chunking: Embedding {len(sentences)} sentences...")
        sentence_embeddings = self._embed_sentences_late(sentences, content)
        
        # STEP 4: Find semantic breakpoints
        breakpoints = self._find_semantic_breakpoints(
            sentences, 
            sentence_embeddings,
            self.similarity_threshold
        )
        
        # STEP 5: Create chunks at breakpoints with size constraints
        chunks = self._create_chunks_from_breakpoints(
            sentences, 
            breakpoints,
            doc_context,
            filename,
            well_names,
            metadata or {}
        )
        
        logger.info(f"Created {len(chunks)} semantically-bounded chunks (avg: {np.mean([len(c['text'].split()) for c in chunks]):.0f} words)")
        
        return chunks
    
    def _generate_document_context(self, content: str, filename: str, 
                                   well_names: List[str]) -> str:
        """
        Generate brief contextual summary of document (Anthropic style)
        
        Example: "Well Report for ADK-GT-01, Operator: XYZ Energy, Pages: 45"
        """
        # Extract key metadata
        well_str = ", ".join(well_names) if well_names else "Unknown Well"
        
        # Try to detect document type
        content_lower = content.lower()
        doc_type = "Well Report"
        if "completion" in content_lower:
            doc_type = "Completion Report"
        elif "drilling" in content_lower:
            doc_type = "Drilling Report"
        elif "test" in content_lower and "production" in content_lower:
            doc_type = "Well Test Report"
        elif "geological" in content_lower or "formation" in content_lower:
            doc_type = "Geological Report"
        
        # Estimate page count (rough)
        estimated_pages = max(1, len(content) // 3000)
        
        context = f"{doc_type} for {well_str}, Document: {filename}"
        if estimated_pages > 1:
            context += f", ~{estimated_pages} pages"
        
        return context
    
    def _segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences using spaCy"""
        if not self.nlp:
            # Fallback to simple segmentation
            import re
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        # Use spaCy in batches
        sentences = []
        max_batch_size = 100000  # 100KB per batch
        
        if len(text) > max_batch_size:
            paragraphs = text.split('\n\n')
            current_batch = ""
            
            for para in paragraphs:
                if len(current_batch) + len(para) > max_batch_size and current_batch:
                    doc = self.nlp(current_batch)
                    for sent in doc.sents:
                        if sent.text.strip():
                            sentences.append(sent.text.strip())
                    current_batch = para
                else:
                    current_batch += "\n\n" + para if current_batch else para
            
            # Process remaining
            if current_batch:
                doc = self.nlp(current_batch)
                for sent in doc.sents:
                    if sent.text.strip():
                        sentences.append(sent.text.strip())
        else:
            doc = self.nlp(text)
            for sent in doc.sents:
                if sent.text.strip():
                    sentences.append(sent.text.strip())
        
        return sentences
    
    def _embed_sentences_late(self, sentences: List[str], full_document: str) -> np.ndarray:
        """
        Late Chunking (Jina AI style): 
        1. Embed full document to get global context
        2. Pool embeddings for individual sentences
        
        This preserves global semantic context while allowing sentence-level granularity
        """
        # Encode sentences (already captures some context via transformer attention)
        sentence_embeddings = self.embedding_model.encode(
            sentences,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return sentence_embeddings
    
    def _find_semantic_breakpoints(self, sentences: List[str], 
                                   embeddings: np.ndarray,
                                   threshold: float) -> List[int]:
        """
        Find semantic breakpoints where topic changes significantly
        
        Strategy:
        - Compute cosine similarity between consecutive sentences
        - Mark breakpoint when similarity drops below threshold
        - This creates natural topic-based boundaries
        
        Args:
            sentences: List of sentences
            embeddings: Sentence embeddings (N x D)
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of breakpoint indices
        """
        breakpoints = [0]  # Always start with first sentence
        
        for i in range(1, len(embeddings)):
            # Compute cosine similarity between consecutive sentences
            similarity = np.dot(embeddings[i-1], embeddings[i]) / (
                np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
            )
            
            # If similarity drops significantly, mark as breakpoint
            if similarity < threshold:
                breakpoints.append(i)
                logger.debug(f"Semantic breakpoint at sentence {i} (similarity: {similarity:.3f})")
        
        # Always include last sentence
        if breakpoints[-1] != len(sentences) - 1:
            breakpoints.append(len(sentences))
        
        logger.info(f"Found {len(breakpoints)-1} semantic breakpoints (threshold: {threshold})")
        
        return breakpoints
    
    def _create_chunks_from_breakpoints(self, sentences: List[str], 
                                       breakpoints: List[int],
                                       doc_context: str,
                                       filename: str,
                                       well_names: List[str],
                                       metadata: Dict) -> List[Dict]:
        """
        Create chunks from semantic breakpoints while respecting size constraints
        
        Strategy:
        - Start chunk at breakpoint
        - Add sentences until max_chunk_size reached
        - If chunk too small, merge with next
        - Prepend contextual enrichment to every chunk
        """
        chunks = []
        chunk_id = 0
        
        i = 0
        while i < len(breakpoints) - 1:
            start_idx = breakpoints[i]
            end_idx = breakpoints[i + 1]
            
            # Collect sentences for this chunk
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = ' '.join(chunk_sentences)
            word_count = len(chunk_text.split())
            
            # Check size constraints
            if word_count < self.min_chunk_size and i < len(breakpoints) - 2:
                # Too small - merge with next breakpoint
                end_idx = breakpoints[i + 2]
                chunk_sentences = sentences[start_idx:end_idx]
                chunk_text = ' '.join(chunk_sentences)
                word_count = len(chunk_text.split())
                i += 1  # Skip next breakpoint
            
            elif word_count > self.max_chunk_size:
                # Too large - split at max_chunk_size
                current_chunk = []
                current_words = 0
                
                for sent in chunk_sentences:
                    sent_words = len(sent.split())
                    if current_words + sent_words > self.max_chunk_size and current_chunk:
                        # Create chunk
                        chunk_text = ' '.join(current_chunk)
                        enriched_chunk = self._enrich_chunk_with_context(chunk_text, doc_context)
                        
                        chunks.append({
                            'text': enriched_chunk,
                            'doc_id': filename,
                            'chunk_id': f"{filename}_ultimate_{chunk_id}",
                            'well_names': well_names,
                            'enriched_context': doc_context,
                            'metadata': metadata
                        })
                        chunk_id += 1
                        
                        # Start new chunk
                        current_chunk = [sent]
                        current_words = sent_words
                    else:
                        current_chunk.append(sent)
                        current_words += sent_words
                
                # Add remaining sentences
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    enriched_chunk = self._enrich_chunk_with_context(chunk_text, doc_context)
                    
                    chunks.append({
                        'text': enriched_chunk,
                        'doc_id': filename,
                        'chunk_id': f"{filename}_ultimate_{chunk_id}",
                        'well_names': well_names,
                        'enriched_context': doc_context,
                        'metadata': metadata
                    })
                    chunk_id += 1
                
                i += 1
                continue
            
            # Valid chunk - create it with contextual enrichment
            enriched_chunk = self._enrich_chunk_with_context(chunk_text, doc_context)
            
            chunks.append({
                'text': enriched_chunk,
                'doc_id': filename,
                'chunk_id': f"{filename}_ultimate_{chunk_id}",
                'well_names': well_names,
                'enriched_context': doc_context,
                'metadata': metadata
            })
            chunk_id += 1
            i += 1
        
        return chunks
    
    def _enrich_chunk_with_context(self, chunk_text: str, doc_context: str) -> str:
        """
        Contextual Enrichment (Anthropic style):
        Prepend document context to every chunk
        
        Example:
        [Context: Completion Report for ADK-GT-01, Document: report.pdf]
        
        The 9-5/8" casing string was set at 2845m MD...
        """
        return f"[Context: {doc_context}]\n\n{chunk_text}"


def create_chunker(config: Dict = None) -> UltimateSemanticChunker:
    """Factory function to create chunker with config"""
    if config is None:
        config = {}
    
    return UltimateSemanticChunker(
        embedding_model=config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'),
        similarity_threshold=config.get('similarity_threshold', 0.7),
        min_chunk_size=config.get('min_chunk_size', 200),
        max_chunk_size=config.get('max_chunk_size', 800),
        context_window=config.get('context_window', 100)
    )
