"""
BM25 Sparse Retrieval - Keyword-based search using BM25Okapi algorithm
Complements dense vector search for exact term matching
"""

import logging
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    BM25Okapi implementation for sparse keyword retrieval
    
    BM25 is particularly good for:
    - Exact term matching (well names, formations, equipment IDs)
    - Technical terminology and acronyms
    - Numeric values and identifiers
    - Complementing semantic search
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever
        
        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        
        # Index storage
        self.documents = []  # List of document dicts
        self.doc_lengths = []  # Length of each document
        self.avgdl = 0  # Average document length
        self.doc_freqs = {}  # Document frequency for each term
        self.idf = {}  # IDF scores for each term
        self.N = 0  # Total number of documents
    
    def index_documents(self, chunks: List[Dict]) -> None:
        """
        Build BM25 index from text chunks
        
        Args:
            chunks: List of chunk dicts with 'text' field
        """
        logger.info(f"Building BM25 index from {len(chunks)} chunks...")
        
        self.documents = chunks
        self.N = len(chunks)
        
        # Tokenize and compute statistics
        self.doc_lengths = []
        term_doc_counts = Counter()
        
        for chunk in chunks:
            tokens = self._tokenize(chunk['text'])
            self.doc_lengths.append(len(tokens))
            
            # Count unique terms in this document
            unique_terms = set(tokens)
            for term in unique_terms:
                term_doc_counts[term] += 1
        
        # Compute average document length
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # Compute IDF scores
        self.doc_freqs = term_doc_counts
        self.idf = {}
        
        for term, df in self.doc_freqs.items():
            # IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
        
        logger.info(f"✓ BM25 index built: {len(self.idf)} unique terms, avg doc length: {self.avgdl:.1f}")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search documents using BM25 scoring
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of chunks with BM25 scores
        """
        if not self.documents:
            logger.warning("BM25 index is empty")
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Compute BM25 scores for all documents
        scores = []
        for i, chunk in enumerate(self.documents):
            score = self._compute_bm25_score(query_tokens, i)
            scores.append((score, i))
        
        # Sort by score (descending)
        scores.sort(reverse=True, key=lambda x: x[0])
        
        # Return top-k results
        results = []
        for score, idx in scores[:top_k]:
            if score > 0:  # Only include documents with positive scores
                chunk = self.documents[idx].copy()
                chunk['bm25_score'] = score
                results.append(chunk)
        
        logger.info(f"BM25 search returned {len(results)} results for query: '{query[:50]}...'")
        
        return results
    
    def _compute_bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """
        Compute BM25 score for a document given query tokens
        
        Formula:
        BM25(D,Q) = Σ(IDF(qi) × (f(qi,D) × (k1 + 1)) / (f(qi,D) + k1 × (1 - b + b × |D| / avgdl)))
        
        Where:
        - qi: query term
        - f(qi,D): term frequency in document D
        - |D|: document length
        - avgdl: average document length
        - k1, b: parameters
        """
        score = 0.0
        doc_text = self.documents[doc_idx]['text']
        doc_tokens = self._tokenize(doc_text)
        doc_len = self.doc_lengths[doc_idx]
        
        # Count term frequencies in document
        term_freqs = Counter(doc_tokens)
        
        for term in query_tokens:
            if term not in self.idf:
                # Term not in vocabulary
                continue
            
            # Term frequency in document
            tf = term_freqs.get(term, 0)
            
            if tf == 0:
                continue
            
            # IDF score
            idf = self.idf[term]
            
            # Length normalization
            norm = 1 - self.b + self.b * (doc_len / self.avgdl)
            
            # BM25 component for this term
            term_score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
            score += term_score
        
        return score
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms
        
        Strategy:
        - Lowercase
        - Split on whitespace and punctuation
        - Keep alphanumeric + hyphens (for well names like ADK-GT-01)
        - Remove very short tokens
        """
        # Lowercase
        text = text.lower()
        
        # Replace punctuation with spaces (except hyphens)
        import re
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Split and filter
        tokens = text.split()
        tokens = [t for t in tokens if len(t) >= 2]  # Keep tokens >= 2 chars
        
        return tokens
    
    def get_term_statistics(self) -> Dict:
        """Get statistics about the BM25 index"""
        if not self.idf:
            return {'status': 'not_indexed'}
        
        # Get top IDF terms
        top_idf_terms = sorted(self.idf.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Get term frequency distribution
        all_terms = []
        for chunk in self.documents:
            tokens = self._tokenize(chunk['text'])
            all_terms.extend(tokens)
        
        term_counts = Counter(all_terms)
        top_freq_terms = term_counts.most_common(20)
        
        return {
            'num_documents': self.N,
            'num_unique_terms': len(self.idf),
            'avg_doc_length': self.avgdl,
            'top_idf_terms': top_idf_terms,
            'top_frequency_terms': top_freq_terms
        }


class HybridDenseSparseRetriever:
    """
    Combines dense (vector) and sparse (BM25) retrieval
    
    Strategy:
    - Dense: Good for semantic similarity and concepts
    - Sparse: Good for exact term matching
    - Hybrid: Weight and combine scores
    """
    
    def __init__(self, dense_retriever, sparse_retriever, 
                 dense_weight: float = 0.7, sparse_weight: float = 0.3):
        """
        Initialize hybrid retriever
        
        Args:
            dense_retriever: RAGRetrievalAgent instance
            sparse_retriever: BM25Retriever instance
            dense_weight: Weight for dense scores (0-1)
            sparse_weight: Weight for sparse scores (0-1)
        """
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        # Normalize weights
        total = dense_weight + sparse_weight
        self.dense_weight /= total
        self.sparse_weight /= total
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Perform hybrid search combining dense and sparse retrieval
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of chunks with combined scores
        """
        # Get dense results
        dense_results = self.dense.retrieve(query, top_k=top_k * 2)
        
        # Get sparse results
        sparse_results = self.sparse.search(query, top_k=top_k * 2)
        
        # Normalize and combine scores
        combined_results = self._combine_results(dense_results, sparse_results)
        
        # Sort by combined score and return top-k
        combined_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return combined_results[:top_k]
    
    def _combine_results(self, dense_results: List[Dict], 
                        sparse_results: List[Dict]) -> List[Dict]:
        """
        Combine dense and sparse results with score normalization
        
        Strategy:
        1. Normalize scores to [0, 1]
        2. Combine with weights
        3. Merge duplicate documents
        """
        # Normalize dense scores (distance -> similarity)
        if dense_results:
            max_distance = max(r.get('distance', 0) for r in dense_results)
            for r in dense_results:
                # Convert distance to similarity (lower distance = higher similarity)
                r['dense_score'] = 1.0 - (r.get('distance', 0) / max_distance) if max_distance > 0 else 1.0
        
        # Normalize sparse scores
        if sparse_results:
            max_bm25 = max(r.get('bm25_score', 0) for r in sparse_results)
            for r in sparse_results:
                r['sparse_score'] = r.get('bm25_score', 0) / max_bm25 if max_bm25 > 0 else 0.0
        
        # Combine results by chunk ID
        combined = {}
        
        for r in dense_results:
            chunk_id = r.get('id', r.get('chunk_id'))
            if chunk_id not in combined:
                combined[chunk_id] = r
                combined[chunk_id]['dense_score'] = r.get('dense_score', 0)
                combined[chunk_id]['sparse_score'] = 0.0
        
        for r in sparse_results:
            chunk_id = r.get('id', r.get('chunk_id'))
            if chunk_id in combined:
                combined[chunk_id]['sparse_score'] = r.get('sparse_score', 0)
            else:
                combined[chunk_id] = r
                combined[chunk_id]['dense_score'] = 0.0
                combined[chunk_id]['sparse_score'] = r.get('sparse_score', 0)
        
        # Compute hybrid scores
        for chunk_id, chunk in combined.items():
            dense_score = chunk.get('dense_score', 0)
            sparse_score = chunk.get('sparse_score', 0)
            chunk['hybrid_score'] = (self.dense_weight * dense_score + 
                                    self.sparse_weight * sparse_score)
        
        return list(combined.values())


def create_bm25_retriever(config: Dict = None) -> BM25Retriever:
    """Factory function to create BM25 retriever"""
    if config is None:
        config = {}
    
    return BM25Retriever(
        k1=config.get('k1', 1.5),
        b=config.get('b', 0.75)
    )
