"""
Reranking System - Improves retrieval result ordering
Uses cross-encoder or LLM-based scoring for better relevance ranking
"""

import logging
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Reranker:
    """
    Rerank retrieval results using cross-encoder or LLM scoring
    
    Why Reranking:
    - Bi-encoders (used in retrieval) encode query and documents independently
    - Cross-encoders process query+document together → better relevance
    - Final reranking ensures most relevant results at top
    """
    
    def __init__(self, method: str = 'cross-encoder',
                 model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                 llm_helper=None):
        """
        Initialize reranker
        
        Args:
            method: 'cross-encoder' or 'llm'
            model_name: Cross-encoder model name
            llm_helper: Optional LLM helper for LLM-based reranking
        """
        self.method = method
        self.llm_helper = llm_helper
        
        # Load cross-encoder model
        if method == 'cross-encoder':
            try:
                logger.info(f"Loading cross-encoder: {model_name}")
                self.cross_encoder = CrossEncoder(model_name)
                logger.info("✓ Cross-encoder loaded")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}. Falling back to score preservation.")
                self.cross_encoder = None
                self.method = 'none'
        else:
            self.cross_encoder = None
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = None) -> List[Dict]:
        """
        Rerank documents based on relevance to query
        
        Args:
            query: Search query
            documents: List of document dicts with 'text' field
            top_k: Number of top results to return (None = all)
            
        Returns:
            Reranked list of documents with updated scores
        """
        if not documents:
            return []
        
        if self.method == 'cross-encoder' and self.cross_encoder:
            return self._rerank_cross_encoder(query, documents, top_k)
        elif self.method == 'llm' and self.llm_helper:
            return self._rerank_llm(query, documents, top_k)
        else:
            # No reranking - return as is
            return documents[:top_k] if top_k else documents
    
    def _rerank_cross_encoder(self, query: str, documents: List[Dict], 
                             top_k: int = None) -> List[Dict]:
        """
        Rerank using cross-encoder model
        
        Strategy:
        - Create (query, doc_text) pairs
        - Score each pair with cross-encoder
        - Sort by score (descending)
        """
        logger.info(f"Reranking {len(documents)} documents with cross-encoder...")
        
        # Prepare query-document pairs
        pairs = [[query, doc.get('text', '')] for doc in documents]
        
        # Score with cross-encoder
        scores = self.cross_encoder.predict(pairs)
        
        # Add scores to documents and sort
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # Sort by rerank score (descending)
        documents_reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        logger.info(f"✓ Reranked documents (top score: {documents_reranked[0]['rerank_score']:.3f})")
        
        return documents_reranked[:top_k] if top_k else documents_reranked
    
    def _rerank_llm(self, query: str, documents: List[Dict], 
                   top_k: int = None) -> List[Dict]:
        """
        Rerank using LLM-based scoring
        
        Strategy:
        - Ask LLM to score relevance on 0-10 scale
        - Sort by LLM scores
        - More expensive but potentially more accurate
        """
        logger.info(f"Reranking {len(documents)} documents with LLM...")
        
        scored_docs = []
        
        for doc in documents:
            doc_text = doc.get('text', '')[:500]  # Truncate for efficiency
            
            # Create scoring prompt
            prompt = f"""Rate the relevance of this document to the query on a scale of 0-10.
Only respond with a number.

Query: {query}

Document: {doc_text}

Relevance Score (0-10):"""
            
            try:
                # Call LLM to score
                response = self.llm_helper._call_ollama(prompt, max_tokens=10)
                
                # Parse score
                import re
                score_match = re.search(r'(\d+(?:\.\d+)?)', response)
                if score_match:
                    score = float(score_match.group(1))
                else:
                    score = 5.0  # Default
                
                doc['rerank_score'] = score
                scored_docs.append(doc)
                
            except Exception as e:
                logger.warning(f"LLM scoring failed for document: {e}")
                doc['rerank_score'] = 5.0  # Default
                scored_docs.append(doc)
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        logger.info(f"✓ LLM reranked documents (top score: {scored_docs[0]['rerank_score']:.1f}/10)")
        
        return scored_docs[:top_k] if top_k else scored_docs
    
    def reciprocal_rank_fusion(self, results_list: List[List[Dict]], k: int = 60) -> List[Dict]:
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion (RRF)
        
        RRF formula: score = Σ(1 / (k + rank_i))
        
        Use case: Combine results from different retrieval strategies
        
        Args:
            results_list: List of ranked result lists
            k: RRF constant (default: 60)
            
        Returns:
            Fused and reranked results
        """
        logger.info(f"Fusing {len(results_list)} result lists with RRF...")
        
        # Collect all unique documents
        doc_scores = {}  # doc_id -> RRF score
        doc_objects = {}  # doc_id -> document dict
        
        for result_list in results_list:
            for rank, doc in enumerate(result_list, start=1):
                doc_id = doc.get('id', doc.get('chunk_id', str(hash(doc.get('text', '')[:100]))))
                
                # Compute RRF score
                rrf_score = 1.0 / (k + rank)
                
                # Accumulate scores
                if doc_id in doc_scores:
                    doc_scores[doc_id] += rrf_score
                else:
                    doc_scores[doc_id] = rrf_score
                    doc_objects[doc_id] = doc
        
        # Sort by RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create final ranked list
        fused_results = []
        for doc_id, score in sorted_docs:
            doc = doc_objects[doc_id]
            doc['rrf_score'] = score
            fused_results.append(doc)
        
        logger.info(f"✓ Fused into {len(fused_results)} unique documents")
        
        return fused_results


def create_reranker(config: Dict = None, llm_helper=None) -> Reranker:
    """Factory function to create reranker"""
    if config is None:
        config = {}
    
    return Reranker(
        method=config.get('method', 'cross-encoder'),
        model_name=config.get('model_name', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
        llm_helper=llm_helper
    )
