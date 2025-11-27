"""
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
Implements hierarchical summarization using clustering and recursive abstraction
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAPTORTree:
    """
    RAPTOR Tree for hierarchical summarization and retrieval
    
    Strategy:
    1. Level 0: Original text chunks (leaves)
    2. Level 1+: Clustered summaries (internal nodes)
    3. Each level provides different abstraction for retrieval
    
    Use cases:
    - High-level summaries: Query upper levels
    - Specific details: Query level 0
    - Comprehensive understanding: Query multiple levels
    """
    
    def __init__(self, llm_helper, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 min_cluster_size: int = 5, max_tree_height: int = 3):
        """
        Initialize RAPTOR tree
        
        Args:
            llm_helper: LLM helper for generating summaries
            embedding_model: Model for computing embeddings
            min_cluster_size: Minimum cluster size for HDBSCAN
            max_tree_height: Maximum tree depth
        """
        self.llm = llm_helper
        self.min_cluster_size = min_cluster_size
        self.max_tree_height = max_tree_height
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Tree structure: level -> list of nodes
        self.tree = {}
        self.level_embeddings = {}
    
    def build_tree(self, chunks: List[Dict]) -> Dict:
        """
        Build RAPTOR tree from base chunks
        
        Args:
            chunks: List of base text chunks (level 0)
            
        Returns:
            Tree metadata dict
        """
        logger.info(f"Building RAPTOR tree from {len(chunks)} base chunks...")
        
        # Level 0: Store base chunks
        self.tree[0] = chunks
        
        # Embed level 0
        level_0_texts = [c['text'] for c in chunks]
        logger.info(f"Embedding level 0 ({len(level_0_texts)} chunks)...")
        self.level_embeddings[0] = self.embedding_model.encode(
            level_0_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Build higher levels recursively
        current_level = 0
        while current_level < self.max_tree_height:
            logger.info(f"Building level {current_level + 1}...")
            
            # Cluster current level
            clusters = self._cluster_chunks(
                self.tree[current_level],
                self.level_embeddings[current_level]
            )
            
            if len(clusters) <= 1:
                # No more clustering possible
                logger.info(f"Stopping at level {current_level} (no more clusters)")
                break
            
            # Summarize each cluster
            next_level_chunks = []
            for cluster_id, cluster_chunks in clusters.items():
                if cluster_id == -1:
                    # Noise cluster - skip
                    continue
                
                summary_chunk = self._summarize_cluster(cluster_chunks, current_level + 1, cluster_id)
                next_level_chunks.append(summary_chunk)
            
            if not next_level_chunks:
                logger.info(f"Stopping at level {current_level} (no valid clusters)")
                break
            
            # Store next level
            self.tree[current_level + 1] = next_level_chunks
            
            # Embed next level
            next_level_texts = [c['text'] for c in next_level_chunks]
            logger.info(f"Embedding level {current_level + 1} ({len(next_level_texts)} summaries)...")
            self.level_embeddings[current_level + 1] = self.embedding_model.encode(
                next_level_texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            logger.info(f"Level {current_level + 1}: {len(next_level_chunks)} summary nodes")
            current_level += 1
        
        # Tree statistics
        tree_height = len(self.tree)
        total_nodes = sum(len(nodes) for nodes in self.tree.values())
        
        logger.info(f"✓ RAPTOR tree built: {tree_height} levels, {total_nodes} total nodes")
        
        return {
            'height': tree_height,
            'total_nodes': total_nodes,
            'level_counts': {level: len(nodes) for level, nodes in self.tree.items()}
        }
    
    def _cluster_chunks(self, chunks: List[Dict], embeddings: np.ndarray) -> Dict[int, List[Dict]]:
        """
        Cluster chunks using HDBSCAN
        
        Args:
            chunks: List of chunks to cluster
            embeddings: Embeddings for chunks
            
        Returns:
            Dict mapping cluster_id -> list of chunks
        """
        if len(chunks) < self.min_cluster_size:
            # Too few chunks to cluster
            return {0: chunks}
        
        # Run HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=2,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Group chunks by cluster
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(chunks[idx])
        
        # Log cluster distribution
        cluster_sizes = {k: len(v) for k, v in clusters.items() if k != -1}
        logger.info(f"Clustered into {len(cluster_sizes)} clusters: {cluster_sizes}")
        
        return clusters
    
    def _summarize_cluster(self, chunks: List[Dict], level: int, cluster_id: int) -> Dict:
        """
        Generate summary for a cluster of chunks
        
        Args:
            chunks: Chunks in this cluster
            level: Tree level being created
            cluster_id: Cluster identifier
            
        Returns:
            Summary chunk dict
        """
        # Combine chunk texts
        combined_text = "\n\n".join([c['text'] for c in chunks])
        
        # Generate summary using LLM
        try:
            summary = self.llm.generate_summary(
                chunks=chunks,
                target_words=300,  # Medium-length summaries
                focus=f"Level {level} cluster {cluster_id}"
            )
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}, using concatenation")
            # Fallback: truncate combined text
            summary = combined_text[:1500] + "..."
        
        # Create summary chunk
        chunk_id = self._generate_chunk_id(level, cluster_id)
        
        # Inherit metadata from constituent chunks
        well_names = set()
        doc_ids = set()
        for chunk in chunks:
            well_names.update(chunk.get('well_names', []))
            doc_ids.add(chunk.get('doc_id', 'unknown'))
        
        return {
            'text': summary,
            'chunk_id': chunk_id,
            'level': level,
            'cluster_id': cluster_id,
            'child_chunks': [c['chunk_id'] for c in chunks],
            'well_names': list(well_names),
            'doc_ids': list(doc_ids),
            'metadata': {
                'is_summary': True,
                'num_children': len(chunks)
            }
        }
    
    def _generate_chunk_id(self, level: int, cluster_id: int) -> str:
        """Generate unique chunk ID for summary node"""
        return f"raptor_L{level}_C{cluster_id}"
    
    def query_tree(self, query: str, level: Optional[int] = None, 
                   top_k: int = 10) -> List[Dict]:
        """
        Query RAPTOR tree at specific level or all levels
        
        Args:
            query: Search query
            level: Tree level to query (None = all levels)
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks with scores
        """
        if not self.tree:
            logger.warning("RAPTOR tree not built yet")
            return []
        
        # Embed query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        
        # Query specific level or all levels
        if level is not None:
            if level not in self.tree:
                logger.warning(f"Level {level} does not exist")
                return []
            return self._query_level(query_embedding, level, top_k)
        else:
            # Query all levels and combine
            all_results = []
            for lvl in self.tree.keys():
                level_results = self._query_level(query_embedding, lvl, top_k // len(self.tree))
                all_results.extend(level_results)
            
            # Sort by score and return top_k
            all_results.sort(key=lambda x: x['score'], reverse=True)
            return all_results[:top_k]
    
    def _query_level(self, query_embedding: np.ndarray, level: int, top_k: int) -> List[Dict]:
        """
        Query specific tree level
        
        Args:
            query_embedding: Query embedding vector
            level: Tree level to query
            top_k: Number of results
            
        Returns:
            List of chunks with scores
        """
        if level not in self.level_embeddings:
            return []
        
        # Compute cosine similarities
        level_embeddings = self.level_embeddings[level]
        similarities = np.dot(level_embeddings, query_embedding) / (
            np.linalg.norm(level_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Prepare results
        results = []
        for idx in top_indices:
            chunk = self.tree[level][idx].copy()
            chunk['score'] = float(similarities[idx])
            chunk['level'] = level
            results.append(chunk)
        
        return results
    
    def get_tree_statistics(self) -> Dict:
        """Get statistics about the RAPTOR tree"""
        if not self.tree:
            return {'status': 'not_built'}
        
        stats = {
            'height': len(self.tree),
            'total_nodes': sum(len(nodes) for nodes in self.tree.values()),
            'level_stats': {}
        }
        
        for level, nodes in self.tree.items():
            texts = [n['text'] for n in nodes]
            word_counts = [len(t.split()) for t in texts]
            
            stats['level_stats'][level] = {
                'num_nodes': len(nodes),
                'avg_words': np.mean(word_counts) if word_counts else 0,
                'total_words': sum(word_counts)
            }
        
        return stats
    
    def traverse_to_leaves(self, summary_chunk_id: str) -> List[str]:
        """
        Traverse tree from summary node to leaf chunks
        
        Args:
            summary_chunk_id: ID of summary chunk
            
        Returns:
            List of leaf chunk IDs
        """
        # Find the summary chunk
        summary_chunk = None
        for level in range(1, len(self.tree)):
            for chunk in self.tree[level]:
                if chunk['chunk_id'] == summary_chunk_id:
                    summary_chunk = chunk
                    break
            if summary_chunk:
                break
        
        if not summary_chunk:
            return []
        
        # Recursively traverse to leaves
        return self._get_leaf_ids(summary_chunk)
    
    def _get_leaf_ids(self, chunk: Dict) -> List[str]:
        """Recursively get leaf chunk IDs"""
        if 'child_chunks' not in chunk:
            # This is a leaf
            return [chunk['chunk_id']]
        
        # Get children and recurse
        leaf_ids = []
        for child_id in chunk['child_chunks']:
            # Find child chunk
            child_chunk = None
            for level in self.tree.values():
                for c in level:
                    if c['chunk_id'] == child_id:
                        child_chunk = c
                        break
                if child_chunk:
                    break
            
            if child_chunk:
                leaf_ids.extend(self._get_leaf_ids(child_chunk))
        
        return leaf_ids


def create_raptor_tree(llm_helper, config: Dict = None) -> RAPTORTree:
    """Factory function to create RAPTOR tree"""
    if config is None:
        config = {}
    
    return RAPTORTree(
        llm_helper=llm_helper,
        embedding_model=config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'),
        min_cluster_size=config.get('min_cluster_size', 5),
        max_tree_height=config.get('max_tree_height', 3)
    )
