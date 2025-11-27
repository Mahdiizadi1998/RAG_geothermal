"""
RAG Retrieval Agent - Advanced Hybrid Retrieval with Multiple Strategies
Implements Dense + Sparse (BM25) + Knowledge Graph + RAPTOR multi-level retrieval
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction, SentenceTransformerEmbeddingFunction
from typing import Dict, List, Optional
import logging
import yaml
from pathlib import Path
import hashlib
import os

# Import advanced retrieval components
from agents.bm25_retrieval import create_bm25_retriever
from agents.knowledge_graph import create_knowledge_graph
from agents.raptor_tree import create_raptor_tree
from agents.reranker import create_reranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGRetrievalAgent:
    """
    Hybrid retrieval agent with multi-strategy collections
    
    Features:
    - Separate ChromaDB collections for each chunking strategy
    - Hybrid search: 0.7 × dense (vector) + 0.3 × sparse (BM25-style)
    - Re-ranking based on query type
    - Source metadata tracking for citations
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize retrieval agent
        
        Args:
            config_path: Path to config.yaml file
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
        
        # Store config path for later use (RAPTOR initialization)
        self.config_path = config_path
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.vector_db_config = self.config['vector_db']
        self.retrieval_config = self.config['retrieval']
        self.ollama_config = self.config['ollama']
        
        # Force CPU-only mode for Ollama (prevent GPU memory issues)
        os.environ['OLLAMA_NUM_GPU'] = '0'
        logger.info("🖥️  Configured for CPU-only mode (OLLAMA_NUM_GPU=0)")
        
        # Initialize embedding function based on config
        embedding_config = self.config.get('embeddings', {'backend': 'ollama'})
        embedding_backend = embedding_config.get('backend', 'ollama')
        
        if embedding_backend == 'sentence-transformers':
            # Use sentence-transformers (2-3x faster on CPU)
            model_name = embedding_config.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
            self.embedding_function = SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
            logger.info(f"📊 Using sentence-transformers: {model_name} (CPU-optimized)")
        else:
            # Use Ollama embeddings (default)
            self.embedding_function = OllamaEmbeddingFunction(
                url=self.ollama_config['host'] + "/api/embeddings",
                model_name=self.ollama_config['model_embedding']
            )
            logger.info(f"📊 Using Ollama embeddings: {self.ollama_config['model_embedding']}")
        
        # Initialize ChromaDB client
        db_path = Path(self.vector_db_config['path'])
        db_path.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Single collection name for fine-grained chunks
        self.collection_name = 'geo_fine_grained'
        
        # Collection will be created during indexing
        self.collection = None
        
        # Initialize advanced retrieval components
        self.bm25 = None
        self.knowledge_graph = None
        self.raptor = None
        self.reranker = None
        
        # BM25 for sparse retrieval
        if self.config.get('bm25', {}).get('enabled', False):
            self.bm25 = create_bm25_retriever(self.config.get('bm25', {}))
            logger.info("✓ BM25 retriever initialized")
        
        # Knowledge Graph for relationship-based retrieval
        if self.config.get('knowledge_graph', {}).get('enabled', False):
            self.knowledge_graph = create_knowledge_graph(self.config.get('knowledge_graph', {}))
            logger.info("✓ Knowledge Graph initialized")
        
        # Reranker for result fusion
        if self.config.get('reranking', {}).get('enabled', False):
            self.reranker = create_reranker(self.config.get('reranking', {}))
            logger.info("✓ Reranker initialized")
        
        logger.info(f"Initialized RAGRetrievalAgent with DB at {db_path}")
    
    def index_chunks(self, chunks_dict: Dict[str, List[Dict]]) -> None:
        """
        Index fine-grained chunks into single ChromaDB collection
        
        Args:
            chunks_dict: Dict with key 'fine_grained' containing list of chunk dicts
        """
        chunks = chunks_dict.get('fine_grained', [])
        
        if not chunks:
            logger.warning("No fine-grained chunks to index")
            return
        
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted existing collection: {self.collection_name}")
        except Exception:
            # Collection doesn't exist, which is fine
            pass
        
        # Create new collection
        try:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.embedding_function
            )
        except Exception as e:
            logger.error(f"Failed to create collection {self.collection_name}: {str(e)}")
            raise
        
        # Prepare data for indexing (minimal metadata)
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            ids.append(chunk['chunk_id'])
            documents.append(chunk['text'])
            
            # Store minimal metadata - well names only
            metadata = {
                'doc_id': chunk['doc_id'],
                'well_names': ','.join(chunk.get('well_names', []))
            }
            metadatas.append(metadata)
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas
            )
        
        logger.info(f"✓ Indexed {len(ids)} chunks into {self.collection_name}")
        
        # Index into BM25 if enabled
        if self.bm25:
            logger.info("Building BM25 index...")
            self.bm25.index_documents(chunks)
            logger.info("✓ BM25 index built")
        
        # Build knowledge graph if enabled
        if self.knowledge_graph:
            logger.info("Building knowledge graph...")
            self.knowledge_graph.build_graph(chunks)
            logger.info("✓ Knowledge graph built")
        
        # Build RAPTOR tree if enabled
        if self.config.get('raptor', {}).get('enabled', False):
            logger.info("Building RAPTOR hierarchical tree...")
            from agents.llm_helper import OllamaHelper
            llm = OllamaHelper(self.config_path)
            self.raptor = create_raptor_tree(llm, self.config.get('raptor', {}))
            self.raptor.build_tree(chunks)
            logger.info("✓ RAPTOR tree built")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Retrieve relevant chunks for a query from single fine-grained collection
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not self.collection:
            # Try to load existing collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
            except:
                logger.error(f"Collection not found: {self.collection_name}. Run indexing first.")
                return []
        
        # Multi-strategy retrieval
        all_results = []
        
        # Dense vector retrieval (ChromaDB)
        try:
            dense_results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k * 2, self.collection.count())
            )
            
            if dense_results['ids'] and dense_results['ids'][0]:
                for i in range(len(dense_results['ids'][0])):
                    chunk = {
                        'text': dense_results['documents'][0][i],
                        'id': dense_results['ids'][0][i],
                        'score': 1 - dense_results['distances'][0][i],  # Convert distance to similarity
                        'metadata': dense_results['metadatas'][0][i],
                        'source': 'dense'
                    }
                    all_results.append(chunk)
            logger.info(f"Dense retrieval: {len(all_results)} chunks")
        except Exception as e:
            logger.error(f"Dense retrieval failed: {str(e)}")
        
        # Sparse retrieval (BM25)
        if self.bm25:
            try:
                bm25_results = self.bm25.search(query, top_k=top_k * 2)
                for result in bm25_results:
                    result['source'] = 'sparse'
                    all_results.append(result)
                logger.info(f"BM25 retrieval: {len(bm25_results)} chunks")
            except Exception as e:
                logger.error(f"BM25 retrieval failed: {str(e)}")
        
        # Knowledge graph traversal
        if self.knowledge_graph:
            try:
                kg_results = self.knowledge_graph.retrieve_related(query, top_k=top_k)
                for result in kg_results:
                    result['source'] = 'knowledge_graph'
                    all_results.append(result)
                logger.info(f"Knowledge graph: {len(kg_results)} chunks")
            except Exception as e:
                logger.error(f"Knowledge graph retrieval failed: {str(e)}")
        
        # RAPTOR hierarchical retrieval
        if self.raptor:
            try:
                raptor_results = self.raptor.retrieve(query, top_k=top_k)
                for result in raptor_results:
                    result['source'] = 'raptor'
                    all_results.append(result)
                logger.info(f"RAPTOR retrieval: {len(raptor_results)} chunks")
            except Exception as e:
                logger.error(f"RAPTOR retrieval failed: {str(e)}")
        
        # Rerank if enabled, otherwise use dense results only
        if self.reranker and len(all_results) > 0:
            try:
                chunks = self.reranker.rerank(query, all_results, top_k=top_k)
                logger.info(f"Reranked to top {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Reranking failed: {str(e)}")
                # Fall back to dense results
                chunks = all_results[:top_k]
        else:
            # No reranking - just take top results from dense search
            chunks = [r for r in all_results if r.get('source') == 'dense'][:top_k]
        
        logger.info(f"Retrieved {len(chunks)} chunks for query='{query[:50]}...'")
        return chunks
    
    def retrieve_two_phase(self, query1: str, query2: str, mode1: str = 'extract', 
                          mode2: str = 'summary', top_k1: int = 15, 
                          top_k2: int = 10, well_name: Optional[str] = None) -> Dict:
        """
        Two-phase retrieval for extraction tasks
        
        Example: Phase 1 gets trajectory data, Phase 2 gets casing design
        
        Args:
            query1: First query (e.g., "trajectory survey")
            query2: Second query (e.g., "casing design")
            mode1: Mode for first query
            mode2: Mode for second query
            top_k1: Top-k for first query
            top_k2: Top-k for second query
            well_name: Optional well name filter
            
        Returns:
            Dict with combined chunks from both queries
        """
        logger.info(f"Two-phase retrieval: Q1='{query1[:30]}...', Q2='{query2[:30]}...'")
        
        # Phase 1
        result1 = self.retrieve(query1, mode=mode1, top_k=top_k1, well_name=well_name)
        
        # Phase 2
        result2 = self.retrieve(query2, mode=mode2, top_k=top_k2, well_name=well_name)
        
        # Combine chunks (remove duplicates by ID)
        all_chunks = []
        seen_ids = set()
        
        for chunk in result1['chunks'] + result2['chunks']:
            if chunk['id'] not in seen_ids:
                all_chunks.append(chunk)
                seen_ids.add(chunk['id'])
        
        # Sort by distance (lower is better)
        all_chunks.sort(key=lambda x: x['distance'])
        
        logger.info(f"Combined {len(all_chunks)} unique chunks from two-phase retrieval")
        
        return {
            'chunks': all_chunks,
            'query': f"Phase1: {query1} | Phase2: {query2}",
            'mode': f"{mode1}+{mode2}",
            'top_k': len(all_chunks)
        }
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about indexed collections"""
        stats = {}
        
        for strategy, collection_name in self.collection_names.items():
            try:
                collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                stats[strategy] = {
                    'name': collection_name,
                    'count': collection.count()
                }
            except:
                stats[strategy] = {
                    'name': collection_name,
                    'count': 0
                }
        
        return stats
    
    def retrieve_hybrid_granularity(self, query: str, mode: str = 'qa', 
                                    well_name: Optional[str] = None) -> Dict:
        """
        Hybrid retrieval using multiple chunk granularities
        
        Combines results from fine-grained, medium-grained, and coarse-grained chunks
        for better coverage and context
        
        Args:
            query: Search query
            mode: 'qa', 'extract', or 'summary'
            well_name: Optional well name to filter
            
        Returns:
            Dict with combined chunks from multiple granularities
        """
        # Get chunks from different granularities
        all_chunks = []
        
        # Primary strategy based on mode
        primary_result = self.retrieve(query, mode=mode, well_name=well_name)
        all_chunks.extend(primary_result['chunks'])
        
        # Add fine-grained chunks for precise details
        try:
            if 'fine_grained' in self.collection_names:
                fine_top_k = self.retrieval_config.get('top_k_fine', 15)
                fine_result = self.retrieve(query, mode='qa', top_k=fine_top_k, well_name=well_name)
                # Mark as fine-grained
                for chunk in fine_result['chunks']:
                    chunk['metadata']['granularity'] = 'fine'
                all_chunks.extend(fine_result['chunks'][:5])  # Take top 5 fine chunks
        except Exception as e:
            logger.debug(f"Fine-grained retrieval skipped: {str(e)}")
        
        # Add coarse-grained chunks for broader context
        try:
            if 'coarse_grained' in self.collection_names:
                coarse_top_k = self.retrieval_config.get('top_k_coarse', 10)
                coarse_result = self.retrieve(query, mode='summary', top_k=coarse_top_k, well_name=well_name)
                # Mark as coarse-grained
                for chunk in coarse_result['chunks']:
                    chunk['metadata']['granularity'] = 'coarse'
                all_chunks.extend(coarse_result['chunks'][:3])  # Take top 3 coarse chunks
        except Exception as e:
            logger.debug(f"Coarse-grained retrieval skipped: {str(e)}")
        
        # Deduplicate and re-rank
        unique_chunks = self._deduplicate_chunks(all_chunks)
        
        logger.info(f"Hybrid retrieval: {len(unique_chunks)} unique chunks from multiple granularities")
        
        return {
            'chunks': unique_chunks,
            'query': query,
            'mode': f'{mode}_hybrid',
            'granularities': ['primary', 'fine', 'coarse']
        }
    
    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Remove duplicate chunks based on text similarity"""
        if not chunks:
            return []
        
        unique = []
        seen_hashes = set()
        
        for chunk in chunks:
            # Create hash of first 100 chars
            text_sample = chunk['text'][:100]
            text_hash = hashlib.md5(text_sample.encode()).hexdigest()
            
            if text_hash not in seen_hashes:
                unique.append(chunk)
                seen_hashes.add(text_hash)
        
        return unique
    
    def clear_all_collections(self) -> None:
        """Delete all collections (useful for reindexing)"""
        for collection_name in self.collection_names.values():
            try:
                self.client.delete_collection(collection_name)
                logger.info(f"Deleted collection: {collection_name}")
            except:
                pass
        
        self.collections = {}
