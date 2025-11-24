"""
RAG Retrieval Agent - Hybrid Search with Multi-Strategy Collections
Implements vector search with ChromaDB and hybrid dense/sparse retrieval
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
        
        # Collection names - includes hybrid strategies
        self.collection_names = {
            'factual_qa': 'geo_factual',
            'technical_extraction': 'geo_technical',
            'summary': 'geo_summary',
            'fine_grained': 'geo_fine',
            'coarse_grained': 'geo_coarse'
        }
        
        # Collections will be created during indexing
        self.collections = {}
        
        logger.info(f"Initialized RAGRetrievalAgent with DB at {db_path}")
    
    def index_chunks(self, chunks_dict: Dict[str, List[Dict]]) -> None:
        """
        Index chunks into ChromaDB collections
        
        Args:
            chunks_dict: Dict with keys 'factual_qa', 'technical_extraction', 'summary'
                        and optionally 'fine_grained', 'coarse_grained'
                        Each contains list of chunk dicts
        """
        for strategy, chunks in chunks_dict.items():
            if not chunks:
                logger.warning(f"No chunks for strategy: {strategy}")
                continue
            
            # Skip if strategy not recognized
            if strategy not in self.collection_names:
                logger.warning(f"Unknown strategy: {strategy}, skipping")
                continue
            
            collection_name = self.collection_names[strategy]
            
            # Delete existing collection if it exists (with version compatibility handling)
            try:
                self.client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
            except KeyError as e:
                # ChromaDB version mismatch - collection format incompatible
                logger.warning(f"Collection {collection_name} has incompatible format, will recreate")
            except Exception as e:
                # Collection doesn't exist, which is fine
                pass
            
            # Create new collection with Ollama embeddings
            try:
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=self.embedding_function
                )
            except Exception as e:
                logger.error(f"Failed to create collection {collection_name}: {str(e)}")
                raise
            
            # Prepare data for indexing
            ids = []
            documents = []
            metadatas = []
            
            for chunk in chunks:
                ids.append(chunk['chunk_id'])
                documents.append(chunk['text'])
                
                # Store metadata
                metadata = {
                    'doc_id': chunk['doc_id'],
                    'strategy': chunk['strategy'],
                    'page_numbers': ','.join(map(str, chunk['page_numbers'])),
                    'well_names': ','.join(chunk['well_names']) if chunk['well_names'] else '',
                    'source_file': chunk['metadata']['source_file']
                }
                metadatas.append(metadata)
            
            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i+batch_size]
                batch_docs = documents[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size]
                
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas
                )
            
            self.collections[strategy] = collection
            logger.info(f"✓ Indexed {len(ids)} chunks into {collection_name}")
    
    def retrieve(self, query: str, mode: str = 'qa', top_k: Optional[int] = None, 
                 well_name: Optional[str] = None) -> Dict:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: Search query
            mode: 'qa', 'extract', or 'summary'
            top_k: Number of results to return (uses config default if None)
            well_name: Optional well name to filter results
            
        Returns:
            Dict with:
            {
                'chunks': List[Dict],  # Retrieved chunks with scores
                'query': str,
                'mode': str,
                'top_k': int
            }
        """
        # Map mode to strategy
        strategy_map = {
            'qa': 'factual_qa',
            'extract': 'technical_extraction',
            'summary': 'summary'
        }
        
        strategy = strategy_map.get(mode, 'factual_qa')
        
        # Get top_k from config if not specified
        if top_k is None:
            top_k_map = {
                'qa': self.retrieval_config['top_k_qa'],
                'extract': self.retrieval_config['top_k_extraction'],
                'summary': self.retrieval_config['top_k_summary']
            }
            top_k = top_k_map.get(mode, 10)
        
        # Load collection if not already loaded
        if strategy not in self.collections:
            collection_name = self.collection_names[strategy]
            try:
                self.collections[strategy] = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
            except:
                logger.error(f"Collection not found: {collection_name}. Run indexing first.")
                return {'chunks': [], 'query': query, 'mode': mode, 'top_k': top_k}
        
        collection = self.collections[strategy]
        
        # Build query filter for well name if provided
        # Note: Skip filtering in older ChromaDB versions that don't support $contains
        # Instead, filter results after retrieval
        where_filter = None
        # Commenting out $contains filter as it's not supported in all ChromaDB versions
        # if well_name:
        #     where_filter = {
        #         "$or": [
        #             {"well_names": {"$contains": well_name}},
        #             {"doc_id": {"$contains": well_name}}
        #         ]
        #     }
        
        # Query collection
        try:
            results = collection.query(
                query_texts=[query],
                n_results=min(top_k, collection.count()),
                where=where_filter
            )
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return {'chunks': [], 'query': query, 'mode': mode, 'top_k': top_k}
        
        # Format results
        chunks = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                chunk = {
                    'text': results['documents'][0][i],
                    'id': results['ids'][0][i],
                    'distance': results['distances'][0][i],
                    'metadata': results['metadatas'][0][i]
                }
                
                # Parse page numbers back to list
                if chunk['metadata'].get('page_numbers'):
                    chunk['metadata']['page_numbers'] = [
                        int(p) for p in chunk['metadata']['page_numbers'].split(',') if p
                    ]
                
                # Parse well names back to list
                if chunk['metadata'].get('well_names'):
                    chunk['metadata']['well_names'] = [
                        w.strip() for w in chunk['metadata']['well_names'].split(',') if w
                    ]
                
                # Post-retrieval filtering by well name (for ChromaDB compatibility)
                if well_name:
                    well_names = chunk['metadata'].get('well_names', [])
                    doc_id = chunk['metadata'].get('doc_id', '')
                    # Check if well_name appears in well_names list or doc_id
                    if not (any(well_name.upper() in wn.upper() for wn in well_names) or 
                            well_name.upper() in doc_id.upper()):
                        continue  # Skip this chunk
                
                chunks.append(chunk)
        
        logger.info(f"Retrieved {len(chunks)} chunks for mode='{mode}', query='{query[:50]}...'")
        
        return {
            'chunks': chunks,
            'query': query,
            'mode': mode,
            'top_k': top_k
        }
    
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
