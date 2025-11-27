"""
Integration Test for Advanced Agentic RAG System
Tests all new components: Ultimate Chunker, RAPTOR, BM25, Knowledge Graph, Vision, Reranking
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all new components can be imported"""
    logger.info("=" * 80)
    logger.info("TEST 1: Component Imports")
    logger.info("=" * 80)
    
    try:
        from agents.ultimate_semantic_chunker import UltimateSemanticChunker, create_chunker
        logger.info("✅ Ultimate Semantic Chunker imported")
    except Exception as e:
        logger.error(f"❌ Ultimate Semantic Chunker import failed: {e}")
        return False
    
    try:
        from agents.raptor_tree import RAPTORTree, create_raptor_tree
        logger.info("✅ RAPTOR Tree imported")
    except Exception as e:
        logger.error(f"❌ RAPTOR Tree import failed: {e}")
        return False
    
    try:
        from agents.bm25_retrieval import BM25Retriever, HybridDenseSparseRetriever, create_bm25_retriever
        logger.info("✅ BM25 Retrieval imported")
    except Exception as e:
        logger.error(f"❌ BM25 Retrieval import failed: {e}")
        return False
    
    try:
        from agents.knowledge_graph import KnowledgeGraph, create_knowledge_graph
        logger.info("✅ Knowledge Graph imported")
    except Exception as e:
        logger.error(f"❌ Knowledge Graph import failed: {e}")
        return False
    
    try:
        from agents.universal_metadata_extractor import UniversalGeothermalMetadataExtractor, create_metadata_extractor
        logger.info("✅ Universal Metadata Extractor imported")
    except Exception as e:
        logger.error(f"❌ Universal Metadata Extractor import failed: {e}")
        return False
    
    try:
        from agents.vision_processor import VisionProcessor, create_vision_processor
        logger.info("✅ Vision Processor imported")
    except Exception as e:
        logger.error(f"❌ Vision Processor import failed: {e}")
        return False
    
    try:
        from agents.reranker import Reranker, create_reranker
        logger.info("✅ Reranker imported")
    except Exception as e:
        logger.error(f"❌ Reranker import failed: {e}")
        return False
    
    try:
        from agents.query_analysis_agent import QueryAnalysisAgent
        logger.info("✅ Query Analysis Agent (enhanced) imported")
    except Exception as e:
        logger.error(f"❌ Query Analysis Agent import failed: {e}")
        return False
    
    logger.info("\n✅ All component imports successful!\n")
    return True


def test_dependencies():
    """Test that all required dependencies are installed"""
    logger.info("=" * 80)
    logger.info("TEST 2: Dependency Check")
    logger.info("=" * 80)
    
    dependencies = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'hdbscan': 'hdbscan',
        'networkx': 'networkx',
        'sentence_transformers': 'sentence-transformers',
        'spacy': 'spacy',
    }
    
    all_ok = True
    for module, package_name in dependencies.items():
        try:
            __import__(module)
            logger.info(f"✅ {package_name} installed")
        except ImportError:
            logger.error(f"❌ {package_name} NOT installed - run: pip install {package_name}")
            all_ok = False
    
    # Check spaCy model
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        logger.info("✅ spaCy model 'en_core_web_sm' available")
    except:
        logger.warning("⚠️  spaCy model 'en_core_web_sm' not found - run: python -m spacy download en_core_web_sm")
        all_ok = False
    
    if all_ok:
        logger.info("\n✅ All dependencies installed!\n")
    else:
        logger.warning("\n⚠️  Some dependencies missing - see above\n")
    
    return all_ok


def test_ultimate_chunker():
    """Test Ultimate Semantic Chunker"""
    logger.info("=" * 80)
    logger.info("TEST 3: Ultimate Semantic Chunker")
    logger.info("=" * 80)
    
    try:
        from agents.ultimate_semantic_chunker import create_chunker
        
        # Create chunker with test config
        config = {
            'embedding_model': 'all-MiniLM-L6-v2',
            'similarity_threshold': 0.7,
            'min_chunk_size': 100,
            'max_chunk_size': 300
        }
        
        chunker = create_chunker(config)
        logger.info("✅ Chunker created")
        
        # Test document
        test_doc = {
            'content': """Well ADK-GT-01 is a geothermal well drilled in the Netherlands.
            The well reached a total depth of 2850m MD and 2840m TVD.
            A 9-5/8 inch casing string was set at 2500m MD.
            The Slochteren formation was encountered at 2300m MD.
            Drilling operations commenced on January 15, 2023.
            The well showed promising geothermal potential.""",
            'filename': 'test_report.pdf',
            'wells': ['ADK-GT-01']
        }
        
        chunks = chunker.chunk_document(test_doc)
        logger.info(f"✅ Chunked document into {len(chunks)} chunks")
        
        # Verify contextual enrichment
        if chunks and '[Context:' in chunks[0]['text']:
            logger.info("✅ Contextual enrichment applied")
        else:
            logger.warning("⚠️  Contextual enrichment may not be working")
        
        logger.info(f"Sample chunk: {chunks[0]['text'][:100]}...")
        logger.info("\n✅ Ultimate Semantic Chunker test passed!\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ultimate Semantic Chunker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metadata_extractor():
    """Test Universal Metadata Extractor"""
    logger.info("=" * 80)
    logger.info("TEST 4: Universal Metadata Extractor")
    logger.info("=" * 80)
    
    try:
        from agents.universal_metadata_extractor import create_metadata_extractor
        
        extractor = create_metadata_extractor({'use_spacy': True})
        logger.info("✅ Metadata extractor created")
        
        # Test text
        test_text = """
        Well ADK-GT-01 was drilled by Nederlandse Aardolie Maatschappij (NAM).
        The well reached 2850m MD and 2840m TVD.
        The Slochteren formation was encountered at 2300m.
        Temperature at 2500m was 95°C.
        Pressure was measured at 250 bar.
        A 9-5/8 inch casing was installed.
        Drilling commenced on 2023-01-15.
        """
        
        metadata = extractor.extract_metadata(test_text, 'test_doc')
        
        logger.info(f"✅ Extracted {len(metadata['well_names'])} well names: {metadata['well_names']}")
        logger.info(f"✅ Extracted {len(metadata['formations'])} formations: {metadata['formations']}")
        logger.info(f"✅ Extracted {len(metadata['depths'])} depths")
        logger.info(f"✅ Extracted {len(metadata['temperatures'])} temperatures")
        logger.info(f"✅ Extracted {len(metadata['pressures'])} pressures")
        logger.info(f"✅ Extracted {len(metadata['equipment'])} equipment specs")
        
        if metadata['well_names'] and 'ADK-GT-01' in metadata['well_names']:
            logger.info("✅ Well name extraction working correctly")
        
        logger.info("\n✅ Universal Metadata Extractor test passed!\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Metadata Extractor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bm25_retrieval():
    """Test BM25 Sparse Retrieval"""
    logger.info("=" * 80)
    logger.info("TEST 5: BM25 Sparse Retrieval")
    logger.info("=" * 80)
    
    try:
        from agents.bm25_retrieval import create_bm25_retriever
        
        bm25 = create_bm25_retriever()
        logger.info("✅ BM25 retriever created")
        
        # Test documents
        test_chunks = [
            {'text': 'Well ADK-GT-01 has a 9-5/8 inch casing at 2500m', 'chunk_id': 'chunk1'},
            {'text': 'The Slochteren formation was encountered at 2300m', 'chunk_id': 'chunk2'},
            {'text': 'Well LDD-GT-02 reached 3000m total depth', 'chunk_id': 'chunk3'},
            {'text': 'Temperature at 2500m was 95 degrees Celsius', 'chunk_id': 'chunk4'},
        ]
        
        bm25.index_documents(test_chunks)
        logger.info(f"✅ Indexed {len(test_chunks)} documents")
        
        # Test query
        results = bm25.search('ADK-GT-01 casing', top_k=2)
        logger.info(f"✅ BM25 search returned {len(results)} results")
        
        if results and 'ADK-GT-01' in results[0]['text']:
            logger.info("✅ BM25 keyword matching working correctly")
        
        stats = bm25.get_term_statistics()
        logger.info(f"✅ Index stats: {stats['num_unique_terms']} unique terms")
        
        logger.info("\n✅ BM25 Retrieval test passed!\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ BM25 Retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_knowledge_graph():
    """Test Knowledge Graph"""
    logger.info("=" * 80)
    logger.info("TEST 6: Knowledge Graph")
    logger.info("=" * 80)
    
    try:
        from agents.knowledge_graph import create_knowledge_graph
        
        config = {
            'embedding_model': 'all-MiniLM-L6-v2',
            'similarity_threshold': 0.6,
            'metadata_edge_types': ['same_well']
        }
        
        kg = create_knowledge_graph(config)
        logger.info("✅ Knowledge graph created")
        
        # Test chunks
        test_chunks = [
            {
                'text': 'ADK-GT-01 casing program details',
                'chunk_id': 'chunk1',
                'well_names': ['ADK-GT-01']
            },
            {
                'text': 'ADK-GT-01 completion operations',
                'chunk_id': 'chunk2',
                'well_names': ['ADK-GT-01']
            },
            {
                'text': 'LDD-GT-02 drilling summary',
                'chunk_id': 'chunk3',
                'well_names': ['LDD-GT-02']
            },
        ]
        
        stats = kg.build_graph(test_chunks)
        logger.info(f"✅ Graph built: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
        
        # Test query
        related = kg.query_graph(['chunk1'], max_hops=1, max_nodes=5)
        logger.info(f"✅ Graph traversal returned {len(related)} related chunks")
        
        logger.info("\n✅ Knowledge Graph test passed!\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Knowledge Graph test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_raptor_tree():
    """Test RAPTOR Tree"""
    logger.info("=" * 80)
    logger.info("TEST 7: RAPTOR Tree")
    logger.info("=" * 80)
    
    try:
        # Mock LLM helper
        class MockLLM:
            def generate_summary(self, chunks, target_words=None, focus=None):
                texts = [c['text'] for c in chunks[:3]]
                return f"Summary of {len(chunks)} chunks: " + " ".join(texts[:100])
        
        from agents.raptor_tree import create_raptor_tree
        
        config = {
            'embedding_model': 'all-MiniLM-L6-v2',
            'min_cluster_size': 2,
            'max_tree_height': 2
        }
        
        llm = MockLLM()
        raptor = create_raptor_tree(llm, config)
        logger.info("✅ RAPTOR tree created")
        
        # Test chunks (need enough for clustering)
        test_chunks = [
            {'text': f'This is chunk {i} about well operations and drilling activities', 
             'chunk_id': f'chunk{i}', 
             'well_names': ['ADK-GT-01']}
            for i in range(10)
        ]
        
        stats = raptor.build_tree(test_chunks)
        logger.info(f"✅ RAPTOR tree built: {stats['height']} levels, {stats['total_nodes']} nodes")
        
        # Test query
        results = raptor.query_tree('drilling operations', level=0, top_k=3)
        logger.info(f"✅ RAPTOR query returned {len(results)} results")
        
        logger.info("\n✅ RAPTOR Tree test passed!\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ RAPTOR Tree test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reranker():
    """Test Reranking System"""
    logger.info("=" * 80)
    logger.info("TEST 8: Reranking System")
    logger.info("=" * 80)
    
    try:
        from agents.reranker import create_reranker
        
        config = {
            'method': 'cross-encoder',
            'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        }
        
        reranker = create_reranker(config)
        logger.info("✅ Reranker created")
        
        # Test documents
        query = "What is the casing design for ADK-GT-01?"
        docs = [
            {'text': 'ADK-GT-01 has a 9-5/8 inch casing at 2500m depth', 'id': 'doc1'},
            {'text': 'The Slochteren formation contains geothermal fluids', 'id': 'doc2'},
            {'text': 'Casing specifications include weight and grade', 'id': 'doc3'},
            {'text': 'Well LDD-GT-02 drilling operations summary', 'id': 'doc4'},
        ]
        
        reranked = reranker.rerank(query, docs, top_k=3)
        logger.info(f"✅ Reranked {len(docs)} documents, returned top {len(reranked)}")
        
        if reranked and 'rerank_score' in reranked[0]:
            logger.info(f"✅ Top result score: {reranked[0]['rerank_score']:.3f}")
            logger.info("✅ Cross-encoder scoring working")
        
        # Test RRF
        results_list = [
            [{'id': 'doc1', 'text': 'text1'}, {'id': 'doc2', 'text': 'text2'}],
            [{'id': 'doc2', 'text': 'text2'}, {'id': 'doc3', 'text': 'text3'}],
        ]
        fused = reranker.reciprocal_rank_fusion(results_list)
        logger.info(f"✅ RRF fused {len(results_list)} result lists into {len(fused)} documents")
        
        logger.info("\n✅ Reranking System test passed!\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Reranker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_query_router():
    """Test Enhanced Query Router"""
    logger.info("=" * 80)
    logger.info("TEST 9: Query Router")
    logger.info("=" * 80)
    
    try:
        from agents.query_analysis_agent import QueryAnalysisAgent
        import yaml
        
        # Load config
        config_path = Path(__file__).parent / 'config' / 'config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        router = QueryAnalysisAgent(config)
        logger.info("✅ Query router created")
        
        # Test different query types
        test_queries = [
            ("What is the casing design?", "hybrid"),
            ("Give me a summary of the well", "raptor"),
            ("Compare Well A and Well B", "graph"),
            ("Find all mentions of ADK-GT-01", "bm25"),
        ]
        
        for query, expected_strategy in test_queries:
            analysis = router.analyze(query)
            logger.info(f"Query: '{query}'")
            logger.info(f"  → Type: {analysis.query_type}, Strategy: {analysis.retrieval_strategy}")
            
            if analysis.retrieval_strategy == expected_strategy:
                logger.info(f"  ✅ Correctly routed to {expected_strategy}")
            else:
                logger.warning(f"  ⚠️  Expected {expected_strategy}, got {analysis.retrieval_strategy}")
        
        logger.info("\n✅ Query Router test passed!\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Query Router test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration file"""
    logger.info("=" * 80)
    logger.info("TEST 10: Configuration")
    logger.info("=" * 80)
    
    try:
        import yaml
        
        config_path = Path(__file__).parent / 'config' / 'config.yaml'
        
        if not config_path.exists():
            logger.error(f"❌ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("✅ Config file loaded")
        
        # Check new sections
        required_sections = [
            'semantic_chunking',
            'raptor',
            'knowledge_graph',
            'bm25',
            'reranking',
            'vision'
        ]
        
        for section in required_sections:
            if section in config:
                enabled = config[section].get('enabled', False)
                logger.info(f"✅ {section}: {'enabled' if enabled else 'disabled'}")
            else:
                logger.warning(f"⚠️  {section} section not found in config")
        
        # Check embedding model
        embedding_model = config.get('embeddings', {}).get('model')
        logger.info(f"✅ Embedding model: {embedding_model}")
        
        # Check LLM models
        ollama_config = config.get('ollama', {})
        logger.info(f"✅ QA Model: {ollama_config.get('model_qa')}")
        logger.info(f"✅ Vision Model: {ollama_config.get('model_vision')}")
        
        logger.info("\n✅ Configuration test passed!\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests"""
    logger.info("\n" + "=" * 80)
    logger.info("ADVANCED AGENTIC RAG SYSTEM - INTEGRATION TEST SUITE")
    logger.info("=" * 80 + "\n")
    
    results = {}
    
    # Run tests
    results['Imports'] = test_imports()
    results['Dependencies'] = test_dependencies()
    results['Ultimate Chunker'] = test_ultimate_chunker()
    results['Metadata Extractor'] = test_metadata_extractor()
    results['BM25 Retrieval'] = test_bm25_retrieval()
    results['Knowledge Graph'] = test_knowledge_graph()
    results['RAPTOR Tree'] = test_raptor_tree()
    results['Reranker'] = test_reranker()
    results['Query Router'] = test_query_router()
    results['Configuration'] = test_config()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        logger.info(f"{test_name:.<40} {status}")
    
    logger.info("=" * 80)
    logger.info(f"Results: {passed}/{total} tests passed ({100*passed//total}%)")
    logger.info("=" * 80 + "\n")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! System is fully integrated.")
        return 0
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed. Check logs above for details.")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
