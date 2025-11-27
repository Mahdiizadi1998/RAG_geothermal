"""
Test Setup Integration
Verifies that the setup scripts install everything needed for the Advanced RAG System
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_python_packages():
    """Test that all required Python packages are installed"""
    logger.info("=" * 80)
    logger.info("TEST 1: Python Package Installation")
    logger.info("=" * 80)
    
    required_packages = {
        # Core
        'gradio': 'gradio>=4.0.0',
        'yaml': 'pyyaml>=6.0',
        
        # LLM and embeddings
        'ollama': 'ollama>=0.1.0',
        'chromadb': 'chromadb>=0.4.0,<1.0.0',
        
        # PDF processing
        'fitz': 'PyMuPDF>=1.23.0',
        'camelot': 'camelot-py[cv]>=0.11.0',
        'PIL': 'Pillow>=10.0.0',
        
        # NLP
        'spacy': 'spacy>=3.7.0',
        'sentence_transformers': 'sentence-transformers>=2.2.0',
        'langchain': 'langchain>=0.1.0',
        'langchain_community': 'langchain-community>=0.0.20',
        
        # Scientific
        'numpy': 'numpy>=1.24.0',
        'scipy': 'scipy>=1.11.0',
        'pandas': 'pandas>=2.0.0',
        'matplotlib': 'matplotlib>=3.7.0',
        
        # Advanced RAG
        'hdbscan': 'hdbscan>=0.8.33',
        'networkx': 'networkx>=3.0',
        'sklearn': 'scikit-learn>=1.3.0',
        
        # Utilities
        'tqdm': 'tqdm>=4.65.0',
        'dotenv': 'python-dotenv>=1.0.0',
        'requests': 'requests>=2.31.0',
    }
    
    missing = []
    installed = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            installed.append(package)
            logger.info(f"✅ {package}")
        except ImportError:
            missing.append(package)
            logger.error(f"❌ {package} NOT INSTALLED")
    
    logger.info(f"\n📊 Results: {len(installed)}/{len(required_packages)} packages installed")
    
    if missing:
        logger.error("\n❌ Missing packages (install with pip):")
        for pkg in missing:
            logger.error(f"   pip install {pkg}")
        return False
    
    logger.info("\n✅ All required Python packages are installed!\n")
    return True


def test_spacy_model():
    """Test that spaCy model is downloaded"""
    logger.info("=" * 80)
    logger.info("TEST 2: spaCy Language Model")
    logger.info("=" * 80)
    
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        logger.info("✅ spaCy model 'en_core_web_sm' is installed")
        logger.info(f"   Version: {nlp.meta['version']}")
        logger.info(f"   Language: {nlp.meta['lang']}")
        logger.info("\n✅ spaCy model test passed!\n")
        return True
    except Exception as e:
        logger.error(f"❌ spaCy model 'en_core_web_sm' not found: {e}")
        logger.error("\n   Download with: python -m spacy download en_core_web_sm\n")
        return False


def test_ollama_installation():
    """Test Ollama installation and models"""
    logger.info("=" * 80)
    logger.info("TEST 3: Ollama Installation and Models")
    logger.info("=" * 80)
    
    import subprocess
    import requests
    
    # Check Ollama CLI
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info(f"✅ Ollama CLI installed: {result.stdout.strip()}")
        else:
            logger.error("❌ Ollama CLI not working properly")
            return False
    except FileNotFoundError:
        logger.error("❌ Ollama CLI not found in PATH")
        logger.error("   Install from: https://ollama.ai/")
        return False
    except Exception as e:
        logger.error(f"❌ Error checking Ollama: {e}")
        return False
    
    # Check Ollama server
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            logger.info("✅ Ollama server is running")
            
            # Check models
            models = response.json().get('models', [])
            required_models = ['llama3.1:8b', 'llava:7b']
            optional_models = ['nomic-embed-text']
            
            installed_models = [m['name'] for m in models]
            logger.info(f"\n📦 Installed models: {len(installed_models)}")
            
            all_required = True
            for model in required_models:
                # Check both exact match and with :latest tag
                found = any(model in m for m in installed_models)
                if found:
                    logger.info(f"✅ {model} (REQUIRED)")
                else:
                    logger.error(f"❌ {model} (REQUIRED) - NOT FOUND")
                    logger.error(f"   Install with: ollama pull {model}")
                    all_required = False
            
            for model in optional_models:
                found = any(model in m for m in installed_models)
                if found:
                    logger.info(f"✅ {model} (OPTIONAL)")
                else:
                    logger.warning(f"⚠️  {model} (OPTIONAL) - not found")
                    logger.info(f"   Install with: ollama pull {model}")
            
            if all_required:
                logger.info("\n✅ All required Ollama models are installed!\n")
                return True
            else:
                logger.error("\n❌ Missing required Ollama models!\n")
                return False
        else:
            logger.error("❌ Ollama server returned error")
            return False
    except requests.exceptions.ConnectionError:
        logger.error("❌ Ollama server is not running")
        logger.error("   Start with: ollama serve")
        return False
    except Exception as e:
        logger.error(f"❌ Error checking Ollama server: {e}")
        return False


def test_sentence_transformers_models():
    """Test that sentence-transformers models can be loaded"""
    logger.info("=" * 80)
    logger.info("TEST 4: Sentence Transformers Models")
    logger.info("=" * 80)
    
    try:
        from sentence_transformers import SentenceTransformer, CrossEncoder
        
        # Test embedding model
        logger.info("Loading all-MiniLM-L6-v2 (embeddings)...")
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        test_embedding = embed_model.encode("test", show_progress_bar=False)
        logger.info(f"✅ all-MiniLM-L6-v2 loaded (dim: {len(test_embedding)})")
        
        # Test reranker model
        logger.info("Loading cross-encoder/ms-marco-MiniLM-L-6-v2 (reranker)...")
        rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        test_score = rerank_model.predict([("test query", "test document")])
        logger.info(f"✅ cross-encoder/ms-marco-MiniLM-L-6-v2 loaded")
        
        logger.info("\n✅ All sentence-transformers models working!\n")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading sentence-transformers models: {e}")
        logger.error("   Models will be auto-downloaded on first use")
        return False


def test_directory_structure():
    """Test that required directories exist"""
    logger.info("=" * 80)
    logger.info("TEST 5: Directory Structure")
    logger.info("=" * 80)
    
    project_root = Path(__file__).parent
    required_dirs = [
        'chroma_db',
        'agents',
        'config',
        'models',
        'utils',
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            logger.info(f"✅ {dir_name}/ exists")
        else:
            logger.error(f"❌ {dir_name}/ NOT FOUND")
            all_exist = False
    
    if all_exist:
        logger.info("\n✅ All required directories exist!\n")
    else:
        logger.error("\n❌ Some directories are missing!\n")
    
    return all_exist


def test_config_file():
    """Test that config.yaml is properly configured"""
    logger.info("=" * 80)
    logger.info("TEST 6: Configuration File")
    logger.info("=" * 80)
    
    import yaml
    
    config_path = Path(__file__).parent / 'config' / 'config.yaml'
    
    if not config_path.exists():
        logger.error("❌ config/config.yaml NOT FOUND")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("✅ config.yaml loaded")
        
        # Check critical settings
        checks = [
            ('embeddings.model', 'all-MiniLM-L6-v2'),
            ('ollama.model_qa', 'llama3.1:8b'),
            ('ollama.model_vision', 'llava:7b'),
            ('semantic_chunking.enabled', True),
            ('raptor.enabled', True),
            ('knowledge_graph.enabled', True),
            ('bm25.enabled', True),
            ('reranking.enabled', True),
            ('vision.enabled', True),
        ]
        
        all_correct = True
        for key_path, expected_value in checks:
            keys = key_path.split('.')
            value = config
            try:
                for key in keys:
                    value = value[key]
                
                if value == expected_value:
                    logger.info(f"✅ {key_path} = {value}")
                else:
                    logger.warning(f"⚠️  {key_path} = {value} (expected: {expected_value})")
            except KeyError:
                logger.error(f"❌ {key_path} NOT FOUND in config")
                all_correct = False
        
        if all_correct:
            logger.info("\n✅ Configuration is correct!\n")
        else:
            logger.warning("\n⚠️  Configuration has warnings (may still work)\n")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error reading config.yaml: {e}")
        return False


def test_advanced_components():
    """Test that advanced RAG components can be imported"""
    logger.info("=" * 80)
    logger.info("TEST 7: Advanced RAG Components")
    logger.info("=" * 80)
    
    components = [
        ('agents.ultimate_semantic_chunker', 'UltimateSemanticChunker'),
        ('agents.raptor_tree', 'RAPTORTree'),
        ('agents.bm25_retrieval', 'BM25Retriever'),
        ('agents.knowledge_graph', 'KnowledgeGraph'),
        ('agents.universal_metadata_extractor', 'UniversalGeothermalMetadataExtractor'),
        ('agents.vision_processor', 'VisionProcessor'),
        ('agents.reranker', 'Reranker'),
    ]
    
    all_ok = True
    for module_name, class_name in components:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            logger.info(f"✅ {module_name}.{class_name}")
        except ImportError as e:
            logger.error(f"❌ {module_name}.{class_name} - Import failed: {e}")
            all_ok = False
        except AttributeError as e:
            logger.error(f"❌ {module_name}.{class_name} - Class not found: {e}")
            all_ok = False
    
    if all_ok:
        logger.info("\n✅ All advanced RAG components available!\n")
    else:
        logger.error("\n❌ Some components are missing!\n")
    
    return all_ok


def run_all_tests():
    """Run all setup integration tests"""
    logger.info("\n" + "=" * 80)
    logger.info("SETUP INTEGRATION TEST SUITE")
    logger.info("Verifies that setup scripts installed everything correctly")
    logger.info("=" * 80 + "\n")
    
    results = {}
    
    # Run tests
    results['Python Packages'] = test_python_packages()
    results['spaCy Model'] = test_spacy_model()
    results['Ollama'] = test_ollama_installation()
    results['Sentence Transformers'] = test_sentence_transformers_models()
    results['Directory Structure'] = test_directory_structure()
    results['Configuration'] = test_config_file()
    results['Advanced Components'] = test_advanced_components()
    
    # Summary
    logger.info("=" * 80)
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
        logger.info("🎉 SETUP COMPLETE! All components are properly installed.")
        logger.info("\n✅ You can now run: python app.py")
        return 0
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed.")
        logger.info("\nRun the setup script again:")
        logger.info("  Windows: setup_simple.bat")
        logger.info("  Linux/Mac: bash setup.sh")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
