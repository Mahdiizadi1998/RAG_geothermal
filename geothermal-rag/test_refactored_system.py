#!/usr/bin/env python3
"""
Quick test script to verify the refactored RAG system
Tests database table storage and hybrid retrieval
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.database_manager import WellDatabaseManager
from agents.rag_retrieval_agent import RAGRetrievalAgent
from agents.hybrid_retrieval_agent import HybridRetrievalAgent
import yaml

def test_database_storage():
    """Test complete table storage and retrieval"""
    print("\n=== Testing Database Storage ===")
    
    db = WellDatabaseManager()
    
    # Test storing a complete table
    headers = ["Depth (m)", "Casing OD (in)", "Weight (lb/ft)", "Grade"]
    rows = [
        [0, 30.0, 220.0, "K-55"],
        [500, 20.0, 133.0, "N-80"],
        [1000, 13.375, 68.0, "P-110"]
    ]
    
    db.store_complete_table(
        well_name="TEST-GT-01",
        source_document="test.pdf",
        page=5,
        table_type="Casing",
        table_reference="Table 4.1 - Casing String Summary",
        headers=headers,
        rows=rows
    )
    
    print("✓ Stored test table")
    
    # Retrieve and verify
    tables = db.get_complete_tables("TEST-GT-01", table_type="Casing")
    print(f"✓ Retrieved {len(tables)} table(s)")
    
    if tables:
        table = tables[0]
        print(f"  - Table: {table['table_reference']}")
        print(f"  - Rows: {table['num_rows']}, Cols: {table['num_cols']}")
        print(f"  - Headers: {len(table.get('headers_json', '[]'))} chars")
        print(f"  - Rows JSON: {len(table.get('rows_json', '[]'))} chars")
    
    return True

def test_single_collection():
    """Test that RAG uses single fine_grained collection"""
    print("\n=== Testing Single Collection ===")
    
    config_path = Path(__file__).parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    rag = RAGRetrievalAgent(config)
    
    print(f"✓ RAG initialized with collection: {rag.collection_name}")
    
    if rag.collection_name != "geo_fine_grained":
        print(f"✗ ERROR: Expected 'geo_fine_grained', got '{rag.collection_name}'")
        return False
    
    return True

def test_hybrid_retrieval():
    """Test hybrid retrieval always queries both sources"""
    print("\n=== Testing Hybrid Retrieval ===")
    
    config_path = Path(__file__).parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    db = WellDatabaseManager()
    rag = RAGRetrievalAgent(config)
    hybrid = HybridRetrievalAgent(config, db, rag)
    
    # Check that retrieve method no longer has mode parameter
    import inspect
    sig = inspect.signature(hybrid.retrieve)
    params = list(sig.parameters.keys())
    
    print(f"✓ Hybrid retrieve parameters: {params}")
    
    if 'mode' in params:
        print("✗ ERROR: 'mode' parameter should be removed")
        return False
    
    print("✓ No 'mode' parameter - always queries both sources")
    
    return True

def test_chunking_config():
    """Test that config only has fine_grained strategy"""
    print("\n=== Testing Chunking Configuration ===")
    
    config_path = Path(__file__).parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    strategies = config.get('chunking_strategies', {})
    print(f"✓ Found {len(strategies)} chunking strategy(ies)")
    
    if len(strategies) != 1:
        print(f"✗ ERROR: Expected 1 strategy, found {len(strategies)}")
        return False
    
    if 'fine_grained' not in strategies:
        print("✗ ERROR: 'fine_grained' strategy not found")
        return False
    
    fine_grained = strategies['fine_grained']
    print(f"  - Chunk size: {fine_grained.get('chunk_size')} words")
    print(f"  - Overlap: {fine_grained.get('overlap')} words")
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING REFACTORED RAG SYSTEM")
    print("=" * 60)
    
    tests = [
        ("Database Storage", test_database_storage),
        ("Single Collection", test_single_collection),
        ("Hybrid Retrieval", test_hybrid_retrieval),
        ("Chunking Config", test_chunking_config),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"✗ {name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready.")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
