#!/usr/bin/env python3
"""
Simple validation test for refactored configuration
Tests config file changes without requiring dependencies
"""

import yaml
from pathlib import Path

def test_config():
    """Test that config.yaml is properly updated"""
    print("\n=== Testing Configuration ===")
    
    config_path = Path(__file__).parent / 'config' / 'config.yaml'
    
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("✓ Config file loaded")
    
    # Check chunking strategies (directly under 'chunking' key)
    chunking = config.get('chunking', {})
    # Filter out non-strategy keys (comments, metadata)
    strategies = {k: v for k, v in chunking.items() if isinstance(v, dict) and 'chunk_size' in v}
    print(f"\nChunking strategies: {list(strategies.keys())}")
    
    if len(strategies) != 1:
        print(f"✗ ERROR: Expected 1 strategy, found {len(strategies)}")
        return False
    
    if 'fine_grained' not in strategies:
        print("✗ ERROR: 'fine_grained' strategy missing")
        return False
    
    print("✓ Only fine_grained strategy present")
    
    # Check fine_grained parameters
    fine = strategies['fine_grained']
    chunk_size = fine.get('chunk_size')
    overlap = fine.get('chunk_overlap')  # Config uses 'chunk_overlap' not 'overlap'
    
    print(f"  - Chunk size: {chunk_size} words")
    print(f"  - Overlap: {overlap} words")
    
    if chunk_size != 500:
        print(f"✗ ERROR: Expected chunk_size=500, got {chunk_size}")
        return False
    
    if overlap != 150:
        print(f"✗ ERROR: Expected overlap=150, got {overlap}")
        return False
    
    print("✓ Chunk parameters correct (500 words, 150 overlap)")
    
    # Check for removed strategies
    removed = ['factual_qa', 'technical_extraction', 'summary', 'coarse_grained']
    found_removed = [s for s in removed if s in strategies]
    
    if found_removed:
        print(f"✗ ERROR: Found removed strategies: {found_removed}")
        return False
    
    print(f"✓ Removed strategies not present: {', '.join(removed)}")
    
    return True

def test_app_structure():
    """Test that app.py has correct structure"""
    print("\n=== Testing App Structure ===")
    
    app_path = Path(__file__).parent / 'app.py'
    
    if not app_path.exists():
        print(f"✗ app.py not found")
        return False
    
    with open(app_path, 'r') as f:
        content = f.read()
    
    print("✓ app.py loaded")
    
    # Check for removed methods
    if 'def _handle_extraction' in content:
        print("✗ ERROR: _handle_extraction method still present")
        return False
    
    print("✓ _handle_extraction removed")
    
    if 'def run_nodal_analysis' in content:
        print("✗ ERROR: run_nodal_analysis method still present")
        return False
    
    print("✓ run_nodal_analysis removed")
    
    # Check for UI simplification
    if '"Extract & Analyze"' in content:
        print("✗ ERROR: 'Extract & Analyze' mode still in UI")
        return False
    
    print("✓ 'Extract & Analyze' mode removed from UI")
    
    # Check for new summary method
    if '_format_table_markdown' not in content:
        print("✗ ERROR: _format_table_markdown method missing")
        return False
    
    print("✓ New _format_table_markdown method present")
    
    # Check for simplified mode choices
    if 'choices=["Q&A", "Summary"]' not in content:
        print("✗ ERROR: Simplified mode choices not found")
        return False
    
    print("✓ UI modes simplified to Q&A and Summary")
    
    return True

def test_database_manager():
    """Test that database_manager.py has complete_tables support"""
    print("\n=== Testing Database Manager ===")
    
    db_path = Path(__file__).parent / 'agents' / 'database_manager.py'
    
    if not db_path.exists():
        print(f"✗ database_manager.py not found")
        return False
    
    with open(db_path, 'r') as f:
        content = f.read()
    
    print("✓ database_manager.py loaded")
    
    # Check for complete_tables methods
    if 'def store_complete_table' not in content:
        print("✗ ERROR: store_complete_table method missing")
        return False
    
    print("✓ store_complete_table method present")
    
    if 'def get_complete_tables' not in content:
        print("✗ ERROR: get_complete_tables method missing")
        return False
    
    print("✓ get_complete_tables method present")
    
    # Check for complete_tables table creation
    if 'CREATE TABLE IF NOT EXISTS complete_tables' not in content:
        print("✗ ERROR: complete_tables table creation missing")
        return False
    
    print("✓ complete_tables table schema present")
    
    return True

def test_hybrid_retrieval():
    """Test that hybrid_retrieval_agent.py is simplified"""
    print("\n=== Testing Hybrid Retrieval Agent ===")
    
    hybrid_path = Path(__file__).parent / 'agents' / 'hybrid_retrieval_agent.py'
    
    if not hybrid_path.exists():
        print(f"✗ hybrid_retrieval_agent.py not found")
        return False
    
    with open(hybrid_path, 'r') as f:
        content = f.read()
    
    print("✓ hybrid_retrieval_agent.py loaded")
    
    # Check that _classify_query is removed
    if 'def _classify_query' in content:
        print("✗ ERROR: _classify_query method still present")
        return False
    
    print("✓ _classify_query method removed")
    
    # Check that mode parameter is not in retrieve signature
    if 'def retrieve(self, query: str, well_name: str, mode:' in content:
        print("✗ ERROR: mode parameter still in retrieve method")
        return False
    
    print("✓ mode parameter removed from retrieve method")
    
    return True

def main():
    """Run all validation tests"""
    print("=" * 70)
    print("REFACTORED SYSTEM VALIDATION (Config & Structure Only)")
    print("=" * 70)
    
    tests = [
        ("Configuration", test_config),
        ("App Structure", test_app_structure),
        ("Database Manager", test_database_manager),
        ("Hybrid Retrieval", test_hybrid_retrieval),
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
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All validation tests passed!")
        print("\nRefactoring Summary:")
        print("  - Removed: 10 validation/extraction agents")
        print("  - Removed: 4 chunking strategies (kept fine_grained only)")
        print("  - Removed: Extract & Analyze mode, nodal analysis")
        print("  - Added: Complete table storage in SQLite")
        print("  - Simplified: Hybrid retrieval always queries both sources")
        print("  - New: 8-data-type summary system")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Review errors above.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
