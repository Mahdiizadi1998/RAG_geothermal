"""
Quick Test Script for Hybrid RAG System
Tests both Mode A (Q&A) and Mode B (Summary)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from agents.database_manager import WellDatabaseManager
from agents.enhanced_table_parser import EnhancedTableParser
from agents.ingestion_agent import IngestionAgent
from agents.hybrid_retrieval_agent import HybridRetrievalAgent
from agents.rag_retrieval_agent import RAGRetrievalAgent
from agents.well_summary_agent import WellSummaryAgent
from agents.llm_helper import OllamaHelper

def test_database_schema():
    """Test 1: Verify database schema includes all 8 data types"""
    print("=" * 80)
    print("TEST 1: Database Schema Verification")
    print("=" * 80)
    
    db = WellDatabaseManager("./test_well_data.db")
    
    # Check tables exist
    cursor = db.conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    required_tables = [
        'wells',           # General Data, Timeline, Depths
        'casing_strings',  # Casing & Tubulars
        'cementing',       # Cementing
        'drilling_fluids', # Fluids
        'formations',      # Geology
        'incidents'        # Incidents
    ]
    
    print("\nRequired tables:")
    for table in required_tables:
        status = "✓" if table in tables else "✗"
        print(f"  {status} {table}")
    
    # Check incidents table schema
    cursor.execute("PRAGMA table_info(incidents)")
    incidents_cols = [row[1] for row in cursor.fetchall()]
    print(f"\nIncidents table columns: {', '.join(incidents_cols)}")
    
    # Check casing table has pipe ID columns
    cursor.execute("PRAGMA table_info(casing_strings)")
    casing_cols = [row[1] for row in cursor.fetchall()]
    has_nominal = 'pipe_id_nominal' in casing_cols
    has_drift = 'pipe_id_drift' in casing_cols
    
    print(f"\nCasing table Pipe ID columns:")
    print(f"  {'✓' if has_nominal else '✗'} pipe_id_nominal")
    print(f"  {'✓' if has_drift else '✗'} pipe_id_drift")
    
    db.close()
    
    all_passed = all(table in tables for table in required_tables) and has_nominal and has_drift
    print(f"\n{'✅ PASSED' if all_passed else '❌ FAILED'}: Database schema test")
    return all_passed

def test_table_parser():
    """Test 2: Verify table parser supports all 8 data types"""
    print("\n" + "=" * 80)
    print("TEST 2: Table Parser Verification")
    print("=" * 80)
    
    parser = EnhancedTableParser()
    
    required_types = [
        'general_data',  # General Data, Timeline, Depths
        'casing',        # Casing & Tubulars
        'cementing',     # Cementing
        'fluids',        # Drilling Fluids
        'formations',    # Geology
        'incidents'      # Incidents
    ]
    
    print("\nSupported table types:")
    for table_type in required_types:
        has_keywords = table_type in parser.table_type_keywords
        status = "✓" if has_keywords else "✗"
        print(f"  {status} {table_type}")
    
    # Check parse methods exist
    parse_methods = [
        'parse_general_data_table',
        'parse_casing_table',
        'parse_cementing_table',
        'parse_fluids_table',
        'parse_formation_table',
        'parse_incidents_table'
    ]
    
    print("\nParse methods:")
    for method in parse_methods:
        has_method = hasattr(parser, method)
        status = "✓" if has_method else "✗"
        print(f"  {status} {method}()")
    
    all_passed = all(t in parser.table_type_keywords for t in required_types) and \
                 all(hasattr(parser, m) for m in parse_methods)
    
    print(f"\n{'✅ PASSED' if all_passed else '❌ FAILED'}: Table parser test")
    return all_passed

def test_ingestion_no_ocr():
    """Test 3: Verify OCR code is removed from ingestion agent"""
    print("\n" + "=" * 80)
    print("TEST 3: Ingestion Agent - No OCR Code")
    print("=" * 80)
    
    # Read the source code
    ingestion_path = Path(__file__).parent / "agents" / "ingestion_agent.py"
    with open(ingestion_path, 'r') as f:
        source = f.read()
    
    # Check for OCR imports
    ocr_imports = ['easyocr', 'pytesseract', 'PIL', 'Image']
    ocr_found = []
    for imp in ocr_imports:
        if imp in source:
            ocr_found.append(imp)
    
    print("\nOCR imports check:")
    if ocr_found:
        print(f"  ✗ Found OCR imports: {', '.join(ocr_found)}")
    else:
        print("  ✓ No OCR imports found")
    
    # Check for OCR methods
    has_ocr_method = '_ocr_page' in source
    print(f"\nOCR methods check:")
    print(f"  {'✗' if has_ocr_method else '✓'} _ocr_page() {'present' if has_ocr_method else 'removed'}")
    
    # Check for incidents handling
    has_incidents = 'add_incident' in source
    print(f"\nIncidents handling:")
    print(f"  {'✓' if has_incidents else '✗'} add_incident() calls present")
    
    all_passed = not ocr_found and not has_ocr_method and has_incidents
    print(f"\n{'✅ PASSED' if all_passed else '❌ FAILED'}: Ingestion agent test")
    return all_passed

def test_hybrid_retrieval():
    """Test 4: Verify hybrid retrieval supports all data types"""
    print("\n" + "=" * 80)
    print("TEST 4: Hybrid Retrieval Agent")
    print("=" * 80)
    
    db = WellDatabaseManager("./test_well_data.db")
    
    # Create a mock RAG agent
    class MockRAG:
        def retrieve(self, query, top_k=10):
            return []
    
    rag = MockRAG()
    hybrid = HybridRetrievalAgent(db, rag)
    
    # Check formatting methods exist
    format_methods = [
        '_format_well_info',
        '_format_casing_data',
        '_format_formation_data',
        '_format_cementing_data',
        '_format_fluids_data',
        '_format_incidents_data'
    ]
    
    print("\nFormat methods:")
    for method in format_methods:
        has_method = hasattr(hybrid, method)
        status = "✓" if has_method else "✗"
        print(f"  {status} {method}()")
    
    db.close()
    
    all_passed = all(hasattr(hybrid, m) for m in format_methods)
    print(f"\n{'✅ PASSED' if all_passed else '❌ FAILED'}: Hybrid retrieval test")
    return all_passed

def test_summary_agent():
    """Test 5: Verify summary agent uses database"""
    print("\n" + "=" * 80)
    print("TEST 5: Well Summary Agent - Mode B")
    print("=" * 80)
    
    db = WellDatabaseManager("./test_well_data.db")
    
    # Initialize summary agent
    try:
        llm = OllamaHelper()
        summary_agent = WellSummaryAgent(llm_helper=llm, database_manager=db)
        
        print("\nSummary agent initialization:")
        print(f"  ✓ Database manager connected")
        print(f"  {'✓' if summary_agent.llm_available else '✗'} LLM available")
        
        # Check methods
        has_generate = hasattr(summary_agent, 'generate_summary')
        has_db_report = hasattr(summary_agent, '_generate_summary_report_from_db')
        has_db_basic = hasattr(summary_agent, '_generate_basic_report_from_db')
        has_db_confidence = hasattr(summary_agent, '_calculate_confidence_from_db')
        
        print("\nRequired methods:")
        print(f"  {'✓' if has_generate else '✗'} generate_summary()")
        print(f"  {'✓' if has_db_report else '✗'} _generate_summary_report_from_db()")
        print(f"  {'✓' if has_db_basic else '✗'} _generate_basic_report_from_db()")
        print(f"  {'✓' if has_db_confidence else '✗'} _calculate_confidence_from_db()")
        
        all_passed = has_generate and has_db_report and has_db_basic and has_db_confidence
        
    except Exception as e:
        print(f"\n  ✗ Error initializing summary agent: {e}")
        all_passed = False
    
    db.close()
    
    print(f"\n{'✅ PASSED' if all_passed else '❌ FAILED'}: Summary agent test")
    return all_passed

def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("HYBRID RAG SYSTEM - TEST SUITE")
    print("=" * 80)
    
    tests = [
        test_database_schema,
        test_table_parser,
        test_ingestion_no_ocr,
        test_hybrid_retrieval,
        test_summary_agent
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for use.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the issues above.")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
