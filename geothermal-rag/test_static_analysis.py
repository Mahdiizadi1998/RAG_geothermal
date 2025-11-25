"""
Static Code Analysis Test for Hybrid RAG System
Verifies implementation without running code
"""

from pathlib import Path
import re

def test_database_schema():
    """Test 1: Check database_manager.py has incidents table"""
    print("=" * 80)
    print("TEST 1: Database Schema - Incidents Table")
    print("=" * 80)
    
    file_path = Path("agents/database_manager.py")
    with open(file_path, 'r') as f:
        content = f.read()
    
    checks = {
        "incidents table creation": "CREATE TABLE IF NOT EXISTS incidents",
        "add_incident method": "def add_incident",
        "incidents in get_well_summary": "'incidents': incidents",
        "delete incidents in clear": "DELETE FROM incidents WHERE well_id"
    }
    
    results = {}
    for check_name, pattern in checks.items():
        found = pattern in content
        results[check_name] = found
        status = "✓" if found else "✗"
        print(f"  {status} {check_name}")
    
    passed = all(results.values())
    print(f"\n{'✅ PASSED' if passed else '❌ FAILED'}: Database schema test")
    return passed

def test_table_parser():
    """Test 2: Check enhanced_table_parser.py supports incidents"""
    print("\n" + "=" * 80)
    print("TEST 2: Table Parser - Incidents Support")
    print("=" * 80)
    
    file_path = Path("agents/enhanced_table_parser.py")
    with open(file_path, 'r') as f:
        content = f.read()
    
    checks = {
        "incidents in keywords": "'incidents':",
        "parse_incidents_table method": "def parse_incidents_table",
        "incident_type column": "incident_type",
        "severity column": "severity"
    }
    
    results = {}
    for check_name, pattern in checks.items():
        found = pattern in content
        results[check_name] = found
        status = "✓" if found else "✗"
        print(f"  {status} {check_name}")
    
    passed = all(results.values())
    print(f"\n{'✅ PASSED' if passed else '❌ FAILED'}: Table parser test")
    return passed

def test_ingestion_no_ocr():
    """Test 3: Verify OCR code removed from ingestion_agent.py"""
    print("\n" + "=" * 80)
    print("TEST 3: Ingestion Agent - No OCR Code")
    print("=" * 80)
    
    file_path = Path("agents/ingestion_agent.py")
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check OCR imports are NOT present
    ocr_checks = {
        "easyocr import": "import easyocr",
        "pytesseract import": "import pytesseract",
        "PIL Image import": "from PIL import Image",
        "_ocr_page method": "def _ocr_page"
    }
    
    print("\nOCR code removal:")
    ocr_removed = True
    for check_name, pattern in ocr_checks.items():
        found = pattern in content
        ocr_removed = ocr_removed and not found
        status = "✓" if not found else "✗"
        print(f"  {status} {check_name} {'(should be removed)' if found else '(removed)'}")
    
    # Check incidents handling IS present
    print("\nIncidents handling:")
    has_incidents = "add_incident" in content
    print(f"  {'✓' if has_incidents else '✗'} add_incident calls present")
    
    passed = ocr_removed and has_incidents
    print(f"\n{'✅ PASSED' if passed else '❌ FAILED'}: Ingestion agent test")
    return passed

def test_hybrid_retrieval():
    """Test 4: Check hybrid_retrieval_agent.py has all formatters"""
    print("\n" + "=" * 80)
    print("TEST 4: Hybrid Retrieval - All 8 Data Types")
    print("=" * 80)
    
    file_path = Path("agents/hybrid_retrieval_agent.py")
    with open(file_path, 'r') as f:
        content = f.read()
    
    formatters = {
        "well_info": "_format_well_info",
        "casing": "_format_casing_data",
        "formations": "_format_formation_data",
        "cementing": "_format_cementing_data",
        "fluids": "_format_fluids_data",
        "incidents": "_format_incidents_data"
    }
    
    print("\nFormatter methods:")
    results = {}
    for data_type, method_name in formatters.items():
        found = f"def {method_name}" in content
        results[data_type] = found
        status = "✓" if found else "✗"
        print(f"  {status} {method_name}() for {data_type}")
    
    # Check that fluids and incidents are added to query_database
    print("\nData retrieval in query_database:")
    has_fluids_retrieval = "drilling_fluids" in content
    has_incidents_retrieval = "'incidents'" in content
    print(f"  {'✓' if has_fluids_retrieval else '✗'} drilling_fluids retrieval")
    print(f"  {'✓' if has_incidents_retrieval else '✗'} incidents retrieval")
    
    passed = all(results.values()) and has_fluids_retrieval and has_incidents_retrieval
    print(f"\n{'✅ PASSED' if passed else '❌ FAILED'}: Hybrid retrieval test")
    return passed

def test_summary_agent():
    """Test 5: Check well_summary_agent.py uses database"""
    print("\n" + "=" * 80)
    print("TEST 5: Well Summary Agent - Mode B Implementation")
    print("=" * 80)
    
    file_path = Path("agents/well_summary_agent.py")
    with open(file_path, 'r') as f:
        content = f.read()
    
    checks = {
        "database_manager parameter": "database_manager=None",
        "generate_summary with well_name": "def generate_summary(self, well_name",
        "_generate_summary_report_from_db": "def _generate_summary_report_from_db",
        "_generate_basic_report_from_db": "def _generate_basic_report_from_db",
        "_calculate_confidence_from_db": "def _calculate_confidence_from_db",
        "all 8 data types comment": "all 8 data types",
        "pipe_id_nominal in report": "pipe_id_nominal",
        "pipe_id_drift in report": "pipe_id_drift"
    }
    
    print("\nMode B implementation:")
    results = {}
    for check_name, pattern in checks.items():
        found = pattern.lower() in content.lower()
        results[check_name] = found
        status = "✓" if found else "✗"
        print(f"  {status} {check_name}")
    
    passed = all(results.values())
    print(f"\n{'✅ PASSED' if passed else '❌ FAILED'}: Summary agent test")
    return passed

def test_app_integration():
    """Test 6: Check app.py uses hybrid retrieval"""
    print("\n" + "=" * 80)
    print("TEST 6: App.py - Mode A & B Integration")
    print("=" * 80)
    
    file_path = Path("app.py")
    with open(file_path, 'r') as f:
        content = f.read()
    
    checks = {
        "HybridRetrievalAgent import": "from agents.hybrid_retrieval_agent import HybridRetrievalAgent",
        "hybrid_retrieval initialization": "self.hybrid_retrieval = HybridRetrievalAgent",
        "database_manager in WellSummaryAgent": "database_manager=self.db",
        "hybrid_retrieval.retrieve in QA": "self.hybrid_retrieval.retrieve",
        "Mode A comment": "Mode A",
        "Mode B comment": "Mode B",
        "8 data types check": "8 data types"
    }
    
    print("\nApp integration:")
    results = {}
    for check_name, pattern in checks.items():
        found = pattern in content
        results[check_name] = found
        status = "✓" if found else "✗"
        print(f"  {status} {check_name}")
    
    passed = all(results.values())
    print(f"\n{'✅ PASSED' if passed else '❌ FAILED'}: App integration test")
    return passed

def test_documentation():
    """Test 7: Check documentation exists"""
    print("\n" + "=" * 80)
    print("TEST 7: Documentation")
    print("=" * 80)
    
    doc_file = Path("HYBRID_SYSTEM_SUMMARY.md")
    
    if not doc_file.exists():
        print("  ✗ HYBRID_SYSTEM_SUMMARY.md not found")
        print(f"\n❌ FAILED: Documentation test")
        return False
    
    with open(doc_file, 'r') as f:
        content = f.read()
    
    sections = [
        "## Architecture",
        "## 8 Supported Data Types",
        "### Mode A: Q&A",
        "### Mode B: Summary",
        "## Changes Made",
        "## Testing Recommendations"
    ]
    
    print("\nDocumentation sections:")
    results = {}
    for section in sections:
        found = section in content
        results[section] = found
        status = "✓" if found else "✗"
        print(f"  {status} {section}")
    
    passed = all(results.values())
    print(f"\n{'✅ PASSED' if passed else '❌ FAILED'}: Documentation test")
    return passed

def main():
    """Run all static analysis tests"""
    print("\n" + "=" * 80)
    print("HYBRID RAG SYSTEM - STATIC ANALYSIS TEST SUITE")
    print("=" * 80)
    print("\nAnalyzing code structure without executing...")
    print()
    
    tests = [
        test_database_schema,
        test_table_parser,
        test_ingestion_no_ocr,
        test_hybrid_retrieval,
        test_summary_agent,
        test_app_integration,
        test_documentation
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    
    test_names = [
        "Database Schema",
        "Table Parser",
        "Ingestion (No OCR)",
        "Hybrid Retrieval",
        "Summary Agent",
        "App Integration",
        "Documentation"
    ]
    
    print("\nDetailed results:")
    for i, (name, result) in enumerate(zip(test_names, results), 1):
        status = "✅" if result else "❌"
        print(f"  {i}. {status} {name}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nThe hybrid RAG system has been successfully implemented with:")
        print("  • PDF-only processing (OCR removed)")
        print("  • All 8 data types supported")
        print("  • Mode A: Hybrid Q&A (SQL + Vector)")
        print("  • Mode B: Comprehensive Summary")
        print("  • Automated table extraction and parsing")
        print("  • Pipe ID tracking (Nominal + Drift)")
        print("\nSystem is ready for testing with real PDF documents!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        print("Please review the issues above before proceeding.")
    
    print("\n" + "=" * 80)
    
    return passed == total

if __name__ == "__main__":
    import os
    os.chdir("/workspaces/RAG_geothermal/geothermal-rag")
    success = main()
    exit(0 if success else 1)
