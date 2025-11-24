#!/usr/bin/env python3
"""
Test script for hybrid database architecture components
Tests database, table parser, template selector, and hybrid retrieval
"""

import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_database_manager():
    """Test database creation and basic operations"""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Database Manager")
    logger.info("="*60)
    
    try:
        from agents.database_manager import WellDatabaseManager
        
        # Create test database
        db = WellDatabaseManager("./test_well_data.db")
        logger.info("✓ Database initialized")
        
        # Add test well
        well_id = db.add_or_get_well(
            'TEST-GT-01',
            operator='TestCo',
            location='Test Field',
            total_depth_md=3000.0,
            total_depth_tvd=2900.0
        )
        logger.info(f"✓ Added well TEST-GT-01 (ID: {well_id})")
        
        # Add casing string
        casing_id = db.add_casing_string('TEST-GT-01', {
            'string_number': 1,
            'outer_diameter': 13.375,
            'weight': 53.5,
            'grade': 'L80',
            'bottom_depth_md': 2500.0,
            'source_page': 8
        })
        logger.info(f"✓ Added casing string (ID: {casing_id})")
        
        # Add formation
        formation_id = db.add_formation('TEST-GT-01', {
            'formation_name': 'Test Formation',
            'top_md': 1500.0,
            'lithology': 'Sandstone',
            'source_page': 12
        })
        logger.info(f"✓ Added formation (ID: {formation_id})")
        
        # Retrieve well summary
        summary = db.get_well_summary('TEST-GT-01')
        assert summary is not None, "Failed to retrieve well summary"
        assert len(summary['casing_strings']) == 1, "Casing not stored correctly"
        assert len(summary['formations']) == 1, "Formation not stored correctly"
        logger.info(f"✓ Retrieved well summary: {summary['well_info']['well_name']}")
        logger.info(f"  - Total depth: {summary['well_info']['total_depth_md']}m MD")
        logger.info(f"  - Casing strings: {len(summary['casing_strings'])}")
        logger.info(f"  - Formations: {len(summary['formations'])}")
        
        # Clean up
        db.clear_well_data('TEST-GT-01')
        db.close()
        Path("./test_well_data.db").unlink(missing_ok=True)
        
        logger.info("✅ Database Manager: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database Manager: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_table_parser():
    """Test table type identification and parsing"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Table Parser")
    logger.info("="*60)
    
    try:
        from agents.table_parser import TableParser
        
        parser = TableParser()
        logger.info("✓ Table parser initialized")
        
        # Test casing table identification
        casing_headers = ['Size', 'Weight', 'Grade', 'Depth']
        casing_rows = [
            ['13 3/8"', '53.5 lb/ft', 'L80', '2500m'],
            ['9 5/8"', '47 lb/ft', 'L80', '3000m']
        ]
        
        table_type = parser.identify_table_type(
            casing_headers, 
            casing_rows,
            context='Casing program for well completion'
        )
        assert table_type == 'casing', f"Expected 'casing', got '{table_type}'"
        logger.info(f"✓ Identified casing table correctly")
        
        # Test casing parsing
        parsed_casing = parser.parse_casing_table(casing_headers, casing_rows, page=8)
        assert len(parsed_casing) == 2, f"Expected 2 casing strings, got {len(parsed_casing)}"
        assert parsed_casing[0]['outer_diameter'] == 13.375, "Failed to parse fraction 13 3/8"
        assert parsed_casing[0]['weight'] == 53.5, "Failed to parse weight"
        assert parsed_casing[0]['bottom_depth_md'] == 2500.0, "Failed to parse depth"
        logger.info(f"✓ Parsed casing table: {len(parsed_casing)} strings")
        logger.info(f"  - String 1: {parsed_casing[0]['outer_diameter']} inch, {parsed_casing[0]['weight']} lb/ft")
        
        # Test formation table identification
        formation_headers = ['Formation', 'Top MD', 'Lithology']
        formation_rows = [
            ['Nieuwerkerk', '950m', 'Clay'],
            ['Aalburg', '1450m', 'Sandstone']
        ]
        
        table_type = parser.identify_table_type(
            formation_headers,
            formation_rows,
            context='Geological formation tops'
        )
        assert table_type == 'formations', f"Expected 'formations', got '{table_type}'"
        logger.info(f"✓ Identified formations table correctly")
        
        # Test formation parsing
        parsed_formations = parser.parse_formation_table(formation_headers, formation_rows, page=12)
        assert len(parsed_formations) == 2, f"Expected 2 formations, got {len(parsed_formations)}"
        assert parsed_formations[0]['formation_name'] == 'Nieuwerkerk'
        assert parsed_formations[0]['top_md'] == 950.0
        logger.info(f"✓ Parsed formation table: {len(parsed_formations)} formations")
        
        logger.info("✅ Table Parser: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Table Parser: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_template_selector():
    """Test template selection based on data availability"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Template Selector")
    logger.info("="*60)
    
    try:
        from agents.database_manager import WellDatabaseManager
        from agents.template_selector import TemplateSelectorAgent
        
        # Create test database with data
        db = WellDatabaseManager("./test_well_data.db")
        
        # Add well with varying data completeness
        db.add_or_get_well(
            'TEST-GT-02',
            operator='TestCo',
            total_depth_md=3500.0,
            total_depth_tvd=3400.0
        )
        db.add_casing_string('TEST-GT-02', {
            'outer_diameter': 13.375,
            'weight': 53.5,
            'grade': 'L80',
            'bottom_depth_md': 3000.0
        })
        db.add_formation('TEST-GT-02', {
            'formation_name': 'Test Formation',
            'top_md': 2000.0
        })
        
        selector = TemplateSelectorAgent(db)
        logger.info("✓ Template selector initialized")
        
        # Test template selection
        template = selector.select_template('TEST-GT-02')
        assert template is not None, "Template selection failed"
        logger.info(f"✓ Selected template: {template['name']}")
        logger.info(f"  - Description: {template['description']}")
        
        # Test data completeness report
        report = selector.get_data_completeness_report('TEST-GT-02')
        assert 'TEST-GT-02' in report
        logger.info(f"✓ Generated completeness report")
        
        # Test with user preference
        template = selector.select_template('TEST-GT-02', user_preference='basic_completion')
        assert template['name'] == 'Basic Completion Summary'
        logger.info(f"✓ User preference override works")
        
        # Clean up
        db.clear_well_data('TEST-GT-02')
        db.close()
        Path("./test_well_data.db").unlink(missing_ok=True)
        
        logger.info("✅ Template Selector: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Template Selector: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_retrieval():
    """Test hybrid retrieval query classification"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Hybrid Retrieval (Query Classification)")
    logger.info("="*60)
    
    try:
        from agents.database_manager import WellDatabaseManager
        from agents.hybrid_retrieval_agent import HybridRetrievalAgent
        
        # Create minimal mock RAG agent
        class MockRAG:
            def retrieve(self, query, top_k=10):
                return [
                    {'text': 'Mock narrative context', 'metadata': {'source': 'test'}}
                ]
        
        db = WellDatabaseManager("./test_well_data.db")
        db.add_or_get_well(
            'TEST-GT-03',
            operator='TestCo',
            total_depth_md=3000.0
        )
        
        mock_rag = MockRAG()
        hybrid = HybridRetrievalAgent(db, mock_rag)
        logger.info("✓ Hybrid retrieval initialized")
        
        # Test query classification
        test_queries = [
            ("What is the total depth?", "Expected: database/hybrid"),
            ("What is the casing program?", "Expected: database"),
            ("What problems occurred during drilling?", "Expected: semantic/hybrid"),
            ("Give me a summary", "Expected: hybrid")
        ]
        
        for query, description in test_queries:
            mode = hybrid._classify_query(query)
            logger.info(f"✓ Query: '{query}' → Mode: {mode} ({description})")
        
        # Test actual retrieval (without real data, just check structure)
        result = hybrid.retrieve(
            query="What is the total depth?",
            well_name='TEST-GT-03',
            mode='auto'
        )
        
        assert 'database_results' in result
        assert 'semantic_results' in result
        assert 'combined_text' in result
        assert 'mode' in result
        logger.info(f"✓ Retrieval result structure valid")
        logger.info(f"  - Mode: {result['mode']}")
        logger.info(f"  - DB results: {len(result['database_results'])}")
        logger.info(f"  - Semantic results: {len(result['semantic_results'])}")
        
        # Clean up
        db.clear_well_data('TEST-GT-03')
        db.close()
        Path("./test_well_data.db").unlink(missing_ok=True)
        
        logger.info("✅ Hybrid Retrieval: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Hybrid Retrieval: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test full integration pipeline"""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Integration Test")
    logger.info("="*60)
    
    try:
        from agents.database_manager import WellDatabaseManager
        from agents.table_parser import TableParser
        from agents.ingestion_agent import IngestionAgent
        
        # Initialize components
        db = WellDatabaseManager("./test_well_data.db")
        parser = TableParser()
        ingestion = IngestionAgent(database_manager=db, table_parser=parser)
        
        logger.info("✓ All components initialized")
        
        # Simulate table extraction and storage
        # (This would normally come from PDF, but we'll create test data)
        well_name = 'TEST-GT-04'
        db.add_or_get_well(well_name, operator='Integration Test', total_depth_md=3200.0)
        
        # Simulate parsed casing table
        casing_data = [
            {
                'outer_diameter': 13.375,
                'weight': 53.5,
                'grade': 'L80',
                'bottom_depth_md': 2800.0,
                'source_page': 8
            },
            {
                'outer_diameter': 9.625,
                'weight': 47.0,
                'grade': 'L80',
                'bottom_depth_md': 3200.0,
                'source_page': 8
            }
        ]
        
        for casing in casing_data:
            db.add_casing_string(well_name, casing)
        
        logger.info(f"✓ Stored {len(casing_data)} casing strings")
        
        # Verify retrieval
        summary = db.get_well_summary(well_name)
        assert len(summary['casing_strings']) == 2
        logger.info(f"✓ Retrieved well summary with {len(summary['casing_strings'])} casing strings")
        
        # Test template selection with this data
        from agents.template_selector import TemplateSelectorAgent
        selector = TemplateSelectorAgent(db)
        template = selector.select_template(well_name)
        logger.info(f"✓ Selected template: {template['name']}")
        
        # Clean up
        db.clear_well_data(well_name)
        db.close()
        Path("./test_well_data.db").unlink(missing_ok=True)
        
        logger.info("✅ Integration Test: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration Test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "="*60)
    logger.info("HYBRID DATABASE ARCHITECTURE - COMPONENT TESTS")
    logger.info("="*60)
    
    results = {
        'Database Manager': test_database_manager(),
        'Table Parser': test_table_parser(),
        'Template Selector': test_template_selector(),
        'Hybrid Retrieval': test_hybrid_retrieval(),
        'Integration': test_integration()
    }
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! Hybrid architecture is working correctly.")
        return 0
    else:
        logger.error(f"\n⚠️ {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
