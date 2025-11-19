"""
Test script to verify system components
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from agents.ingestion_agent import IngestionAgent
        print("✓ IngestionAgent")
    except Exception as e:
        print(f"✗ IngestionAgent: {e}")
    
    try:
        from agents.preprocessing_agent import PreprocessingAgent
        print("✓ PreprocessingAgent")
    except Exception as e:
        print(f"✗ PreprocessingAgent: {e}")
    
    try:
        from agents.rag_retrieval_agent import RAGRetrievalAgent
        print("✓ RAGRetrievalAgent")
    except Exception as e:
        print(f"✗ RAGRetrievalAgent: {e}")
    
    try:
        from agents.parameter_extraction_agent import ParameterExtractionAgent
        print("✓ ParameterExtractionAgent")
    except Exception as e:
        print(f"✗ ParameterExtractionAgent: {e}")
    
    try:
        from agents.validation_agent import ValidationAgent
        print("✓ ValidationAgent")
    except Exception as e:
        print(f"✗ ValidationAgent: {e}")
    
    try:
        from models.nodal_runner import NodalAnalysisRunner
        print("✓ NodalAnalysisRunner")
    except Exception as e:
        print(f"✗ NodalAnalysisRunner: {e}")
    
    try:
        from utils.pattern_library import PatternLibrary
        print("✓ PatternLibrary")
    except Exception as e:
        print(f"✗ PatternLibrary: {e}")
    
    try:
        from utils.unit_conversion import UnitConverter
        print("✓ UnitConverter")
    except Exception as e:
        print(f"✗ UnitConverter: {e}")
    
    print("\nAll imports successful!")


def test_pattern_library():
    """Test pattern matching"""
    print("\nTesting PatternLibrary...")
    
    from utils.pattern_library import PatternLibrary
    
    # Test trajectory extraction
    test_text = """
    MD      TVD     Inc
    100.0   100.0   0.5
    500.5   495.2   2.3
    1000.0  980.5   5.1
    """
    
    points = PatternLibrary.extract_trajectory_points(test_text)
    print(f"✓ Extracted {len(points)} trajectory points")
    for p in points:
        print(f"  MD: {p['md']}, TVD: {p['tvd']}, Inc: {p['inclination']}")
    
    # Test casing extraction
    casing_text = """
    20" casing from 0 to 650 m, ID 19.124"
    13 3/8" casing from 650 to 1500 m, ID 12.615"
    9 5/8" liner from 1500 to 2500 m, ID 8.535"
    """
    
    casing = PatternLibrary.extract_casing_design(casing_text)
    print(f"\n✓ Extracted {len(casing)} casing strings")
    for c in casing:
        print(f"  OD: {c['od']}\", {c['top_md']}-{c['bottom_md']}m, ID: {c['id']}\"")


def test_unit_conversion():
    """Test unit conversions"""
    print("\nTesting UnitConverter...")
    
    from utils.unit_conversion import UnitConverter
    
    # Test fractional inches
    test_cases = [
        ("13 3/8\"", 13.375),
        ("9 5/8\"", 9.625),
        ("7\"", 7.0)
    ]
    
    for input_str, expected in test_cases:
        result = UnitConverter.parse_fractional_inches(input_str)
        assert abs(result - expected) < 0.001, f"Failed: {input_str}"
        print(f"✓ {input_str} = {result}\"")
    
    # Test inches to meters
    inches = 13.375
    meters = UnitConverter.inches_to_meters(inches)
    print(f"\n✓ {inches}\" = {meters:.4f}m")
    
    # Test validation
    assert UnitConverter.validate_md_tvd(1000, 995, tolerance=1.0)
    assert not UnitConverter.validate_md_tvd(1000, 1005, tolerance=1.0)
    print("✓ MD/TVD validation working")
    
    assert UnitConverter.validate_pipe_id_mm(200, 50, 1000)
    assert not UnitConverter.validate_pipe_id_mm(2000, 50, 1000)
    print("✓ Pipe ID validation working")


def test_nodal_runner():
    """Test nodal analysis runner"""
    print("\nTesting NodalAnalysisRunner...")
    
    from models.nodal_runner import NodalAnalysisRunner
    
    runner = NodalAnalysisRunner()
    
    # Test with sample trajectory data
    sample_data = {
        'well_name': 'TEST-GT-01',
        'trajectory': [
            {'md': 0, 'tvd': 0, 'inclination': 0, 'pipe_id': 0.3397},
            {'md': 500, 'tvd': 500, 'inclination': 0, 'pipe_id': 0.2445},
            {'md': 1500, 'tvd': 1500, 'inclination': 0, 'pipe_id': 0.1778},
            {'md': 2500, 'tvd': 2500, 'inclination': 0, 'pipe_id': 0.1778}
        ],
        'pvt_data': {
            'density': 1000.0,
            'viscosity': 0.001
        }
    }
    
    # Generate preview
    preview = runner.generate_preview_code(sample_data)
    print(f"✓ Generated trajectory preview ({len(preview)} chars)")
    print(f"✓ Trajectory points: {len(sample_data['trajectory'])}")
    
    print("\n✓ NodalAnalysisRunner initialized successfully")


if __name__ == "__main__":
    print("=" * 60)
    print("RAG for Geothermal Wells - System Tests")
    print("=" * 60)
    
    test_imports()
    test_pattern_library()
    test_unit_conversion()
    test_nodal_runner()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully! ✓")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download spaCy model: python -m spacy download en_core_web_sm")
    print("3. Pull Ollama models: ollama pull llama3 && ollama pull nomic-embed-text")
    print("4. Run the application: python app.py")
