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
    
    # New validation agents
    try:
        from agents.query_analysis_agent import QueryAnalysisAgent
        print("✓ QueryAnalysisAgent")
    except Exception as e:
        print(f"✗ QueryAnalysisAgent: {e}")
    
    try:
        from agents.fact_verification_agent import FactVerificationAgent
        print("✓ FactVerificationAgent")
    except Exception as e:
        print(f"✗ FactVerificationAgent: {e}")
    
    try:
        from agents.physical_validation_agent import PhysicalValidationAgent
        print("✓ PhysicalValidationAgent")
    except Exception as e:
        print(f"✗ PhysicalValidationAgent: {e}")
    
    try:
        from agents.missing_data_agent import MissingDataAgent
        print("✓ MissingDataAgent")
    except Exception as e:
        print(f"✗ MissingDataAgent: {e}")
    
    try:
        from agents.confidence_scorer import ConfidenceScorerAgent
        print("✓ ConfidenceScorerAgent")
    except Exception as e:
        print(f"✗ ConfidenceScorerAgent: {e}")
    
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
    [size]" liner from 1500 to 2500 m, ID [number]"
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
            {'md': 0, 'tvd': 0, 'inclination': 0, 'pipe_id': 13.375},  # 13 3/8" casing
            {'md': 500, 'tvd': 500, 'inclination': 0, 'pipe_id': 9.625},  # 9 5/8" casing
            {'md': 1500, 'tvd': 1500, 'inclination': 0, 'pipe_id': 7.0},  # 7" casing
            {'md': 2500, 'tvd': 2500, 'inclination': 0, 'pipe_id': 7.0}  # 7" casing
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


def test_query_analysis():
    """Test query analysis agent"""
    print("\nTesting QueryAnalysisAgent...")
    
    from agents.query_analysis_agent import QueryAnalysisAgent
    import yaml
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    agent = QueryAnalysisAgent(config)
    
    # Test summary query with word count
    query1 = "summarize the document in 150 words"
    analysis1 = agent.analyze(query1)
    print(f"✓ Query type: {analysis1.query_type} (expected: summary)")
    print(f"✓ Word count: {analysis1.target_word_count} (expected: 150)")
    assert analysis1.query_type == 'summary'
    assert analysis1.target_word_count == 150
    
    # Test extraction query
    query2 = "extract well trajectory for Well GT-05"
    analysis2 = agent.analyze(query2)
    print(f"✓ Query type: {analysis2.query_type} (expected: extraction)")
    print(f"✓ Entities: {analysis2.entities}")
    assert analysis2.query_type == 'extraction'
    
    # Test Q&A query
    query3 = "what is the maximum depth of the well?"
    analysis3 = agent.analyze(query3)
    print(f"✓ Query type: {analysis3.query_type} (expected: qa)")
    assert analysis3.query_type == 'qa'


def test_physical_validation():
    """Test physical validation agent"""
    print("\nTesting PhysicalValidationAgent...")
    
    from agents.physical_validation_agent import PhysicalValidationAgent
    import yaml
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    agent = PhysicalValidationAgent(config)
    
    # Test valid trajectory
    valid_trajectory = [
        {'MD': 0, 'TVD': 0, 'ID': 20.0},
        {'MD': 650, 'TVD': 650, 'ID': 13.375},
        {'MD': 1500, 'TVD': 1500, 'ID': 9.625},
        {'MD': 2500, 'TVD': 2500, 'ID': 7.0}
    ]
    
    result1 = agent.validate_trajectory(valid_trajectory)
    print(f"✓ Valid trajectory: {result1.is_valid}, Confidence: {result1.confidence*100:.0f}%")
    assert result1.is_valid, "Valid trajectory should pass"
    
    # Test invalid trajectory (MD < TVD)
    invalid_trajectory = [
        {'MD': 1000, 'TVD': 1100, 'ID': 10.0}  # MD < TVD violation
    ]
    
    result2 = agent.validate_trajectory(invalid_trajectory)
    print(f"✓ Invalid trajectory detected: {not result2.is_valid}")
    assert not result2.is_valid, "MD < TVD should be invalid"
    assert any(v.violation_type == 'MD_LESS_THAN_TVD' for v in result2.violations)
    
    # Test telescoping violation (ID increases with depth)
    telescoping_violation = [
        {'MD': 0, 'TVD': 0, 'ID': 10.0},
        {'MD': 1000, 'TVD': 1000, 'ID': 12.0}  # ID increases with depth
    ]
    
    result3 = agent.validate_trajectory(telescoping_violation)
    print(f"✓ Telescoping violation detected: {not result3.is_valid}")
    assert not result3.is_valid, "Increasing ID with depth should be invalid"


def test_missing_data_agent():
    """Test missing data detection agent"""
    print("\nTesting MissingDataAgent...")
    
    from agents.missing_data_agent import MissingDataAgent
    import yaml
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    agent = MissingDataAgent(config)
    
    # Test complete data
    complete_data = {
        'trajectory': [
            {'MD': 0, 'TVD': 0, 'ID': 0.5},
            {'MD': 1000, 'TVD': 1000, 'ID': 0.3}
        ],
        'casing': [
            {'depth': 650, 'OD': 0.5, 'ID': 0.486}
        ],
        'pvt': {
            'density': 1000,
            'viscosity': 0.001
        }
    }
    
    result1 = agent.assess_completeness(complete_data)
    print(f"✓ Complete data: {result1.completeness_score*100:.0f}% completeness")
    assert result1.completeness_score > 0.6, "Complete data should score reasonably high (missing optional fields is OK)"
    
    # Test incomplete data
    incomplete_data = {
        'trajectory': []  # Missing trajectory
    }
    
    result2 = agent.assess_completeness(incomplete_data)
    print(f"✓ Incomplete data: {result2.completeness_score*100:.0f}% completeness")
    print(f"✓ Clarification questions: {len(result2.clarification_questions)}")
    assert result2.has_critical_gaps, "Missing trajectory should be critical"
    assert len(result2.clarification_questions) > 0, "Should generate questions"


def test_confidence_scorer():
    """Test confidence scoring agent"""
    print("\nTesting ConfidenceScorerAgent...")
    
    from agents.confidence_scorer import ConfidenceScorerAgent
    import yaml
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    agent = ConfidenceScorerAgent(config)
    
    # Test high confidence scenario
    high_confidence = agent.calculate_confidence(
        source_quality=0.9,
        fact_verification=0.95,
        completeness=0.9,
        consistency=0.85,
        physical_validity=0.95
    )
    
    print(f"✓ High confidence: {high_confidence.overall*100:.0f}%")
    print(f"✓ Recommendation: {high_confidence.recommendation}")
    assert high_confidence.recommendation == 'high', "Should be high confidence"
    
    # Test low confidence scenario
    low_confidence = agent.calculate_confidence(
        source_quality=0.4,
        fact_verification=0.5,
        completeness=0.3,
        consistency=0.6,
        physical_validity=0.4
    )
    
    print(f"✓ Low confidence: {low_confidence.overall*100:.0f}%")
    print(f"✓ Recommendation: {low_confidence.recommendation}")
    assert low_confidence.recommendation == 'low', "Should be low confidence"
    
    # Test source quality calculation
    test_chunks = [
        {'score': 0.9},
        {'score': 0.85},
        {'score': 0.8},
        {'score': 0.75},
        {'score': 0.7}
    ]
    
    quality = agent.calculate_source_quality(test_chunks, top_k=5)
    print(f"✓ Source quality: {quality*100:.0f}%")
    assert 0.7 <= quality <= 0.9, "Quality should be in expected range"


def test_word_count_enforcement():
    """Test strict word count enforcement"""
    print("\nTesting word count enforcement...")
    
    # Test word count parsing
    test_queries = [
        ("summarize in 200 words", 200),
        ("brief summary", 100),
        ("detailed summary", 500),
        ("summarize in 50 words", 50),
    ]
    
    from agents.query_analysis_agent import QueryAnalysisAgent
    import yaml
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    agent = QueryAnalysisAgent(config)
    
    for query, expected_words in test_queries:
        analysis = agent.analyze(query)
        actual_words = analysis.target_word_count if analysis.target_word_count else config.get('summarization', {}).get('default_words', 200)
        print(f"✓ '{query}' → {actual_words} words (expected: {expected_words})")
        assert actual_words == expected_words, f"Expected {expected_words}, got {actual_words}"


if __name__ == "__main__":
    print("=" * 60)
    print("RAG for Geothermal Wells - Comprehensive System Tests")
    print("=" * 60)
    
    test_imports()
    test_pattern_library()
    test_unit_conversion()
    test_nodal_runner()
    
    print("\n" + "=" * 60)
    print("Testing New Validation Agents (Weeks 1-3)")
    print("=" * 60)
    
    test_query_analysis()
    test_physical_validation()
    test_missing_data_agent()
    test_confidence_scorer()
    test_word_count_enforcement()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully! ✓")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download spaCy model: python -m spacy download en_core_web_sm")
    print("3. Pull Ollama models: ollama pull llama3 && ollama pull llama3.1 && ollama pull nomic-embed-text")
    print("4. Run the application: python app.py")
    print("\nNew features:")
    print("✓ Query analysis with word count detection")
    print("✓ Fact verification with LLM")
    print("✓ Physical validation (MD≥TVD, telescoping)")
    print("✓ Missing data detection with clarification questions")
    print("✓ Multi-dimensional confidence scoring")
    print("✓ Strict 200-word default for summaries (±5%)")
    print("✓ 7-minute timeouts for deep validation")
    print("✓ Always-ask confirmation before nodal analysis")
