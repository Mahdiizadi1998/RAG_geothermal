"""Integration tests for the full RAG pipeline."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import (
    IngestionAgent,
    PreprocessingAgent,
    ParameterExtractionAgent,
    ValidationAgent
)
from utils import load_config

# Sample well report text simulating PDF content
SAMPLE_WELL_REPORT = """
--- Page 1 ---
End of Well Report
Well: TEST-GT-01
Location: Test Location

--- Page 8 ---
Tubular Design

13 3/8" surface casing to 1331m
9 5/8" production casing to 2500m
7" liner to 3000m

--- Page 19 ---
Directional Survey Data

MD (m)    TVD (m)    Inc (deg)
100.0     100.0      0.5
500.0     499.8      0.7
1000.0    999.2      1.0
1331.0    1329.5     1.5
2000.0    1997.0     2.0
2500.0    2495.5     2.5
3000.0    2994.0     3.0
"""

class TestFullPipeline:
    """Test the full extraction pipeline without actual PDF files."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = load_config()
    
    def test_extraction_from_text(self):
        """Test extraction from sample text data."""
        from utils.extraction_patterns import (
            extract_well_names,
            extract_trajectory_data,
            extract_casing_data
        )
        
        # Extract well names
        well_names = extract_well_names(SAMPLE_WELL_REPORT)
        assert len(well_names) > 0
        assert 'TEST-GT-01' in well_names
        
        # Extract trajectory
        trajectory = extract_trajectory_data(SAMPLE_WELL_REPORT)
        assert len(trajectory) >= 5
        assert trajectory[0]['md'] == 100.0
        assert trajectory[0]['tvd'] == 100.0
        
        # Extract casing
        casing = extract_casing_data(SAMPLE_WELL_REPORT)
        assert len(casing) >= 1
        
        # All casing should have valid pipe IDs
        for interval in casing:
            assert 'pipe_id' in interval
            assert interval['pipe_id'] > 0
    
    def test_merge_trajectory_casing(self):
        """Test merging trajectory and casing data."""
        from utils.extraction_patterns import (
            extract_trajectory_data,
            extract_casing_data
        )
        
        extraction_agent = ParameterExtractionAgent(self.config)
        
        trajectory = extract_trajectory_data(SAMPLE_WELL_REPORT)
        casing = extract_casing_data(SAMPLE_WELL_REPORT)
        
        # Merge
        merged = extraction_agent._merge_trajectory_casing(trajectory, casing)
        
        # Should have at least as many points as casing strings
        assert len(merged) >= len(casing)
        
        # Each merged point should have all required fields
        for point in merged:
            assert 'md' in point
            assert 'tvd' in point
            assert 'inc' in point
            assert 'pipe_id' in point
    
    def test_validation_pipeline(self):
        """Test validation of extracted data."""
        from utils.extraction_patterns import extract_trajectory_data
        
        trajectory = extract_trajectory_data(SAMPLE_WELL_REPORT)
        
        validation_agent = ValidationAgent(self.config)
        
        # Create extraction result
        extraction_result = {
            'trajectory': trajectory,
            'casing': [],
            'merged': [],
            'confidence': 0.8
        }
        
        # Validate
        report = validation_agent.validate_extraction(extraction_result)
        
        # Should be valid (trajectory exists and is valid)
        assert 'is_valid' in report
        assert 'errors' in report
        assert 'warnings' in report
        
        # Trajectory should be valid
        if trajectory:
            # Should have some warnings about missing casing, but trajectory itself is valid
            assert len(report['errors']) == 0  # No critical errors
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        extraction_agent = ParameterExtractionAgent(self.config)
        
        # Test with complete data
        result_complete = {
            'trajectory': [{'md': 100, 'tvd': 100, 'inc': 0.5, 'pipe_id': 0.3}],
            'casing': [{'md': 100, 'pipe_id': 0.3}],
            'merged': [{'md': 100, 'tvd': 100, 'inc': 0.5, 'pipe_id': 0.3}]
        }
        confidence = extraction_agent._calculate_confidence(result_complete)
        assert confidence >= 0.7  # Should have high confidence with complete data
        
        # Test with partial data
        result_partial = {
            'trajectory': [{'md': 100, 'tvd': 100, 'inc': 0.5}],
            'casing': [],
            'merged': []
        }
        confidence = extraction_agent._calculate_confidence(result_partial)
        assert confidence < 0.7  # Should have lower confidence with partial data

class TestChunking:
    """Test chunking strategies."""
    
    def test_multi_strategy_chunking(self):
        """Test that all chunking strategies are applied."""
        config = load_config()
        preprocessing_agent = PreprocessingAgent(config)
        
        # Create mock document
        mock_doc = {
            'content': SAMPLE_WELL_REPORT,
            'pages': 19,
            'wells': ['TEST-GT-01'],
            'metadata': {
                'filename': 'test.pdf',
                'source': '/tmp/test.pdf',
                'page_contents': {}
            }
        }
        
        # Chunk
        chunks_by_strategy = preprocessing_agent.chunk_documents([mock_doc])
        
        # Should have all three strategies
        assert 'factual_qa' in chunks_by_strategy
        assert 'technical_extraction' in chunks_by_strategy
        assert 'summary' in chunks_by_strategy
        
        # Each strategy should have chunks
        for strategy, chunks in chunks_by_strategy.items():
            assert len(chunks) > 0
            
            # Each chunk should have required fields
            for chunk in chunks:
                assert 'content' in chunk
                assert 'metadata' in chunk
                assert chunk['metadata']['strategy'] == strategy
