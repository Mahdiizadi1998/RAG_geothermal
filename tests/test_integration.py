"""Integration tests for the complete RAG pipeline."""

import pytest
import json
from pathlib import Path
from agents.ingestion_agent import IngestionAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.extraction_agent import ParameterExtractionAgent


class TestIntegration:
    """Integration tests for end-to-end pipeline."""
    
    @pytest.fixture
    def test_pdf_path(self):
        """Path to test PDF."""
        return "data/test_well.pdf"
    
    def test_complete_pipeline(self, test_pdf_path):
        """Test complete extraction pipeline."""
        # Skip if test PDF doesn't exist
        if not Path(test_pdf_path).exists():
            pytest.skip("Test PDF not found")
        
        # Step 1: Ingestion
        ingestion = IngestionAgent()
        doc_data = ingestion.process_document(test_pdf_path)
        
        assert doc_data is not None
        assert 'pages' in doc_data
        assert len(doc_data['pages']) > 0
        assert doc_data['metadata'].well_name == 'TEST-GT-01'
        
        # Step 2: Preprocessing
        preprocessing = PreprocessingAgent()
        chunks = preprocessing.process_document(doc_data['pages'], strategy='table-aware')
        
        assert len(chunks) > 0
        assert all(hasattr(chunk, 'text') for chunk in chunks)
        
        # Step 3: Extraction
        extraction = ParameterExtractionAgent()
        params = extraction.extract_from_chunks(chunks)
        
        # Verify extracted parameters
        assert len(params.trajectory) > 0
        assert len(params.casing) > 0
        assert params.reservoir_pressure is not None
        assert params.reservoir_temperature is not None
        
        # Verify specific values
        assert params.reservoir_pressure == 245.0
        assert params.reservoir_temperature == 155.0
        
        # Check trajectory data
        assert any(p.md == 0.0 and p.tvd == 0.0 for p in params.trajectory)
        assert any(p.md == 2000.0 for p in params.trajectory)
        
        # Check validation passed
        assert len(params.validation_results) > 0
        assert all(result.is_valid for result in params.validation_results)
    
    def test_unit_conversions_in_pipeline(self, test_pdf_path):
        """Test that unit conversions work correctly in pipeline."""
        if not Path(test_pdf_path).exists():
            pytest.skip("Test PDF not found")
        
        # Run extraction
        ingestion = IngestionAgent()
        doc_data = ingestion.process_document(test_pdf_path)
        
        preprocessing = PreprocessingAgent()
        chunks = preprocessing.process_document(doc_data['pages'])
        
        extraction = ParameterExtractionAgent()
        params = extraction.extract_from_chunks(chunks)
        
        # Check casing diameter conversions
        # 13 3/8" should be ~0.34m
        large_casing = [c for c in params.casing if c.diameter > 0.3]
        assert len(large_casing) > 0
        assert abs(large_casing[0].diameter - 0.33973) < 0.001
        
        # 9 5/8" should be ~0.24m
        small_casing = [c for c in params.casing if c.diameter < 0.3]
        assert len(small_casing) > 0
        assert abs(small_casing[0].diameter - 0.24448) < 0.001
    
    def test_validation_in_pipeline(self, test_pdf_path):
        """Test that validation works in pipeline."""
        if not Path(test_pdf_path).exists():
            pytest.skip("Test PDF not found")
        
        # Run extraction
        ingestion = IngestionAgent()
        doc_data = ingestion.process_document(test_pdf_path)
        
        preprocessing = PreprocessingAgent()
        chunks = preprocessing.process_document(doc_data['pages'])
        
        extraction = ParameterExtractionAgent()
        params = extraction.extract_from_chunks(chunks)
        
        # All trajectory points should satisfy MD >= TVD
        for point in params.trajectory:
            assert point.md >= point.tvd, f"MD ({point.md}) should be >= TVD ({point.tvd})"
        
        # Inclinations should be in valid range
        for point in params.trajectory:
            if point.inclination is not None:
                assert 0 <= point.inclination <= 90
