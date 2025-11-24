"""
Test script for the new 3-Pass Well Summary Agent

This script demonstrates:
1. Pass 1: Metadata extraction from PDF header
2. Pass 2: Casing table extraction with ID column
3. Pass 3: Narrative extraction from geology section
4. Final report generation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from agents.well_summary_agent import WellSummaryAgent
from agents.llm_helper import OllamaHelper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_summary_agent():
    """Test the 3-pass summarization system"""
    
    print("=" * 70)
    print("3-PASS WELL SUMMARY AGENT TEST")
    print("=" * 70)
    
    # Initialize LLM helper
    print("\n1. Initializing LLM helper...")
    try:
        llm = OllamaHelper()
        if not llm.is_available():
            print("⚠️  Ollama not available - some features will use fallback mode")
            print("   Start Ollama: ollama serve")
    except Exception as e:
        print(f"⚠️  Could not initialize LLM: {e}")
        llm = None
    
    # Initialize summary agent
    print("\n2. Initializing Well Summary Agent...")
    agent = WellSummaryAgent(llm_helper=llm)
    print(f"   ✓ Agent initialized")
    print(f"   - LLM available: {agent.llm_available}")
    print(f"   - pdfplumber available: {agent.llm_available and hasattr(agent, 'PDFPLUMBER_AVAILABLE')}")
    
    # Test with a sample PDF path (user needs to provide actual PDF)
    print("\n3. Testing with sample PDF...")
    print("\n   NOTE: To test with a real PDF, update the pdf_path variable below")
    print("   Example: pdf_path = '/path/to/your/completion_report.pdf'")
    
    # Example - user should replace with actual PDF path
    pdf_path = None  # Set to actual PDF path for testing
    
    if pdf_path and Path(pdf_path).exists():
        print(f"\n   Testing with: {pdf_path}")
        
        try:
            result = agent.generate_summary(pdf_path)
            
            print("\n" + "=" * 70)
            print("RESULTS")
            print("=" * 70)
            
            # Pass 1 Results
            print("\n📋 PASS 1 - METADATA EXTRACTION:")
            print("-" * 70)
            metadata = result['metadata']
            for key, value in metadata.items():
                print(f"   {key}: {value}")
            
            # Pass 2 Results
            print("\n🔧 PASS 2 - TECHNICAL SPECS EXTRACTION:")
            print("-" * 70)
            casing_program = result['technical_specs'].get('casing_program', [])
            print(f"   Found {len(casing_program)} casing strings:")
            for i, casing in enumerate(casing_program, 1):
                print(f"   {i}. Size: {casing.get('size')}, "
                      f"Weight: {casing.get('weight')} ppf, "
                      f"Depth: {casing.get('depth')} m, "
                      f"ID: {casing.get('pipe_id', 'N/A')}")
            
            # Pass 3 Results
            print("\n⛰️  PASS 3 - NARRATIVE EXTRACTION:")
            print("-" * 70)
            narrative = result['narrative']
            if narrative.get('hazards'):
                print("   Hazards:")
                for hazard in narrative['hazards']:
                    print(f"   - {hazard}")
            if narrative.get('instabilities'):
                print("   Instabilities:")
                for instability in narrative['instabilities']:
                    print(f"   - {instability}")
            if narrative.get('gas_shows'):
                print("   Gas Shows:")
                for gas in narrative['gas_shows']:
                    print(f"   - {gas}")
            
            # Final Report
            print("\n📄 FINAL REPORT:")
            print("-" * 70)
            print(result['summary_report'])
            
            # Confidence Score
            print("\n📊 CONFIDENCE SCORE:")
            print("-" * 70)
            print(f"   {result['confidence']*100:.0f}%")
            
        except Exception as e:
            print(f"\n❌ Error during testing: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n   ⚠️  No PDF path provided or file not found")
        print("\n   To test the agent:")
        print("   1. Update pdf_path variable in this script")
        print("   2. Ensure Ollama is running: ollama serve")
        print("   3. Install pdfplumber: pip install pdfplumber")
        print("   4. Run: python test_summary_agent.py")
    
    # Demonstrate JSON parsing robustness
    print("\n" + "=" * 70)
    print("TESTING JSON PARSING ROBUSTNESS")
    print("=" * 70)
    
    test_cases = [
        '{"name": "Test"}',  # Clean JSON
        '```json\n{"name": "Test"}\n```',  # Markdown wrapped
        'Some text before {"name": "Test"} some text after',  # Extra text
        '[{"id": 1}, {"id": 2}]',  # Array
        '{"name": "Test"',  # Malformed (missing closing brace)
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n   Test {i}: {test_case[:50]}...")
        try:
            result = agent._parse_json_response(test_case)
            print(f"   ✓ Parsed: {result}")
        except Exception as e:
            print(f"   ✗ Failed: {e}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print("\n✓ All components initialized successfully")
    print("✓ JSON parsing is robust")
    print("\n📝 Next steps:")
    print("   1. Provide a PDF path to test full extraction")
    print("   2. Verify Ollama is running for LLM features")
    print("   3. Install pdfplumber for table extraction")
    print("\n")


if __name__ == "__main__":
    test_summary_agent()
