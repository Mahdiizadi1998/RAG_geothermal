"""
Test the new LLM-based summary extraction system
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.llm_helper import OllamaHelper
from agents.database_manager import WellDatabaseManager
import yaml

def test_extraction():
    """Test the new extract_information method"""
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize LLM
    llm = OllamaHelper('config/config.yaml')
    
    if not llm.is_available():
        print("❌ LLM not available - please start Ollama")
        return
    
    print("✅ LLM is available")
    
    # Test context (sample casing table data)
    test_context = """
Table from page 15 (Casing type):
  Type: Surface Casing | OD: 13⅜" | Weight: 68 lb/ft | Grade: K-55 | ID Nominal: 12.615" | ID Drift: 12.515" | Top: 0 m | Bottom: 450 m
  Type: Intermediate Casing | OD: 9⅝" | Weight: 47 lb/ft | Grade: L-80 | ID Nominal: 8.835" | ID Drift: 8.735" | Top: 0 m | Bottom: 2,150 m
  Type: Production Casing | OD: 7" | Weight: 29 lb/ft | Grade: P-110 | ID Nominal: 6.184" | ID Drift: 6.094" | Top: 0 m | Bottom: 3,456 m

Table from page 22:
  Casing String: Conductor | Size: 20" | Depth: 50 m
"""
    
    # Test extraction prompt
    extraction_prompt = """Extract casing and tubular information. For EACH casing string, extract:
- Type (Conductor, Surface, Intermediate, Production, Liner, etc.)
- OD (Outside Diameter in inches)
- Weight (lb/ft)
- Grade (e.g., K-55, L-80, P-110)
- Connection type
- **Pipe ID - Both Nominal ID AND Drift ID** (very important)
- Top Depth (mAH)
- Bottom Depth (mAH)

Format as a numbered list, one entry per casing string. Include Nominal and Drift ID for each string."""
    
    print("\n📊 Testing information extraction...")
    print("\n" + "="*60)
    
    result = llm.extract_information(extraction_prompt, test_context)
    
    print("Extracted Information:")
    print(result)
    print("="*60)
    
    # Check if it includes critical information
    if "Nominal" in result and "Drift" in result:
        print("\n✅ SUCCESS: Both Nominal ID and Drift ID extracted!")
    else:
        print("\n⚠️ WARNING: May be missing ID information")
    
    if "Surface" in result and "Production" in result:
        print("✅ Multiple casing strings identified")
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    test_extraction()
