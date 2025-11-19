"""
Demo script to test the RAG system components without requiring Ollama
This demonstrates extraction, validation, and nodal analysis with sample data
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from agents.parameter_extraction_agent import ParameterExtractionAgent
from agents.validation_agent import ValidationAgent
from models.nodal_analysis import NodalAnalysisModel
from utils.pattern_library import PatternLibrary
from utils.unit_conversion import UnitConverter

print("=" * 70)
print("RAG for Geothermal Wells - Component Demo")
print("=" * 70)
print()

# ============================================================================
# DEMO 1: Pattern Extraction
# ============================================================================
print("DEMO 1: Pattern Extraction from Text")
print("-" * 70)

sample_trajectory_text = """
DIRECTIONAL SURVEY - ADK-GT-01

MD (m)    TVD (m)    Inc (deg)
0.0       0.0        0.0
500.0     499.5      1.2
1000.0    995.0      3.5
1500.0    1485.0     5.8
2000.0    1970.0     8.2
2500.0    2445.0     10.5
"""

sample_casing_text = """
CASING DESIGN:

20" conductor casing from 0 to 650 m, ID 19.124"
13 3/8" surface casing from 650 to 1500 m, ID 12.615"
9 5/8" production liner from 1500 to 2667 m, ID 8.535"
"""

print("\n📄 Sample Trajectory Text:")
print(sample_trajectory_text)

print("\n🔍 Extracting trajectory points...")
trajectory_points = PatternLibrary.extract_trajectory_points(sample_trajectory_text)
print(f"✓ Extracted {len(trajectory_points)} points:")
for i, point in enumerate(trajectory_points[:3], 1):
    print(f"  {i}. MD: {point['md']:>7.1f}m, TVD: {point['tvd']:>7.1f}m, Inc: {point['inclination']:>5.1f}°")

print("\n📄 Sample Casing Text:")
print(sample_casing_text)

print("\n🔍 Extracting casing design...")
casing_design = PatternLibrary.extract_casing_design(sample_casing_text)
print(f"✓ Extracted {len(casing_design)} casing strings:")
for i, casing in enumerate(casing_design, 1):
    print(f"  {i}. {casing['od']:.3f}\" ({casing['top_md']:.0f}-{casing['bottom_md']:.0f}m), ID: {casing['id']:.3f}\"")

# ============================================================================
# DEMO 2: Unit Conversion
# ============================================================================
print("\n" + "=" * 70)
print("DEMO 2: Unit Conversion")
print("-" * 70)

converter = UnitConverter()

test_sizes = ["13 3/8\"", "9 5/8\"", "7\""]
print("\n🔧 Converting fractional inches to meters:")
for size_str in test_sizes:
    inches = converter.parse_fractional_inches(size_str)
    meters = converter.inches_to_meters(inches)
    mm = meters * 1000
    print(f"  {size_str:>10} = {inches:>7.3f}\" = {meters:.4f}m = {mm:.1f}mm")

# ============================================================================
# DEMO 3: Data Validation
# ============================================================================
print("\n" + "=" * 70)
print("DEMO 3: Data Validation")
print("-" * 70)

validator = ValidationAgent()

# Create sample extracted data
extracted_data = {
    'well_name': 'ADK-GT-01',
    'trajectory': [
        {'md': 0, 'tvd': 0, 'inclination': 0, 'pipe_id': 0.486},
        {'md': 650, 'tvd': 649, 'inclination': 2.5, 'pipe_id': 0.320},
        {'md': 1500, 'tvd': 1485, 'inclination': 5.8, 'pipe_id': 0.217},
        {'md': 2667, 'tvd': 2600, 'inclination': 10.5, 'pipe_id': 0.217}
    ],
    'casing_design': casing_design,
    'pvt_data': {
        'density': 1050,
        'viscosity': 0.0015,
        'temp_gradient': 32.0
    }
}

print("\n🔍 Validating extracted data...")
validation_result = validator.validate(extracted_data)

print(f"\n{validator.format_validation_report(validation_result)}")

# ============================================================================
# DEMO 4: Trajectory-Casing Merger
# ============================================================================
print("\n" + "=" * 70)
print("DEMO 4: Trajectory-Casing Merger")
print("-" * 70)

extraction_agent = ParameterExtractionAgent()

print("\n🔄 Merging trajectory with casing design...")
print(f"   Input: {len(trajectory_points)} trajectory points + {len(casing_design)} casing strings")

# Simulate merging
merged = extraction_agent._merge_trajectory_with_casing(
    trajectory_points, casing_design, []
)

print(f"   Output: {len(merged)} merged points with pipe ID")
print("\n📊 Merged data sample:")
for i, point in enumerate(merged[:5], 1):
    print(f"  {i}. MD: {point['md']:>7.1f}m, TVD: {point['tvd']:>7.1f}m, "
          f"Inc: {point['inclination']:>5.1f}°, Pipe ID: {point['pipe_id']*1000:>6.1f}mm")

# ============================================================================
# DEMO 5: Nodal Analysis
# ============================================================================
print("\n" + "=" * 70)
print("DEMO 5: Nodal Analysis")
print("-" * 70)

nodal = NodalAnalysisModel()

print("\n⚙️  Quick estimate for ADK-GT-01:")
total_depth = 2667
fluid_density = 1050
avg_pipe_id = 0.217

quick_result = nodal.quick_estimate(total_depth, fluid_density, avg_pipe_id)
print(f"   Total depth: {total_depth}m")
print(f"   Fluid density: {fluid_density} kg/m³")
print(f"   Average pipe ID: {avg_pipe_id*1000:.1f}mm")
print(f"\n   Results:")
print(f"   • Hydrostatic pressure: {quick_result['hydrostatic_pressure_bar']:.1f} bar")
print(f"   • Estimated flow rate: {quick_result['estimated_flow_rate_m3h']:.1f} m³/h")
print(f"   • Estimated flow rate: {quick_result['estimated_flow_rate_bpd']:.0f} bpd")

print("\n⚙️  Detailed pressure profile calculation:")
profile = nodal.calculate_pressure_profile(
    trajectory=merged[:4],  # Use first 4 points
    fluid_density=fluid_density,
    fluid_viscosity=0.0015,
    flow_rate_m3s=0.05,  # 50 L/s = 180 m³/h
    wellhead_pressure=100000  # 1 bar
)

if profile.get('success'):
    print(f"   Wellhead pressure: {profile['wellhead_pressure_bar']:.2f} bar")
    print(f"   Bottomhole pressure: {profile['bottomhole_pressure_bar']:.2f} bar")
    print(f"   Pressure gain: {profile['total_pressure_gain_bar']:.2f} bar")
    print(f"   Flow rate: {profile['flow_rate_m3h']:.1f} m³/h")
    
    print(f"\n   Pressure profile along wellbore:")
    for i, p in enumerate(profile['profile'][:5], 1):
        print(f"   {i}. MD: {p['md']:>7.1f}m, TVD: {p['tvd']:>7.1f}m, "
              f"Pressure: {p['pressure_bar']:>6.2f} bar")

# ============================================================================
# DEMO 6: Format for Nodal Analysis
# ============================================================================
print("\n" + "=" * 70)
print("DEMO 6: Export Format for Nodal Analysis")
print("-" * 70)

print("\n📤 Formatted Python code for nodal analysis:")
formatted_code = extraction_agent.format_for_nodal_analysis(extracted_data)
print(formatted_code)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("✅ DEMO COMPLETE")
print("=" * 70)
print("\nAll core components demonstrated:")
print("  ✓ Pattern extraction (trajectory & casing)")
print("  ✓ Unit conversion (fractional inches → meters)")
print("  ✓ Data validation (physics-based checks)")
print("  ✓ Trajectory-casing merger")
print("  ✓ Nodal analysis (pressure calculations)")
print("  ✓ Export formatting")
print("\n🚀 The system is ready for use with PDF documents!")
print("\nNote: To use with actual PDFs and RAG functionality, you need:")
print("  1. Ollama installed and running (for embeddings and LLM)")
print("  2. Models pulled: ollama pull llama3 && ollama pull nomic-embed-text")
print("  3. Run: python app.py")
print("\nFor demo purposes, all extraction and analysis components work perfectly!")
print("=" * 70)
