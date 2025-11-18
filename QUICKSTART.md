# Quick Start Guide

This guide will help you get started with the RAG Geothermal Wells system in 5 minutes.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

```bash
# Clone the repository
git clone https://github.com/SinaMad98/RAG_geothermal.git
cd RAG_geothermal

# Install dependencies
pip install -r requirements.txt
```

## Quick Test

Run the example script to see the system in action:

```bash
python example.py
```

This will demonstrate:
- Well name extraction
- Trajectory data extraction
- Casing design extraction
- Data validation
- Trajectory-casing merging
- Unit conversions

Expected output:
```
======================================================================
RAG Geothermal Wells System - Example Usage
======================================================================

1. Extracting Well Names:
   Found: ['ADK-GT-01']

2. Extracting Trajectory Data:
   Extracted 8 trajectory points:
   [1] MD: 100.0m, TVD: 100.0m, Inc: 0.5°
   ...

3. Extracting Casing Design:
   Extracted 1 casing strings:
   [1] MD: 1331.0m, Pipe ID: 0.3227m (322.7mm)

5. Merging Trajectory + Casing:
       MD (m)    TVD (m)    Inc (°)  Pipe ID (m)
   ----------------------------------------------
      1331.00    1328.00       1.80       0.3227
      2750.00    2735.00       4.20       0.3227
```

## Processing PDF Files

### Command Line

```bash
# Process a single PDF
python main.py data/your_well_report.pdf

# Process multiple PDFs
python main.py data/*.pdf
```

### Python API

```python
from main import RAGGeothermalSystem

# Initialize system
system = RAGGeothermalSystem()

# Process PDF documents
results = system.process_documents(['well_report.pdf'])

# Extract parameters for a specific well
well_data = system.extract_well_parameters('ADK-GT-01')

# Access extracted data
trajectory = well_data['extraction']['trajectory']
casing = well_data['extraction']['casing']
merged = well_data['extraction']['merged']

# Print results
for point in merged:
    print(f"MD: {point['md']:.1f}m, TVD: {point['tvd']:.1f}m, "
          f"Inc: {point['inc']:.1f}°, Pipe ID: {point['pipe_id']:.4f}m")
```

## Understanding the Output

The system extracts three types of data:

### 1. Trajectory Points
- **MD**: Measured Depth (meters) - length along the wellbore
- **TVD**: True Vertical Depth (meters) - vertical depth from surface
- **Inc**: Inclination (degrees) - 0° = vertical, 90° = horizontal

### 2. Casing Intervals
- **MD**: Depth where casing string starts (meters)
- **Pipe ID**: Internal diameter (meters) - converted from inches

### 3. Merged Data
Combines trajectory and casing for nodal analysis calculations.

## Running Tests

Verify everything is working:

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_extraction_patterns.py -v
```

Expected: **41 tests passed** ✅

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# Example: Adjust chunking for larger tables
chunking:
  technical_extraction:
    chunk_size: 3000      # Increase for very large tables
    chunk_overlap: 500

# Example: Adjust validation tolerance
validation:
  md_tvd_tolerance: 2.0   # Allow 2m instead of 1m rounding error
```

## Common Use Cases

### Extract from a Single Well Report

```python
from main import RAGGeothermalSystem

system = RAGGeothermalSystem()
system.process_documents(['ADK-GT-01.pdf'])
result = system.extract_well_parameters('ADK-GT-01')

print(f"Confidence: {result['extraction']['confidence']:.2%}")
print(f"Valid: {result['validation']['is_valid']}")
```

### Batch Process Multiple Wells

```python
import glob
from main import RAGGeothermalSystem

system = RAGGeothermalSystem()

# Process all PDFs in directory
pdf_files = glob.glob('data/*.pdf')
system.process_documents(pdf_files)

# Extract each well
for pdf_file in pdf_files:
    # Assuming filename contains well name
    well_name = pdf_file.split('/')[-1].split('.')[0]
    result = system.extract_well_parameters(well_name)
    # Process result...
```

### Query System for Information

```python
from main import RAGGeothermalSystem

system = RAGGeothermalSystem()
system.process_documents(['well_report.pdf'])

# Query for specific information
chunks = system.query(
    "What is the total depth of ADK-GT-01?",
    strategy='factual_qa'
)

for chunk in chunks[:3]:  # Top 3 results
    print(chunk['content'])
```

## Troubleshooting

### Issue: No well names detected

**Solution**: Check that your PDF contains well names in format:
- `ABC-GT-01` (Dutch geothermal format)
- `WXYZ-123` (Generic format)

### Issue: No trajectory extracted

**Solution**: Ensure your PDF contains a table with columns:
- MD / Measured Depth
- TVD / True Vertical Depth
- Inc / Inclination

### Issue: No casing extracted

**Solution**: Check for casing descriptions like:
- `13 3/8" casing to 1331m`
- `9.625" liner from 1298m to 2500m`

### Issue: Low confidence score

**Possible causes**:
- Partial data extraction (only trajectory or only casing)
- Physics validation failures
- Very few data points

**Solution**: Review warnings in validation report and manually verify extracted data.

## Next Steps

- Read the [full documentation](README.md)
- Explore the [configuration options](config/config.yaml)
- Check out [example.py](example.py) for more usage patterns
- Review the [test suite](tests/) to understand system capabilities

## Support

For issues or questions:
1. Check the [README.md](README.md) for detailed documentation
2. Review [tests/](tests/) for usage examples
3. Open an issue on GitHub

## Quick Reference

| Task | Command |
|------|---------|
| Run example | `python example.py` |
| Process PDF | `python main.py your_file.pdf` |
| Run tests | `pytest tests/` |
| View config | `cat config/config.yaml` |

---

**Ready to extract well data?** Run `python example.py` to see it in action! 🚀
