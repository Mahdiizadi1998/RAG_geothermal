# RAG Geothermal Wells System

A Retrieval-Augmented Generation (RAG) system for extracting and analyzing geothermal well parameters from PDF documents.

## Overview

This system automates the extraction of critical parameters from geothermal well reports, including:
- **Trajectory data** (Measured Depth, True Vertical Depth, Inclination)
- **Casing design** (depths and pipe diameters)
- **Reservoir properties** (pressure, temperature)
- **Fluid properties**

The system processes PDF reports from sources like the Dutch Oil & Gas Portal (NLOG) and extracts structured data that can be used for nodal analysis and production capacity calculations.

## Features

- **Multi-strategy text chunking**: Semantic, sentence-based, and table-aware chunking
- **Robust parameter extraction**: Regex-based patterns with unit conversion
- **Physics-based validation**: Ensures extracted data satisfies physical constraints
- **Unit conversion**: Handles mixed units (fractional inches, bar/psi, Celsius/Fahrenheit)
- **Table detection**: Identifies and preserves table structures in PDFs
- **Confidence scoring**: Each extracted parameter has an associated confidence score

## Project Structure

```
RAG_geothermal/
├── agents/                    # Core agent modules
│   ├── ingestion_agent.py    # PDF text extraction
│   ├── preprocessing_agent.py # Text chunking
│   └── extraction_agent.py    # Parameter extraction
├── utils/                     # Utility modules
│   ├── unit_converter.py     # Unit conversion utilities
│   └── validators.py          # Physics-based validation
├── config/                    # Configuration files
│   └── config.yaml           # Main configuration
├── tests/                     # Test suite
│   ├── test_unit_converter.py
│   └── test_validators.py
├── data/                      # Data directory (PDFs, test data)
├── main.py                    # Main CLI entry point
└── requirements.txt           # Python dependencies
```

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/SinaMad98/RAG_geothermal.git
cd RAG_geothermal
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download spaCy language model** (if needed):
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Command Line Interface

Process a PDF document:
```bash
python main.py path/to/document.pdf
```

Save extracted parameters to JSON:
```bash
python main.py path/to/document.pdf --output results.json
```

Use custom configuration:
```bash
python main.py path/to/document.pdf --config custom_config.yaml
```

### Python API

```python
from agents.ingestion_agent import IngestionAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.extraction_agent import ParameterExtractionAgent

# Step 1: Ingest PDF
ingestion = IngestionAgent()
doc_data = ingestion.process_document("well_report.pdf")

# Step 2: Preprocess and chunk
preprocessing = PreprocessingAgent()
chunks = preprocessing.process_document(doc_data['pages'], strategy='table-aware')

# Step 3: Extract parameters
extraction = ParameterExtractionAgent()
params = extraction.extract_from_chunks(chunks)

# Access extracted data
print(f"Trajectory points: {len(params.trajectory)}")
print(f"Casing sections: {len(params.casing)}")
print(f"Reservoir pressure: {params.reservoir_pressure} bar")
```

## Configuration

The system is configured via `config/config.yaml`:

### Key Configuration Sections

- **extraction.patterns**: Regex patterns for parameter extraction
- **extraction.units**: Unit conversion factors
- **validation**: Physics-based validation rules and ranges
- **retrieval**: RAG retrieval settings (for future vector search implementation)
- **llm**: Language model settings (for future LLM integration)

Example configuration snippet:
```yaml
extraction:
  chunking:
    semantic:
      max_chunk_size: 500
      overlap: 50
    table_aware:
      detect_tables: true
      preserve_structure: true

validation:
  trajectory:
    md_tvd_relationship: "MD >= TVD"
    inclination_range: [0, 90]
    max_depth: 10000
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=. tests/
```

## Example Output

```
Processing PDF: ADK-GT-01.pdf
  [1/3] Ingesting PDF...
       Extracted 27 pages
  [2/3] Preprocessing and chunking...
       Created 45 chunks
  [3/3] Extracting parameters...

=== Extraction Results ===
Well Name: ADK-GT-01
Trajectory Points: 35
Casing Sections: 4
Reservoir Pressure: 245.0 bar
Reservoir Temperature: 155.0°C

=== Validation ===
Validation 1: PASSED (confidence: 0.95)
  WARNING: Large difference between MD (2500m) and TVD (2300m). Check for highly deviated well.
```

## Development Roadmap

### Current Status (Phase 1-3 Complete)
- [x] Project setup and infrastructure
- [x] Core agents implementation
- [x] Unit conversion and validation
- [x] Basic parameter extraction
- [x] Unit tests

### Future Enhancements (Phase 4-6)
- [ ] RAG vector database integration (ChromaDB)
- [ ] LLM-based extraction enhancement
- [ ] Interactive validation with user prompts
- [ ] Nodal analysis calculations
- [ ] Web interface (Gradio)
- [ ] Advanced table extraction (Camelot/Tabula)

## Technical Details

### Supported Unit Conversions

- **Length**: meters ↔ feet
- **Diameter**: inches (fractional/decimal) ↔ meters
- **Pressure**: bar, psi, kPa, MPa
- **Temperature**: Celsius ↔ Fahrenheit

### Validation Rules

- **Trajectory**: MD ≥ TVD (fundamental constraint)
- **Inclination**: 0° to 90° (from vertical)
- **Casing diameter**: 50mm to 1000mm
- **Pressure**: 0 to 1000 bar (reservoir context-dependent)
- **Temperature**: 0°C to 300°C

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## References

- Dutch Oil & Gas Portal (NLOG): www.nlog.nl
- Geothermal well engineering best practices
- Nodal analysis for well performance prediction

## Contact

For questions or issues, please open an issue on GitHub.