# 3-Pass Summarization System - Quick Start Guide

## Overview

The new **3-Pass Summarization System** generates professional "End of Well Summary" reports from PDF completion reports with structured data extraction.

## Installation

### 1. Install pdfplumber (Required for Table Extraction)

```bash
pip install pdfplumber
```

Or update from requirements:
```bash
cd /workspaces/RAG_geothermal/geothermal-rag
pip install -r requirements.txt
```

### 2. Ensure Ollama is Running

```bash
ollama serve
```

Verify:
```bash
curl http://localhost:11434/api/tags
```

## Usage

### Via Gradio UI

1. **Start the application:**
   ```bash
   cd /workspaces/RAG_geothermal/geothermal-rag
   python app.py
   ```

2. **Upload PDF:**
   - Go to "Document Upload" tab
   - Upload completion report PDF
   - Click "Index Documents"

3. **Generate Summary:**
   - Go to "Query Interface" tab
   - Select "Summary" mode
   - Enter query with trigger keywords:
     - `"Generate detailed End of Well Summary"`
     - `"Professional completion report summary"`
     - `"EOW summary"`
     - `"Detailed drilling report"`

4. **View Results:**
   - Summary report with metadata, casing table, and narrative
   - Confidence score
   - Debug information

### Via Python API

```python
from agents.well_summary_agent import WellSummaryAgent
from agents.llm_helper import OllamaHelper

# Initialize
llm = OllamaHelper()
agent = WellSummaryAgent(llm_helper=llm)

# Generate summary
result = agent.generate_summary(
    pdf_path='/path/to/completion_report.pdf'
)

# Access results
print(result['summary_report'])  # Final formatted report
print(f"Confidence: {result['confidence']*100:.0f}%")

# Access individual passes
metadata = result['metadata']  # Pass 1 results
technical = result['technical_specs']  # Pass 2 results
narrative = result['narrative']  # Pass 3 results
```

### Testing

Run the test script:
```bash
cd /workspaces/RAG_geothermal/geothermal-rag
python test_summary_agent.py
```

## Features

### Pass 1: Metadata Extraction
- Operator Name
- Well Name
- Rig Name
- Spud Date
- End Date
- Days Total (computed)

### Pass 2: Technical Specs
- Casing/Tubing/Liner tables
- Pipe dimensions (OD, Weight, Depth)
- **Inner Diameter (ID)** - extracted from ID columns
- Markdown table formatting

### Pass 3: Narrative
- Formation instabilities
- Gas shows and peaks
- Drilling hazards
- Geology/lithology findings

### Final Output
- Professional drilling engineer report
- Bold Markdown headers
- Complete casing table with IDs
- Confidence scoring

## Trigger Keywords

Use these keywords in Summary mode to activate 3-pass system:
- `"end of well"`
- `"eow"`
- `"detailed"`
- `"professional"`
- `"comprehensive"`
- `"completion report"`
- `"well report"`
- `"drilling report"`

## Example Output

```markdown
✅ HIGH CONFIDENCE END OF WELL SUMMARY

## End of Well Summary

### Well Information
**Well Name:** ADK-GT-01
**Operator:** Aardwarmte Delft BV
**Rig:** Drillmec HH-220
**Spud Date:** 15 January 2023
**End Date:** 24 March 2023
**Total Days:** 68 days

### Casing Program

| Size | Weight (ppf) | Depth | ID |
|------|--------------|-------|-----|
| 20 inch | 133 | 450 m | 19.124 in |
| 13 3/8 inch | 72 | 1834 m | 12.615 in |
| 9 5/8 inch | 53.5 | 2642 m | 8.535 in |

### Geology & Drilling Hazards
**Formation Instabilities:**
- Shale swelling observed at 1200-1400m

**Gas Shows:**
- Minor H2S detected at 1850m

---
*Summary generated using 3-pass extraction system*

**Confidence Score:** 85%
```

## Fallback Modes

### No pdfplumber
- Skips table extraction
- Uses existing text-based extraction
- Still generates metadata and narrative

### No Ollama/LLM
- Uses regex-based metadata extraction
- Generates basic report template
- Still extracts dates and computes days

### Missing Sections
- Returns partial results
- Lower confidence score
- Documents missing data

## Performance

| Pass | Time (CPU) | Description |
|------|------------|-------------|
| 1 | 5-10s | Metadata + regex/LLM |
| 2 | 15-30s | Table extraction + LLM |
| 3 | 10-15s | Section search + LLM |
| Final | 20-40s | Report generation |
| **Total** | **50-95s** | End-to-end |

## Troubleshooting

### "No PDF path available"
- Ensure documents are indexed before querying
- Check that PDF upload was successful

### "pdfplumber not available"
- Install: `pip install pdfplumber`
- System will work but skip table extraction

### "Ollama not available"
- Start Ollama: `ollama serve`
- Check connection: `curl http://localhost:11434/api/tags`
- Fallback mode will activate automatically

### Low Confidence Score
- Check if PDF contains expected sections
- Verify table formats are recognized
- Review debug output for missing data

## Advanced Configuration

### Customize Date Patterns

Edit `well_summary_agent.py`:
```python
self.date_patterns = {
    'spud_date': [
        r'Your custom pattern here',
        ...
    ]
}
```

### Customize Section Keywords

```python
self.geology_keywords = [
    'your', 'custom', 'keywords'
]
```

### Adjust LLM Timeouts

Edit `config/config.yaml`:
```yaml
ollama:
  timeout_summary: 180  # seconds
  timeout_extraction: 600
```

## Integration with Existing System

The 3-pass system **extends** the existing summarization without breaking it:

- **Standard summaries** still work as before
- **Keyword-triggered** activation for detailed EOW summaries
- **Dual-mode operation** for flexibility
- **Backward compatible** with all existing queries

## Files

### Created
- `agents/well_summary_agent.py` - Main implementation
- `test_summary_agent.py` - Test script
- `SUMMARIZATION_SYSTEM_UPDATE.md` - Detailed docs
- `3PASS_SUMMARY_QUICKSTART.md` - This file

### Modified
- `app.py` - Integration and UI updates
- `requirements.txt` - Added pdfplumber

## Support

For issues or questions:
1. Check logs for detailed error messages
2. Verify all dependencies installed
3. Test with `test_summary_agent.py`
4. Review `SUMMARIZATION_SYSTEM_UPDATE.md` for details

---

**Ready to generate professional well summaries!** 🚀
