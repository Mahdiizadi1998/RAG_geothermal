# Hybrid Database + Semantic RAG Architecture

## Overview

The system now uses a **hybrid approach** combining:

1. **SQLite Database** (structured data): Exact numerical values, table data, technical specs
2. **ChromaDB Semantic Search** (narrative context): Text descriptions, problems, operations
3. **LLM with Templates**: Fill templates with exact data, enrich with context

**Why this architecture?**
- ❌ **Old**: Semantic embeddings for everything → numbers/specs get fuzzy/inaccurate
- ✅ **New**: Database for exact data, embeddings for narratives → best of both worlds

---

## Architecture Components

### 1. Database Layer (`database_manager.py`)

**SQLite schema** with 7 tables:

```
wells
├── well_id (PK)
├── well_name (UNIQUE)
├── operator, location, country
├── spud_date, completion_date
├── total_depth_md, total_depth_tvd
└── well_type, status

casing_strings (FK: well_id)
├── outer_diameter, weight, grade
├── top_depth_md, bottom_depth_md
└── source_page, source_table

formations (FK: well_id)
├── formation_name
├── top_md, top_tvd, bottom_md, bottom_tvd
├── lithology, age
└── source_page

cementing (FK: well_id)
├── stage_number, cement_type
├── top_of_cement_md, bottom_of_cement_md
├── volume, density
└── source_page

operations (FK: well_id)
├── operation_date, operation_type
├── description, depth_md
└── duration_hours, status

measurements (FK: well_id)
├── measurement_type (temp, pressure, etc.)
├── depth_md, value, unit
└── measurement_date

documents (FK: well_id)
└── filename, filepath, document_type, total_pages
```

**Key methods:**
- `add_or_get_well(well_name, **kwargs)` → well_id
- `add_casing_string(well_name, casing_data)` → casing_id
- `add_formation(well_name, formation_data)` → formation_id
- `get_well_summary(well_name)` → full well dict with all tables
- `query_sql(query, params)` → custom SQL queries

---

### 2. Table Parser (`table_parser.py`)

**Intelligent table type identification** using keyword matching:

```python
table_types = {
    'casing': ['casing', 'pipe', 'od', 'weight', 'grade'],
    'cementing': ['cement', 'toc', 'stage', 'slurry'],
    'formations': ['formation', 'lithology', 'top', 'geology'],
    'trajectory': ['survey', 'inclination', 'azimuth', 'md', 'tvd'],
    'fluids': ['mud', 'density', 'viscosity'],
    'operations': ['activity', 'time', 'duration']
}
```

**Parsing methods:**
- `identify_table_type(headers, rows, context)` → 'casing', 'formations', etc.
- `parse_casing_table(headers, rows, page)` → list of casing dict
- `parse_formation_table(headers, rows, page)` → list of formation dict

**Smart extraction:**
- Handles fractions: "13 3/8 inch" → 13.375
- Multiple units: m, ft, inch, lb/ft, kg/m
- Flexible column order (finds columns by keywords)

---

### 3. Enhanced Ingestion (`ingestion_agent.py`)

**Now extracts tables AND stores in database:**

```python
ingestion = IngestionAgent(
    database_manager=db,
    table_parser=parser
)

# Process PDF
documents = ingestion.process([pdf_path])

# Extract + store tables in database
tables_stored = ingestion.process_and_store_tables(
    pdf_path, 
    well_names=['ADK-GT-01']
)
```

**Workflow:**
1. Extract tables using `pdfplumber` (handles invisible grids)
2. Identify table type (casing, formations, etc.)
3. Parse table into structured data
4. Link to well using well name detection
5. Insert into SQLite database with source page tracking

---

### 4. Summary Templates (`summary_templates.py`)

**Three templates with placeholders:**

#### **basic_completion**
```
The {well_name} well{operator_text} reached {total_depth_md}m MD{tvd_text}.
{casing_summary}
{completion_date_text}
```
**Required:** well_name, total_depth_md  
**Optional:** operator, casing_strings

#### **detailed_technical**
```
The {well_name} well{operator_text}{location_text} was drilled to {total_depth_md}m MD{tvd_text}.

{casing_summary}
{formation_summary}
{cement_summary}
{dates_summary}
```
**Required:** well_name, total_depth_md, casing_strings  
**Optional:** formations, cementing, operator

#### **comprehensive**
```
{well_header}

WELL DATA: {well_data_section}
CASING PROGRAM: {casing_section}
GEOLOGICAL INFORMATION: {geology_section}
CEMENTING OPERATIONS: {cementing_section}
OPERATIONS SUMMARY: {operations_section}
ADDITIONAL INFORMATION: {narrative_context}
```
**Required:** well_name, total_depth_md  
**Optional:** All tables + narrative context

**Helper functions:**
- `format_casing_summary(casings, include_source=True)` → formatted string with [Source: Page X]
- `format_formation_summary(formations)` → formatted string
- `format_cement_summary(cementing)` → formatted string

---

### 5. Template Selector (`template_selector.py`)

**Automatic template selection** based on data availability:

```python
selector = TemplateSelectorAgent(database_manager=db)

# Auto-select best template
template = selector.select_template('ADK-GT-01')

# Or user can specify
template = selector.select_template('ADK-GT-01', user_preference='comprehensive')
```

**Scoring algorithm:**
- Required data present: +10 points each
- Optional data present: +5 points each
- Missing required data: disqualify template (-1 score)
- Highest score wins

**Example:**
```
ADK-GT-01 has: well_name, total_depth_md, casing_strings, formations

basic_completion:      +10 (well_name) +10 (depth) +5 (casing) = 25 points
detailed_technical:    +10 +10 +10 (casing req) +5 (formations) = 35 points ✓ WINNER
comprehensive:         +10 +10 +5 +5 = 30 points (missing many optionals)
```

---

### 6. Hybrid Retrieval (`hybrid_retrieval_agent.py`)

**Smart routing** based on query type:

```python
hybrid = HybridRetrievalAgent(
    database_manager=db,
    rag_retrieval_agent=semantic_rag
)

# Auto-detect mode
results = hybrid.retrieve(
    query="What is the casing program?",
    well_name="ADK-GT-01",
    mode='auto'  # or 'database', 'semantic', 'hybrid'
)
```

**Query classification patterns:**

| Query Type | Examples | Route |
|------------|----------|-------|
| **Numerical** | "total depth", "casing size", "when drilled" | Database |
| **Table** | "casing program", "formation tops", "list all" | Database |
| **Narrative** | "what problems occurred", "describe operations" | Semantic |
| **Complex** | "depth and problems", "casing and issues" | Hybrid (both) |

**Result structure:**
```python
{
    'database_results': [
        {'type': 'well_info', 'data': {...}, 'text': "Well: ADK-GT-01\nOperator: ECW..."},
        {'type': 'casing', 'data': [...], 'text': "Casing Program:\n  - 13.375 inch..."}
    ],
    'semantic_results': [
        {'text': "During drilling operations...", 'metadata': {...}},
        ...
    ],
    'combined_text': "=== EXACT DATA FROM DATABASE ===\n...\n=== SUPPORTING CONTEXT ===\n...",
    'mode': 'hybrid'
}
```

**Priority:** Database (exact) > Semantic (context)

---

## Integration with Existing System

### Update `app.py`:

```python
from agents.database_manager import WellDatabaseManager
from agents.table_parser import TableParser
from agents.template_selector import TemplateSelectorAgent
from agents.hybrid_retrieval_agent import HybridRetrievalAgent

# Initialize in __init__
self.db = WellDatabaseManager("./well_data.db")
self.table_parser = TableParser()
self.ingestion = IngestionAgent(
    database_manager=self.db,
    table_parser=self.table_parser
)
self.template_selector = TemplateSelectorAgent(self.db)
self.hybrid_retrieval = HybridRetrievalAgent(self.db, self.rag)

# In index_documents()
for pdf_path in pdf_paths:
    # Extract text + metadata
    documents = self.ingestion.process([pdf_path])
    
    # Extract + store tables in database
    for doc in documents:
        self.ingestion.process_and_store_tables(
            doc['filepath'],
            doc['wells']
        )
    
    # Continue with text chunking for semantic search
    # (only for narrative text, not tables)
    ...

# In _handle_summary()
# 1. Get well name from query
well_name = self._extract_well_name(user_query)

# 2. Select template
template = self.template_selector.select_template(well_name)

# 3. Hybrid retrieval
retrieval_results = self.hybrid_retrieval.retrieve(
    query=user_query,
    well_name=well_name,
    mode='hybrid'
)

# 4. Generate summary using template + data
summary = self.llm_helper.generate_summary_from_template(
    template=template,
    database_data=retrieval_results['database_results'],
    narrative_context=retrieval_results['semantic_results'],
    well_name=well_name,
    target_words=target_words
)

# 5. Verify against database
verification = self._verify_against_database(summary, well_name)
```

---

## Benefits

### ✅ Accuracy
- **Exact numbers** from database (no embedding fuzzing)
- **Source tracking** (page numbers for all table data)
- **Well-specific** (foreign key ensures correct well)

### ✅ Completeness
- **Template-driven** (ensures all important fields covered)
- **Smart selection** (uses best template for available data)
- **Narrative enrichment** (adds context from semantic search)

### ✅ Verifiability
- **Database verification** (check all numbers against source)
- **Citation generation** (automatic [Source: Page X] from database)
- **Confidence scoring** (based on data completeness)

### ✅ Performance
- **Fast SQL queries** (indexed lookups vs vector search)
- **Reduced embedding load** (only narratives, not tables)
- **Scalable** (SQLite handles millions of rows)

---

## Example Workflow

### 1. Upload PDF
```
User uploads: "NLOG_GS_PUB_EOWR ADK-GT-01 SODM v1.1.pdf"
```

### 2. Ingestion
```
→ Extract text (PyMuPDF + EasyOCR)
→ Detect well names: ['ADK-GT-01']
→ Extract tables (pdfplumber): 5 tables found
   - Table 1 (Page 8): Casing Program → parse_casing_table()
   - Table 2 (Page 12): Formation Tops → parse_formation_table()
   - Table 3 (Page 15): Cementing → parse_cementing_table()
→ Store in database:
   - wells: INSERT ADK-GT-01
   - casing_strings: INSERT 4 rows
   - formations: INSERT 8 rows
   - cementing: INSERT 3 rows
→ Chunk narrative text for semantic search
```

### 3. Summary Query
```
User: "Give a summary of well ADK-GT-01"
```

### 4. Template Selection
```
→ Check database: ADK-GT-01 has casing, formations, cementing
→ Score templates:
   - basic_completion: 25
   - detailed_technical: 35 ✓
   - comprehensive: 30
→ Selected: detailed_technical
```

### 5. Hybrid Retrieval
```
→ Query classification: "summary" → hybrid mode
→ Database query:
   SELECT * FROM wells WHERE well_name = 'ADK-GT-01'
   → Depth: 2667.5m MD, 2358m TVD
   
   SELECT * FROM casing_strings WHERE well_id = 1
   → 13.375 inch, 53.5 lb/ft, L80, set at 2642m MD [Page 8]
   → 9.625 inch, 47 lb/ft, L80, set at 2667m MD [Page 8]
   
→ Semantic search for context:
   → "Drilling commenced January 2017..."
   → "Operations completed without major issues..."
```

### 6. Summary Generation
```
→ Fill template with database data:
   "The ADK-GT-01 well, operated by ECW Geomanagement BV, was drilled
    to a total measured depth of 2667.5m MD (2358m TVD).
    
    CASING PROGRAM:
    - 13 3/8 inch, 53.5 lb/ft, L80, set at 2642m MD [Source: Page 8]
    - 9 5/8 inch, 47 lb/ft, L80, set at 2667m MD [Source: Page 8]
    
    FORMATIONS:
    - Nieuwerkerk Formation at 950m MD [Source: Page 12]
    - Aalburg Formation at 1450m MD [Source: Page 12]
    ..."
    
→ LLM enriches with semantic context:
   "Drilling operations commenced in January 2017 and were completed
    without major incidents..."
```

### 7. Verification
```
→ Check all numbers against database:
   ✓ 2667.5m MD matches wells.total_depth_md
   ✓ 2358m TVD matches wells.total_depth_tvd  
   ✓ 13.375 inch matches casing_strings.outer_diameter
   ✓ All page citations valid
   
→ Overall confidence: 95% (all data verified)
```

---

## Next Steps

### Immediate:
1. Update `app.py` to initialize new components
2. Modify indexing workflow to call `process_and_store_tables()`
3. Update summary generation to use templates + hybrid retrieval
4. Add database verification step

### Future Enhancements:
1. Add parsers for cementing, trajectory, fluids tables
2. Implement SQL query builder for complex questions
3. Add database export (JSON, CSV) functionality
4. Create web UI for database browsing
5. Add multi-well comparison queries

---

## Files Created

```
agents/
├── database_manager.py          # SQLite handler (7 tables)
├── table_parser.py              # Table type identification + parsing
├── summary_templates.py         # 3 templates with placeholders
├── template_selector.py         # Auto-select best template
└── hybrid_retrieval_agent.py    # Database + semantic routing
```

**Database location:** `./well_data.db` (SQLite file)

---

## Testing

```python
# Test database
from agents.database_manager import WellDatabaseManager
db = WellDatabaseManager()
well_id = db.add_or_get_well('TEST-GT-01', operator='TestCo', total_depth_md=3000)
db.add_casing_string('TEST-GT-01', {
    'outer_diameter': 13.375,
    'weight': 53.5,
    'grade': 'L80',
    'bottom_depth_md': 2500
})
summary = db.get_well_summary('TEST-GT-01')
print(summary)

# Test table parser
from agents.table_parser import TableParser
parser = TableParser()
table_type = parser.identify_table_type(
    headers=['Size', 'Weight', 'Grade', 'Depth'],
    rows=[['13 3/8"', '53.5 lb/ft', 'L80', '2500m']],
    context='Casing program for well'
)
print(table_type)  # → 'casing'

# Test hybrid retrieval
from agents.hybrid_retrieval_agent import HybridRetrievalAgent
hybrid = HybridRetrievalAgent(db, rag)
results = hybrid.retrieve("What is the casing program?", well_name='TEST-GT-01')
print(results['combined_text'])
```
