# Remaining Refactoring Tasks

## Completed ✅
1. ✅ Updated config.yaml - removed all strategies except fine_grained
2. ✅ Updated database_manager.py - added complete_tables table with store/retrieve methods
3. ✅ Updated ingestion_agent.py - removed page-by-page extraction, simplified to store complete tables
4. ✅ Updated preprocessing_agent.py - only uses fine_grained strategy, stores text+well_names+section_headers
5. ✅ Updated rag_retrieval_agent.py - uses single collection, simplified retrieve method

## Remaining Tasks 🔨

### 1. Update hybrid_retrieval_agent.py
**Location:** `/workspaces/RAG_geothermal/geothermal-rag/agents/hybrid_retrieval_agent.py`

**Changes needed:**
- Remove `_classify_query` method - always query both database AND semantic search
- Update `retrieve` method to:
  - Always call `_query_database()` AND `_query_semantic()` (no classification)
  - Use `rag_retrieval_agent.retrieve()` with new signature (no mode parameter)
- Update `_query_database` to:
  - Use `db.get_complete_tables(well_name)` to get ALL tables
  - Convert tables to text format for LLM
  - Include all data types (text and numbers)

**Example changes:**
```python
def retrieve(self, query: str, well_name: Optional[str] = None, top_k: int = 10) -> Dict[str, Any]:
    """Always retrieve from BOTH database and semantic search"""
    results = {
        'database_results': [],
        'semantic_results': [],
        'combined_text': ''
    }
    
    # ALWAYS query database
    db_results = self._query_database(query, well_name)
    results['database_results'] = db_results
    
    # ALWAYS query semantic
    semantic_results = self.semantic_rag.retrieve(query, top_k)
    results['semantic_results'] = semantic_results
    
    # Combine results
    results['combined_text'] = self._combine_results(results)
    return results

def _query_database(self, query: str, well_name: Optional[str]) -> List[Dict]:
    """Get ALL complete tables for the well"""
    if not well_name:
        return []
    
    # Get all tables
    tables = self.db.get_complete_tables(well_name)
    
    # Convert to text format
    db_results = []
    for table in tables:
        table_text = self._format_table_as_text(table)
        db_results.append({
            'type': 'table',
            'table_type': table['table_type'],
            'page': table['source_page'],
            'text': table_text,
            'headers': table['headers'],
            'rows': table['rows']
        })
    
    return db_results
```

### 2. Update app.py - Remove unused agents and simplify workflow
**Location:** `/workspaces/RAG_geothermal/geothermal-rag/app.py`

**Changes in `__init__`:**
```python
# REMOVE these agents (lines ~90-103):
- self.query_analyzer = QueryAnalysisAgent(self.config)
- self.fact_verifier = FactVerificationAgent(self.config, database_manager=self.db)
- self.extraction = ParameterExtractionAgent(...)
- self.validation = ValidationAgent(config_path)
- self.missing_data_agent = MissingDataAgent(self.config)
- self.confidence_scorer = ConfidenceScorerAgent(self.config)
- self.nodal_runner = NodalAnalysisRunner()

# KEEP only:
- self.ingestion
- self.preprocessing
- self.rag (RAGRetrievalAgent)
- self.hybrid_retrieval
- self.llm
- self.db
- self.memory
```

**Changes in `_handle_qa` method (lines ~270-380):**
```python
def _handle_qa(self, query: str) -> Tuple[str, str]:
    """Handle Q&A queries - query BOTH database AND semantic always"""
    
    # Get conversation context
    context = self.memory.get_context_string(last_n=3)
    enhanced_query = f"{context}\n\n{query}" if context else query
    
    # Extract well name
    well_name = self._extract_well_name_from_query(enhanced_query)
    
    # ALWAYS use hybrid retrieval (database + semantic)
    hybrid_result = self.hybrid_retrieval.retrieve(
        enhanced_query,
        well_name=well_name,
        top_k=10
    )
    
    combined_text = hybrid_result.get('combined_text', '')
    
    if not combined_text:
        return "⚠️ No relevant information found", ""
    
    # Generate answer with LLM
    if self.llm_available:
        answer = self.llm.generate_answer(query, [{'text': combined_text}])
        return answer, f"Retrieved from database and semantic search"
    else:
        return combined_text, "LLM not available, showing raw context"
```

**Remove `_handle_extraction` method entirely** (lines ~450-600)
- No longer needed since we don't do parameter extraction

**Remove `run_nodal_analysis` method** (lines ~750-850)
- Not needed

### 3. Implement New Summary System
**Location:** `/workspaces/RAG_geothermal/geothermal-rag/app.py`

**Replace `_handle_summary` method:**
```python
def _handle_summary(self, query: str) -> Tuple[str, str]:
    """
    Generate summary by retrieving 8 data types from database and chunks
    
    8 Data Types:
    1. General Data
    2. Drilling Timeline
    3. Depths
    4. Casing & Tubulars
    5. Cementing
    6. Fluids
    7. Geology/Formations
    8. Incidents
    """
    well_name = self._extract_well_name_from_query(query)
    
    if not well_name:
        return "⚠️ Please specify a well name for summary", ""
    
    summary_parts = []
    summary_parts.append(f"# Well Summary: {well_name}\n")
    
    # 1. Get data from DATABASE (complete tables)
    tables = self.db.get_complete_tables(well_name)
    
    # Process tables by type
    casing_tables = [t for t in tables if 'casing' in t['table_type'].lower() or 'tubular' in str(t['headers']).lower()]
    cement_tables = [t for t in tables if 'cement' in t['table_type'].lower()]
    fluid_tables = [t for t in tables if 'fluid' in t['table_type'].lower() or 'mud' in str(t['headers']).lower()]
    formation_tables = [t for t in tables if 'formation' in t['table_type'].lower() or 'geology' in str(t['headers']).lower()]
    
    # Format tables as text
    if casing_tables:
        summary_parts.append("\n## 4. Casing & Tubulars\n")
        for table in casing_tables:
            summary_parts.append(self._format_table_markdown(table))
    
    if cement_tables:
        summary_parts.append("\n## 5. Cementing\n")
        for table in cement_tables:
            summary_parts.append(self._format_table_markdown(table))
    
    if fluid_tables:
        summary_parts.append("\n## 6. Drilling Fluids\n")
        for table in fluid_tables:
            summary_parts.append(self._format_table_markdown(table))
    
    if formation_tables:
        summary_parts.append("\n## 7. Geology/Formations\n")
        for table in formation_tables:
            summary_parts.append(self._format_table_markdown(table))
    
    # 2. Get narrative data from SEMANTIC SEARCH
    # Search for specific sections
    searches = [
        ("general data well information operator", "1. General Data"),
        ("spud date completion timeline days", "2. Drilling Timeline"),
        ("total depth TD TVD measured", "3. Depths"),
        ("incident problem stuck pipe gas peak mud loss", "8. Incidents")
    ]
    
    for search_query, section_title in searches:
        chunks = self.rag.retrieve(f"{well_name} {search_query}", top_k=3)
        if chunks:
            summary_parts.append(f"\n## {section_title}\n")
            # Use LLM to summarize chunks
            combined_text = "\n\n".join([c['text'] for c in chunks])
            if self.llm_available:
                section_summary = self.llm.generate_answer(
                    f"Summarize {section_title} for {well_name}",
                    [{'text': combined_text}]
                )
                summary_parts.append(section_summary)
            else:
                summary_parts.append(combined_text[:500])
    
    final_summary = "\n".join(summary_parts)
    return final_summary, f"Generated from {len(tables)} tables and semantic search"

def _format_table_markdown(self, table: Dict) -> str:
    """Convert table to markdown format"""
    md = f"\n**{table.get('table_reference', 'Table')}** (Page {table['source_page']})\n\n"
    
    # Headers
    md += "| " + " | ".join(table['headers']) + " |\n"
    md += "| " + " | ".join(["---"] * len(table['headers'])) + " |\n"
    
    # Rows
    for row in table['rows'][:20]:  # Limit to 20 rows
        md += "| " + " | ".join([str(cell) for cell in row]) + " |\n"
    
    if len(table['rows']) > 20:
        md += f"\n*({len(table['rows']) - 20} more rows...)*\n"
    
    return md
```

### 4. Update app.py status messages
**Location:** Around line 180

Change:
```python
# Show base strategies
base_strategies = ['factual_qa', 'technical_extraction', 'summary']
```

To:
```python
# Show single strategy
base_strategies = ['fine_grained']
```

Remove hybrid statistics section (lines ~195-205).

### 5. Update Gradio UI
**Location:** `app.py` lines ~1000-1100

Remove "Extract & Analyze" mode from dropdown:
```python
query_mode = gr.Dropdown(
    choices=["Q&A", "Summary"],  # Removed "Extract & Analyze"
    value="Q&A",
    label="Query Mode"
)
```

Remove nodal analysis button and description (lines ~1060-1090).

### 6. Clean up imports in app.py
**Location:** Top of file (lines 1-35)

Remove unused imports:
```python
# REMOVE:
from agents.parameter_extraction_agent import ParameterExtractionAgent
from agents.validation_agent import ValidationAgent
from agents.ensemble_judge_agent import EnsembleJudgeAgent
from agents.query_analysis_agent import QueryAnalysisAgent
from agents.fact_verification_agent import FactVerificationAgent
from agents.physical_validation_agent import PhysicalValidationAgent
from agents.missing_data_agent import MissingDataAgent
from agents.confidence_scorer import ConfidenceScorerAgent
from models.nodal_runner import NodalAnalysisRunner
```

## Testing After Changes

1. **Test ingestion:**
   ```python
   python -c "from agents.ingestion_agent import IngestionAgent; from agents.database_manager import WellDatabaseManager; db = WellDatabaseManager(); agent = IngestionAgent(db); docs = agent.process(['test.pdf'])"
   ```

2. **Test chunking:**
   ```python
   python -c "from agents.preprocessing_agent import PreprocessingAgent; agent = PreprocessingAgent(); chunks = agent.process([{'filename': 'test.pdf', 'content': 'test text', 'wells': ['TEST-01'], 'pages': 1}]); print(f'Created {len(chunks[\"fine_grained\"])} chunks')"
   ```

3. **Test retrieval:**
   ```python
   python -c "from agents.rag_retrieval_agent import RAGRetrievalAgent; agent = RAGRetrievalAgent(); results = agent.retrieve('test query', top_k=5); print(f'Retrieved {len(results)} chunks')"
   ```

4. **Test full system:**
   ```bash
   python app.py
   # Upload a PDF
   # Try Q&A query
   # Try Summary query
   ```

## Summary of Architectural Changes

**Before:**
- 5 chunking strategies with different sizes
- 5 separate ChromaDB collections
- Complex query routing and classification
- Row-by-row table parsing
- Parameter extraction with validation pipeline
- Fact verification and confidence scoring
- Extraction mode with nodal analysis

**After:**
- 1 chunking strategy (fine_grained: 500 words)
- 1 ChromaDB collection
- Always query BOTH database AND semantic
- Complete table storage (all cells preserved)
- Simple Q&A and Summary modes only
- Summary retrieves 8 data types from DB + chunks
- Minimal metadata (text, well_names, section_headers)
