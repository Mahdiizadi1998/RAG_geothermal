# Advanced Agentic RAG System - Integration Guide

## Overview

This project has been enhanced with state-of-the-art (SOTA) techniques from 2024 research to implement a fully **Agentic, Multi-Modal, Hybrid RAG** architecture for geothermal well engineering.

## New Components

### 1. **Ultimate Semantic Chunker** (`ultimate_semantic_chunker.py`)
Implements three advanced chunking techniques:
- **Late Chunking (Jina AI)**: Embeds full document first to capture global context
- **Contextual Enrichment (Anthropic)**: Prepends document context to every chunk
- **Semantic Breakpoints**: Splits at semantic similarity drops (not fixed sizes)

**Benefits:**
- Better context preservation
- Natural topic boundaries
- Improved retrieval accuracy

### 2. **RAPTOR Tree** (`raptor_tree.py`)
Hierarchical summarization using clustering and recursive abstraction:
- Level 0: Original chunks (detailed facts)
- Level 1+: Cluster summaries (progressively higher-level)
- Query different levels for different abstraction needs

**Benefits:**
- High-level summaries without losing details
- Multi-level retrieval
- Better summary quality

### 3. **BM25 Sparse Retrieval** (`bm25_retrieval.py`)
Keyword-based search using BM25Okapi algorithm:
- Complements dense vector search
- Excellent for exact term matching (well names, IDs, equipment)
- Hybrid scoring combines both approaches

**Benefits:**
- Better exact-match retrieval
- Improved technical terminology matching
- More robust hybrid search

### 4. **Knowledge Graph** (`knowledge_graph.py`)
Document relationship graph using NetworkX:
- Nodes: Document chunks
- Edges: Semantic similarity + metadata connections
- Multi-hop traversal for related documents

**Benefits:**
- Find related documents across the corpus
- Multi-hop reasoning
- Community detection

### 5. **Universal Metadata Extractor** (`universal_metadata_extractor.py`)
Advanced entity extraction using spaCy NER + Regex:
- Well names, formations, depths, pressures, temperatures
- Operators, dates, equipment specifications
- Context-aware extraction

**Benefits:**
- Richer metadata for filtering
- Better entity-based queries
- Improved structured retrieval

### 6. **Vision Processor** (`vision_processor.py`)
Image/plot captioning using llava:7b VLM:
- Extracts images from PDFs
- Classifies image types (plot, schematic, diagram)
- Generates detailed text captions
- Enables semantic search over images

**Benefits:**
- Multi-modal RAG (text + images)
- Searchable plots and diagrams
- Complete document understanding

### 7. **Query Router** (enhanced `query_analysis_agent.py`)
Intelligent query routing to appropriate strategy:
- Factual Q&A → Hybrid (Vector + BM25)
- Summaries → RAPTOR tree
- Extractions → Structured DB + BM25
- Comparisons → Knowledge Graph
- Relationships → Knowledge Graph traversal

**Benefits:**
- Optimal retrieval for each query type
- Better performance and accuracy
- Adaptive system behavior

### 8. **Reranking System** (`reranker.py`)
Improves result ordering using cross-encoder:
- Re-scores retrieved results
- Reciprocal Rank Fusion (RRF) for combining strategies
- LLM-based scoring option

**Benefits:**
- More relevant results at top
- Better multi-strategy fusion
- Improved user experience

## Architecture Comparison

### Before (Current System)
```
User Query
    ↓
Query Analysis (basic)
    ↓
Hybrid Retrieval: Database (tables) + Vector Search (text chunks)
    ↓
LLM Generation
    ↓
Answer
```

### After (Enhanced System)
```
User Query
    ↓
Universal Metadata Extraction (entities)
    ↓
Query Router (intelligent routing)
    ├─→ Factual Q&A: Hybrid (Dense + Sparse) + Reranking
    ├─→ Summary: RAPTOR Tree (multi-level)
    ├─→ Extraction: Structured DB + BM25
    ├─→ Comparison: Knowledge Graph (multi-hop)
    └─→ Images: Vision Processing (VLM captions)
    ↓
Result Fusion (RRF) + Reranking
    ↓
LLM Generation (with enriched context)
    ↓
Answer
```

## Key Improvements

| Feature | Before | After |
|---------|--------|-------|
| **Chunking** | Fixed-size recursive | Ultimate Semantic (Late + Contextual + Breakpoints) |
| **Retrieval** | Vector only | Hybrid (Dense + Sparse + Graph + RAPTOR) |
| **Routing** | Simple type detection | Intelligent strategy routing |
| **Metadata** | Basic regex | Universal NER + Regex |
| **Images** | Not processed | VLM captioning + embedding |
| **Summaries** | Single-level | Multi-level hierarchical (RAPTOR) |
| **Ranking** | Distance-based | Cross-encoder reranking |
| **Graph** | None | Knowledge Graph with multi-hop |

## Models Used

Following the reference architecture:

| Role | Model | Purpose |
|------|-------|---------|
| **Reasoning & Generation** | `llama3.1:8b` | Core "brain" for agents, Q&A, and summaries |
| **Vision Processing** | `llava:7b` | Image/plot captioning and understanding |
| **Embeddings** | `all-MiniLM-L6-v2` | Fast sentence embeddings for search |
| **Clustering** | `HDBSCAN` | Density-based clustering for RAPTOR |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Result reranking |

## Installation

1. **Install new dependencies:**
```bash
cd geothermal-rag
pip install -r requirements.txt
```

2. **Pull new Ollama models:**
```bash
ollama pull llama3.1:8b   # ~4.7GB
ollama pull llava:7b      # ~4.7GB
```

3. **Download spaCy model:**
```bash
python -m spacy download en_core_web_sm
```

## Usage Examples

### 1. Standard Q&A (Automatic Hybrid Retrieval)
```python
# System automatically uses hybrid vector+BM25 retrieval
query = "What is the casing program for well ADK-GT-01?"
answer = app.ask_question(query, well_name="ADK-GT-01")
```

### 2. High-Level Summary (RAPTOR Level 2)
```python
# System routes to RAPTOR tree for high-level summary
query = "Give me a brief overview of the well"
answer = app.ask_question(query, well_name="ADK-GT-01")
```

### 3. Detailed Summary (RAPTOR Level 1)
```python
# System routes to RAPTOR tree for detailed summary
query = "Give me a comprehensive summary of all operations"
answer = app.ask_question(query, well_name="ADK-GT-01")
```

### 4. Exact Term Matching (BM25)
```python
# System uses BM25 for exact well name/ID matching
query = "Find all mentions of XYZ-GT-02 in the documents"
answer = app.ask_question(query)
```

### 5. Relationship Query (Knowledge Graph)
```python
# System uses knowledge graph for related document traversal
query = "What other wells are related to ADK-GT-01?"
answer = app.ask_question(query, well_name="ADK-GT-01")
```

### 6. Image Search (Vision Processing)
```python
# System searches over image captions generated by llava:7b
query = "Show me plots of temperature vs depth"
answer = app.ask_question(query, well_name="ADK-GT-01")
```

## Integration Steps

### Step 1: Update Preprocessing Agent
Replace simple chunking with Ultimate Semantic Chunker:

```python
from agents.ultimate_semantic_chunker import create_chunker

# In preprocessing_agent.py
chunker = create_chunker(config.get('semantic_chunking'))
chunks = chunker.chunk_document(document)
```

### Step 2: Build Indices
Build all indices (vector, BM25, RAPTOR, knowledge graph):

```python
# Vector index (existing)
rag_agent.index_chunks(chunks)

# BM25 index (new)
from agents.bm25_retrieval import create_bm25_retriever
bm25 = create_bm25_retriever()
bm25.index_documents(chunks)

# RAPTOR tree (new)
from agents.raptor_tree import create_raptor_tree
raptor = create_raptor_tree(llm_helper)
raptor.build_tree(chunks)

# Knowledge graph (new)
from agents.knowledge_graph import create_knowledge_graph
kg = create_knowledge_graph()
kg.build_graph(chunks)
```

### Step 3: Use Query Router
Let the router decide retrieval strategy:

```python
from agents.query_analysis_agent import QueryAnalysisAgent

query_analyzer = QueryAnalysisAgent(config)
analysis = query_analyzer.analyze(query)

# Route based on strategy
if analysis.retrieval_strategy == 'hybrid':
    results = hybrid_retriever.search(query)
elif analysis.retrieval_strategy == 'raptor':
    results = raptor.query_tree(query, level=analysis.raptor_level)
elif analysis.retrieval_strategy == 'graph':
    # Get initial results, then traverse graph
    initial = vector_retriever.retrieve(query)
    seed_ids = [r['chunk_id'] for r in initial[:3]]
    results = kg.query_graph(seed_ids, max_hops=2)
elif analysis.retrieval_strategy == 'bm25':
    results = bm25.search(query)
```

### Step 4: Apply Reranking
Rerank results for better ordering:

```python
from agents.reranker import create_reranker

reranker = create_reranker(config.get('reranking'))
reranked_results = reranker.rerank(query, results, top_k=10)
```

## Configuration

All new features are controlled via `config/config.yaml`:

```yaml
# Enable/disable features
semantic_chunking:
  enabled: true
  similarity_threshold: 0.7

raptor:
  enabled: true
  max_tree_height: 3

knowledge_graph:
  enabled: true
  similarity_threshold: 0.7

bm25:
  enabled: true

reranking:
  enabled: true
  method: "cross-encoder"

vision:
  enabled: true
  model: "llava:7b"
```

## Performance Considerations

- **Memory**: RAPTOR tree and Knowledge Graph increase memory usage (~2x)
- **Indexing Time**: Initial indexing slower due to clustering and graph building
- **Query Time**: Routing adds minimal overhead (<100ms)
- **Storage**: Additional indices increase disk usage (~1.5x)

## Backward Compatibility

All enhancements are **backward compatible**:
- Existing table-based retrieval unchanged
- Can disable new features via config
- Fallback to simple retrieval if components unavailable

## Testing

Test each component individually:

```bash
# Test semantic chunker
python test_ultimate_chunker.py

# Test RAPTOR tree
python test_raptor_tree.py

# Test BM25 retrieval
python test_bm25.py

# Test knowledge graph
python test_knowledge_graph.py

# Test vision processing
python test_vision_processor.py

# Test full integration
python test_advanced_rag_system.py
```

## Troubleshooting

**Issue**: Vision model not found
```bash
# Solution: Pull the model
ollama pull llava:7b
```

**Issue**: spaCy model missing
```bash
# Solution: Download model
python -m spacy download en_core_web_sm
```

**Issue**: Cross-encoder loading slow
```bash
# Solution: First load takes time, subsequent loads are fast
# Or disable reranking in config
```

**Issue**: Out of memory during RAPTOR clustering
```bash
# Solution: Reduce max_tree_height or min_cluster_size in config
```

## Next Steps

1. **Test with your data**: Run full pipeline on geothermal well reports
2. **Tune parameters**: Adjust thresholds and weights in config
3. **Monitor performance**: Track query times and memory usage
4. **Iterate**: Refine based on actual usage patterns

## References

- **Late Chunking**: Jina AI research (2024)
- **Contextual Enrichment**: Anthropic contextual retrieval (2024)
- **RAPTOR**: Stanford/Berkeley recursive abstractive processing (2023)
- **BM25**: Robertson & Zaragoza (2009)
- **Knowledge Graphs**: Various research on graph-based RAG (2023-2024)
- **Vision-Language Models**: LLaVA (Liu et al., 2023)
