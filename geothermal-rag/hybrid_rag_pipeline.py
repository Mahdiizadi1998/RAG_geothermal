"""
Hybrid RAG Pipeline for End of Well Reports
============================================
CPU-Optimized Solution: 8-core CPU, 16GB RAM, no GPU

STRATEGY: "Hybrid Parsing" - Treat tables and narrative text differently

Stream A (Tables):
  - Use pdfplumber to detect tables with invisible grid lines
  - DO NOT chunk tables (destroys row context)
  - Convert each table row to semantic summary string for embedding
  - Store raw table data (JSON/Dict) in metadata for exact retrieval

Stream B (Narrative Text):
  - Extract full page text, clean headers/footers
  - Split semantically by section headers ("1. GENERAL DATA", "4. GEOLOGY")
  - Use RecursiveCharacterTextSplitter with ~800 tokens, 100 overlap

WHY THIS WORKS:
  - Tables need row-level preservation for accuracy
  - Narrative text benefits from semantic chunking
  - Embeddings work on both semantic summaries and text chunks
  - LLM can retrieve exact numbers from metadata
"""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

import pdfplumber
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


# ============================================================================
# TARGET DATA STRUCTURE (Pydantic Models)
# ============================================================================

class CasingTubular(BaseModel):
    """
    Single casing/tubing/liner entry
    WHY: Crucial for well integrity - need exact pipe specs
    """
    type: str = Field(..., description="Conductor, Surface, Intermediate, Production, Liner")
    od_inches: float = Field(..., description="Outer Diameter in inches")
    weight_lbs_ft: Optional[float] = Field(None, description="Weight in lbs/ft")
    grade: Optional[str] = Field(None, description="Steel grade (K55, L80, P110)")
    connection: Optional[str] = Field(None, description="Connection type (BTC, LTC)")
    
    # CRITICAL: Both nominal and drift ID
    pipe_id_nominal_inches: Optional[float] = Field(None, description="Nominal Inner Diameter")
    pipe_id_drift_inches: Optional[float] = Field(None, description="Drift Inner Diameter")
    
    top_depth_mah: float = Field(..., description="Top depth in meters (measured along hole)")
    bottom_depth_mah: float = Field(..., description="Bottom depth in meters")


class CementJob(BaseModel):
    """Cementing operation details"""
    stage: str = Field(..., description="Primary, Squeeze, Remedial")
    lead_volume_m3: Optional[float] = None
    lead_density_sg: Optional[float] = None
    tail_volume_m3: Optional[float] = None
    tail_density_sg: Optional[float] = None
    toc_mah: Optional[float] = Field(None, description="Top of Cement in meters")


class Formation(BaseModel):
    """Geological formation encountered"""
    name: str
    top_depth_mah: float
    bottom_depth_mah: Optional[float] = None
    lithology: Optional[str] = None
    notes: Optional[str] = Field(None, description="Gas shows, instability, etc.")


class Incident(BaseModel):
    """Operational incidents"""
    date: Optional[str] = None
    type: str = Field(..., description="Gas peak, stuck pipe, mud loss, etc.")
    description: str
    depth_mah: Optional[float] = None


class WellReport(BaseModel):
    """
    Complete End of Well Report structure
    WHY: This schema guides the extraction process
    """
    # General Data
    well_name: str
    license: Optional[str] = None
    well_type: str = Field(..., description="Vertical, Directional, Horizontal")
    location: Optional[str] = None
    coordinates_x: Optional[float] = None
    coordinates_y: Optional[float] = None
    operator: Optional[str] = None
    rig_name: Optional[str] = None
    target_formation: Optional[str] = None
    
    # Drilling Timeline
    spud_date: Optional[str] = None
    end_operations_date: Optional[str] = None
    total_days: Optional[int] = None
    
    # Depths
    td_mah: Optional[float] = Field(None, description="Total Depth (measured along hole)")
    tvd: Optional[float] = Field(None, description="True Vertical Depth")
    sidetrack_start_depth: Optional[float] = None
    
    # Tubulars (CRUCIAL - most queries are about this)
    casing_tubulars: List[CasingTubular] = Field(default_factory=list)
    
    # Cementing
    cement_jobs: List[CementJob] = Field(default_factory=list)
    
    # Fluids
    hole_sizes: List[Dict[str, Any]] = Field(default_factory=list, 
                                             description="[{size_inches, from_mah, to_mah, fluid_type, density_range}]")
    
    # Geology
    formations: List[Formation] = Field(default_factory=list)
    
    # Incidents
    incidents: List[Incident] = Field(default_factory=list)


# ============================================================================
# HYBRID PARSER CLASS
# ============================================================================

@dataclass
class TableChunk:
    """
    Represents a single table row as a searchable chunk
    WHY: Tables aren't chunked - each row is a semantic unit
    """
    content: str  # Semantic summary: "Casing Table: 13 3/8 inch casing set at 1331m"
    metadata: Dict[str, Any]  # Raw table data as JSON
    page_num: int
    table_index: int
    row_index: int


@dataclass
class TextChunk:
    """
    Represents a narrative text chunk
    WHY: Narrative text can be chunked semantically
    """
    content: str
    metadata: Dict[str, Any]
    page_num: int
    section: Optional[str] = None  # "1. GENERAL DATA", "4. GEOLOGY"


class HybridParser:
    """
    Hybrid Parsing for End of Well Reports
    WHY: Tables and text need different treatment for accuracy
    """
    
    def __init__(self):
        # WHY all-MiniLM-L6-v2: Fast on CPU, good quality, small footprint (80MB)
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # WHY RecursiveCharacterTextSplitter: Tries natural breaks (\n\n, \n, space)
        # 800 tokens ≈ 3200 chars (4 chars/token avg for technical text)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3200,  # ~800 tokens
            chunk_overlap=400,  # ~100 tokens
            separators=["\n\n", "\n", ". ", " ", ""],  # Natural breaks
            length_function=len
        )
        
        # Header/footer patterns to remove
        self.noise_patterns = [
            r"Page \d+ of \d+",
            r"SodM EOWR",
            r"End of Well Report",
            r"Confidential",
            r"^\s*\d+\s*$",  # Page numbers alone
        ]
    
    def extract_tables(self, pdf_path: str) -> List[TableChunk]:
        """
        Stream A: Extract tables using pdfplumber
        
        WHY pdfplumber: Better than PyMuPDF/camelot for Well Reports
        WHY vertical_strategy="text": Handles invisible grid lines
        WHY NOT chunk: Chunking destroys row context - we need complete rows
        
        Returns:
            List of TableChunk objects, one per table row
        """
        table_chunks = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # WHY table_settings: Well Reports often have invisible grid lines
                # vertical_strategy="text" uses text alignment instead of lines
                tables = page.extract_tables(table_settings={
                    "vertical_strategy": "text",
                    "horizontal_strategy": "lines",
                    "intersection_tolerance": 5
                })
                
                for table_idx, table in enumerate(tables):
                    if not table or len(table) < 2:  # Skip empty or header-only
                        continue
                    
                    # First row is usually headers
                    headers = [str(cell).strip() if cell else f"col_{i}" 
                              for i, cell in enumerate(table[0])]
                    
                    # Process each data row
                    for row_idx, row in enumerate(table[1:], start=1):
                        if not any(row):  # Skip empty rows
                            continue
                        
                        # Create row dictionary (raw data for metadata)
                        row_dict = {headers[i]: str(cell).strip() if cell else ""
                                   for i, cell in enumerate(row)}
                        
                        # Create semantic summary for embedding
                        # Example: "Casing Table: 13 3/8 inch OD, 68 lbs/ft, set at 1331m MD"
                        semantic_summary = self._create_table_row_summary(
                            row_dict, page_num, table_idx
                        )
                        
                        # WHY: Semantic summary goes into vector store for search
                        # Raw data goes into metadata for exact LLM retrieval
                        table_chunks.append(TableChunk(
                            content=semantic_summary,
                            metadata={
                                "type": "table",
                                "page": page_num,
                                "table_index": table_idx,
                                "row_data": row_dict,  # Exact data for LLM
                                "headers": headers
                            },
                            page_num=page_num,
                            table_index=table_idx,
                            row_index=row_idx
                        ))
        
        print(f"✓ Extracted {len(table_chunks)} table rows from {pdf_path}")
        return table_chunks
    
    def _create_table_row_summary(self, row_dict: Dict[str, str], 
                                  page_num: int, table_idx: int) -> str:
        """
        Convert table row to semantic summary for embedding
        
        WHY: Embeddings work better with natural language than raw table cells
        Example Input: {"OD": "13 3/8", "Weight": "68", "Depth": "1331"}
        Example Output: "Casing Table: 13 3/8 inch casing, 68 lbs/ft weight, set at 1331 meters MD"
        """
        # Detect table type from common column names
        cols_lower = [k.lower() for k in row_dict.keys()]
        
        if any(term in ' '.join(cols_lower) for term in ['casing', 'od', 'pipe', 'tubular']):
            table_type = "Casing/Tubular"
        elif any(term in ' '.join(cols_lower) for term in ['cement', 'density', 'volume']):
            table_type = "Cementing"
        elif any(term in ' '.join(cols_lower) for term in ['fluid', 'mud', 'density']):
            table_type = "Drilling Fluid"
        elif any(term in ' '.join(cols_lower) for term in ['formation', 'lithology', 'geology']):
            table_type = "Geology"
        else:
            table_type = "Data Table"
        
        # Build natural language summary
        parts = [f"{table_type} (Page {page_num}):"]
        
        for key, value in row_dict.items():
            if value and value.strip():
                parts.append(f"{key}: {value}")
        
        return " | ".join(parts)
    
    def extract_narrative_text(self, pdf_path: str) -> List[TextChunk]:
        """
        Stream B: Extract narrative text with semantic chunking
        
        WHY: Narrative sections (geology, daily ops) need different handling than tables
        
        Returns:
            List of TextChunk objects
        """
        text_chunks = []
        section_pattern = re.compile(r'^(\d+\.)\s+([A-Z\s]+)$', re.MULTILINE)
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue
                
                # Clean headers/footers
                text = self._clean_text(text)
                
                # Detect section headers (e.g., "1. GENERAL DATA", "4. GEOLOGY")
                sections = self._split_by_sections(text, page_num)
                
                if sections:
                    # Split each section using RecursiveCharacterTextSplitter
                    for section_name, section_text in sections:
                        chunks = self.text_splitter.split_text(section_text)
                        for chunk in chunks:
                            if len(chunk.strip()) < 50:  # Skip tiny chunks
                                continue
                            
                            text_chunks.append(TextChunk(
                                content=chunk,
                                metadata={
                                    "type": "narrative",
                                    "page": page_num,
                                    "section": section_name
                                },
                                page_num=page_num,
                                section=section_name
                            ))
                else:
                    # No clear sections, just chunk the whole page
                    chunks = self.text_splitter.split_text(text)
                    for chunk in chunks:
                        if len(chunk.strip()) < 50:
                            continue
                        
                        text_chunks.append(TextChunk(
                            content=chunk,
                            metadata={
                                "type": "narrative",
                                "page": page_num,
                                "section": None
                            },
                            page_num=page_num,
                            section=None
                        ))
        
        print(f"✓ Extracted {len(text_chunks)} narrative chunks from {pdf_path}")
        return text_chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Remove headers/footers that add noise
        WHY: "Page 5 of 20" doesn't help with semantic search
        """
        for pattern in self.noise_patterns:
            text = re.sub(pattern, "", text, flags=re.MULTILINE)
        
        # Remove extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def _split_by_sections(self, text: str, page_num: int) -> List[tuple]:
        """
        Split text by major section headers
        WHY: Keeps related content together (all geology info in one section)
        
        Returns:
            List of (section_name, section_text) tuples
        """
        # Pattern for "1. GENERAL DATA", "4. GEOLOGY", etc.
        section_pattern = re.compile(r'^(\d+\.)\s+([A-Z\s]{3,})$', re.MULTILINE)
        
        matches = list(section_pattern.finditer(text))
        if not matches:
            return []
        
        sections = []
        for i, match in enumerate(matches):
            section_num = match.group(1)
            section_name = match.group(2).strip()
            start = match.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            
            section_text = text[start:end].strip()
            if section_text:
                sections.append((f"{section_num} {section_name}", section_text))
        
        return sections


# ============================================================================
# VECTOR STORE MANAGER
# ============================================================================

class HybridVectorStore:
    """
    Manages ChromaDB for hybrid table + narrative storage
    WHY ChromaDB: Fast, runs locally on disk, no server needed
    """
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        
        # WHY persist_directory: Data saved to disk, survives restarts
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create separate collections for tables and narrative
        # WHY: Different retrieval strategies for each
        self.table_collection = self.client.get_or_create_collection(
            name="well_tables",
            metadata={"description": "Table rows from Well Reports"}
        )
        
        self.text_collection = self.client.get_or_create_collection(
            name="well_narrative",
            metadata={"description": "Narrative text from Well Reports"}
        )
        
        print(f"✓ ChromaDB initialized at {persist_dir}")
    
    def add_table_chunks(self, chunks: List[TableChunk], embeddings: List[List[float]]):
        """
        Add table chunks to vector store
        WHY: Table rows stored with raw data in metadata
        """
        if not chunks:
            return
        
        self.table_collection.add(
            ids=[f"table_{c.page_num}_{c.table_index}_{c.row_index}" for c in chunks],
            embeddings=embeddings,
            documents=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks]
        )
        print(f"✓ Added {len(chunks)} table chunks to vector store")
    
    def add_text_chunks(self, chunks: List[TextChunk], embeddings: List[List[float]]):
        """
        Add narrative text chunks to vector store
        """
        if not chunks:
            return
        
        self.text_collection.add(
            ids=[f"text_{c.page_num}_{i}" for i, c in enumerate(chunks)],
            embeddings=embeddings,
            documents=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks]
        )
        print(f"✓ Added {len(chunks)} text chunks to vector store")
    
    def query_hybrid(self, query: str, embedder: SentenceTransformer, 
                    top_k_tables: int = 10, top_k_text: int = 10) -> Dict[str, Any]:
        """
        Query both table and narrative collections
        WHY: Some queries need tables (pipe specs), others need narrative (geology)
        
        Returns:
            Dict with 'tables' and 'text' results
        """
        # Embed the query
        query_embedding = embedder.encode([query])[0].tolist()
        
        # Query both collections
        table_results = self.table_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k_tables
        )
        
        text_results = self.text_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k_text
        )
        
        return {
            "tables": table_results,
            "text": text_results
        }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def build_rag_pipeline(pdf_paths: List[str], persist_dir: str = "./chroma_db"):
    """
    Build complete hybrid RAG pipeline
    
    Args:
        pdf_paths: List of paths to End of Well Report PDFs
        persist_dir: Where to store ChromaDB
    
    WHY THIS ARCHITECTURE:
        1. Tables extracted as complete rows (no chunking)
        2. Each row becomes a semantic summary for search
        3. Raw table data stored in metadata for exact retrieval
        4. Narrative text split semantically by sections
        5. Both streams embedded with same model (all-MiniLM-L6-v2)
        6. LLM can retrieve exact numbers from table metadata
    """
    print("="*70)
    print("HYBRID RAG PIPELINE FOR END OF WELL REPORTS")
    print("CPU-Optimized: 8-core, 16GB RAM, no GPU")
    print("="*70)
    print()
    
    # Initialize components
    parser = HybridParser()
    vector_store = HybridVectorStore(persist_dir)
    
    # Process each PDF
    for pdf_path in pdf_paths:
        print(f"\nProcessing: {pdf_path}")
        print("-" * 70)
        
        # Stream A: Extract tables
        print("\n[Stream A] Extracting tables...")
        table_chunks = parser.extract_tables(pdf_path)
        
        if table_chunks:
            # Embed table summaries
            print("  Embedding table summaries...")
            table_embeddings = parser.embedder.encode(
                [chunk.content for chunk in table_chunks],
                show_progress_bar=True
            )
            
            # Store in vector DB with metadata
            vector_store.add_table_chunks(table_chunks, table_embeddings)
        
        # Stream B: Extract narrative text
        print("\n[Stream B] Extracting narrative text...")
        text_chunks = parser.extract_narrative_text(pdf_path)
        
        if text_chunks:
            # Embed text chunks
            print("  Embedding narrative chunks...")
            text_embeddings = parser.embedder.encode(
                [chunk.content for chunk in text_chunks],
                show_progress_bar=True
            )
            
            # Store in vector DB
            vector_store.add_text_chunks(text_chunks, text_embeddings)
        
        print(f"\n✓ Completed: {pdf_path}")
    
    print("\n" + "="*70)
    print("PIPELINE BUILD COMPLETE")
    print("="*70)
    print(f"\nVector store saved to: {persist_dir}")
    print("\nYou can now query the system with:")
    print("  - Table queries: 'What is the pipe ID of the production casing?'")
    print("  - Narrative queries: 'Describe the geological formations encountered'")
    print("  - Hybrid queries: 'What casing was set in the Paleozoic formation?'")
    
    return vector_store, parser


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Process multiple Well Reports
    pdf_files = [
        "path/to/well_report_1.pdf",
        "path/to/well_report_2.pdf",
    ]
    
    # Build the pipeline
    vector_store, parser = build_rag_pipeline(
        pdf_paths=pdf_files,
        persist_dir="./chroma_db_hybrid"
    )
    
    # Example query
    query = "What is the pipe ID of the 9 5/8 inch casing?"
    results = vector_store.query_hybrid(
        query=query,
        embedder=parser.embedder,
        top_k_tables=10,  # Retrieve 10 table rows
        top_k_text=5      # Retrieve 5 text chunks
    )
    
    print(f"\n\nQuery: {query}")
    print("-" * 70)
    print("\nTable Results:")
    for i, doc in enumerate(results['tables']['documents'][0][:3]):
        print(f"\n{i+1}. {doc}")
    
    print("\n\nText Results:")
    for i, doc in enumerate(results['text']['documents'][0][:3]):
        print(f"\n{i+1}. {doc[:200]}...")
