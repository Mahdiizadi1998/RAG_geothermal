"""
RAG for Geothermal Wells - Main Application
Gradio UI with simplified workflow for document Q&A and summarization
"""

import gradio as gr
import sys
from pathlib import Path
import yaml
import logging
from typing import List, Dict, Optional, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import core agents only
from agents.ingestion_agent import IngestionAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.rag_retrieval_agent import RAGRetrievalAgent
from agents.chat_memory import ChatMemory
from agents.llm_helper import OllamaHelper

# Import hybrid database components
from agents.database_manager import WellDatabaseManager
from agents.hybrid_retrieval_agent import HybridRetrievalAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log Gradio version for troubleshooting
try:
    import importlib.metadata
    gradio_version = importlib.metadata.version('gradio')
    logger.info(f"Gradio version: {gradio_version}")
except Exception:
    logger.warning("Could not determine Gradio version")


class GeothermalRAGSystem:
    """
    Main RAG system orchestrator
    
    Handles:
    - Document ingestion and indexing
    - Multi-mode queries (Q&A, Summary, Extraction)
    - Parameter extraction with validation
    - Nodal analysis integration
    - Conversation memory
    """
    
    def __init__(self, config_path: str = None):
        """Initialize RAG system"""
        if config_path is None:
            config_path = Path(__file__).parent / 'config' / 'config.yaml'
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize agents
        logger.info("Initializing agents...")
        
        # Initialize database
        db_path = Path(__file__).parent / 'well_data.db'
        self.db = WellDatabaseManager(str(db_path))
        
        # Initialize core agents
        self.ingestion = IngestionAgent(database_manager=self.db)
        self.preprocessing = PreprocessingAgent(config_path)
        self.rag = RAGRetrievalAgent(config_path)
        self.llm = OllamaHelper(config_path)
        self.memory = ChatMemory()
        
        # Initialize hybrid retrieval (queries both DB and semantic)
        self.hybrid_retrieval = HybridRetrievalAgent(self.db, self.rag, config=config_path)
        
        self.indexed_documents = []
        self.llm_available = self.llm.is_available()
        if not self.llm_available:
            logger.warning("⚠️ Ollama not available - Q&A and Summary will use fallback mode")
        logger.info("✓ System initialized successfully")
    
    def ingest_and_index(self, pdf_files: List) -> str:
        """
        Ingest PDF files and index them
        
        Args:
            pdf_files: List of file objects from Gradio upload
            
        Returns:
            Status message
        """
        try:
            if not pdf_files:
                return "❌ No files uploaded"
            
            # Get file paths
            file_paths = [file.name for file in pdf_files]
            
            logger.info(f"Processing {len(file_paths)} files...")
            
            # Step 1: Ingestion
            documents = self.ingestion.process(file_paths)
            
            if not documents:
                return "❌ Failed to process any documents"
            
            # Step 1.5: Extract and store complete tables in database
            tables_stored = 0
            for doc in documents:
                if doc['wells']:  # Only process if well names detected
                    try:
                        stored = self.ingestion.process_and_store_complete_tables(
                            doc['filepath'],
                            doc['wells']
                        )
                        tables_stored += stored
                    except Exception as e:
                        logger.warning(f"Table extraction failed for {doc['filename']}: {e}")
            
            if tables_stored > 0:
                logger.info(f"✓ Stored {tables_stored} complete tables in database")
            
            # Step 2: Chunking (only narrative text, not tables)
            chunks_dict = self.preprocessing.process(documents)
            
            # Get statistics
            stats = self.preprocessing.get_chunk_statistics(chunks_dict)
            
            # Step 3: Indexing
            self.rag.index_chunks(chunks_dict)
            
            # Store document names
            self.indexed_documents = [doc['filename'] for doc in documents]
            self.memory.set_documents(self.indexed_documents)
            
            # Build status message
            status = "✓ Successfully indexed documents!\n\n"
            status += f"Documents processed: {len(documents)}\n"
            for doc in documents:
                status += f"  • {doc['filename']} ({doc['pages']} pages, wells: {', '.join(doc['wells']) if doc['wells'] else 'none detected'})\n"
            
            status += "\nChunking statistics:\n"
            
            # Show fine-grained strategy only
            if 'fine_grained' in stats:
                stat = stats['fine_grained']
                status += f"  • fine_grained: {stat['count']} chunks (avg {stat['avg_words']:.0f} words)\n"
            
            status += "\n✓ Ready for queries!"
            
            # Add database statistics
            if tables_stored > 0:
                status += f"\n\n**Database System:**\n"
                status += f"  • {tables_stored} complete tables stored in SQLite\n"
                status += f"  • All data types preserved (text and numbers)\n"
                status += f"  • Semantic search available for narrative content\n"
            
            return status
            
        except Exception as e:
            logger.error(f"Indexing failed: {str(e)}", exc_info=True)
            return f"❌ Error: {str(e)}"
    
    def _extract_well_name_from_query(self, query: str) -> Optional[str]:
        """Extract well name from query or use document context"""
        import re
        
        # Try to find well name in query
        well_pattern = re.compile(r'\b([A-Z]{2,10}-GT-\d{2}(?:-S\d+)?)\b')
        match = well_pattern.search(query)
        
        if match:
            return match.group(1)
        
        # If not in query, check indexed documents for well names
        for doc_name in self.indexed_documents:
            match = well_pattern.search(doc_name)
            if match:
                return match.group(1)
        
        return None
    
    def query(self, user_query: str, mode: str = "Q&A") -> Tuple[str, str]:
        """
        Process user query
        
        Args:
            user_query: User's question
            mode: "Q&A" or "Summary"
            
        Returns:
            Tuple of (response, debug_info)
        """
        try:
            if not self.indexed_documents:
                return "⚠️ Please index documents first", ""
            
            if not user_query.strip():
                return "⚠️ Please enter a query", ""
            
            logger.info(f"Processing query in mode '{mode}': {user_query[:50]}...")
            
            # Route to appropriate handler
            if mode == "Q&A":
                response, debug = self._handle_qa(user_query)
            elif mode == "Summary":
                response, debug = self._handle_summary(user_query)
            else:
                response = f"❌ Unknown mode: {mode}"
                debug = ""
            
            # Store in memory
            self.memory.add_exchange(user_query, response)
            
            return response, debug
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}", exc_info=True)
            return f"❌ Error processing query: {str(e)}", str(e)
    
    def _handle_qa(self, query: str) -> Tuple[str, str]:
        """Handle Q&A queries - always queries both database and semantic search"""
        # Get conversation context
        context = self.memory.get_context_string(last_n=3)
        enhanced_query = f"{context}\n\nCurrent question: {query}" if context else query
        
        # Extract well name
        well_name = self._extract_well_name_from_query(enhanced_query)
        
        # ALWAYS use hybrid retrieval (database + semantic)
        logger.info("⏳ Querying database and semantic search...")
        hybrid_result = self.hybrid_retrieval.retrieve(
            enhanced_query,
            well_name=well_name,
            top_k=10
        )
        
        # Get combined context
        combined_text = hybrid_result.get('combined_text', '')
        semantic_results = hybrid_result.get('semantic_results', [])
        database_results = hybrid_result.get('database_results', [])
        
        logger.info(f"Retrieved {len(database_results)} tables, {len(semantic_results)} text chunks")
        
        if not combined_text:
            return "⚠️ No relevant information found", ""
        
        # Generate answer with LLM if available
        if self.llm_available:
            try:
                logger.info("⏳ Generating answer with LLM...")
                
                # Format context chunks properly for LLM
                formatted_chunks = []
                
                # Add database results
                for db_result in database_results:
                    formatted_chunks.append({
                        'text': db_result.get('text', ''),
                        'metadata': {
                            'source_file': 'Database',
                            'page_numbers': [db_result.get('page', '?')]
                        }
                    })
                
                # Add semantic results (they already have proper format)
                for chunk in semantic_results:
                    # Ensure metadata exists
                    if 'metadata' not in chunk:
                        chunk['metadata'] = {'source_file': 'Document', 'page_numbers': ['?']}
                    formatted_chunks.append(chunk)
                
                # If no formatted chunks, create one from combined text
                if not formatted_chunks:
                    formatted_chunks = [{
                        'text': combined_text,
                        'metadata': {'source_file': 'Document', 'page_numbers': ['?']}
                    }]
                
                answer = self.llm.generate_answer(query, formatted_chunks)
                
                debug_info = f"Retrieved {len(database_results)} tables, {len(semantic_results)} chunks\n"
                debug_info += f"Generated answer using LLM ({self.llm.model_qa})\n"
                
                return answer, debug_info
                
            except Exception as e:
                logger.error(f"LLM answer generation failed: {str(e)}")
                import traceback
                traceback.print_exc()
                return f"Error generating answer: {str(e)}", str(e)
        
        # Fallback: return raw context
        debug_info = f"Retrieved {len(database_results)} tables, {len(semantic_results)} chunks (LLM not available)\n"
        return combined_text[:2000], debug_info
    
    def _handle_summary(self, query: str) -> Tuple[str, str]:
        """
        Generate comprehensive well summary by extracting 8 key data types
        
        8 Data Types:
        1. General Data: Well Name, License, Well Type, Location, Coordinates (X/Y), Operator, Rig Name, Target Formation
        2. Drilling Timeline: Spud Date, End of Operations, Total Days
        3. Depths: TD (mAH), TVD, Sidetrack Start Depth
        4. Casing & Tubulars: Type, OD, Weight, Grade, Connection, Pipe ID (Nominal + Drift), Top/Bottom Depths
        5. Cementing: Lead/Tail volumes, Densities, TOC
        6. Fluids: Hole Size, Fluid Type, Density Range
        7. Geology: Formation names, depths, lithology, notes (gas shows, instability)
        8. Incidents: Gas peaks, stuck pipe events, mud losses
        """
        well_name = self._extract_well_name_from_query(query)
        
        if not well_name:
            return "⚠️ Please specify a well name for summary", ""
        
        logger.info(f"📝 Generating summary for {well_name}")
        
        # Get all data sources
        tables = self.db.get_complete_tables(well_name)
        logger.info(f"Retrieved {len(tables)} tables from database")
        
        # Organize tables by type
        tables_by_type = {}
        for table in tables:
            table_type = table.get('table_type', 'General')
            if table_type not in tables_by_type:
                tables_by_type[table_type] = []
            tables_by_type[table_type].append(table)
        
        summary_parts = []
        summary_parts.append(f"# Well Summary: {well_name}\n")
        
        # Define extraction tasks for each section
        extraction_tasks = [
            {
                'title': '## 1. General Data',
                'table_types': ['General', 'auto_detected'],
                'search_query': f"{well_name} well name license type location coordinates operator rig target formation",
                'extraction_prompt': f"""Extract the following information FOR WELL {well_name} ONLY:
- Well Name
- License Number
- Well Type
- Location/Field
- Coordinates (X/Y or Lat/Long)
- Operator
- Rig Name
- Target Formation

CRITICAL: If the context contains data for multiple wells (e.g., comparison tables with columns for different wells), extract ONLY the data for {well_name}. Ignore all other wells.

Format as a bullet list. Only include information that is found."""
            },
            {
                'title': '## 2. Drilling Timeline',
                'table_types': ['Timeline', 'General'],
                'search_query': f"{well_name} spud date completion date end operations drilling timeline duration days",
                'extraction_prompt': f"""Extract the following timeline information FOR WELL {well_name} ONLY:
- Spud Date
- End of Operations / Completion Date
- Total Days / Duration

CRITICAL: If the context contains data for multiple wells, extract ONLY the data for {well_name}. Ignore all other wells.

Format as a bullet list. Only include information that is found."""
            },
            {
                'title': '## 3. Depths',
                'table_types': ['Depths', 'General'],
                'search_query': f"{well_name} total depth TD TVD measured depth true vertical sidetrack kickoff",
                'extraction_prompt': f"""Extract depth information FOR WELL {well_name} ONLY:
- TD (Total Depth in mAH)
- TVD (True Vertical Depth)
- Sidetrack Start Depth (if applicable)

CRITICAL: If the context contains data for multiple wells, extract ONLY the data for {well_name}. Ignore all other wells.

Format as a bullet list. Only include information that is found."""
            },
            {
                'title': '## 4. Casing & Tubulars',
                'table_types': ['Casing'],
                'search_query': f"{well_name} casing tubing liner conductor surface intermediate production size OD ID weight grade",
                'extraction_prompt': f"""Extract casing and tubular information FOR WELL {well_name} ONLY. For EACH casing string, extract:
- Type (Conductor, Surface, Intermediate, Production, Liner, etc.)
- OD (Outside Diameter in inches)
- Weight (lb/ft)
- Grade (e.g., K-55, L-80, P-110)
- Connection type
- **Pipe ID - Both Nominal ID AND Drift ID** (very important)
- Top Depth (mAH)
- Bottom Depth (mAH)

CRITICAL: If the context contains comparison tables with multiple wells (e.g., columns like "ABC-GT-01" and "XYZ-GT-02"), extract ONLY the {well_name} column data. Ignore all other wells.

Format as a numbered list, one entry per casing string. Include Nominal and Drift ID for each string."""
            },
            {
                'title': '## 5. Cementing',
                'table_types': ['Cementing'],
                'search_query': f"{well_name} cement lead tail slurry volume density TOC top of cement",
                'extraction_prompt': f"""Extract cementing information FOR WELL {well_name} ONLY for each cement job:
- Lead Slurry Volume
- Tail Slurry Volume
- Lead Density (sg or ppg)
- Tail Density (sg or ppg)
- TOC (Top of Cement)

CRITICAL: If the context contains data for multiple wells, extract ONLY the data for {well_name}. Ignore all other wells.

Format as a bullet list or numbered list if multiple jobs. Only include information that is found."""
            },
            {
                'title': '## 6. Drilling Fluids',
                'table_types': ['Fluids'],
                'search_query': f"{well_name} drilling fluid mud type density hole size water oil synthetic",
                'extraction_prompt': f"""Extract drilling fluid information FOR WELL {well_name} ONLY:
- Hole Size (inches)
- Fluid Type (WBM, OBM, SBM, etc.)
- Density Range (sg or ppg)

CRITICAL: If the context contains data for multiple wells, extract ONLY the data for {well_name}. Ignore all other wells.

Format as a bullet list. List multiple hole sections if present. Only include information that is found."""
            },
            {
                'title': '## 7. Geology & Formations',
                'table_types': ['Formations', 'General'],
                'search_query': f"{well_name} formation geology lithology gas show instability overpressure shale sandstone",
                'extraction_prompt': f"""Extract geological information FOR WELL {well_name} ONLY:
- Formation names and depths
- Lithology (rock types)
- Key notes: Gas shows, instability issues, overpressure zones

CRITICAL: If the context contains data for multiple wells, extract ONLY the data for {well_name}. Ignore all other wells.

Format as a bullet list. Only include information that is found."""
            },
            {
                'title': '## 8. Incidents & Issues',
                'table_types': ['Incidents'],
                'search_query': f"{well_name} incident stuck pipe gas kick loss circulation mud loss pressure",
                'extraction_prompt': f"""Extract incidents and issues FOR WELL {well_name} ONLY:
- Gas peaks/kicks
- Stuck pipe events
- Mud losses / Lost circulation
- Other significant drilling problems

CRITICAL: If the context contains data for multiple wells, extract ONLY the data for {well_name}. Ignore all other wells.

Format as a bullet list. Only include information that is found."""
            }
        ]
        
        # Process each section
        for task in extraction_tasks:
            logger.info(f"Processing: {task['title']}")
            
            # Gather relevant tables
            relevant_tables = []
            for table_type in task['table_types']:
                if table_type in tables_by_type:
                    relevant_tables.extend(tables_by_type[table_type])
            
            # Get narrative context from semantic search
            semantic_chunks = self.rag.retrieve(task['search_query'], top_k=3)
            
            # Combine all context
            context_parts = []
            
            # Add table data
            for table in relevant_tables[:5]:  # Limit to 5 tables per section
                table_text = self._format_table_for_extraction(table)
                context_parts.append(table_text)
            
            # Add semantic context
            for chunk in semantic_chunks:
                context_parts.append(chunk['text'][:500])
            
            if not context_parts:
                # No data found for this section - skip it
                logger.info(f"No data found for {task['title']}")
                continue
            
            combined_context = "\n\n".join(context_parts)
            
            # Use LLM to extract key information
            if self.llm_available:
                try:
                    extraction = self.llm.extract_information(
                        task['extraction_prompt'],
                        combined_context
                    )
                    
                    if extraction and len(extraction.strip()) > 10:
                        summary_parts.append(f"\n{task['title']}")
                        summary_parts.append(extraction)
                    else:
                        logger.info(f"No relevant data extracted for {task['title']}")
                except Exception as e:
                    logger.error(f"LLM extraction failed for {task['title']}: {str(e)}")
                    # Fallback: show abbreviated raw data
                    summary_parts.append(f"\n{task['title']}")
                    summary_parts.append(combined_context[:800])
            else:
                # No LLM: show abbreviated raw data
                summary_parts.append(f"\n{task['title']}")
                summary_parts.append(combined_context[:600])
        
        final_summary = "\n\n".join(summary_parts)
        debug_info = f"Generated from {len(tables)} tables across {len(tables_by_type)} types"
        
        return final_summary, debug_info
    
    def _format_table_for_extraction(self, table: Dict) -> str:
        """Format table data for LLM extraction (compact format)"""
        import json
        
        headers = json.loads(table.get('headers_json', '[]'))
        rows = json.loads(table.get('rows_json', '[]'))
        
        # Create compact text representation
        table_text = f"Table from page {table['source_page']} ({table.get('table_type', 'Unknown')} type):\n"
        
        # Limit rows to prevent context overflow
        max_rows = 15
        for i, row in enumerate(rows[:max_rows]):
            if i == 0 or len(headers) == 0:
                # First row or no headers: treat as data row
                row_text = " | ".join(str(cell) for cell in row)
            else:
                # Create key-value pairs
                row_pairs = []
                for header, value in zip(headers, row):
                    if value and str(value).strip():
                        row_pairs.append(f"{header}: {value}")
                row_text = ", ".join(row_pairs)
            
            table_text += f"  {row_text}\n"
        
        if len(rows) > max_rows:
            table_text += f"  ... ({len(rows) - max_rows} more rows)\n"
        
        return table_text
    
    def _extract_well_name(self, query: str) -> Optional[str]:
        """Extract well name from query"""
        import re
        match = re.search(r'\b([A-Z]{2,10}-GT-\d{2}(?:-S\d+)?)\b', query)
        return match.group(1) if match else None
    
    def clear_index(self) -> str:
        """Clear all indexed data"""
        try:
            self.rag.clear_all_collections()
            self.indexed_documents = []
            self.memory.clear()
            return "✓ Index cleared successfully"
        except Exception as e:
            return f"❌ Error clearing index: {str(e)}"


# ============================================================================
# GRADIO UI
# ============================================================================

def create_ui():
    """Create Gradio UI"""
    
    # Initialize system
    system = GeothermalRAGSystem()
    
    # Create Gradio app with theme if supported (Gradio >=4.0)
    try:
        app = gr.Blocks(title="RAG for Geothermal Wells", theme=gr.themes.Soft())
    except TypeError:
        # Fallback for older Gradio versions without theme support
        app = gr.Blocks(title="RAG for Geothermal Wells")
    
    with app:
        gr.Markdown("# 🌋 RAG for Geothermal Wells")
        gr.Markdown("Intelligent document analysis for geothermal well completion reports")
        
        with gr.Tab("📁 Document Upload"):
            with gr.Row():
                with gr.Column():
                    file_upload = gr.File(
                        label="Upload PDF Reports",
                        file_count="multiple",
                        file_types=[".pdf"]
                    )
                    index_btn = gr.Button("Index Documents", variant="primary")
                    clear_btn = gr.Button("Clear Index", variant="secondary")
                
                with gr.Column():
                    index_status = gr.Textbox(
                        label="Indexing Status",
                        lines=15,
                        interactive=False
                    )
            
            # Button actions
            index_btn.click(
                fn=system.ingest_and_index,
                inputs=[file_upload],
                outputs=[index_status]
            )
            
            clear_btn.click(
                fn=system.clear_index,
                outputs=[index_status]
            )
        
        with gr.Tab("💬 Query Interface"):
            with gr.Row():
                with gr.Column(scale=2):
                    query_mode = gr.Radio(
                        choices=["Q&A", "Summary"],
                        value="Q&A",
                        label="Query Mode"
                    )
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., What is the total depth of well [WELL-NAME]?",
                        lines=3
                    )
                    
                    query_btn = gr.Button("Submit Query", variant="primary")
                    
                    gr.Markdown("""
                    **Query Mode Guide:**
                    - **Q&A**: Ask specific questions about the documents (with conversation memory)
                    - **Summary**: Get a comprehensive summary with data from database tables and document text
                    
                    **Example queries:**
                    - Q&A: "What is the casing design for [WELL-NAME]?"
                    - Q&A: "What was the drilling fluid density?"
                    - Summary: "Summarize well [WELL-NAME]"
                    - Summary: "Generate summary for [WELL-NAME]"
                    
                    **Summary includes 8 data types:**
                    1. General Data (Operator, Location, Rig)
                    2. Timeline (Spud, Completion, Total Days)
                    3. Depths (TD, TVD)
                    4. Casing & Tubulars (all tables)
                    5. Cementing
                    6. Fluids
                    7. Geology/Formations
                    8. Incidents
                    """)
                    
                    # Conversation history
                    with gr.Accordion("💭 Conversation History", open=False):
                        history_display = gr.Markdown("No conversation yet...")
                
                with gr.Column(scale=3):
                    response_output = gr.Markdown(label="Response")
                    
                    with gr.Accordion("Debug Information", open=False):
                        debug_output = gr.Textbox(
                            label="Debug Info",
                            lines=10,
                            interactive=False
                        )
            
            # Define helper function for updating history
            def query_with_history_update(query, mode):
                response, debug = system.query(query, mode)
                
                # Get updated history
                history = system.memory.get_history(last_n=5)
                if history:
                    history_md = "## Recent Conversations\n\n"
                    for i, exchange in enumerate(history[-5:], 1):
                        history_md += f"**{i}. User:** {exchange['user'][:100]}...\n\n"
                        history_md += f"**Assistant:** {exchange['assistant'][:200]}...\n\n"
                        history_md += "---\n\n"
                else:
                    history_md = "No conversation yet..."
                
                return response, debug, history_md
            
            query_btn.click(
                fn=query_with_history_update,
                inputs=[query_input, query_mode],
                outputs=[response_output, debug_output, history_display]
            )
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## RAG System for Geothermal Wells
            
            This system provides intelligent analysis of geothermal well completion reports using:
            
            ### Features
            - **Hybrid Retrieval**: Always searches both database tables AND document text
            - **Complete Table Storage**: Stores entire tables with all columns in SQLite database
            - **Fine-Grained Chunking**: Optimized 500-word chunks with 150-word overlap
            - **8-Data-Type Summary**: Comprehensive summaries covering all aspects of well data
            - **Conversation Memory**: Multi-turn Q&A with context retention
            - **LLM-Powered Answers**: Uses Ollama for natural language generation
            
            ### Summary System (8 Data Types)
            When you request a summary, the system retrieves:
            1. **General Data**: Well name, operator, location, rig
            2. **Timeline**: Spud date, end date, total days
            3. **Depths**: Total depth, TVD, sidetrack depths
            4. **Casing & Tubulars**: Complete tables with all specifications
            5. **Cementing**: Volumes, densities, TOC
            6. **Fluids**: Hole sizes, fluid types, densities
            7. **Geology**: Formation names, depths, lithology
            8. **Incidents**: Gas peaks, stuck pipe, losses
            
            ### Architecture
            - **Vector DB**: ChromaDB (single collection)
            - **Relational DB**: SQLite (complete tables)
            - **Embeddings**: nomic-embed-text (384 dims)
            - **LLM**: Ollama (llama3/llama3.1)
            - **PDF Processing**: PyMuPDF + pdfplumber
            - **Chunking**: Fine-grained only (500 words, 150 overlap)
            
            ### Usage Tips
            1. Upload PDF completion reports in the "Document Upload" tab
            2. Wait for indexing to complete (typically 20-40 seconds)
            3. Switch to "Query Interface"
            4. For Q&A: Ask specific questions about well data
            5. For Summary: Include well name (e.g., "Summarize well [WELL-NAME]")
            6. System always searches both database and document text for best results
            
            Developed with ❤️ for geothermal engineering
            """)
    
    return app


if __name__ == "__main__":
    # Load UI config
    config_path = Path(__file__).parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    ui_config = config.get('ui', {})
    
    # Create and launch app
    app = create_ui()
    app.launch(
        server_name=ui_config.get('server_name', '0.0.0.0'),
        server_port=ui_config.get('port', 7860),
        share=ui_config.get('share', False)
    )
