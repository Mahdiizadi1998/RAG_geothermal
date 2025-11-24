"""
RAG for Geothermal Wells - Main Application
Gradio UI with agentic workflow for document Q&A, summarization, and parameter extraction
"""

import gradio as gr
import sys
from pathlib import Path
import yaml
import logging
from typing import List, Dict, Optional, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import agents
from agents.ingestion_agent import IngestionAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.rag_retrieval_agent import RAGRetrievalAgent
from agents.parameter_extraction_agent import ParameterExtractionAgent
from agents.validation_agent import ValidationAgent
from agents.chat_memory import ChatMemory
from agents.ensemble_judge_agent import EnsembleJudgeAgent
from agents.llm_helper import OllamaHelper
from agents.well_summary_agent import WellSummaryAgent
from models.nodal_runner import NodalAnalysisRunner

# Import new validation agents
from agents.query_analysis_agent import QueryAnalysisAgent
from agents.fact_verification_agent import FactVerificationAgent
from agents.physical_validation_agent import PhysicalValidationAgent
from agents.missing_data_agent import MissingDataAgent
from agents.confidence_scorer import ConfidenceScorerAgent

# Import hybrid database components
from agents.database_manager import WellDatabaseManager
from agents.enhanced_table_parser import EnhancedTableParser
from agents.template_selector import TemplateSelectorAgent
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
        
        # Initialize hybrid database system
        db_path = Path(__file__).parent / 'well_data.db'
        self.db = WellDatabaseManager(str(db_path))
        self.table_parser = EnhancedTableParser()
        
        self.ingestion = IngestionAgent(
            database_manager=self.db,
            table_parser=self.table_parser
        )
        self.preprocessing = PreprocessingAgent(config_path)
        self.rag = RAGRetrievalAgent(config_path)
        self.extraction = ParameterExtractionAgent(
            enable_llm_fallback=self.config['extraction']['enable_llm_fallback']
        )
        self.validation = ValidationAgent(config_path)
        self.memory = ChatMemory()
        self.judge = EnsembleJudgeAgent()
        self.llm = OllamaHelper(config_path)
        self.well_summary_agent = WellSummaryAgent(llm_helper=self.llm)
        self.nodal_runner = NodalAnalysisRunner()
        
        # Initialize new validation agents
        self.query_analyzer = QueryAnalysisAgent(self.config)
        self.fact_verifier = FactVerificationAgent(self.config, database_manager=self.db)
        self.physical_validator = PhysicalValidationAgent(self.config)
        self.missing_data_agent = MissingDataAgent(self.config)
        self.confidence_scorer = ConfidenceScorerAgent(self.config)
        
        # Initialize hybrid retrieval and template selector
        self.template_selector = TemplateSelectorAgent(self.db)
        self.hybrid_retrieval = HybridRetrievalAgent(self.db, self.rag)
        
        # Store pending extraction for confirmation
        self.pending_extraction = None
        self.pending_clarifications = []  # Store clarification questions
        
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
            
            # Step 1.5: Extract and store tables in database
            tables_stored = 0
            for doc in documents:
                if doc['wells']:  # Only process if well names detected
                    try:
                        stored = self.ingestion.process_and_store_tables(
                            doc['filepath'],
                            doc['wells']
                        )
                        tables_stored += stored
                    except Exception as e:
                        logger.warning(f"Table extraction failed for {doc['filename']}: {e}")
            
            if tables_stored > 0:
                logger.info(f"✓ Stored {tables_stored} table records in database")
            
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
            
            # Show base strategies
            base_strategies = ['factual_qa', 'technical_extraction', 'summary']
            for strategy in base_strategies:
                if strategy in stats:
                    stat = stats[strategy]
                    status += f"  • {strategy}: {stat['count']} chunks (avg {stat['avg_words']:.0f} words)\n"
            
            # Show hybrid strategies if present
            hybrid_strategies = ['fine_grained', 'coarse_grained']
            hybrid_present = any(s in stats for s in hybrid_strategies)
            
            if hybrid_present:
                status += "\n  **Hybrid chunking enabled:**\n"
                for strategy in hybrid_strategies:
                    if strategy in stats:
                        stat = stats[strategy]
                        status += f"  • {strategy}: {stat['count']} chunks (avg {stat['avg_words']:.0f} words)\n"
            
            status += "\n✓ Ready for queries!"
            if hybrid_present:
                status += " (Using multi-granularity hybrid chunking for better retrieval)"
            
            # Add database statistics
            if tables_stored > 0:
                status += f"\n\n**Hybrid Database System:**\n"
                status += f"  • {tables_stored} table records stored in SQLite\n"
                status += f"  • Exact numerical data ready for queries\n"
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
        Process user query with automatic mode detection
        
        Args:
            user_query: User's question
            mode: "Q&A", "Summary", or "Extract & Analyze"
            
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
            elif mode == "Extract & Analyze":
                response, debug = self._handle_extraction(user_query)
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
        """Handle Q&A queries with LLM-generated answers and fact verification"""
        # Analyze query
        query_analysis = self.query_analyzer.analyze(query)
        logger.info(f"Query type: {query_analysis.query_type}, Priority: {query_analysis.priority}")
        
        # Get conversation context for better answers
        context = self.memory.get_context_string(last_n=3)
        
        # Enhance query with context if available
        enhanced_query = query
        if context:
            enhanced_query = f"{context}\n\nCurrent question: {query}"
        
        # Retrieve relevant chunks using hybrid granularity (multiple chunk sizes)
        retrieval_result = self.rag.retrieve_hybrid_granularity(enhanced_query, mode='qa')
        chunks = retrieval_result['chunks']
        logger.info(f"Retrieved {len(chunks)} chunks using hybrid granularity strategy")
        
        if not chunks:
            return "⚠️ No relevant information found in documents", ""
        
        # Calculate source quality
        source_quality = self.confidence_scorer.calculate_source_quality(chunks, top_k=5)
        
        # Generate answer using LLM if available
        if self.llm_available:
            try:
                logger.info("⏳ Generating answer with LLM...")
                answer = self.llm.generate_answer(query, chunks)
                
                # Fact verification
                logger.info("⏳ Verifying facts...")
                verification = self.fact_verifier.verify(answer, chunks)
                
                # Calculate confidence
                confidence = self.confidence_scorer.calculate_confidence(
                    source_quality=source_quality,
                    fact_verification=verification.overall_confidence,
                    consistency=0.85,  # Default - no extraction to check
                    context={'query_type': 'qa'}
                )
                
                # Build response with warnings
                response_parts = []
                
                # Add confidence header
                if confidence.recommendation == 'high':
                    response_parts.append("✅ **HIGH CONFIDENCE ANSWER**\n")
                elif confidence.recommendation == 'review':
                    response_parts.append("⚠️ **REVIEW RECOMMENDED** - Verify before critical use\n")
                else:
                    response_parts.append("⚠️ **LOW CONFIDENCE** - Results may be unreliable\n")
                
                response_parts.append(answer)
                
                # Add verification warnings
                if verification.warnings:
                    response_parts.append("\n\n**⚠️ Verification Warnings:**")
                    for warning in verification.warnings:
                        response_parts.append(f"- {warning}")
                
                # Add sources
                sources = "\n\n**Sources:**\n"
                seen_sources = set()
                for i, chunk in enumerate(chunks[:5], 1):
                    source = chunk['metadata'].get('source_file', 'unknown')
                    pages = chunk['metadata'].get('page_numbers', ['?'])
                    page_str = ', '.join(map(str, pages[:3]))
                    source_key = f"{source}-{page_str}"
                    
                    if source_key not in seen_sources:
                        sources += f"- {source}, pages {page_str}\n"
                        seen_sources.add(source_key)
                
                response_parts.append(sources)
                
                # Add confidence breakdown
                response_parts.append(f"\n**Confidence Assessment:**\n{confidence.explanation}")
                
                response_text = "\n".join(response_parts)
                
                # Debug info
                debug_info = f"Retrieved {len(chunks)} chunks\n"
                debug_info += f"Generated answer using LLM ({self.llm.model_qa})\n"
                debug_info += f"Fact verification: {verification.support_rate*100:.0f}% claims supported\n"
                debug_info += f"Overall confidence: {confidence.overall*100:.0f}%\n"
                debug_info += f"Recommendation: {confidence.recommendation.upper()}\n"
                
                return response_text, debug_info
                
            except Exception as e:
                logger.error(f"LLM answer generation failed: {str(e)}")
                # Fall through to fallback mode
        
        # Fallback: return excerpts from chunks
        response_parts = []
        response_parts.append(f"Based on the indexed documents:\n")
        
        for i, chunk in enumerate(chunks[:4], 1):
            response_parts.append(f"\n**Excerpt {i}** (from {chunk['metadata'].get('source_file', 'unknown')}," 
                                f" page {chunk['metadata'].get('page_numbers', ['?'])[0]}):")
            
            excerpt = chunk['text'][:400].strip()
            if len(chunk['text']) > 400:
                excerpt += "..."
            response_parts.append(f"{excerpt}\n")
        
        response_text = "\n".join(response_parts)
        quality = self.judge.evaluate_response(query, response_text, chunks)
        
        debug_info = f"Retrieved {len(chunks)} chunks (fallback mode - no LLM)\n"
        debug_info += f"Quality score: {quality['quality_score']:.2f}\n"
        
        return response_text, debug_info
    
    def _handle_summary(self, query: str) -> Tuple[str, str]:
        """Handle document summarization using 3-pass extraction system"""
        # Analyze query to check if user wants detailed End of Well Summary
        query_analysis = self.query_analyzer.analyze(query)
        query_lower = query.lower()
        
        # Check if user wants detailed/professional End of Well Summary
        detailed_summary = any(kw in query_lower for kw in [
            'end of well', 'eow', 'detailed', 'professional', 'comprehensive',
            'completion report', 'well report', 'drilling report'
        ])
        
        # If detailed summary requested and we have indexed documents, use 3-pass system
        if detailed_summary and self.indexed_documents:
            logger.info("📝 Generating detailed End of Well Summary using 3-pass system...")
            
            try:
                # Get the most recent PDF file path
                pdf_path = None
                document_data = None
                
                if self.indexed_documents:
                    # Use first indexed document (can be enhanced to select specific well)
                    doc_info = self.indexed_documents[0]
                    pdf_path = doc_info.get('filepath')
                    document_data = doc_info
                
                if not pdf_path:
                    logger.warning("No PDF path available for 3-pass summary")
                    # Fall through to standard summary
                else:
                    # Generate 3-pass summary
                    summary_result = self.well_summary_agent.generate_summary(
                        pdf_path=pdf_path,
                        document_data=document_data
                    )
                    
                    # Build response
                    response_parts = []
                    
                    # Add confidence header
                    confidence = summary_result['confidence']
                    if confidence >= 0.8:
                        response_parts.append("✅ **HIGH CONFIDENCE END OF WELL SUMMARY**\n")
                    elif confidence >= 0.6:
                        response_parts.append("⚠️ **REVIEW RECOMMENDED**\n")
                    else:
                        response_parts.append("⚠️ **LOW CONFIDENCE - VERIFY DATA**\n")
                    
                    # Add the summary report
                    response_parts.append(summary_result['summary_report'])
                    
                    # Add confidence details
                    response_parts.append(f"\n\n**Confidence Score:** {confidence*100:.0f}%")
                    response_parts.append("\n*Generated using 3-pass extraction: Metadata → Technical Specs → Narrative*")
                    
                    summary_text = "\n".join(response_parts)
                    
                    # Debug info
                    debug_info = f"3-Pass Summary System Used\n"
                    debug_info += f"Pass 1 - Metadata: {len(summary_result['metadata'])} fields\n"
                    debug_info += f"Pass 2 - Casing Program: {len(summary_result['technical_specs'].get('casing_program', []))} strings\n"
                    debug_info += f"Pass 3 - Narrative: {len(summary_result['narrative'])} sections\n"
                    debug_info += f"Confidence: {confidence*100:.0f}%\n"
                    debug_info += f"PDF: {pdf_path}\n"
                    
                    return summary_text, debug_info
                    
            except Exception as e:
                logger.error(f"3-pass summary generation failed: {str(e)}")
                logger.exception(e)
                # Fall through to standard summary
        
        # Standard summary approach (original code)
        logger.info("📝 Generating standard summary...")
        
        # Extract target word count (use analysis result or default)
        target_words = query_analysis.target_word_count
        if target_words is None:
            target_words = self.config.get('summarization', {}).get('default_words', 200)
        
        # Handle null default (no limit) - use 0 as signal for comprehensive mode
        if target_words is None:
            target_words = 0
            logger.info(f"Target: No word limit (comprehensive summary)")
        else:
            logger.info(f"Target: {target_words} words")
            # Ensure reasonable range only if a limit is specified
            target_words = max(50, min(target_words, 1000))
        
        # Extract focus area from query analysis
        focus = ', '.join(query_analysis.detected_focus) if query_analysis.detected_focus else None
        
        # Extract well name for hybrid retrieval
        well_name = self._extract_well_name_from_query(query)
        
        # Use hybrid retrieval if well name is known
        if well_name:
            logger.info(f"Using hybrid retrieval for well: {well_name}")
            hybrid_result = self.hybrid_retrieval.retrieve(
                query=query,
                well_name=well_name,
                mode='hybrid',  # Use both database and semantic search
                top_k=15
            )
            
            # Extract chunks from hybrid result
            chunks = hybrid_result.get('semantic_results', [])
            
            # Add database info as context
            db_context = []
            for db_result in hybrid_result.get('database_results', []):
                db_context.append({
                    'text': db_result.get('text', ''),
                    'metadata': {
                        'source_file': 'Database',
                        'type': db_result.get('type', 'unknown')
                    }
                })
            
            # Combine database context with semantic chunks (database first for priority)
            chunks = db_context + chunks
            logger.info(f"Hybrid retrieval: {len(db_context)} DB results + {len(chunks)-len(db_context)} semantic chunks")
        else:
            # Fallback to traditional retrieval if no well name
            logger.info("No well name detected, using traditional semantic retrieval")
            retrieval_result = self.rag.retrieve_hybrid_granularity(query, mode='summary')
            chunks = retrieval_result['chunks']
            logger.info(f"Retrieved {len(chunks)} chunks using hybrid granularity strategy")
        
        if not chunks:
            return "⚠️ No content found for summarization", ""
        
        # Calculate source quality
        source_quality = self.confidence_scorer.calculate_source_quality(chunks, top_k=10)
        
        # Generate summary using LLM if available
        if self.llm_available:
            try:
                logger.info("⏳ Generating summary with citations and strict word count enforcement...")
                summary = self.llm.generate_summary(chunks, target_words, focus)
                
                # Count actual words
                actual_words = len(summary.split())
                logger.info(f"✓ Generated {actual_words} words (target: {target_words})")
                
                # Perform fact verification if enabled and enough chunks
                summary_config = self.config.get('summarization', {})
                enable_verification = summary_config.get('enable_verification', True)
                min_chunks = summary_config.get('min_chunks_for_verification', 5)
                
                fact_verification_score = 1.0  # Default to perfect if not verified
                if enable_verification and len(chunks) >= min_chunks:
                    logger.info("⏳ Performing fact verification on summary...")
                    try:
                        # Fixed: Use verify() method not verify_facts()
                        verification_result = self.fact_verifier.verify(summary, chunks)
                        fact_verification_score = verification_result.overall_confidence
                        logger.info(f"✓ Fact verification complete: {fact_verification_score*100:.0f}% ({verification_result.support_rate*100:.0f}% supported)")
                    except Exception as e:
                        logger.warning(f"Fact verification failed: {e}")
                        import traceback
                        traceback.print_exc()
                        fact_verification_score = 0.5  # Assume moderate confidence
                else:
                    logger.info("Skipping fact verification (not enabled or insufficient chunks)")
                
                # Calculate confidence with verification
                confidence = self.confidence_scorer.calculate_confidence(
                    source_quality=source_quality,
                    fact_verification=fact_verification_score,
                    completeness=0.9,  # High for summaries
                    consistency=0.85,
                    context={'query_type': 'summary', 'word_count': actual_words}
                )
                
                # Build response with metadata
                response_parts = []
                
                # Add confidence header
                if confidence.recommendation == 'high':
                    response_parts.append("✅ **HIGH CONFIDENCE SUMMARY**\n")
                elif confidence.recommendation == 'review':
                    response_parts.append("⚠️ **REVIEW RECOMMENDED**\n")
                else:
                    response_parts.append("⚠️ **LOW CONFIDENCE**\n")
                
                response_parts.append(summary)
                
                # Add warnings if any
                if confidence.warnings:
                    response_parts.append("\n\n**⚠️ Warnings:**")
                    for warning in confidence.warnings:
                        response_parts.append(f"- {warning}")
                
                # Add metadata
                sources = set(chunk['metadata'].get('source_file', 'unknown') for chunk in chunks[:10])
                metadata = f"\n\n---\n*Summary of {', '.join(sources)} ({len(chunks)} sections, {actual_words} words)*"
                response_parts.append(metadata)
                
                # Add confidence breakdown
                response_parts.append(f"\n**Confidence Assessment:**\n{confidence.explanation}")
                
                summary_text = "\n".join(response_parts)
                
                # Debug info
                debug_info = f"Summarized from {len(chunks)} chunks\n"
                debug_info += f"Target words: {target_words}, Actual: {actual_words}\n"
                debug_info += f"Focus: {focus or 'general'}\n"
                debug_info += f"Generated using LLM ({self.llm.model_summary})\n"
                debug_info += f"Confidence: {confidence.overall*100:.0f}%\n"
                
                return summary_text, debug_info
                
            except Exception as e:
                logger.error(f"LLM summary generation failed: {str(e)}")
                # Fall through to fallback mode
        
        # Fallback: return bullet points from chunks
        summary_parts = []
        summary_parts.append(f"**Summary** (from {len(chunks)} sections):\n")
        
        word_count = 0
        for chunk in chunks[:12]:
            # Take sentences until we reach target word count
            sentences = chunk['text'].split('. ')
            for sentence in sentences:
                if word_count + len(sentence.split()) > target_words:
                    break
                summary_parts.append(f"• {sentence.strip()}.")
                word_count += len(sentence.split())
            
            if word_count >= target_words:
                break
        
        summary_text = "\n".join(summary_parts)
        summary_text += f"\n\n---\n*Fallback summary (~{word_count} words, Ollama not available)*"
        
        debug_info = f"Summarized from {len(chunks)} chunks (fallback mode)\n"
        debug_info += f"Target words: {target_words}\n"
        
        return summary_text, debug_info
    
    def _handle_extraction(self, query: str) -> Tuple[str, str]:
        """Handle parameter extraction with comprehensive validation and clarification"""
        # Extract well name from query
        well_name = self._extract_well_name(query)
        
        # Check cache first
        cached = self.memory.get_cached_extraction(well_name) if well_name else None
        
        if not cached:
            # Two-phase retrieval with increased chunks for better data extraction
            logger.info("Performing two-phase retrieval with high chunk counts...")
            
            query1 = f"trajectory survey directional {well_name or ''}"
            query2 = f"casing design well schematic pipe ID {well_name or ''}"
            
            retrieval_result = self.rag.retrieve_two_phase(
                query1, query2,
                mode1='extract', mode2='summary',
                top_k1=40,  # Increased from 15 for better trajectory coverage
                top_k2=30,  # Increased from 10 for better casing data
                well_name=well_name
            )
            
            chunks = retrieval_result['chunks']
            
            if not chunks:
                return "⚠️ No trajectory or casing data found", ""
            
            # Extract parameters
            logger.info("Extracting parameters...")
            extraction_log = []
            extracted_data = self.extraction.extract(chunks, well_name, extraction_log)
            
            # Cache results
            if well_name:
                self.memory.cache_extraction(well_name, extracted_data)
        else:
            extracted_data = cached
            extraction_log = cached.get('extraction_log', [])
        
        # Step 1: Basic validation
        logger.info("⏳ Validating extracted data...")
        validation_result = self.validation.validate(extracted_data)
        
        # Step 2: Physical validation
        logger.info("⏳ Running physical validation...")
        trajectory_data = []
        for point in extracted_data.get('trajectory', []):
            trajectory_data.append({
                'MD': point.get('md', 0),
                'TVD': point.get('tvd', 0),
                'ID': point.get('pipe_id', 0)  # Already in inches
            })
        
        physical_validation = self.physical_validator.validate_trajectory(trajectory_data)
        
        # Step 3: Completeness assessment
        logger.info("⏳ Assessing data completeness...")
        completeness = self.missing_data_agent.assess_completeness(extracted_data)
        
        # Step 4: Calculate confidence
        source_quality = self.confidence_scorer.calculate_source_quality(chunks, top_k=10)
        consistency = self.confidence_scorer.calculate_consistency(extracted_data)
        
        confidence = self.confidence_scorer.calculate_confidence(
            source_quality=source_quality,
            completeness=completeness.completeness_score,
            consistency=consistency,
            physical_validity=physical_validation.confidence,
            context={'query_type': 'extraction'}
        )
        
        # Apply defaults if needed
        if validation_result['suggestions'] and not validation_result['critical_errors']:
            extracted_data = self.validation.apply_defaults(extracted_data, validation_result['suggestions'])
        
        # Store for potential nodal analysis
        self.pending_extraction = extracted_data
        self.pending_clarifications = completeness.clarification_questions
        
        # Build comprehensive response
        response_parts = []
        response_parts.append(f"# Extraction Results for {extracted_data.get('well_name', 'Unknown Well')}\n")
        
        # Confidence header
        if confidence.recommendation == 'high':
            response_parts.append("✅ **HIGH CONFIDENCE EXTRACTION**\n")
        elif confidence.recommendation == 'review':
            response_parts.append("⚠️ **REVIEW REQUIRED BEFORE USE**\n")
        else:
            response_parts.append("⚠️ **LOW CONFIDENCE - VERIFY ALL DATA**\n")
        
        # Trajectory summary
        trajectory = extracted_data.get('trajectory', [])
        if trajectory:
            response_parts.append(f"\n## Trajectory Data")
            response_parts.append(f"Points extracted: {len(trajectory)}")
            response_parts.append(f"Depth range: {trajectory[0]['md']:.1f} - {trajectory[-1]['md']:.1f} m MD")
            response_parts.append(f"\nFirst 5 points:")
            for i, point in enumerate(trajectory[:5], 1):
                response_parts.append(
                    f"  {i}. MD: {point['md']:.1f}m, TVD: {point['tvd']:.1f}m, "
                    f"Inc: {point['inclination']:.1f}°, ID: {point['pipe_id']:.2f}\""
                )
            
            if len(trajectory) > 5:
                response_parts.append(f"  ... ({len(trajectory) - 5} more points)")
            
            # Show full trajectory format for approval
            response_parts.append(f"\n### Complete Trajectory Data (for nodal analysis):")
            preview_code = self.nodal_runner.generate_preview_code(extracted_data)
            response_parts.append(f"```python\n{preview_code}\n```")
        
        # PVT data
        pvt = extracted_data.get('pvt_data', {})
        if pvt:
            response_parts.append(f"\n## Fluid Properties")
            if 'density' in pvt:
                response_parts.append(f"  • Density: {pvt['density']:.0f} kg/m³")
            if 'viscosity' in pvt:
                response_parts.append(f"  • Viscosity: {pvt['viscosity']:.4f} Pa·s")
            if 'temp_gradient' in pvt:
                response_parts.append(f"  • Temperature gradient: {pvt['temp_gradient']:.1f} °C/km")
        
        # Physical validation results
        response_parts.append(f"\n## Physical Validation")
        if physical_validation.is_valid:
            response_parts.append("✓ All physical constraints satisfied")
        else:
            response_parts.append(f"✗ Physical violations detected:")
            for violation in physical_validation.violations[:5]:  # Show top 5
                response_parts.append(f"  • [{violation.severity.upper()}] {violation.description}")
                response_parts.append(f"    → {violation.suggestion}")
        
        # Completeness assessment
        response_parts.append(f"\n## Data Completeness")
        response_parts.append(completeness.summary)
        
        # Show clarification questions if needed
        if completeness.clarification_questions:
            response_parts.append(f"\n### ⚠️ Clarification Needed:")
            for i, question in enumerate(completeness.clarification_questions[:5], 1):
                response_parts.append(f"  {i}. {question}")
        
        # Legacy validation report
        response_parts.append(f"\n## Legacy Validation")
        if validation_result['valid']:
            response_parts.append("✓ All validations passed")
        else:
            response_parts.append("✗ Validation failed:")
            for error in validation_result['critical_errors']:
                response_parts.append(f"  {error}")
        
        if validation_result['warnings']:
            response_parts.append("\nWarnings:")
            for warning in validation_result['warnings']:
                response_parts.append(f"  {warning}")
        
        # Confidence breakdown
        response_parts.append(f"\n## Confidence Assessment")
        response_parts.append(confidence.explanation)
        
        if confidence.warnings:
            response_parts.append(f"\n**⚠️ Overall Warnings:**")
            for warning in confidence.warnings:
                response_parts.append(f"- {warning}")
        
        # Next steps - ALWAYS ask for confirmation
        response_parts.append(f"\n---")
        if physical_validation.is_valid and not completeness.has_critical_gaps:
            response_parts.append(f"\n**✓ Data extraction complete with {confidence.overall*100:.0f}% confidence**")
            response_parts.append(f"\n⚠️ **IMPORTANT:** Review the extracted data above carefully.")
            if completeness.clarification_questions:
                response_parts.append(f"⚠️ **Consider answering the clarification questions** for better results.")
            response_parts.append(f"\nIf the data looks correct, click **'Run Nodal Analysis'** below to proceed.")
        else:
            response_parts.append(f"\n**⚠️ Data has critical issues - address before analysis:**")
            if not physical_validation.is_valid:
                response_parts.append(f"  • Physical constraint violations detected")
            if completeness.has_critical_gaps:
                response_parts.append(f"  • Critical data missing")
            response_parts.append(f"\nReview issues above and verify source documents.")
        
        # Debug info
        debug_info = "Extraction Log:\n"
        debug_info += "\n".join(extraction_log[-20:])  # Last 20 log messages
        
        return "\n".join(response_parts), debug_info
    
    def run_nodal_analysis(self) -> Tuple[str, str]:
        """Run nodal analysis with previously extracted data"""
        if not self.pending_extraction:
            return "⚠️ No extraction data available. Please run 'Extract & Analyze' mode first.", ""
        
        extracted_data = self.pending_extraction
        trajectory = extracted_data.get('trajectory', [])
        pvt = extracted_data.get('pvt_data', {})
        
        if not trajectory:
            return "⚠️ No trajectory data found in extraction results", ""
        
        logger.info(f"Running nodal analysis with {len(trajectory)} trajectory points...")
        
        response_parts = []
        response_parts.append(f"# Nodal Analysis Results\n")
        response_parts.append(f"Well: {extracted_data.get('well_name', 'Unknown')}\n")
        
        try:
            success, output, plot_path = self.nodal_runner.run_with_extracted_data(extracted_data)
            
            if success:
                response_parts.append(f"\n**✓ Nodal Analysis Completed Successfully**\n")
                response_parts.append(f"```\n{output}\n```")
                
                # Parse key results from output
                import re
                flowrate_match = re.search(r'Flowrate:\s*([\d.]+)\s*m3/hr', output)
                bhp_match = re.search(r'Bottomhole pressure:\s*([\d.]+)\s*bar', output)
                pump_head_match = re.search(r'Pump head:\s*([\d.]+)\s*m', output)
                
                if flowrate_match and bhp_match and pump_head_match:
                    response_parts.append(f"\n## Key Results:")
                    response_parts.append(f"- **Flow Rate:** {flowrate_match.group(1)} m³/hr")
                    response_parts.append(f"- **Bottomhole Pressure:** {bhp_match.group(1)} bar")
                    response_parts.append(f"- **Pump Head:** {pump_head_match.group(1)} m")
                
                response_parts.append(f"\n✓ Analysis complete. Check console output for full results.")
            else:
                response_parts.append(f"\n**⚠️ Nodal Analysis Error:**")
                response_parts.append(f"```\n{output}\n```")
                response_parts.append(f"\nThe nodal analysis script encountered an error. Please check the error message above.")
                
        except Exception as e:
            response_parts.append(f"\n⚠️ Nodal analysis execution error: {str(e)}")
            logger.error(f"Nodal analysis failed: {str(e)}", exc_info=True)
        
        debug_info = f"Trajectory points: {len(trajectory)}\n"
        debug_info += f"Depth range: {trajectory[0]['md']:.1f} - {trajectory[-1]['md']:.1f} m\n"
        debug_info += f"Fluid density: {pvt.get('density', 1000.0)} kg/m³\n"
        
        return "\n".join(response_parts), debug_info
    
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
                        choices=["Q&A", "Summary", "Extract & Analyze"],
                        value="Q&A",
                        label="Query Mode"
                    )
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., What is the total depth of well ADK-GT-01?",
                        lines=3
                    )
                    
                    with gr.Row():
                        query_btn = gr.Button("Submit Query", variant="primary")
                        nodal_btn = gr.Button("Run Nodal Analysis", variant="secondary")
                    
                    gr.Markdown("""
                    **Query Mode Guide:**
                    - **Q&A**: Ask specific questions about the documents (with conversation memory)
                    - **Summary**: Get a summary of document contents
                    - **Extract & Analyze**: Extract trajectory/casing data for review
                    
                    **Example queries:**
                    - Q&A: "What is the casing design for ADK-GT-01?"
                    - Summary: "Summarize the completion report in 300 words"
                    - Summary (Detailed): "Generate detailed End of Well Summary" or "Professional completion report summary"
                    - Extract: "Extract trajectory data for ADK-GT-01"
                    
                    **✨ NEW: 3-Pass End of Well Summary**
                    Use keywords like "End of Well", "EOW", "detailed", "professional", or "completion report" to activate:
                    - **Pass 1**: Metadata (Operator, Well, Rig, Dates, Days Total)
                    - **Pass 2**: Technical Specs (Casing/Tubing tables with ID column)
                    - **Pass 3**: Narrative (Geology, hazards, instabilities)
                    
                    **Workflow for Nodal Analysis:**
                    1. Use "Extract & Analyze" mode to extract trajectory data
                    2. Review the displayed trajectory format
                    3. Click "Run Nodal Analysis" to execute the analysis with extracted data
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
            
            nodal_btn.click(
                fn=system.run_nodal_analysis,
                inputs=[],
                outputs=[response_output, debug_output]
            )
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## RAG System for Geothermal Wells
            
            This system provides intelligent analysis of geothermal well completion reports using:
            
            ### Features
            - **✨ NEW: 3-Pass End of Well Summary**: Professional drilling reports with metadata, casing tables (with ID), and narrative
            - **Multi-strategy chunking**: Optimized for Q&A, summarization, and data extraction
            - **Two-phase retrieval**: Separate queries for trajectory and casing data
            - **Regex-first extraction**: Fast, reliable table parsing with LLM fallback
            - **Data validation**: Physics-based checks for MD≥TVD, realistic pipe sizes, etc.
            - **Nodal analysis**: Production capacity estimation from extracted parameters
            - **Conversation memory**: Multi-turn interactions with context
            
            ### 3-Pass Summary System
            **Pass 1 - Metadata (Key-Value):**
            - Scans first 3 pages
            - Extracts: Operator, Well Name, Rig Name, Dates (Spud/End)
            - Computes Days_Total
            
            **Pass 2 - Technical Specs (Table-to-Markdown):**
            - Uses pdfplumber to extract Casing/Tubing/Liner tables
            - Converts to Markdown format
            - Extracts pipe_id (Inner Diameter) from ID columns
            
            **Pass 3 - Narrative (Section-Restricted):**
            - Locates Geology/Lithology sections
            - Extracts formation instabilities, gas peaks, drilling hazards
            
            ### Architecture
            - **Vector DB**: ChromaDB (embedded)
            - **Embeddings**: nomic-embed-text (384 dims)
            - **LLM**: Ollama (llama3/llama3.1)
            - **PDF Processing**: PyMuPDF + pdfplumber (tables)
            
            ### Usage Tips
            1. Upload PDF completion reports in the "Document Upload" tab
            2. Wait for indexing to complete (20-40 seconds for typical reports)
            3. Switch to "Query Interface" and select appropriate mode
            4. For extraction, include well name in your query (e.g., "ADK-GT-01")
            5. For detailed summaries, use keywords: "End of Well", "EOW", "detailed", or "professional"
            
            ### Validation Rules
            - MD ≥ TVD (±1m tolerance)
            - Pipe ID: 2-30 inches
            - Inclination: 0-90°
            - Well depth: 500-5000m (typical geothermal range)
            
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
