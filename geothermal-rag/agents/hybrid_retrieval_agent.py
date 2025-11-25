"""
Hybrid Retrieval Agent - Combines structured database queries with semantic search
Routes queries to appropriate backend based on query type
"""

import logging
import re
from typing import Dict, List, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRetrievalAgent:
    """
    Hybrid retrieval that ALWAYS queries:
    1. Database for complete tables (all data types: text, numbers, etc.)
    2. Semantic vector search for narrative content
    
    No classification - always queries both sources
    """
    
    def __init__(self, database_manager, rag_retrieval_agent, config=None):
        """
        Initialize hybrid retrieval
        
        Args:
            database_manager: WellDatabaseManager instance
            rag_retrieval_agent: RAGRetrievalAgent instance (for semantic search)
            config: Optional config dict or path for LLM helper
        """
        self.db = database_manager
        self.semantic_rag = rag_retrieval_agent
        
        # Initialize LLM helper for SQL generation
        try:
            from agents.llm_helper import OllamaHelper
            self.llm = OllamaHelper(config)
            self.llm_available = self.llm.is_available()
            if self.llm_available:
                logger.info("LLM available for intelligent SQL query generation")
        except Exception as e:
            logger.warning(f"LLM not available for SQL generation: {e}")
            self.llm = None
            self.llm_available = False
    
    def retrieve(self, query: str, well_name: Optional[str] = None, 
                top_k: int = 10) -> Dict[str, Any]:
        """
        Retrieve information - ALWAYS queries both database AND semantic search
        
        Args:
            query: User query
            well_name: Well name to focus on (if known)
            top_k: Number of results for semantic search
            
        Returns:
            Dict with 'database_results', 'semantic_results', 'combined_text'
        """
        logger.info(f"Hybrid retrieval for query: '{query[:50]}...'")
        
        results = {
            'database_results': [],
            'semantic_results': [],
            'combined_text': ''
        }
        
        # ALWAYS query database
        db_results = self._query_database(query, well_name)
        results['database_results'] = db_results
        
        # ALWAYS query semantic search
        semantic_results = self._query_semantic(query, top_k)
        results['semantic_results'] = semantic_results
        
        # Combine results with priority: database > semantic
        results['combined_text'] = self._combine_results(results)
        
        return results
    
    def _query_database(self, query: str, well_name: Optional[str]) -> List[Dict]:
        """
        Query database with intelligent SQL generation
        
        The database stores:
        - Complete tables with ALL data (text, numbers, measurements)
        - Each table has headers_json, rows_json, and table_type fields
        
        Uses LLM to generate smart SQL filters based on question
        """
        results = []
        
        if not well_name:
            logger.warning("No well name provided for database query")
            return results
        
        # Try intelligent SQL generation if LLM available
        if self.llm_available and self.llm:
            try:
                # Generate SQL filter based on query
                sql_filter = self.llm.generate_sql_filter(query, well_name)
                
                # Execute smart SQL query
                sql = f"SELECT * FROM complete_tables WHERE well_name = ? AND ({sql_filter}) ORDER BY source_page"
                tables = self.db.query_sql(sql, (well_name,))
                
                logger.info(f"Smart SQL query returned {len(tables)} tables (filter: {sql_filter})")
                
            except Exception as e:
                logger.warning(f"Smart SQL generation failed: {e}, falling back to simple query")
                # Fallback: Get all tables
                tables = self.db.get_complete_tables(well_name)
        else:
            # Fallback: Get all tables
            tables = self.db.get_complete_tables(well_name)
        
        if not tables:
            logger.warning(f"No tables found for {well_name}")
            return results
        
        # Convert each table to text format
        for table in tables:
            table_text = self._format_table_as_text(table)
            results.append({
                'type': 'table',
                'table_type': table.get('table_type', 'unknown'),
                'page': table.get('source_page'),
                'text': table_text,
                'headers': table.get('headers', []),
                'rows': table.get('rows', [])
            })
        
        logger.info(f"Database query returned {len(results)} complete tables")
        return results
    
    def _query_semantic(self, query: str, top_k: int) -> List[Dict]:
        """Query semantic vector database"""
        try:
            # Use simplified RAG retrieval agent (returns list of chunks)
            chunks = self.semantic_rag.retrieve(query, top_k=top_k)
            logger.info(f"Semantic search returned {len(chunks)} chunks")
            return chunks
        
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return []
    
    def _format_table_as_text(self, table: Dict) -> str:
        """Convert complete table to text format for LLM"""
        parts = []
        
        # Table reference and page
        ref = table.get('table_reference', 'Table')
        page = table.get('source_page', '?')
        parts.append(f"{ref} (Page {page})")
        parts.append("=" * 60)
        
        # Headers
        headers = table.get('headers', [])
        if headers:
            parts.append(" | ".join(str(h) for h in headers))
            parts.append("-" * 60)
        
        # Rows (all data: text and numbers)
        rows = table.get('rows', [])
        for row in rows[:50]:  # Limit to 50 rows to avoid overwhelming context
            parts.append(" | ".join(str(cell) for cell in row))
        
        if len(rows) > 50:
            parts.append(f"... ({len(rows) - 50} more rows)")
        
        parts.append("")
        return "\n".join(parts)
    
    def _combine_results(self, results: Dict) -> str:
        """
        Combine database and semantic results into unified text
        Priority: Database (exact) > Semantic (context)
        """
        combined = []
        
        # Add database results first (highest priority - exact data)
        if results['database_results']:
            combined.append("=== EXACT DATA FROM DATABASE ===")
            for db_result in results['database_results']:
                combined.append(db_result.get('text', ''))
            combined.append("")
        
        # Add semantic results (supporting context)
        if results['semantic_results']:
            combined.append("=== SUPPORTING CONTEXT ===")
            for chunk in results['semantic_results'][:5]:  # Top 5 semantic chunks
                text = chunk.get('text', '') if isinstance(chunk, dict) else str(chunk)
                combined.append(text)
                combined.append("---")
        
        return '\n'.join(combined)
    
    def _format_well_info(self, well_info: Dict) -> str:
        """Format well basic info"""
        parts = [f"Well: {well_info.get('well_name')}"]
        
        if well_info.get('operator'):
            parts.append(f"Operator: {well_info.get('operator')}")
        
        if well_info.get('location'):
            parts.append(f"Location: {well_info.get('location')}")
        
        if well_info.get('total_depth_md'):
            tvd_text = f", {well_info.get('total_depth_tvd')}m TVD" if well_info.get('total_depth_tvd') else ""
            parts.append(f"Total Depth: {well_info.get('total_depth_md')}m MD{tvd_text}")
        
        if well_info.get('spud_date'):
            parts.append(f"Spud Date: {well_info.get('spud_date')}")
        
        if well_info.get('completion_date'):
            parts.append(f"Completion Date: {well_info.get('completion_date')}")
        
        return '\n'.join(parts)
    
    def _format_casing_data(self, casings: List[Dict]) -> str:
        """Format casing program"""
        lines = ["Casing Program:"]
        for casing in casings:
            size = casing.get('outer_diameter')
            weight = casing.get('weight')
            grade = casing.get('grade')
            depth = casing.get('bottom_depth_md')
            page = casing.get('source_page')
            
            parts = []
            if size:
                parts.append(f"{size} inch")
            if weight:
                parts.append(f"{weight} lb/ft")
            if grade:
                parts.append(grade)
            if depth:
                parts.append(f"set at {depth}m MD")
            
            line = f"  - {', '.join(parts)}"
            if page:
                line += f" [Source: Page {page}]"
            lines.append(line)
        
        return '\n'.join(lines)
    
    def _format_formation_data(self, formations: List[Dict]) -> str:
        """Format formation tops"""
        lines = ["Formation Tops:"]
        for fm in formations:
            name = fm.get('formation_name')
            top_md = fm.get('top_md')
            lithology = fm.get('lithology')
            page = fm.get('source_page')
            
            parts = [f"  - {name}"]
            if top_md:
                parts.append(f"at {top_md}m MD")
            if lithology:
                parts.append(f"({lithology})")
            
            line = ' '.join(parts)
            if page:
                line += f" [Source: Page {page}]"
            lines.append(line)
        
        return '\n'.join(lines)
    
    def _format_cementing_data(self, cementing: List[Dict]) -> str:
        """Format cementing operations"""
        lines = ["Cementing Operations:"]
        for cement in cementing:
            stage = cement.get('stage_number')
            top = cement.get('top_of_cement_md')
            bottom = cement.get('bottom_of_cement_md')
            volume = cement.get('volume')
            page = cement.get('source_page')
            
            line = f"  - Stage {stage}: " if stage else "  - Cement job: "
            if top and bottom:
                line += f"{bottom}m to {top}m MD"
            if volume:
                line += f", {volume}m³"
            
            if page:
                line += f" [Source: Page {page}]"
            lines.append(line)
        
        return '\n'.join(lines)
    
    def _format_fluids_data(self, fluids: List[Dict]) -> str:
        """Format drilling fluids"""
        lines = ["Drilling Fluids:"]
        for fluid in fluids:
            hole_size = fluid.get('hole_size')
            fluid_type = fluid.get('fluid_type')
            density_min = fluid.get('density_min')
            density_max = fluid.get('density_max')
            page = fluid.get('source_page')
            
            parts = []
            if hole_size:
                parts.append(f"{hole_size} inch hole")
            if fluid_type:
                parts.append(f"{fluid_type}")
            if density_min and density_max:
                parts.append(f"density: {density_min}-{density_max} kg/m³")
            elif density_min:
                parts.append(f"density: {density_min} kg/m³")
            
            line = f"  - {', '.join(parts)}"
            if page:
                line += f" [Source: Page {page}]"
            lines.append(line)
        
        return '\n'.join(lines)
    
    def _format_incidents_data(self, incidents: List[Dict]) -> str:
        """Format incidents"""
        lines = ["Incidents:"]
        for incident in incidents:
            date = incident.get('date')
            incident_type = incident.get('incident_type')
            description = incident.get('description')
            depth = incident.get('depth_md')
            severity = incident.get('severity')
            page = incident.get('source_page')
            
            parts = []
            if date:
                parts.append(f"Date: {date}")
            if incident_type:
                parts.append(f"Type: {incident_type}")
            if depth:
                parts.append(f"at {depth}m MD")
            if severity:
                parts.append(f"Severity: {severity}")
            
            line = f"  - {', '.join(parts)}"
            if description:
                line += f"\n    Description: {description}"
            if page:
                line += f" [Source: Page {page}]"
            lines.append(line)
        
        return '\n'.join(lines)
