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
    Intelligent retrieval that combines:
    1. SQL database queries for exact numerical/tabular data
    2. Semantic vector search for narrative/descriptive content
    
    Query routing logic:
    - Numerical queries (depth, size, weight, dates) → Database
    - Table queries (casing program, formations) → Database
    - Narrative queries (problems, operations, geology description) → Semantic search
    - Complex queries → Both (database + semantic context)
    """
    
    def __init__(self, database_manager, rag_retrieval_agent):
        """
        Initialize hybrid retrieval
        
        Args:
            database_manager: WellDatabaseManager instance
            rag_retrieval_agent: RAGRetrievalAgent instance (for semantic search)
        """
        self.db = database_manager
        self.semantic_rag = rag_retrieval_agent
        
        # Query classification patterns
        self.numerical_patterns = [
            r'\b(?:depth|md|tvd|measured|vertical)\b',
            r'\b(?:size|diameter|od|id|inch)\b',
            r'\b(?:weight|lb/ft|kg/m)\b',
            r'\b(?:date|when|spud|completion)\b',
            r'\b(?:how deep|how long|how much)\b',
            r'\b\d+(?:\.\d+)?\s*(?:m|ft|inch|mm)\b'
        ]
        
        self.table_patterns = [
            r'\b(?:casing|tubular|pipe)\s+(?:program|string|design)\b',
            r'\b(?:formation|lithology|geology)\s+(?:tops|layers)\b',
            r'\b(?:cement|cementing)\s+(?:job|stage|operation)\b',
            r'\b(?:list|show|display)\s+(?:all|the)\b'
        ]
        
        self.narrative_patterns = [
            r'\b(?:problem|issue|challenge|difficulty)\b',
            r'\b(?:describe|explain|discuss)\b',
            r'\b(?:why|how|reason)\b',
            r'\b(?:operation|activity|procedure)\b',
            r'\b(?:summary|overview|description)\b'
        ]
    
    def retrieve(self, query: str, well_name: Optional[str] = None, 
                mode: str = 'auto', top_k: int = 10) -> Dict[str, Any]:
        """
        Retrieve information using hybrid approach
        
        Args:
            query: User query
            well_name: Well name to focus on (if known)
            mode: 'auto', 'database', 'semantic', or 'hybrid'
            top_k: Number of results for semantic search
            
        Returns:
            Dict with 'database_results', 'semantic_results', 'combined_text'
        """
        # Classify query type if mode is auto
        if mode == 'auto':
            mode = self._classify_query(query)
        
        logger.info(f"Hybrid retrieval mode: {mode} for query: '{query[:50]}...'")
        
        results = {
            'database_results': [],
            'semantic_results': [],
            'combined_text': '',
            'mode': mode
        }
        
        # Execute retrieval based on mode
        if mode in ['database', 'hybrid']:
            db_results = self._query_database(query, well_name)
            results['database_results'] = db_results
        
        if mode in ['semantic', 'hybrid']:
            semantic_results = self._query_semantic(query, top_k)
            results['semantic_results'] = semantic_results
        
        # Combine results with priority: database > semantic
        results['combined_text'] = self._combine_results(results)
        
        return results
    
    def _classify_query(self, query: str) -> str:
        """
        Classify query to determine retrieval strategy
        
        Returns:
            'database', 'semantic', or 'hybrid'
        """
        query_lower = query.lower()
        
        # Count pattern matches
        numerical_matches = sum(1 for p in self.numerical_patterns if re.search(p, query_lower))
        table_matches = sum(1 for p in self.table_patterns if re.search(p, query_lower))
        narrative_matches = sum(1 for p in self.narrative_patterns if re.search(p, query_lower))
        
        logger.debug(f"Query classification - Numerical: {numerical_matches}, Table: {table_matches}, Narrative: {narrative_matches}")
        
        # Scoring logic
        if (numerical_matches + table_matches) >= 2 and narrative_matches == 0:
            return 'database'
        elif narrative_matches >= 2 and (numerical_matches + table_matches) == 0:
            return 'semantic'
        elif (numerical_matches + table_matches) > 0 and narrative_matches > 0:
            return 'hybrid'
        else:
            # Default to hybrid for safety
            return 'hybrid'
    
    def _query_database(self, query: str, well_name: Optional[str]) -> List[Dict]:
        """Query database for structured data"""
        results = []
        
        if not well_name:
            logger.warning("No well name provided for database query")
            return results
        
        # Get comprehensive well data
        well_summary = self.db.get_well_summary(well_name)
        
        if not well_summary:
            logger.warning(f"No database entry found for {well_name}")
            return results
        
        # Format database data into structured results
        well_info = well_summary.get('well_info', {})
        casings = well_summary.get('casing_strings', [])
        formations = well_summary.get('formations', [])
        cementing = well_summary.get('cementing', [])
        
        # Add well basic info
        if well_info:
            results.append({
                'type': 'well_info',
                'data': well_info,
                'text': self._format_well_info(well_info)
            })
        
        # Add casing data if relevant
        if casings and re.search(r'\b(?:casing|pipe|tubular|string)\b', query, re.IGNORECASE):
            results.append({
                'type': 'casing',
                'data': casings,
                'text': self._format_casing_data(casings)
            })
        
        # Add formation data if relevant
        if formations and re.search(r'\b(?:formation|lithology|geology|layer)\b', query, re.IGNORECASE):
            results.append({
                'type': 'formations',
                'data': formations,
                'text': self._format_formation_data(formations)
            })
        
        # Add cementing data if relevant
        if cementing and re.search(r'\b(?:cement|cementing|stage)\b', query, re.IGNORECASE):
            results.append({
                'type': 'cementing',
                'data': cementing,
                'text': self._format_cementing_data(cementing)
            })
        
        logger.info(f"Database query returned {len(results)} result sets")
        return results
    
    def _query_semantic(self, query: str, top_k: int) -> List[Dict]:
        """Query semantic vector database"""
        try:
            # Use existing RAG retrieval agent
            retrieval_result = self.semantic_rag.retrieve(query, top_k=top_k)
            
            if isinstance(retrieval_result, dict):
                chunks = retrieval_result.get('chunks', [])
            else:
                chunks = retrieval_result
            
            logger.info(f"Semantic search returned {len(chunks)} chunks")
            return chunks
        
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return []
    
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
