"""
Query Analysis Agent with Intelligent Routing
Analyzes user queries to detect intent, extract entities, and route to appropriate retrieval strategy
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class QueryAnalysis:
    """Structured query analysis result"""
    query_type: str  # 'qa', 'summary', 'extraction', 'comparison', 'relationship'
    retrieval_strategy: str  # 'vector', 'bm25', 'hybrid', 'raptor', 'graph', 'structured'
    target_word_count: Optional[int]  # Explicit word count if specified
    entities: Dict[str, List[str]]  # Extracted well names, depths, parameters
    priority: str  # 'high', 'medium', 'low'
    needs_clarification: bool
    detected_focus: List[str]  # Key topics/aspects
    raptor_level: Optional[int]  # RAPTOR tree level to query (if applicable)


class QueryAnalysisAgent:
    """
    Analyzes queries and routes to appropriate retrieval strategy
    
    Routing Logic:
    1. Factual Q&A → Hybrid (Vector + BM25)
    2. Summary → RAPTOR (upper levels)
    3. Extraction → Structured DB + BM25
    4. Comparison → Knowledge Graph traversal
    5. Relationship queries → Knowledge Graph
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.default_word_count = config.get('summarization', {}).get('default_words', 200)
        
        # Query type patterns
        self.summary_patterns = [
            r'\bsummar[yi]ze\b',
            r'\bsummary\b',
            r'\boverview\b',
            r'\bgive me (?:a |an )?(?:brief |detailed )?(?:summary|overview)\b',
            r'\bwhat (?:does|is) (?:the )?(?:document|paper|report)\b'
        ]
        
        self.extraction_patterns = [
            r'\bextract\b',
            r'\bget (?:the )?(?:trajectory|casing|tubing|well)\b',
            r'\bwell\s+trajectory\b',
            r'\bcasing\s+(?:design|program)\b',
            r'\btubing\s+(?:design|size)\b',
            r'\bpvt\s+data\b'
        ]
        
        # Length constraint patterns
        self.length_patterns = [
            r'(\d+)\s*words?\b',
            r'in\s+(\d+)\s*words?\b',
            r'(?:brief|short|concise)',  # Maps to 100 words
            r'(?:detailed|comprehensive|thorough)',  # Maps to 500 words
        ]
        
        # Entity patterns
        self.well_name_pattern = r'\b(?:well|hole)\s+(?:[A-Z0-9\-]+)\b'
        self.depth_pattern = r'\b(\d+(?:\.\d+)?)\s*(?:m|ft|meter|feet)\b'
        self.parameter_pattern = r'\b(?:temperature|pressure|flowrate|density|viscosity)\b'
        
    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze query to extract intent, entities, constraints, and route to retrieval strategy.
        
        Args:
            query: User query string
            
        Returns:
            QueryAnalysis object with structured information and routing decision
        """
        query_lower = query.lower()
        
        # Detect query type
        query_type = self._detect_query_type(query_lower)
        
        # Route to appropriate retrieval strategy
        retrieval_strategy, raptor_level = self._route_query(query_lower, query_type)
        
        # Extract word count constraint
        target_word_count = self._extract_word_count(query_lower)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Determine priority
        priority = self._determine_priority(query_lower, entities)
        
        # Check if clarification needed
        needs_clarification = self._needs_clarification(query_lower, entities, query_type)
        
        # Detect focus areas
        detected_focus = self._detect_focus(query_lower)
        
        return QueryAnalysis(
            query_type=query_type,
            retrieval_strategy=retrieval_strategy,
            target_word_count=target_word_count,
            entities=entities,
            priority=priority,
            needs_clarification=needs_clarification,
            detected_focus=detected_focus,
            raptor_level=raptor_level
        )
    
    def _detect_query_type(self, query_lower: str) -> str:
        """Detect if query is Q&A, summary, or extraction"""
        # Check extraction first (most specific)
        for pattern in self.extraction_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return 'extraction'
        
        # Check summary
        for pattern in self.summary_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return 'summary'
        
        # Default to Q&A
        return 'qa'
    
    def _extract_word_count(self, query_lower: str) -> Optional[int]:
        """Extract explicit word count or map qualitative constraints"""
        # Check for numeric word count
        for pattern in self.length_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                if match.groups():  # Has numeric capture
                    return int(match.group(1))
        
        # Map qualitative to numeric
        if re.search(r'(?:brief|short|concise)', query_lower):
            return 100
        elif re.search(r'(?:detailed|comprehensive|thorough)', query_lower):
            return 500
        
        # No explicit constraint - will use default
        return None
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract well names, depths, parameters"""
        entities = {
            'wells': [],
            'depths': [],
            'parameters': []
        }
        
        # Extract well names
        well_matches = re.findall(self.well_name_pattern, query, re.IGNORECASE)
        entities['wells'] = list(set(well_matches))
        
        # Extract depths
        depth_matches = re.findall(self.depth_pattern, query, re.IGNORECASE)
        entities['depths'] = [m[0] for m in depth_matches] if depth_matches else []
        
        # Extract parameters
        param_matches = re.findall(self.parameter_pattern, query, re.IGNORECASE)
        entities['parameters'] = list(set(param_matches))
        
        return entities
    
    def _determine_priority(self, query_lower: str, entities: Dict[str, List[str]]) -> str:
        """Determine query priority based on keywords and entities"""
        # High priority: Safety, critical operations
        high_keywords = ['safety', 'critical', 'urgent', 'failure', 'risk', 'emergency']
        if any(kw in query_lower for kw in high_keywords):
            return 'high'
        
        # Medium priority: Multiple entities or extraction
        if len(entities['wells']) > 1 or len(entities['parameters']) > 2:
            return 'medium'
        
        # Low priority: Simple Q&A
        return 'low'
    
    def _needs_clarification(self, query_lower: str, entities: Dict[str, List[str]], 
                           query_type: str) -> bool:
        """Check if query is ambiguous or incomplete"""
        # Extraction without specific well name
        if query_type == 'extraction' and not entities['wells']:
            return True
        
        # Vague questions
        vague_patterns = [
            r'\bwhat about\b',
            r'\btell me about\b',
            r'\banything\b',
            r'\bstuff\b'
        ]
        if any(re.search(p, query_lower) for p in vague_patterns):
            return True
        
        return False
    
    def _route_query(self, query_lower: str, query_type: str) -> tuple[str, Optional[int]]:
        """
        Route query to appropriate retrieval strategy based on intent
        
        Returns:
            (retrieval_strategy, raptor_level)
        """
        # Summary queries → RAPTOR tree (upper levels for high-level summaries)
        if query_type == 'summary':
            if 'brief' in query_lower or 'overview' in query_lower:
                return ('raptor', 2)  # High-level summary from level 2
            elif 'detailed' in query_lower or 'comprehensive' in query_lower:
                return ('raptor', 1)  # More detailed from level 1
            else:
                return ('raptor', None)  # Query all levels
        
        # Comparison queries → Knowledge Graph
        if 'compar' in query_lower or 'differ' in query_lower or 'between' in query_lower:
            return ('graph', None)
        
        # Relationship queries → Knowledge Graph
        if 'related' in query_lower or 'connection' in query_lower or 'similar' in query_lower:
            return ('graph', None)
        
        # Extraction with specific terms → BM25 + Structured
        if query_type == 'extraction':
            # Check for exact term matching needs
            has_ids = bool(re.search(r'[A-Z]{2,}-GT-\d{2}', query_lower.upper()))
            has_numbers = bool(re.search(r'\b\d+\.?\d*\s*(?:inch|m|bar|psi)\b', query_lower))
            
            if has_ids or has_numbers:
                return ('bm25', None)  # Exact term matching
            else:
                return ('structured', None)  # Database query
        
        # Default: Hybrid dense+sparse retrieval for factual Q&A
        return ('hybrid', None)
    
    def _detect_focus(self, query_lower: str) -> List[str]:
        """Detect key focus areas in query"""
        focus_areas = []
        
        focus_map = {
            'trajectory': r'\b(?:trajectory|wellbore|path|deviation)\b',
            'casing': r'\b(?:casing|cement|annulus)\b',
            'tubing': r'\b(?:tubing|production\s+string)\b',
            'pressure': r'\b(?:pressure|bhp|thp|drawdown)\b',
            'temperature': r'\b(?:temperature|thermal|heat)\b',
            'flow': r'\b(?:flow|rate|production|injection)\b',
            'pvt': r'\b(?:pvt|fluid|properties|viscosity|density)\b',
            'nodal': r'\b(?:nodal|ipr|tpr|operating\s+point)\b'
        }
        
        for focus, pattern in focus_map.items():
            if re.search(pattern, query_lower):
                focus_areas.append(focus)
        
        return focus_areas if focus_areas else ['general']


def create_agent(config: Dict[str, Any]) -> QueryAnalysisAgent:
    """Factory function to create agent"""
    return QueryAnalysisAgent(config)
