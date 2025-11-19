"""
Chat Memory Agent - Conversation Tracking for Multi-Turn Interactions
"""

from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatMemory:
    """
    Simple conversation memory for RAG system
    
    Tracks:
    - User queries and assistant responses
    - Document context (which documents are being discussed)
    - Extracted parameters (to avoid re-extraction)
    """
    
    def __init__(self, max_history: int = 10):
        """
        Initialize chat memory
        
        Args:
            max_history: Maximum number of exchanges to remember
        """
        self.max_history = max_history
        self.history: List[Dict] = []
        self.current_documents: List[str] = []
        self.cached_extractions: Dict = {}
    
    def add_exchange(self, user_query: str, assistant_response: str, 
                    metadata: Optional[Dict] = None):
        """
        Add a conversation exchange
        
        Args:
            user_query: User's question
            assistant_response: Assistant's answer
            metadata: Optional metadata (retrieved chunks, extraction results, etc.)
        """
        exchange = {
            'user': user_query,
            'assistant': assistant_response,
            'metadata': metadata or {}
        }
        
        self.history.append(exchange)
        
        # Keep only last N exchanges
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """
        Get conversation history
        
        Args:
            last_n: Number of recent exchanges to return (None = all)
            
        Returns:
            List of exchange dicts
        """
        if last_n is None:
            return self.history
        return self.history[-last_n:]
    
    def get_context_string(self, last_n: int = 3) -> str:
        """
        Get formatted context string for LLM
        
        Args:
            last_n: Number of recent exchanges to include
            
        Returns:
            Formatted conversation history
        """
        recent = self.get_history(last_n)
        
        if not recent:
            return ""
        
        context_parts = ["Previous conversation:"]
        for exchange in recent:
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"Assistant: {exchange['assistant'][:200]}...")  # Truncate long responses
        
        return "\n".join(context_parts)
    
    def set_documents(self, document_names: List[str]):
        """Set current document context"""
        self.current_documents = document_names
        logger.info(f"Set document context: {document_names}")
    
    def get_documents(self) -> List[str]:
        """Get current document context"""
        return self.current_documents
    
    def cache_extraction(self, well_name: str, extraction_data: Dict):
        """Cache extraction results to avoid re-extraction"""
        self.cached_extractions[well_name] = extraction_data
        logger.info(f"Cached extraction for {well_name}")
    
    def get_cached_extraction(self, well_name: str) -> Optional[Dict]:
        """Retrieve cached extraction if available"""
        return self.cached_extractions.get(well_name)
    
    def clear(self):
        """Clear all memory"""
        self.history = []
        self.current_documents = []
        self.cached_extractions = {}
        logger.info("Cleared chat memory")
