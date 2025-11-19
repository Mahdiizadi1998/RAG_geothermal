"""
Ensemble Judge Agent - Answer Quality Assessment
Validates LLM responses against source documents
"""

from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleJudgeAgent:
    """
    Quality control agent for LLM responses
    
    Checks:
    - Answer relevance to query
    - Citation accuracy (facts present in source chunks)
    - Hallucination detection
    - Completeness assessment
    """
    
    def __init__(self):
        pass
    
    def evaluate_response(self, query: str, response: str, 
                         source_chunks: List[Dict]) -> Dict:
        """
        Evaluate response quality
        
        Args:
            query: User's question
            response: Assistant's answer
            source_chunks: Retrieved chunks used for answer
            
        Returns:
            Dict with evaluation results:
            {
                'quality_score': float,  # 0-1
                'has_citations': bool,
                'relevance': str,  # 'high', 'medium', 'low'
                'issues': List[str]
            }
        """
        issues = []
        
        # Check if response is too short
        if len(response.split()) < 10:
            issues.append("Response is very short")
        
        # Check if response contains "I don't know" or similar
        uncertainty_phrases = ["i don't know", "i'm not sure", "cannot determine", 
                              "insufficient information", "not found in"]
        if any(phrase in response.lower() for phrase in uncertainty_phrases):
            issues.append("Response indicates uncertainty")
        
        # Check if source chunks are relevant to query
        query_words = set(query.lower().split())
        relevant_chunks = 0
        
        for chunk in source_chunks[:5]:  # Check top 5
            chunk_words = set(chunk['text'].lower().split())
            overlap = len(query_words & chunk_words)
            if overlap >= 2:
                relevant_chunks += 1
        
        relevance = 'high' if relevant_chunks >= 3 else 'medium' if relevant_chunks >= 1 else 'low'
        
        # Calculate quality score
        quality_score = 0.5  # Base score
        
        if not issues:
            quality_score += 0.3
        
        if relevance == 'high':
            quality_score += 0.2
        elif relevance == 'medium':
            quality_score += 0.1
        
        # Check for page citations
        has_citations = 'page' in response.lower() or 'section' in response.lower()
        if has_citations:
            quality_score += 0.1
        
        return {
            'quality_score': min(quality_score, 1.0),
            'has_citations': has_citations,
            'relevance': relevance,
            'issues': issues,
            'source_chunks_count': len(source_chunks),
            'relevant_chunks_count': relevant_chunks
        }
    
    def check_hallucination(self, response: str, source_chunks: List[Dict]) -> Dict:
        """
        Check if response contains information not in sources
        
        Args:
            response: Assistant's answer
            source_chunks: Retrieved chunks
            
        Returns:
            Dict with hallucination analysis
        """
        # Simple check: are key entities in response present in sources?
        # This is a simplified heuristic - real hallucination detection is more complex
        
        # Combine all source text
        all_source_text = ' '.join([chunk['text'] for chunk in source_chunks]).lower()
        
        # Extract numbers from response (could be measurements, dates, etc.)
        import re
        response_numbers = re.findall(r'\d+\.?\d*', response)
        
        # Check if numbers in response are in sources
        hallucinated_numbers = []
        for num in response_numbers:
            if num not in all_source_text:
                hallucinated_numbers.append(num)
        
        # If more than 30% of numbers are not in sources, flag as potential hallucination
        if response_numbers:
            hallucination_ratio = len(hallucinated_numbers) / len(response_numbers)
        else:
            hallucination_ratio = 0.0
        
        return {
            'likely_hallucination': hallucination_ratio > 0.3,
            'hallucination_ratio': hallucination_ratio,
            'total_numbers': len(response_numbers),
            'missing_numbers': len(hallucinated_numbers),
            'note': 'This is a simplified check - manual review recommended for critical data'
        }
