"""
Confidence Scoring Agent
Provides multi-dimensional confidence assessment for RAG responses.
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConfidenceScore:
    """Multi-dimensional confidence score"""
    overall: float  # 0.0 to 1.0
    dimensions: Dict[str, float]  # Individual dimension scores
    recommendation: str  # 'high', 'review', 'low'
    explanation: str
    warnings: List[str]


class ConfidenceScorerAgent:
    """Calculates multi-dimensional confidence scores for RAG responses"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Thresholds
        validation_config = config.get('validation', {})
        self.high_confidence_threshold = validation_config.get('high_confidence', 0.85)
        self.low_confidence_threshold = validation_config.get('low_confidence', 0.50)
        
        # Dimension weights (must sum to 1.0)
        self.weights = {
            'source_quality': 0.20,      # Quality of retrieved sources
            'fact_verification': 0.25,   # Fact checking results
            'completeness': 0.20,        # Data completeness
            'consistency': 0.15,         # Internal consistency
            'physical_validity': 0.20    # Physical constraints
        }
        
        logger.info(f"Confidence scoring: High≥{self.high_confidence_threshold*100:.0f}%, Low<{self.low_confidence_threshold*100:.0f}%")
    
    def calculate_confidence(self, 
                            source_quality: float = None,
                            fact_verification: float = None,
                            completeness: float = None,
                            consistency: float = None,
                            physical_validity: float = None,
                            context: Dict[str, Any] = None) -> ConfidenceScore:
        """
        Calculate overall confidence score from multiple dimensions
        
        Args:
            source_quality: Score for source document quality (0-1)
            fact_verification: Score from fact verification (0-1)
            completeness: Score for data completeness (0-1)
            consistency: Score for internal consistency (0-1)
            physical_validity: Score for physical validation (0-1)
            context: Additional context for explanation
            
        Returns:
            ConfidenceScore with overall score and recommendations
        """
        # Build dimension scores dict
        dimensions = {}
        warnings = []
        
        # Source quality
        if source_quality is not None:
            dimensions['source_quality'] = source_quality
            if source_quality < 0.6:
                warnings.append("⚠️ Low source quality - retrieved documents may not be highly relevant")
        else:
            dimensions['source_quality'] = 0.5
            warnings.append("⚠️ Source quality not assessed")
        
        # Fact verification
        if fact_verification is not None:
            dimensions['fact_verification'] = fact_verification
            if fact_verification < 0.7:
                warnings.append("⚠️ Some claims could not be verified against sources")
        else:
            dimensions['fact_verification'] = 0.5
            warnings.append("⚠️ Fact verification not performed")
        
        # Completeness
        if completeness is not None:
            dimensions['completeness'] = completeness
            if completeness < 0.7:
                warnings.append("⚠️ Data is incomplete - results may be limited")
        else:
            dimensions['completeness'] = 0.5
            warnings.append("⚠️ Completeness not assessed")
        
        # Consistency
        if consistency is not None:
            dimensions['consistency'] = consistency
            if consistency < 0.7:
                warnings.append("⚠️ Internal inconsistencies detected")
        else:
            dimensions['consistency'] = 0.7  # Default assume consistent
        
        # Physical validity
        if physical_validity is not None:
            dimensions['physical_validity'] = physical_validity
            if physical_validity < 0.7:
                warnings.append("⚠️ Physical constraint violations detected")
        else:
            dimensions['physical_validity'] = 0.7  # Default assume valid
        
        # Calculate weighted overall score
        overall = sum(dimensions[dim] * self.weights[dim] 
                     for dim in dimensions if dim in self.weights)
        
        # Determine recommendation
        if overall >= self.high_confidence_threshold:
            recommendation = 'high'
            rec_text = "HIGH CONFIDENCE - Results are reliable"
        elif overall >= self.low_confidence_threshold:
            recommendation = 'review'
            rec_text = "REVIEW RECOMMENDED - Verify results before use"
        else:
            recommendation = 'low'
            rec_text = "LOW CONFIDENCE - Results may be unreliable"
        
        # Generate explanation
        explanation = self._generate_explanation(dimensions, overall, recommendation, context)
        
        logger.info(f"Confidence score: {overall*100:.0f}% ({recommendation.upper()}) - {len(warnings)} warnings")
        
        return ConfidenceScore(
            overall=overall,
            dimensions=dimensions,
            recommendation=recommendation,
            explanation=f"{rec_text}\n\n{explanation}",
            warnings=warnings
        )
    
    def _generate_explanation(self, dimensions: Dict[str, float], 
                            overall: float, recommendation: str,
                            context: Dict[str, Any]) -> str:
        """Generate human-readable explanation of confidence score"""
        lines = [f"Overall Confidence: {overall*100:.0f}%"]
        lines.append("")
        lines.append("Dimension Breakdown:")
        
        # Sort dimensions by score (lowest first to highlight issues)
        sorted_dims = sorted(dimensions.items(), key=lambda x: x[1])
        
        for dim_name, score in sorted_dims:
            # Format dimension name
            display_name = dim_name.replace('_', ' ').title()
            bar_length = int(score * 20)  # 20-char bar
            bar = '█' * bar_length + '░' * (20 - bar_length)
            
            # Add indicator
            if score >= 0.85:
                indicator = '✓'
            elif score >= 0.70:
                indicator = '○'
            else:
                indicator = '✗'
            
            lines.append(f"  {indicator} {display_name:20s} {bar} {score*100:3.0f}%")
        
        lines.append("")
        
        # Add context-specific notes
        if context:
            query_type = context.get('query_type', 'unknown')
            if query_type == 'extraction':
                lines.append("Note: Extraction results require user confirmation before use.")
            elif query_type == 'summary':
                word_count = context.get('word_count', 'N/A')
                lines.append(f"Summary length: ~{word_count} words")
        
        return "\n".join(lines)
    
    def calculate_source_quality(self, chunks: List[Dict], top_k: int = 5) -> float:
        """
        Calculate source quality score based on retrieval scores
        
        Args:
            chunks: Retrieved chunks with 'score' field
            top_k: Number of top chunks to consider
            
        Returns:
            Quality score 0.0 to 1.0
        """
        if not chunks:
            return 0.0
        
        # Use top_k chunks
        top_chunks = chunks[:top_k]
        
        # Extract scores (assuming cosine similarity 0-1)
        scores = [chunk.get('score', 0.5) for chunk in top_chunks]
        
        if not scores:
            return 0.5
        
        # Average score
        avg_score = sum(scores) / len(scores)
        
        # Check for score degradation
        if len(scores) > 1:
            first_score = scores[0]
            last_score = scores[-1]
            degradation = first_score - last_score
            
            # Penalize if scores drop too quickly
            if degradation > 0.3:
                avg_score *= 0.8
        
        return min(1.0, max(0.0, avg_score))
    
    def calculate_consistency(self, extracted_data: Dict[str, Any]) -> float:
        """
        Calculate internal consistency score for extracted data
        
        Args:
            extracted_data: Extracted well data
            
        Returns:
            Consistency score 0.0 to 1.0
        """
        score = 1.0
        
        trajectory = extracted_data.get('trajectory', [])
        casing = extracted_data.get('casing', [])
        
        # Check trajectory consistency
        if trajectory:
            # Check for duplicate depths
            mds = [p.get('MD', 0) for p in trajectory]
            if len(mds) != len(set(mds)):
                score -= 0.2  # Duplicates detected
            
            # Check for gaps
            if len(trajectory) > 1:
                sorted_mds = sorted(mds)
                gaps = [sorted_mds[i+1] - sorted_mds[i] for i in range(len(sorted_mds)-1)]
                avg_gap = sum(gaps) / len(gaps)
                max_gap = max(gaps)
                
                # Large gap inconsistency
                if max_gap > avg_gap * 5:
                    score -= 0.1
        
        # Check casing consistency
        if casing:
            # Check for overlapping depths
            depths = [c.get('depth', 0) for c in casing]
            sorted_depths = sorted(depths)
            
            # Casings should be at increasing depths
            for i in range(len(sorted_depths) - 1):
                if sorted_depths[i] >= sorted_depths[i+1] * 0.95:  # 5% tolerance
                    score -= 0.15
        
        # Check trajectory vs casing consistency
        if trajectory and casing:
            max_traj_md = max(p.get('MD', 0) for p in trajectory)
            max_casing_depth = max(c.get('depth', 0) for c in casing)
            
            # Trajectory should extend to at least deepest casing
            if max_traj_md < max_casing_depth * 0.9:
                score -= 0.2
        
        return max(0.0, min(1.0, score))


def create_agent(config: Dict[str, Any]) -> ConfidenceScorerAgent:
    """Factory function to create agent"""
    return ConfidenceScorerAgent(config)
