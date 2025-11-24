"""
Fact Verification Agent
Verifies factual claims in LLM responses against source documents.
"""

import re
import logging
from typing import List, Dict, Any, Tuple
import requests
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of fact verification"""
    claim: str
    is_supported: bool
    confidence: float  # 0.0 to 1.0
    supporting_sources: List[str]
    explanation: str


@dataclass
class OverallVerification:
    """Overall verification assessment"""
    verified_claims: List[VerificationResult]
    support_rate: float  # Percentage of claims supported
    overall_confidence: float
    warnings: List[str]
    is_trustworthy: bool


class FactVerificationAgent:
    """Verifies factual claims against source documents using database + LLM"""
    
    def __init__(self, config: Dict[str, Any], database_manager=None, ollama_host: str = "http://localhost:11434"):
        self.config = config
        self.ollama_host = ollama_host
        self.db = database_manager  # Database for exact numerical verification
        
        # Model selection
        ollama_config = config.get('ollama', {})
        self.model = ollama_config.get('model_verification', 'llama3.1')
        self.timeout = ollama_config.get('timeout_verification', 420)
        
        # Verification thresholds
        validation_config = config.get('validation', {})
        self.min_support_rate = validation_config.get('min_support_rate', 0.8)  # 80% claims must be supported
        self.min_confidence = validation_config.get('min_confidence', 0.7)  # 70% confidence threshold
        
        # Patterns for numerical claims that should be verified against database
        self.numerical_patterns = [
            r'\d+(?:\.\d+)?\s*(?:m|ft|meter|feet|inch|in|")',  # Depths, sizes
            r'\d+(?:\.\d+)?\s*(?:lb/ft|kg/m|ppf)',  # Weights
            r'\d+\s+\d+/\d+',  # Fractions (casing sizes)
            r'TD|TVD|total depth|true vertical',  # Depth references
            r'OD|ID|outer diameter|inner diameter|pipe\s+id',  # Diameter references
        ]
        
        logger.info(f"Fact verification using {self.model} + database, thresholds: support≥{self.min_support_rate*100}%, confidence≥{self.min_confidence*100}%")
    
    def verify(self, answer: str, source_chunks: List[Dict]) -> OverallVerification:
        """
        Verify factual claims in answer against source chunks
        
        Args:
            answer: Generated answer to verify
            source_chunks: Source chunks used to generate answer
            
        Returns:
            OverallVerification with detailed results
        """
        logger.info(f"Starting fact verification for answer ({len(answer)} chars) against {len(source_chunks)} sources")
        
        # Extract claims from answer
        claims = self._extract_claims(answer)
        logger.info(f"Extracted {len(claims)} factual claims")
        
        if not claims:
            logger.warning("No factual claims detected in answer")
            return OverallVerification(
                verified_claims=[],
                support_rate=1.0,  # No claims = vacuously true
                overall_confidence=0.5,
                warnings=["No factual claims detected in answer"],
                is_trustworthy=True
            )
        
        # Verify each claim
        verified_claims = []
        for claim in claims:
            result = self._verify_claim(claim, source_chunks)
            verified_claims.append(result)
            logger.info(f"  Claim: '{claim[:80]}...' → {'✓' if result.is_supported else '✗'} ({result.confidence:.2f})")
        
        # Calculate overall metrics
        supported_count = sum(1 for c in verified_claims if c.is_supported)
        support_rate = supported_count / len(verified_claims) if verified_claims else 0.0
        
        avg_confidence = sum(c.confidence for c in verified_claims) / len(verified_claims)
        
        # Generate warnings
        warnings = []
        if support_rate < self.min_support_rate:
            warnings.append(f"⚠️ Only {support_rate*100:.0f}% of claims are supported (threshold: {self.min_support_rate*100:.0f}%)")
        
        if avg_confidence < self.min_confidence:
            warnings.append(f"⚠️ Average confidence {avg_confidence*100:.0f}% below threshold ({self.min_confidence*100:.0f}%)")
        
        unsupported = [c for c in verified_claims if not c.is_supported]
        if unsupported:
            warnings.append(f"⚠️ {len(unsupported)} unsupported claims detected")
        
        is_trustworthy = (support_rate >= self.min_support_rate and 
                         avg_confidence >= self.min_confidence)
        
        logger.info(f"Verification complete: {support_rate*100:.0f}% supported, {avg_confidence*100:.0f}% confidence, {'✓ TRUSTWORTHY' if is_trustworthy else '✗ NEEDS REVIEW'}")
        
        return OverallVerification(
            verified_claims=verified_claims,
            support_rate=support_rate,
            overall_confidence=avg_confidence,
            warnings=warnings,
            is_trustworthy=is_trustworthy
        )
    
    def _extract_claims(self, answer: str) -> List[str]:
        """
        Extract factual claims from answer
        Focuses on statements with numbers, measurements, well names, etc.
        Skips meta-statements about missing information.
        """
        claims = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', answer)
        
        # Patterns indicating meta-statements about missing info (not factual claims)
        negative_patterns = [
            r'not provided',
            r'not mentioned',
            r'not explicitly',
            r'no information',
            r'no specific',
            r'not available',
            r'not stated',
            r'not given',
            r'does not contain',
            r'lack[s]? the answer',
            r'cannot be determined',
            r'are only given',
            r'is important to note that there are no',
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short fragments
                continue
            
            # Skip negative/meta statements - these aren't claims to verify
            is_negative = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in negative_patterns)
            if is_negative:
                continue
            
            # Check if sentence contains factual content
            has_number = bool(re.search(r'\d+', sentence))
            has_well_name = bool(re.search(r'\b(?:Well|well|hole)\s+[A-Z0-9\-]+\b', sentence))
            has_measurement = bool(re.search(r'\d+(?:\.\d+)?\s*(?:m|ft|inch|bar|psi|°C|°F|kg/m³|bbl|m³)', sentence))
            has_technical_term = bool(re.search(r'\b(?:casing|tubing|trajectory|depth|diameter|pressure|temperature|MD|TVD|ID|OD)\b', sentence, re.IGNORECASE))
            
            # Consider it a factual claim if it has specific technical content
            if has_measurement or (has_number and has_technical_term) or has_well_name:
                claims.append(sentence)
        
        return claims
    
    def _check_claim_in_database(self, claim: str) -> Optional[Tuple[bool, str]]:
        """
        Check if claim contains numerical data that can be verified against database
        
        Returns:
            Tuple of (is_supported, explanation) if database check applicable, None otherwise
        """
        if not self.db:
            return None
        
        # Check if claim contains numerical patterns
        has_numerical = any(re.search(pattern, claim, re.IGNORECASE) 
                           for pattern in self.numerical_patterns)
        
        if not has_numerical:
            return None  # Not a numerical claim, use semantic verification
        
        # Extract well name from claim
        well_pattern = re.compile(r'\b([A-Z]{2,10}-GT-\d{2}(?:-S\d+)?)\b')
        well_match = well_pattern.search(claim)
        
        if not well_match:
            return None  # No well name, can't query database
        
        well_name = well_match.group(1)
        
        try:
            # Get well data from database
            well_data = self.db.get_well_summary(well_name)
            
            if not well_data or not well_data.get('well_info'):
                return None  # No data in database
            
            # Convert claim and database data to comparable format
            claim_lower = claim.lower()
            
            # Check various technical data points
            well_info = well_data['well_info']
            casing_strings = well_data.get('casing_strings', [])
            
            # Simple heuristic checks (can be made more sophisticated)
            verified = False
            explanation = ""
            
            # Check TD/TVD
            if 'total depth' in claim_lower or 'td' in claim_lower:
                td_md = well_info.get('total_depth_md')
                if td_md:
                    # Extract depth from claim
                    depth_match = re.search(r'(\d+(?:\.\d+)?)\s*m', claim)
                    if depth_match:
                        claimed_depth = float(depth_match.group(1))
                        # Allow 1% tolerance
                        if abs(claimed_depth - td_md) / td_md < 0.01:
                            verified = True
                            explanation = f"Total depth {td_md}m matches database"
            
            # Check casing sizes
            if 'casing' in claim_lower or 'od' in claim_lower:
                for casing in casing_strings:
                    od = casing.get('outer_diameter')
                    if od:
                        od_str = str(od)
                        if od_str in claim or f"{od:.3f}" in claim:
                            verified = True
                            explanation = f"Casing OD {od}\" found in database"
                            break
            
            if verified:
                return (True, explanation)
            else:
                # Database has data but claim not verified
                return (False, "Numerical claim not found in database records")
                
        except Exception as e:
            logger.warning(f"Database verification error: {str(e)}")
            return None
    
    def _verify_claim(self, claim: str, source_chunks: List[Dict]) -> VerificationResult:
        """
        Verify a single claim against database (priority) and source chunks using LLM
        
        Args:
            claim: Factual claim to verify
            source_chunks: Source documents
            
        Returns:
            VerificationResult with support assessment
        """
        # First, try database verification for numerical claims
        db_result = self._check_claim_in_database(claim)
        if db_result is not None:
            is_supported, explanation = db_result
            logger.info(f"Database verification: {claim[:60]}... → {'✓' if is_supported else '✗'}")
            return VerificationResult(
                claim=claim,
                is_supported=is_supported,
                confidence=0.95 if is_supported else 0.85,  # High confidence from database
                supporting_sources=["Database"],
                explanation=f"[DB] {explanation}"
            )
        
        # Fall back to semantic verification via LLM
        # Build context from source chunks
        context_parts = []
        for i, chunk in enumerate(source_chunks[:10], 1):  # Use top 10 sources
            source = chunk['metadata'].get('source_file', 'unknown')
            pages = chunk['metadata'].get('page_numbers', ['?'])
            page_str = f"p.{pages[0]}" if pages else "p.?"
            context_parts.append(f"[Source {i} from {source}, {page_str}]\n{chunk['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Create verification prompt
        prompt = f"""You are a fact-checking assistant for geothermal well engineering documents. Your task is to verify if a claim is supported by the provided source documents.

CLAIM TO VERIFY:
{claim}

SOURCE DOCUMENTS:
{context}

INSTRUCTIONS:
1. Carefully read the claim and all source documents
2. Determine if the claim is directly supported by the sources
3. A claim is SUPPORTED if:
   - The exact information appears in the sources
   - Numbers, measurements, and units match
   - Well names and identifiers match
   - Technical details are consistent
4. A claim is NOT SUPPORTED if:
   - Information is not present in sources
   - Numbers or measurements differ
   - Details are added that aren't in sources
   - Claim contradicts source information

Provide your assessment in this EXACT format:
SUPPORTED: [Yes/No]
CONFIDENCE: [0-100]
SOURCES: [List source numbers that support this, e.g., "1, 3, 5" or "None"]
EXPLANATION: [Brief explanation of your reasoning in 1-2 sentences]

Your assessment:"""
        
        try:
            # Call LLM for verification
            response = self._call_ollama(prompt)
            
            # Parse response
            supported = self._parse_yes_no(response, 'SUPPORTED')
            confidence = self._parse_confidence(response)
            sources = self._parse_sources(response, source_chunks)
            explanation = self._parse_explanation(response)
            
            return VerificationResult(
                claim=claim,
                is_supported=supported,
                confidence=confidence,
                supporting_sources=sources,
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Verification failed for claim: {str(e)}")
            # Return conservative result on error
            return VerificationResult(
                claim=claim,
                is_supported=False,
                confidence=0.0,
                supporting_sources=[],
                explanation=f"Verification error: {str(e)}"
            )
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API for verification"""
        url = f"{self.ollama_host}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Low temperature for factual verification
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        
        result = response.json()
        return result['response'].strip()
    
    def _parse_yes_no(self, response: str, field: str) -> bool:
        """Parse Yes/No field from response"""
        pattern = f"{field}:\\s*(Yes|No|yes|no|YES|NO)"
        match = re.search(pattern, response)
        if match:
            return match.group(1).lower() == 'yes'
        return False  # Conservative default
    
    def _parse_confidence(self, response: str) -> float:
        """Parse confidence score from response"""
        pattern = r"CONFIDENCE:\s*(\d+)"
        match = re.search(pattern, response)
        if match:
            return int(match.group(1)) / 100.0
        return 0.5  # Default to medium confidence
    
    def _parse_sources(self, response: str, chunks: List[Dict]) -> List[str]:
        """Parse supporting source references"""
        pattern = r"SOURCES:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, response)
        if not match:
            return []
        
        sources_text = match.group(1).strip()
        if sources_text.lower() in ['none', 'n/a', '-']:
            return []
        
        # Extract source numbers
        source_nums = re.findall(r'\d+', sources_text)
        
        # Map to source file names
        source_files = []
        for num in source_nums:
            idx = int(num) - 1
            if 0 <= idx < len(chunks):
                source = chunks[idx]['metadata'].get('source_file', f'Source {num}')
                if source not in source_files:
                    source_files.append(source)
        
        return source_files
    
    def _parse_explanation(self, response: str) -> str:
        """Parse explanation from response"""
        pattern = r"EXPLANATION:\s*(.+?)(?:\n\n|\Z)"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return "No explanation provided"


def create_agent(config: Dict[str, Any]) -> FactVerificationAgent:
    """Factory function to create agent"""
    ollama_host = config.get('ollama', {}).get('host', 'http://localhost:11434')
    return FactVerificationAgent(config, ollama_host)
