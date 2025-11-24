"""
LLM Helper - Ollama Integration for Q&A and Summarization
Provides functions to generate coherent answers using retrieved context
"""

import requests
import logging
from typing import List, Dict, Optional
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaHelper:
    """Helper class for Ollama LLM interactions"""
    
    def __init__(self, config_path: str = None):
        """Initialize Ollama helper"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.ollama_config = self.config['ollama']
        self.host = self.ollama_config['host']
        
        # Model selection per task (more capable models for validation)
        self.model_qa = self.ollama_config.get('model_qa', 'llama3')
        self.model_summary = self.ollama_config.get('model_summary', 'llama3.1')
        self.model_verification = self.ollama_config.get('model_verification', 'llama3.1')
        self.model_extraction = self.ollama_config.get('model_extraction', 'llama3')
        
        # Timeouts (extended for deep validation)
        self.timeout = self.ollama_config.get('timeout', 420)  # 7 minutes default
        self.timeout_summary = self.ollama_config.get('timeout_summary', 420)
        self.timeout_extraction = self.ollama_config.get('timeout_extraction', 420)
        self.timeout_verification = self.ollama_config.get('timeout_verification', 420)
        
        # Summarization config
        summary_config = self.config.get('summarization', {})
        self.default_word_count = summary_config.get('default_words')  # Can be null
        self.word_count_tolerance = summary_config.get('tolerance_percent', 20) / 100.0
        self.max_retries = summary_config.get('max_retries', 1)
        
        logger.info(f"Ollama models: QA={self.model_qa}, Summary={self.model_summary}, Verification={self.model_verification}")
        if self.default_word_count:
            logger.info(f"Timeouts: {self.timeout}s, Default summary: {self.default_word_count} words (±{int(self.word_count_tolerance*100)}%)")
        else:
            logger.info(f"Timeouts: {self.timeout}s, Default summary: No word limit (as much as needed)")
    
    def generate_answer(self, question: str, context_chunks: List[Dict], 
                       max_tokens: Optional[int] = None) -> str:
        """
        Generate answer to question using context from chunks
        
        Args:
            question: User's question
            context_chunks: List of chunk dicts with 'text' field
            max_tokens: Optional limit on response length
            
        Returns:
            Generated answer as string
        """
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks[:5], 1):  # Use top 5 chunks
            source = chunk['metadata'].get('source_file', 'unknown')
            pages = chunk['metadata'].get('page_numbers', ['?'])
            page_str = f"p.{pages[0]}" if pages else "p.?"
            
            context_parts.append(
                f"[Source {i} from {source}, {page_str}]\n{chunk['text']}\n"
            )
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = self._create_qa_prompt(question, context)
        
        try:
            response = self._call_ollama(prompt, max_tokens)
            return response
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            # Fallback: return first chunk excerpt
            if context_chunks:
                return f"Based on the document: {context_chunks[0]['text'][:500]}..."
            return "Unable to generate answer."
    
    def generate_summary(self, chunks: List[Dict], target_words: int = None, 
                        focus: Optional[str] = None) -> str:
        """
        Generate summary from chunks with STRICT word count enforcement
        
        Args:
            chunks: List of chunk dicts with 'text' field
            target_words: Target word count (uses default if not specified)
            focus: Optional focus area (e.g., "well trajectory and casing")
            
        Returns:
            Generated summary as string with strict word count
        """
        # Use default if not specified (can be null = no limit)
        if target_words is None:
            target_words = self.default_word_count
            if target_words is None:
                # No limit - generate comprehensive summary
                logger.info("No word count limit - generating comprehensive summary")
                target_words = 0  # Signal to skip word count enforcement
            else:
                logger.info(f"No word count specified - using default: {target_words} words")
        
        # Get citation settings
        summary_config = self.config.get('summarization', {})
        enable_citations = summary_config.get('enable_citations', True)
        
        # Combine chunks - use more chunks for comprehensive summaries with source metadata
        content_parts = []
        for i, chunk in enumerate(chunks[:20], 1):  # Use up to 20 chunks for comprehensive summary
            # Add source markers with page numbers for citations
            source = chunk['metadata'].get('source_file', 'unknown')
            pages = chunk['metadata'].get('page_numbers', [])
            page_str = f", Page {pages[0]}" if pages else ""
            
            content_parts.append(f"[Section {i} from {source}{page_str}]\n{chunk['text']}")
        
        content = "\n\n".join(content_parts)
        
        logger.info(f"Preparing summary from {len(content_parts)} chunks (~{len(content)} chars)")
        logger.info(f"Citations enabled: {enable_citations}")
        
        # Try with retries (only if word count specified)
        max_attempts = (self.max_retries + 1) if target_words > 0 else 1
        
        for attempt in range(max_attempts):
            # Create prompt with word count instruction and citations
            prompt = self._create_summary_prompt(content, target_words, focus, attempt > 0, enable_citations)
            
            try:
                if target_words > 0:
                    logger.info(f"Generating summary (target: {target_words} words, attempt {attempt+1}/{max_attempts}, timeout={self.timeout_summary}s)...")
                    max_tokens = int(target_words * 2.5)
                else:
                    logger.info(f"Generating comprehensive summary (no word limit, timeout={self.timeout_summary}s)...")
                    max_tokens = 4000  # Allow long comprehensive summaries
                    
                response = self._call_ollama(
                    prompt, 
                    max_tokens=max_tokens,
                    timeout=self.timeout_summary,
                    model=self.model_summary
                )
                
                # Count words and validate (only if target specified)
                word_count = len(response.split())
                
                if target_words == 0:
                    # No word count enforcement - accept result
                    logger.info(f"✓ Generated comprehensive summary ({word_count} words)")
                    return response
                else:
                    tolerance = int(target_words * self.word_count_tolerance)
                    min_words = target_words - tolerance
                    max_words = target_words + tolerance
                    
                    logger.info(f"Generated {word_count} words (target: {target_words} ±{tolerance})")
                    
                    if min_words <= word_count <= max_words:
                        logger.info(f"✓ Word count within tolerance")
                        return response
                    elif attempt < self.max_retries:
                        logger.warning(f"✗ Word count {word_count} outside range [{min_words}, {max_words}], retrying...")
                    else:
                        logger.warning(f"✗ Word count {word_count} outside range after {self.max_retries} retries, accepting result")
                        return response
                    
            except Exception as e:
                logger.error(f"Summary generation failed (attempt {attempt+1}): {str(e)}")
                if attempt == self.max_retries:
                    # Fallback: return bullet points from chunks
                    summary = f"Summary (from {len(chunks)} sections):\n\n"
                    for i, chunk in enumerate(chunks[:8], 1):
                        summary += f"• {chunk['text'][:200]}...\n\n"
                    return summary
    
    def _create_qa_prompt(self, question: str, context: str) -> str:
        """Create prompt for Q&A with strict grounding requirements"""
        prompt = f"""You are a technical assistant for geothermal well engineering. Answer the question using ONLY information from the provided context.

Context from well reports:
{context}

Question: {question}

CRITICAL INSTRUCTIONS:
1. GROUNDING REQUIREMENT:
   - Every fact, number, date, or detail in your answer MUST come directly from the context above
   - Quote exact phrases when possible (e.g., "The report states '...'")
   - Do NOT add information from general knowledge
   - Do NOT make assumptions or inferences beyond what's explicitly stated
   - Do NOT use placeholder values or typical ranges

2. ANSWER FORMAT:
   - Start with direct answer to the question
   - Include ALL relevant numbers with units (e.g., "2642 m MD", "9 5/8 inch")
   - Cite specific sections: "According to [document name], page X..."
   - If multiple sources mention the same fact, cite all

3. MISSING INFORMATION:
   - If context lacks the answer: "The provided documents do not contain information about [specific detail]"
   - If context has partial info: State what IS available, then note what's missing
   - Never fabricate or estimate missing data

4. TECHNICAL PRECISION:
   - Use exact terminology from the documents
   - Preserve all significant figures and units
   - Include well names, operator names, dates when mentioned
   - Note any uncertainties or ranges stated in documents

Answer (grounded strictly in context):"""
        return prompt
    
    def _create_summary_prompt(self, content: str, target_words: int, 
                              focus: Optional[str] = None, is_retry: bool = False,
                              enable_citations: bool = True) -> str:
        """Create prompt for summarization with strict word count, grounding, and citations"""
        focus_text = f" with emphasis on {focus}" if focus else ""
        
        # Handle no word count limit
        if target_words == 0:
            word_count_instruction = "Write a COMPREHENSIVE and DETAILED summary. Include ALL important information. No word limit."
            word_count_range = "comprehensive (no limit)"
        else:
            # Stricter instructions on retry
            strictness = ""
            if is_retry:
                strictness = f"\n⚠️ CRITICAL: Previous attempt had incorrect word count. You MUST produce EXACTLY {target_words} words (±{int(self.word_count_tolerance*100)}%). Count carefully."
            word_count_instruction = f"Your summary MUST be EXACTLY {target_words} words (±{int(self.word_count_tolerance*100)}% = {int(target_words * (1 - self.word_count_tolerance))}-{int(target_words * (1 + self.word_count_tolerance))} words){strictness}"
            word_count_range = f"{target_words} words"
        
        # Citation instructions
        citation_text = ""
        if enable_citations:
            citation_text = """

6. CITATIONS (MANDATORY - NO EXCEPTIONS):
   - After EVERY sentence with factual content, add: [Source: filename, Page X]
   - EVERY depth value MUST have a citation
   - EVERY pipe specification MUST have a citation
   - EVERY measurement MUST have a citation
   - EVERY date MUST have a citation
   - EVERY operator/well name MUST have a citation
   
   Examples of CORRECT citation format:
   ✓ "The ADK-GT-01 well reached 2667.5m MD and 2358m TVD [Source: NLOG_GS_PUB_EOWR ADK-GT-01 SODM v1.1.pdf, Page 12]"
   ✓ "A 9 5/8 inch casing with 53.5 lb/ft weight was set at 2642m MD [Source: completion_report.pdf, Page 8]"
   ✓ "Drilling commenced on January 15, 2017 [Source: operations_log.pdf, Page 3]"
   
   ✗ WRONG: "The well reached 2667.5m MD" (missing citation)
   ✗ WRONG: "Casing installed at 2642m" (missing citation)
   
   Format: [Source: EXACT_FILENAME.pdf, Page XX]"""
        
        prompt = f"""You are a technical assistant for geothermal well engineering. Create a factual summary of the following well report content{focus_text}.

Content from well report:
{content}

STRICT REQUIREMENTS:
1. GROUNDING (Most Important):
   - Include ONLY information explicitly present in the content above
   - Use exact numbers, dates, and names from the content
   - Do NOT add general knowledge about geothermal wells
   - Do NOT infer or assume details not stated
   - Do NOT use placeholder values like "approximately" or "around"
   - When mentioning operations/equipment, use exact terminology from content
   - EVERY claim must be verifiable from the content above

2. WORD COUNT: {word_count_instruction}

3. CONTENT PRIORITY (include in order until word limit):
   - Well name, operator, dates (if present)
   - Total depth: EXACT Measured Depth (MD) and True Vertical Depth (TVD) values
     ⚠️ CRITICAL: MD (Measured Depth) is ALWAYS ≥ TVD (True Vertical Depth)
     MD = length along wellbore path, TVD = straight vertical depth
     In vertical wells: MD ≈ TVD. In deviated wells: MD > TVD
     Example: "2667.5m MD, 2358m TVD" ✓ CORRECT (MD > TVD)
     Example: "2358m MD, 2667.5m TVD" ✗ WRONG (TVD cannot exceed MD)
   - Casing program: EXACT specifications (e.g., "9 5/8 inch, 53.5 lb/ft, L80, set at 2642m MD")
   - Key operations mentioned (drilling, completion, testing) with dates
   - Equipment used (exact names/types from content)
   - Measurements, test results, or findings with exact units
   - Formation tops and geology (exact depths and formation names)
   - Issues or special conditions noted (with depths/dates)

4. FORMAT:
   - Write as continuous technical prose (not bullet points)
   - Use exact units from content (m, inch, bar, ft, etc.)
   - Preserve all significant figures (e.g., "2667.5 m" not "2668 m")
   - Name specific formations, zones, or targets if mentioned
   - Include well name prominently{citation_text}

5. FORBIDDEN:
   - ❌ Generic statements like "typical completion procedures"
   - ❌ Vague terms: "approximately", "around", "about"
   - ❌ Assumed values or estimates
   - ❌ Operations not mentioned in content
   - ❌ Equipment not specifically named
   - ❌ Standard industry practice not stated in content
   - ❌ Claiming information without citation (if citations enabled)

Factual Summary ({word_count_range}, grounded only in provided content, with citations for EVERY claim):"""
        return prompt
    
    def _call_ollama(self, prompt: str, max_tokens: Optional[int] = None, 
                     timeout: Optional[int] = None, model: Optional[str] = None) -> str:
        """
        Call Ollama API
        
        Args:
            prompt: Prompt string
            max_tokens: Maximum tokens in response
            timeout: Custom timeout (uses default if not specified)
            model: Model to use (overrides default)
            
        Returns:
            Generated text
        """
        url = f"{self.host}/api/generate"
        
        payload = {
            "model": model if model else self.model_qa,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,  # Balanced creativity
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        timeout_value = timeout if timeout is not None else self.timeout
        
        response = requests.post(url, json=payload, timeout=timeout_value)
        response.raise_for_status()
        
        result = response.json()
        return result['response'].strip()
    
    def is_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

