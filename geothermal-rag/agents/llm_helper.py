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
        self.model = self.ollama_config['model_qa']
        self.timeout = self.ollama_config.get('timeout', 300)
        self.timeout_summary = self.ollama_config.get('timeout_summary', 600)
        self.timeout_extraction = self.ollama_config.get('timeout_extraction', 300)
        
        logger.info(f"Ollama timeouts: Q&A={self.timeout}s, Summary={self.timeout_summary}s, Extraction={self.timeout_extraction}s")
    
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
    
    def generate_summary(self, chunks: List[Dict], target_words: int = 200, 
                        focus: Optional[str] = None) -> str:
        """
        Generate summary from chunks with specific word count
        
        Args:
            chunks: List of chunk dicts with 'text' field
            target_words: Target word count for summary
            focus: Optional focus area (e.g., "well trajectory and casing")
            
        Returns:
            Generated summary as string
        """
        # Combine chunks - use more chunks for comprehensive summaries
        content_parts = []
        for i, chunk in enumerate(chunks[:20], 1):  # Use up to 20 chunks for comprehensive summary
            # Add source markers for traceability
            source = chunk['metadata'].get('source_file', 'unknown')
            content_parts.append(f"[Section {i} from {source}]\n{chunk['text']}")
        
        content = "\n\n".join(content_parts)
        
        logger.info(f"Preparing summary from {len(content_parts)} chunks (~{len(content)} chars)")
        
        # Create prompt with word count instruction
        prompt = self._create_summary_prompt(content, target_words, focus)
        
        try:
            logger.info(f"Generating summary (~{target_words} words, timeout={self.timeout_summary}s)...")
            response = self._call_ollama(prompt, max_tokens=target_words * 2, timeout=self.timeout_summary)
            return response
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            # Fallback: return bullet points from chunks
            summary = f"Summary (from {len(chunks)} sections):\n\n"
            for i, chunk in enumerate(chunks[:8], 1):
                summary += f"• {chunk['text'][:200]}...\n\n"
            return summary
    
    def _create_qa_prompt(self, question: str, context: str) -> str:
        """Create prompt for Q&A"""
        prompt = f"""You are a technical assistant for geothermal well engineering. Answer the question based ONLY on the provided context from well reports.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer directly and precisely based on the context
- Cite which source(s) you used (e.g., "According to Source 1...")
- If the context doesn't contain the answer, say "The provided documents do not contain information about..."
- Be technical and specific with numbers, units, and well names
- Keep your answer focused and concise

Answer:"""
        return prompt
    
    def _create_summary_prompt(self, content: str, target_words: int, 
                              focus: Optional[str] = None) -> str:
        """Create prompt for summarization"""
        focus_text = f" with emphasis on {focus}" if focus else ""
        
        prompt = f"""You are a technical assistant for geothermal well engineering. Create a comprehensive summary of the following well report content{focus_text}.

Content from well report:
{content}

Instructions for creating an excellent technical summary:
1. Target length: Approximately {target_words} words (±20% is acceptable for completeness)
2. Structure your summary logically:
   - Well identification and overview
   - Key depths and trajectory characteristics
   - Casing design and specifications
   - Equipment and completion details
   - Fluid properties and production parameters (if mentioned)
   - Any significant findings or observations

3. Technical requirements:
   - Use proper engineering terminology
   - Include ALL specific numbers, depths, diameters, and units
   - Mention well names, operators, and dates when present
   - Cite key measurements and test results
   - Note any anomalies or special conditions

4. Quality standards:
   - Write in clear, professional technical language
   - Organize information hierarchically (general → specific)
   - Ensure factual accuracy - only include what's in the content
   - Make it useful for engineers who need quick understanding

Technical Summary:"""
        return prompt
    
    def _call_ollama(self, prompt: str, max_tokens: Optional[int] = None, 
                     timeout: Optional[int] = None) -> str:
        """
        Call Ollama API
        
        Args:
            prompt: Prompt string
            max_tokens: Maximum tokens in response
            timeout: Custom timeout (uses default if not specified)
            
        Returns:
            Generated text
        """
        url = f"{self.host}/api/generate"
        
        payload = {
            "model": self.model,
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

