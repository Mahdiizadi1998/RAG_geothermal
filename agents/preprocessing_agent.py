"""Text preprocessing and chunking agent."""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_id: str
    page_num: Optional[int] = None
    chunk_type: str = "text"  # 'text', 'table', 'heading', etc.
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PreprocessingAgent:
    """
    Agent responsible for text preprocessing and chunking.
    
    Implements multiple chunking strategies:
    1. Semantic chunking (paragraph-based)
    2. Table-aware chunking (preserves table structure)
    3. Sentence-based chunking
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessing agent.
        
        Args:
            config: Configuration dict with chunking parameters
        """
        self.config = config or {}
        self.max_chunk_size = self.config.get('max_chunk_size', 500)
        self.overlap = self.config.get('overlap', 50)
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (heuristic)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n')
        
        return text.strip()
    
    def detect_tables(self, text: str) -> List[Dict]:
        """
        Detect table-like structures in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected tables with start/end positions
        """
        tables = []
        lines = text.split('\n')
        
        in_table = False
        table_start = 0
        table_lines = []
        
        for i, line in enumerate(lines):
            # Heuristics for table detection:
            # 1. Multiple numbers separated by whitespace
            # 2. Contains pipe characters |
            # 3. Consistent column structure
            
            has_pipes = '|' in line
            has_multiple_numbers = len(re.findall(r'\d+\.?\d*', line)) >= 3
            has_tabs = '\t' in line
            
            is_table_line = has_pipes or (has_multiple_numbers and (has_tabs or len(line.split()) >= 4))
            
            if is_table_line and not in_table:
                in_table = True
                table_start = i
                table_lines = [line]
            elif is_table_line and in_table:
                table_lines.append(line)
            elif not is_table_line and in_table:
                # End of table
                tables.append({
                    'start_line': table_start,
                    'end_line': i - 1,
                    'content': '\n'.join(table_lines)
                })
                in_table = False
                table_lines = []
        
        # Handle table at end of text
        if in_table:
            tables.append({
                'start_line': table_start,
                'end_line': len(lines) - 1,
                'content': '\n'.join(table_lines)
            })
        
        return tables
    
    def chunk_by_semantics(self, text: str, page_num: Optional[int] = None) -> List[Chunk]:
        """
        Chunk text by semantic units (paragraphs, sections).
        
        Args:
            text: Text to chunk
            page_num: Page number for metadata
            
        Returns:
            List of chunks
        """
        chunks = []
        
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds max size, save current chunk
            if len(current_chunk) + len(para) > self.max_chunk_size and current_chunk:
                chunks.append(Chunk(
                    text=current_chunk,
                    chunk_id=f"semantic_{chunk_id}",
                    page_num=page_num,
                    chunk_type="semantic"
                ))
                chunk_id += 1
                
                # Keep overlap
                words = current_chunk.split()
                if len(words) > 10:
                    overlap_text = ' '.join(words[-10:])
                    current_chunk = overlap_text + ' ' + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += '\n\n' + para
                else:
                    current_chunk = para
        
        # Add final chunk
        if current_chunk:
            chunks.append(Chunk(
                text=current_chunk,
                chunk_id=f"semantic_{chunk_id}",
                page_num=page_num,
                chunk_type="semantic"
            ))
        
        return chunks
    
    def chunk_by_sentences(self, text: str, page_num: Optional[int] = None) -> List[Chunk]:
        """
        Chunk text by sentences.
        
        Args:
            text: Text to chunk
            page_num: Page number for metadata
            
        Returns:
            List of chunks
        """
        chunks = []
        
        # Simple sentence splitting (can be improved with spaCy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        chunk_id = 0
        sentence_count = 0
        min_sentences = 3
        max_sentences = 10
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            if sentence_count >= min_sentences and (
                len(current_chunk) + len(sent) > self.max_chunk_size or 
                sentence_count >= max_sentences
            ):
                chunks.append(Chunk(
                    text=current_chunk,
                    chunk_id=f"sentence_{chunk_id}",
                    page_num=page_num,
                    chunk_type="sentence"
                ))
                chunk_id += 1
                current_chunk = sent
                sentence_count = 1
            else:
                if current_chunk:
                    current_chunk += ' ' + sent
                else:
                    current_chunk = sent
                sentence_count += 1
        
        # Add final chunk
        if current_chunk:
            chunks.append(Chunk(
                text=current_chunk,
                chunk_id=f"sentence_{chunk_id}",
                page_num=page_num,
                chunk_type="sentence"
            ))
        
        return chunks
    
    def process_page(self, page_text: str, page_num: int, strategy: str = "semantic") -> List[Chunk]:
        """
        Process a single page with specified chunking strategy.
        
        Args:
            page_text: Text from the page
            page_num: Page number
            strategy: Chunking strategy ('semantic', 'sentence', 'table-aware')
            
        Returns:
            List of chunks
        """
        # Clean text
        text = self.clean_text(page_text)
        
        if strategy == "semantic":
            return self.chunk_by_semantics(text, page_num)
        elif strategy == "sentence":
            return self.chunk_by_sentences(text, page_num)
        elif strategy == "table-aware":
            # Detect tables first
            tables = self.detect_tables(text)
            
            if not tables:
                return self.chunk_by_semantics(text, page_num)
            
            chunks = []
            lines = text.split('\n')
            
            # Process non-table sections
            last_end = 0
            for table in tables:
                # Text before table
                if table['start_line'] > last_end:
                    pre_text = '\n'.join(lines[last_end:table['start_line']])
                    chunks.extend(self.chunk_by_semantics(pre_text, page_num))
                
                # Table itself
                chunks.append(Chunk(
                    text=table['content'],
                    chunk_id=f"table_{page_num}_{table['start_line']}",
                    page_num=page_num,
                    chunk_type="table"
                ))
                
                last_end = table['end_line'] + 1
            
            # Text after last table
            if last_end < len(lines):
                post_text = '\n'.join(lines[last_end:])
                chunks.extend(self.chunk_by_semantics(post_text, page_num))
            
            return chunks
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def process_document(self, pages: List, strategy: str = "table-aware") -> List[Chunk]:
        """
        Process all pages in document.
        
        Args:
            pages: List of PageContent objects
            strategy: Chunking strategy
            
        Returns:
            List of all chunks from the document
        """
        all_chunks = []
        
        for page in pages:
            chunks = self.process_page(page.text, page.page_num, strategy)
            all_chunks.extend(chunks)
        
        return all_chunks
