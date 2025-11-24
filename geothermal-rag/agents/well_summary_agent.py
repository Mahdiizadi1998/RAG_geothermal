"""
Well Summary Agent - 3-Pass Document Summarization
Generates End of Well Summary reports using structured extraction approach

Three-pass strategy:
1. Metadata Extraction (Key-Value): Operator, Well, Rig, Dates
2. Technical Specs (Table-to-Markdown): Casing/Tubing/Liner with ID
3. Narrative Extraction (Section-Restricted): Geology hazards and instabilities
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import pdfplumber for table extraction
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
    logger.info("pdfplumber available for table extraction")
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available - table extraction will be limited. Install: pip install pdfplumber")


class WellSummaryAgent:
    """
    Generates comprehensive End of Well summaries using a three-pass approach
    
    Pass 1: Metadata Extraction (Key-Value)
    - Scan first 3 pages
    - Use regex for dates (Spud Date, End Date)
    - Use LLM for Operator, Well Name, Rig Name
    - Compute Days_Total
    
    Pass 2: Technical Specs Extraction (Table-to-Markdown)
    - Search for Casing/Tubing/Liner tables
    - Extract tables with pdfplumber
    - Convert to Markdown
    - Use LLM with JSON schema to extract pipe_id (ID column)
    
    Pass 3: Narrative Extraction (Section-Restricted)
    - Locate Geology/Lithology sections
    - Extract formation instabilities, gas peaks, drilling hazards
    """
    
    def __init__(self, llm_helper=None):
        """
        Initialize Well Summary Agent
        
        Args:
            llm_helper: OllamaHelper instance for LLM calls (optional)
        """
        self.llm = llm_helper
        self.llm_available = llm_helper is not None and llm_helper.is_available()
        
        # Regex patterns for Pass 1 (Metadata)
        self.date_patterns = {
            'spud_date': [
                r'Spud\s+Date[:\s]+(\d{1,2}[\s\-/\.]\w+[\s\-/\.]\d{2,4})',
                r'Start\s+Date[:\s]+(\d{1,2}[\s\-/\.]\w+[\s\-/\.]\d{2,4})',
                r'Commenced[:\s]+(\d{1,2}[\s\-/\.]\w+[\s\-/\.]\d{2,4})',
            ],
            'end_date': [
                r'(?:End|Completion|Finish)\s+Date[:\s]+(\d{1,2}[\s\-/\.]\w+[\s\-/\.]\d{2,4})',
                r'Rig\s+Release[:\s]+(\d{1,2}[\s\-/\.]\w+[\s\-/\.]\d{2,4})',
                r'TD\s+Reached[:\s]+(\d{1,2}[\s\-/\.]\w+[\s\-/\.]\d{2,4})',
            ]
        }
        
        # Section keywords for Pass 3 (Narrative)
        self.geology_keywords = [
            'geology', 'lithology', 'formation', 'stratigraphy',
            'drilling hazards', 'gas shows', 'formation instability'
        ]
        
        logger.info(f"WellSummaryAgent initialized (LLM available: {self.llm_available}, pdfplumber: {PDFPLUMBER_AVAILABLE})")
    
    def generate_summary(self, pdf_path: str, document_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate comprehensive End of Well Summary
        
        Args:
            pdf_path: Path to PDF file
            document_data: Optional pre-processed document data from IngestionAgent
            
        Returns:
            Dict containing:
            {
                'metadata': Dict (Pass 1 results),
                'technical_specs': Dict (Pass 2 results),
                'narrative': Dict (Pass 3 results),
                'summary_report': str (formatted final report),
                'confidence': float
            }
        """
        logger.info(f"Generating 3-pass summary for: {pdf_path}")
        
        # Pass 1: Metadata Extraction
        logger.info("⏳ Pass 1: Extracting metadata...")
        metadata = self._pass1_metadata_extraction(pdf_path)
        logger.info(f"✓ Pass 1 complete: Found {len(metadata)} metadata fields")
        
        # Pass 2: Technical Specs (Tables)
        logger.info("⏳ Pass 2: Extracting technical specs...")
        technical_specs = self._pass2_technical_specs_extraction(pdf_path)
        logger.info(f"✓ Pass 2 complete: Found {len(technical_specs.get('casing_program', []))} casing strings")
        
        # Pass 3: Narrative Extraction
        logger.info("⏳ Pass 3: Extracting narrative...")
        narrative = self._pass3_narrative_extraction(pdf_path, document_data)
        logger.info(f"✓ Pass 3 complete: Extracted narrative sections")
        
        # Generate final report
        logger.info("⏳ Generating final summary report...")
        summary_report = self._generate_summary_report(metadata, technical_specs, narrative)
        logger.info("✓ Summary report generated")
        
        # Calculate confidence
        confidence = self._calculate_confidence(metadata, technical_specs, narrative)
        
        return {
            'metadata': metadata,
            'technical_specs': technical_specs,
            'narrative': narrative,
            'summary_report': summary_report,
            'confidence': confidence
        }
    
    def _pass1_metadata_extraction(self, pdf_path: str) -> Dict[str, Any]:
        """
        Pass 1: Metadata Extraction (Key-Value Way)
        
        Logic:
        1. Scan first 3 pages only
        2. Use Regex for dates (Spud Date, End Date)
        3. Use ollama to extract Operator, Well Name, Rig Name from header text
        4. Compute Days_Total = (End Date - Spud Date)
        
        Returns:
            Dict with keys: operator_name, well_name, rig_name, spud_date, end_date, days_total
        """
        metadata = {}
        
        # Open PDF and extract first 3 pages
        doc = fitz.open(pdf_path)
        header_text = ""
        for page_num in range(min(3, len(doc))):
            header_text += doc[page_num].get_text() + "\n\n"
        doc.close()
        
        # Extract dates using regex first (more reliable than LLM)
        spud_date_str = self._extract_date(header_text, 'spud_date')
        end_date_str = self._extract_date(header_text, 'end_date')
        
        metadata['spud_date'] = spud_date_str
        metadata['end_date'] = end_date_str
        
        # Compute Days_Total if both dates found
        if spud_date_str and end_date_str:
            try:
                spud = self._parse_date(spud_date_str)
                end = self._parse_date(end_date_str)
                if spud and end:
                    days_total = (end - spud).days
                    metadata['days_total'] = days_total
                    logger.info(f"Computed Days_Total: {days_total} days")
            except Exception as e:
                logger.warning(f"Could not compute days_total: {e}")
                metadata['days_total'] = None
        else:
            metadata['days_total'] = None
        
        # Use LLM to extract names (Operator, Well, Rig) if available
        if self.llm_available:
            try:
                names_data = self._extract_names_with_llm(header_text)
                metadata.update(names_data)
            except Exception as e:
                logger.error(f"LLM name extraction failed: {e}")
                # Fallback to regex patterns
                metadata.update(self._extract_names_with_regex(header_text))
        else:
            # Use regex fallback
            metadata.update(self._extract_names_with_regex(header_text))
        
        return metadata
    
    def _extract_date(self, text: str, date_type: str) -> Optional[str]:
        """Extract date using regex patterns"""
        patterns = self.date_patterns.get(date_type, [])
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1).strip()
                logger.info(f"Found {date_type}: {date_str}")
                return date_str
        
        return None
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object"""
        # Try multiple date formats
        date_formats = [
            '%d %B %Y', '%d %b %Y',  # 15 January 2023
            '%d-%m-%Y', '%d/%m/%Y', '%d.%m.%Y',  # 15-01-2023
            '%d-%m-%y', '%d/%m/%y', '%d.%m.%y',  # 15-01-23
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        logger.warning(f"Could not parse date: {date_str}")
        return None
    
    def _extract_names_with_llm(self, header_text: str) -> Dict[str, Optional[str]]:
        """Extract operator, well, and rig names using LLM"""
        prompt = f"""Extract the following information from this well report header text:

Header Text:
{header_text[:2000]}

Extract these fields (return "null" if not found):
1. Operator Name (company drilling the well)
2. Well Name (e.g., ADK-GT-01)
3. Rig Name (drilling rig used)

Return your answer as valid JSON only, no other text:
{{
    "operator_name": "...",
    "well_name": "...",
    "rig_name": "..."
}}

JSON response:"""
        
        response = self.llm._call_ollama(prompt, max_tokens=200, model=self.llm.model_summary)
        
        # Parse JSON robustly
        names_data = self._parse_json_response(response)
        
        return {
            'operator_name': names_data.get('operator_name'),
            'well_name': names_data.get('well_name'),
            'rig_name': names_data.get('rig_name')
        }
    
    def _extract_names_with_regex(self, text: str) -> Dict[str, Optional[str]]:
        """Fallback regex extraction for names"""
        names = {
            'operator_name': None,
            'well_name': None,
            'rig_name': None
        }
        
        # Well name pattern (e.g., ADK-GT-01)
        well_pattern = r'\b([A-Z]{2,10}-GT-\d{2}(?:-S\d+)?)\b'
        well_match = re.search(well_pattern, text)
        if well_match:
            names['well_name'] = well_match.group(1)
        
        # Operator pattern (look for "Operator:" or "Company:")
        operator_pattern = r'(?:Operator|Company)[:\s]+([A-Z][A-Za-z\s&\.]+?)(?:\n|,|\s{2,})'
        operator_match = re.search(operator_pattern, text, re.IGNORECASE)
        if operator_match:
            names['operator_name'] = operator_match.group(1).strip()
        
        # Rig pattern
        rig_pattern = r'Rig[:\s]+([A-Za-z0-9\s\-]+?)(?:\n|,|\s{2,})'
        rig_match = re.search(rig_pattern, text, re.IGNORECASE)
        if rig_match:
            names['rig_name'] = rig_match.group(1).strip()
        
        return names
    
    def _pass2_technical_specs_extraction(self, pdf_path: str) -> Dict[str, Any]:
        """
        Pass 2: Technical Specs Extraction (Table-to-Markdown Way)
        
        Logic:
        1. Search for tables containing keywords: "Casing", "Tubing", "Liner"
        2. Use pdfplumber to extract tables
        3. Convert to Markdown strings
        4. Send Markdown to ollama with JSON schema instruction to extract:
           - size (Inner Diameter)
           - weight (ppf)
           - depth (Setting Depth)
           - pipe_id (ID column - look for 'ID', 'I.D.', 'Inside Diam')
        
        Returns:
            Dict with key 'casing_program': List[Dict] with extracted data
        """
        casing_program = []
        
        if not PDFPLUMBER_AVAILABLE:
            logger.warning("pdfplumber not available - skipping table extraction")
            return {'casing_program': casing_program, 'tables_markdown': []}
        
        try:
            # Open PDF with pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                tables_markdown = []
                
                # Search all pages for tables with casing keywords
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    
                    # Check if page contains casing/tubing/liner keywords
                    if not any(kw in page_text.lower() for kw in ['casing', 'tubing', 'liner', 'pipe']):
                        continue
                    
                    # Extract tables from this page
                    tables = page.extract_tables()
                    
                    if not tables:
                        continue
                    
                    logger.info(f"Found {len(tables)} table(s) on page {page_num}")
                    
                    # Convert each table to Markdown
                    for table_idx, table in enumerate(tables):
                        if not table or len(table) < 2:
                            continue
                        
                        markdown = self._table_to_markdown(table)
                        tables_markdown.append({
                            'page': page_num,
                            'table_index': table_idx,
                            'markdown': markdown
                        })
                        
                        logger.debug(f"Table {table_idx+1} on page {page_num}:\n{markdown[:200]}...")
                
                # Extract structured data from Markdown tables using LLM
                if tables_markdown and self.llm_available:
                    casing_program = self._extract_casing_from_markdown(tables_markdown)
                
                return {
                    'casing_program': casing_program,
                    'tables_markdown': tables_markdown
                }
                
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            return {'casing_program': [], 'tables_markdown': []}
    
    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """
        Convert table (list of lists) to Markdown format
        
        Args:
            table: List of rows, each row is list of cell values
            
        Returns:
            Markdown formatted table string
        """
        if not table or len(table) < 2:
            return ""
        
        # Clean cells (handle None values)
        cleaned_table = []
        for row in table:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            cleaned_table.append(cleaned_row)
        
        # Build Markdown
        markdown_lines = []
        
        # Header row
        header = cleaned_table[0]
        markdown_lines.append("| " + " | ".join(header) + " |")
        
        # Separator row
        markdown_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        
        # Data rows
        for row in cleaned_table[1:]:
            # Pad row to match header length
            while len(row) < len(header):
                row.append("")
            markdown_lines.append("| " + " | ".join(row[:len(header)]) + " |")
        
        return "\n".join(markdown_lines)
    
    def _extract_casing_from_markdown(self, tables_markdown: List[Dict]) -> List[Dict]:
        """
        Extract casing program from Markdown tables using LLM with JSON schema
        
        LLM Instruction:
        "Extract the Casing Program into a JSON list. For each row, capture:
        - size (Outer Diameter)
        - weight (ppf)
        - depth (Setting Depth)
        - pipe_id (Inner Diameter / ID). Look for columns labeled 'ID', 'I.D.', or 'Inside Diam'. If not found, return null."
        """
        all_casing = []
        
        for table_info in tables_markdown:
            markdown = table_info['markdown']
            
            prompt = f"""Extract the Casing Program from this table into a JSON list.

Table (Markdown format):
{markdown}

For each row in the table, extract these fields:
1. size: Outer Diameter (OD) of the casing (look for columns like "Size", "OD", "Diameter")
2. weight: Weight per foot in ppf (look for "Weight", "lb/ft", "ppf")
3. depth: Setting Depth in meters or feet (look for "Depth", "Setting Depth", "MD")
4. pipe_id: Inner Diameter (ID) - LOOK CAREFULLY for columns labeled "ID", "I.D.", or "Inside Diam". If not found, return null.

Return ONLY valid JSON array, no other text:
[
    {{
        "size": "9 5/8 inch",
        "weight": 53.5,
        "depth": 1234.5,
        "pipe_id": 8.535
    }},
    ...
]

JSON response:"""
            
            try:
                response = self.llm._call_ollama(prompt, max_tokens=1000, model=self.llm.model_extraction)
                
                # Parse JSON robustly
                casing_list = self._parse_json_response(response)
                
                if isinstance(casing_list, list):
                    all_casing.extend(casing_list)
                    logger.info(f"Extracted {len(casing_list)} casing strings from table on page {table_info['page']}")
                
            except Exception as e:
                logger.error(f"Failed to extract casing from table on page {table_info['page']}: {e}")
                continue
        
        return all_casing
    
    def _pass3_narrative_extraction(self, pdf_path: str, document_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Pass 3: Narrative Extraction (Section-Restricted Way)
        
        Logic:
        1. Locate the "Geology" or "Lithology" section headers
        2. Chunk only that text
        3. Ask ollama: "What formation instabilities, gas peaks, or drilling hazards were reported?"
        
        Returns:
            Dict with keys: geology_section, hazards, instabilities, gas_shows
        """
        narrative = {
            'geology_section': None,
            'hazards': [],
            'instabilities': [],
            'gas_shows': []
        }
        
        # Extract full text if not provided
        if document_data and 'content' in document_data:
            full_text = document_data['content']
        else:
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n\n"
            doc.close()
        
        # Locate geology/lithology section
        geology_text = self._locate_geology_section(full_text)
        
        if not geology_text:
            logger.warning("No Geology/Lithology section found")
            return narrative
        
        narrative['geology_section'] = geology_text[:1000]  # Store excerpt
        
        # Use LLM to extract hazards and instabilities if available
        if self.llm_available:
            try:
                hazards_data = self._extract_hazards_with_llm(geology_text)
                narrative.update(hazards_data)
            except Exception as e:
                logger.error(f"LLM hazard extraction failed: {e}")
        
        return narrative
    
    def _locate_geology_section(self, text: str) -> Optional[str]:
        """
        Locate the Geology/Lithology section in the document
        
        Returns:
            Section text (up to 3000 chars after header)
        """
        # Look for section headers
        patterns = [
            r'(?:^|\n)(#+\s*)?(?:Geology|Lithology|Formation|Stratigraphy)(?:\s*:|\s+Summary|\s+Description)?(?:\n|$)',
            r'(?:^|\n)\d+[\.\)]\s*(?:Geology|Lithology|Formation)(?:\s*:|\s+Summary)?(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                # Extract text starting from this header
                start_pos = match.end()
                # Extract up to 3000 chars or until next major section
                section_text = text[start_pos:start_pos + 3000]
                
                logger.info(f"Found Geology section at position {start_pos}")
                return section_text
        
        # Fallback: search for paragraphs mentioning geology keywords
        geology_mentions = []
        for keyword in self.geology_keywords:
            pattern = rf'[^\n]*{re.escape(keyword)}[^\n]*(?:\n[^\n]+){{0,5}}'
            matches = re.findall(pattern, text, re.IGNORECASE)
            geology_mentions.extend(matches)
        
        if geology_mentions:
            return "\n".join(geology_mentions[:10])  # First 10 mentions
        
        return None
    
    def _extract_hazards_with_llm(self, geology_text: str) -> Dict[str, List[str]]:
        """Extract drilling hazards and formation instabilities using LLM"""
        prompt = f"""Analyze this geology/lithology section from a well report and identify:

Geology Section:
{geology_text}

Extract the following (return empty array [] if not found):
1. Formation instabilities (e.g., shale swelling, lost circulation, stuck pipe)
2. Gas peaks or gas shows (e.g., H2S, methane, CO2 levels)
3. Drilling hazards (e.g., overpressure zones, fractured formations, wellbore stability issues)

Return ONLY valid JSON, no other text:
{{
    "instabilities": ["...", "..."],
    "gas_shows": ["...", "..."],
    "hazards": ["...", "..."]
}}

JSON response:"""
        
        response = self.llm._call_ollama(prompt, max_tokens=500, model=self.llm.model_summary)
        
        # Parse JSON robustly
        hazards_data = self._parse_json_response(response)
        
        return {
            'instabilities': hazards_data.get('instabilities', []),
            'gas_shows': hazards_data.get('gas_shows', []),
            'hazards': hazards_data.get('hazards', [])
        }
    
    def _generate_summary_report(self, metadata: Dict, technical_specs: Dict, narrative: Dict) -> str:
        """
        Final Output Generator: generate_summary_report(data_object)
        
        Logic:
        - Take structured JSON from Passes 1, 2, 3
        - Send to ollama with prompt:
          "You are a Drilling Engineer. Write a professional 'End of Well Summary' based on this JSON data.
           Use bold headers. Include the Casing Table with ID values."
        
        Constraints:
        - Handle JSON parsing errors robustly
        - Use Python type hints
        - Do not execute external APIs
        """
        if not self.llm_available:
            # Fallback: generate basic report without LLM
            return self._generate_basic_report(metadata, technical_specs, narrative)
        
        # Prepare data object
        data_object = {
            'metadata': metadata,
            'technical_specs': technical_specs,
            'narrative': narrative
        }
        
        # Convert to JSON string
        data_json = json.dumps(data_object, indent=2)
        
        prompt = f"""You are a Drilling Engineer. Write a professional "End of Well Summary" based on this extracted data.

Extracted Data (JSON):
{data_json}

Requirements:
1. Use bold Markdown headers (## for sections, ### for subsections)
2. Include all available metadata (Operator, Well Name, Rig, Dates, Days Total)
3. Include a Casing Program table with the following format:
   | Size | Weight (ppf) | Depth (m) | ID (inches) |
   Include pipe_id (Inner Diameter) values if available
4. Include drilling hazards and formation issues if mentioned
5. Write in professional technical style
6. Use exact values from the data (do not estimate or round)
7. If a field is missing or null, write "Not specified" or omit it

Write a comprehensive End of Well Summary (300-400 words):"""
        
        try:
            response = self.llm._call_ollama(
                prompt,
                max_tokens=800,
                model=self.llm.model_summary,
                timeout=120
            )
            return response.strip()
        except Exception as e:
            logger.error(f"LLM report generation failed: {e}")
            return self._generate_basic_report(metadata, technical_specs, narrative)
    
    def _generate_basic_report(self, metadata: Dict, technical_specs: Dict, narrative: Dict) -> str:
        """Fallback report generator without LLM"""
        lines = []
        
        lines.append("## End of Well Summary")
        lines.append("")
        
        # Metadata section
        lines.append("### Well Information")
        if metadata.get('well_name'):
            lines.append(f"**Well Name:** {metadata['well_name']}")
        if metadata.get('operator_name'):
            lines.append(f"**Operator:** {metadata['operator_name']}")
        if metadata.get('rig_name'):
            lines.append(f"**Rig:** {metadata['rig_name']}")
        if metadata.get('spud_date'):
            lines.append(f"**Spud Date:** {metadata['spud_date']}")
        if metadata.get('end_date'):
            lines.append(f"**End Date:** {metadata['end_date']}")
        if metadata.get('days_total'):
            lines.append(f"**Total Days:** {metadata['days_total']} days")
        lines.append("")
        
        # Casing program
        casing_program = technical_specs.get('casing_program', [])
        if casing_program:
            lines.append("### Casing Program")
            lines.append("")
            lines.append("| Size | Weight (ppf) | Depth | ID |")
            lines.append("|------|--------------|-------|-----|")
            
            for casing in casing_program:
                size = casing.get('size', 'N/A')
                weight = casing.get('weight', 'N/A')
                depth = casing.get('depth', 'N/A')
                pipe_id = casing.get('pipe_id', 'N/A')
                
                if pipe_id and pipe_id != 'N/A':
                    pipe_id_str = f"{pipe_id} in" if isinstance(pipe_id, (int, float)) else str(pipe_id)
                else:
                    pipe_id_str = "Not specified"
                
                lines.append(f"| {size} | {weight} | {depth} | {pipe_id_str} |")
            
            lines.append("")
        
        # Narrative section
        if narrative.get('hazards') or narrative.get('instabilities') or narrative.get('gas_shows'):
            lines.append("### Geology & Drilling Hazards")
            
            if narrative.get('instabilities'):
                lines.append("**Formation Instabilities:**")
                for item in narrative['instabilities']:
                    lines.append(f"- {item}")
                lines.append("")
            
            if narrative.get('gas_shows'):
                lines.append("**Gas Shows:**")
                for item in narrative['gas_shows']:
                    lines.append(f"- {item}")
                lines.append("")
            
            if narrative.get('hazards'):
                lines.append("**Drilling Hazards:**")
                for item in narrative['hazards']:
                    lines.append(f"- {item}")
                lines.append("")
        
        lines.append("---")
        lines.append("*Summary generated using 3-pass extraction system*")
        
        return "\n".join(lines)
    
    def _calculate_confidence(self, metadata: Dict, technical_specs: Dict, narrative: Dict) -> float:
        """
        Calculate confidence score for the summary
        
        Score components:
        - Metadata completeness: 40% (dates, names)
        - Technical specs: 40% (casing program with IDs)
        - Narrative: 20% (geology/hazards)
        """
        score = 0.0
        
        # Metadata (40%)
        metadata_fields = ['well_name', 'operator_name', 'spud_date', 'end_date', 'days_total']
        metadata_count = sum(1 for field in metadata_fields if metadata.get(field))
        score += 0.40 * (metadata_count / len(metadata_fields))
        
        # Technical specs (40%)
        casing_program = technical_specs.get('casing_program', [])
        if casing_program:
            # Check if IDs are present
            ids_present = sum(1 for c in casing_program if c.get('pipe_id'))
            id_rate = ids_present / len(casing_program) if casing_program else 0
            score += 0.40 * id_rate
        
        # Narrative (20%)
        narrative_fields = ['hazards', 'instabilities', 'gas_shows']
        narrative_count = sum(1 for field in narrative_fields if narrative.get(field))
        score += 0.20 * (narrative_count / len(narrative_fields))
        
        return round(score, 2)
    
    def _parse_json_response(self, response: str) -> Any:
        """
        Robustly parse JSON from LLM response
        
        Constraints:
        - Handle JSON parsing errors robustly (find first { and last })
        
        Args:
            response: LLM response string
            
        Returns:
            Parsed JSON object (dict or list)
        """
        # Remove markdown code blocks if present
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        # Find first { or [ and last } or ]
        start_brace = response.find('{')
        start_bracket = response.find('[')
        
        # Determine which comes first
        if start_brace == -1 and start_bracket == -1:
            logger.error("No JSON found in response")
            return {}
        
        if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
            # Object
            start = start_brace
            end = response.rfind('}')
            if end == -1:
                logger.error("No closing } found")
                return {}
            json_str = response[start:end+1]
        else:
            # Array
            start = start_bracket
            end = response.rfind(']')
            if end == -1:
                logger.error("No closing ] found")
                return []
            json_str = response[start:end+1]
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.debug(f"Attempted to parse: {json_str[:200]}...")
            return {} if start == start_brace else []
