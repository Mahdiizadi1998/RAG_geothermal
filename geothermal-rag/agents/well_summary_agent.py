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
    Generates comprehensive End of Well summaries using hybrid data sources
    
    Mode B: Summary Generation
    - Uses all 8 data types from database (structured) and vector store (narrative)
    - No word limit but concise
    - If data not found in PDF, don't mention it
    
    Data Sources:
    1. Database: General Data, Timeline, Depths, Casing, Cementing, Fluids, Geology (formations), Incidents
    2. Vector Store: Narrative context, descriptions, additional details
    """
    
    def __init__(self, llm_helper=None, database_manager=None):
        """
        Initialize Well Summary Agent
        
        Args:
            llm_helper: OllamaHelper instance for LLM calls (optional)
            database_manager: WellDatabaseManager for structured data access
        """
        self.llm = llm_helper
        self.db = database_manager
        self.llm_available = llm_helper is not None and llm_helper.is_available()
        
        logger.info(f"WellSummaryAgent initialized (LLM available: {self.llm_available}, DB available: {database_manager is not None})")
    
    def generate_summary(self, well_name: str, narrative_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive End of Well Summary (Mode B)
        
        Uses all 8 data types:
        1. General Data (from DB)
        2. Timeline (from DB)
        3. Depths (from DB)
        4. Casing (from DB)
        5. Cementing (from DB)
        6. Fluids (from DB)
        7. Geology - Formations (from DB) + Narrative (from vector store)
        8. Incidents (from DB)
        
        Args:
            well_name: Well name to generate summary for
            narrative_context: Optional narrative text from vector store
            
        Returns:
            Dict containing:
            {
                'well_data': Dict (all 8 data types from database),
                'summary_report': str (formatted final report),
                'confidence': float
            }
        """
        if not self.db:
            logger.error("Database manager not initialized - cannot generate summary")
            return {
                'well_data': {},
                'summary_report': "Error: Database not available",
                'confidence': 0.0
            }
        
        logger.info(f"Generating summary for: {well_name}")
        
        # Get all data from database
        logger.info("⏳ Fetching data from database...")
        well_summary = self.db.get_well_summary(well_name)
        
        if not well_summary:
            logger.warning(f"No data found for {well_name}")
            return {
                'well_data': {},
                'summary_report': f"No data found for well: {well_name}",
                'confidence': 0.0
            }
        
        logger.info("✓ Database fetch complete")
        
        # Generate final report
        logger.info("⏳ Generating summary report...")
        summary_report = self._generate_summary_report_from_db(well_summary, narrative_context)
        logger.info("✓ Summary report generated")
        
        # Calculate confidence
        confidence = self._calculate_confidence_from_db(well_summary)
        
        return {
            'well_data': well_summary,
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
2. Well Name (e.g., [WELL-ID-##])
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
        
        # Well name pattern (e.g., [CODE]-GT-##)
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
        "size": "[size] inch",
        "weight": [number],
        "depth": [number],
        "pipe_id": [number]
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
    
    def _generate_summary_report_from_db(self, well_summary: Dict, narrative_context: Optional[str] = None) -> str:
        """
        Generate summary report from database data (all 8 data types)
        
        Args:
            well_summary: Complete well data from database
            narrative_context: Optional narrative text from vector store
            
        Returns:
            Formatted Markdown summary report
        """
        if not self.llm_available:
            # Fallback: generate basic report without LLM
            return self._generate_basic_report_from_db(well_summary, narrative_context)
        
        # Convert to JSON string
        data_json = json.dumps(well_summary, indent=2, default=str)
        
        prompt = f"""You are a Drilling Engineer. Write a professional "End of Well Summary" based on this database data.

Complete Well Data (JSON):
{data_json}

Additional Narrative Context:
{narrative_context if narrative_context else "No additional narrative available"}

Requirements:
1. Use bold Markdown headers (## for sections, ### for subsections)
2. Include ALL available data from these 8 categories:
   - General Data (well name, license, well type, location, coordinates, operator, rig name, target formation)
   - Drilling Timeline (spud date, end of operations, total days)
   - Depths (TD measured, TVD, sidetrack start depth if applicable)
   - Casing & Tubulars (create a table with: Type, OD, Weight, Grade, Connection, Pipe ID Nominal, Pipe ID Drift, Depths)
   - Cementing (lead/tail volumes, densities, TOC)
   - Drilling Fluids (hole size, fluid type, density range)
   - Geology/Formations (formation tops, lithology)
   - Incidents (gas peaks, stuck pipe, mud losses, NPT)
3. If data is not available for any category, DO NOT mention that category
4. Use exact values from the data (do not estimate or round)
5. Include Pipe ID values (both Nominal and Drift) in the casing table - this is CRUCIAL
6. Write in professional technical style
7. Keep it concise but don't skip important data

Write a comprehensive End of Well Summary:"""
        
        try:
            response = self.llm._call_ollama(
                prompt,
                max_tokens=1200,
                model=self.llm.model_summary,
                timeout=120
            )
            return response.strip()
        except Exception as e:
            logger.error(f"LLM report generation failed: {e}")
            return self._generate_basic_report_from_db(well_summary, narrative_context)
    
    def _generate_basic_report_from_db(self, well_summary: Dict, narrative_context: Optional[str] = None) -> str:
        """Fallback report generator without LLM using database data"""
        lines = []
        
        lines.append("## End of Well Summary")
        lines.append("")
        
        # 1. General Data
        well_info = well_summary.get('well_info', {})
        if well_info:
            lines.append("### General Information")
            if well_info.get('well_name'):
                lines.append(f"**Well Name:** {well_info['well_name']}")
            if well_info.get('license_number'):
                lines.append(f"**License:** {well_info['license_number']}")
            if well_info.get('well_type'):
                lines.append(f"**Well Type:** {well_info['well_type']}")
            if well_info.get('location'):
                lines.append(f"**Location:** {well_info['location']}")
            if well_info.get('coordinate_x') and well_info.get('coordinate_y'):
                lines.append(f"**Coordinates:** X={well_info['coordinate_x']}, Y={well_info['coordinate_y']}")
            if well_info.get('operator'):
                lines.append(f"**Operator:** {well_info['operator']}")
            if well_info.get('rig_name'):
                lines.append(f"**Rig:** {well_info['rig_name']}")
            if well_info.get('target_formation'):
                lines.append(f"**Target Formation:** {well_info['target_formation']}")
            lines.append("")
            
            # 2. Timeline
            lines.append("### Drilling Timeline")
            if well_info.get('spud_date'):
                lines.append(f"**Spud Date:** {well_info['spud_date']}")
            if well_info.get('end_of_operations'):
                lines.append(f"**End of Operations:** {well_info['end_of_operations']}")
            if well_info.get('total_days'):
                lines.append(f"**Total Days:** {well_info['total_days']} days")
            lines.append("")
            
            # 3. Depths
            lines.append("### Depths")
            if well_info.get('total_depth_md'):
                lines.append(f"**Total Depth (MD):** {well_info['total_depth_md']} m")
            if well_info.get('total_depth_tvd'):
                lines.append(f"**Total Depth (TVD):** {well_info['total_depth_tvd']} m")
            if well_info.get('sidetrack_start_depth_md'):
                lines.append(f"**Sidetrack Start Depth:** {well_info['sidetrack_start_depth_md']} m")
            lines.append("")
        
        # 4. Casing & Tubulars
        casings = well_summary.get('casing_strings', [])
        if casings:
            lines.append("### Casing & Tubulars")
            lines.append("")
            lines.append("| Type | OD (in) | Weight (lb/ft) | Grade | Connection | ID Nominal (in) | ID Drift (in) | Top (m) | Bottom (m) |")
            lines.append("|------|---------|----------------|-------|------------|-----------------|---------------|---------|------------|")
            
            for casing in casings:
                ctype = casing.get('casing_type', 'N/A')
                od = casing.get('outer_diameter', 'N/A')
                weight = casing.get('weight', 'N/A')
                grade = casing.get('grade', 'N/A')
                connection = casing.get('connection_type', 'N/A')
                id_nom = casing.get('pipe_id_nominal', 'N/A')
                id_drift = casing.get('pipe_id_drift', 'N/A')
                top = casing.get('top_depth_md', 'N/A')
                bottom = casing.get('bottom_depth_md', 'N/A')
                
                lines.append(f"| {ctype} | {od} | {weight} | {grade} | {connection} | {id_nom} | {id_drift} | {top} | {bottom} |")
            
            lines.append("")
        
        # 5. Cementing
        cementing = well_summary.get('cementing', [])
        if cementing:
            lines.append("### Cementing Operations")
            for cement in cementing:
                stage = cement.get('stage_number', 'N/A')
                lines.append(f"**Stage {stage}:**")
                if cement.get('lead_volume'):
                    lines.append(f"- Lead: {cement['lead_volume']} m³ @ {cement.get('lead_density', 'N/A')} kg/m³")
                if cement.get('tail_volume'):
                    lines.append(f"- Tail: {cement['tail_volume']} m³ @ {cement.get('tail_density', 'N/A')} kg/m³")
                if cement.get('top_of_cement_md'):
                    lines.append(f"- TOC: {cement['top_of_cement_md']} m MD")
                lines.append("")
        
        # 6. Drilling Fluids
        fluids = well_summary.get('drilling_fluids', [])
        if fluids:
            lines.append("### Drilling Fluids")
            for fluid in fluids:
                hole_size = fluid.get('hole_size', 'N/A')
                fluid_type = fluid.get('fluid_type', 'N/A')
                density_min = fluid.get('density_min')
                density_max = fluid.get('density_max')
                
                line = f"**{hole_size} in hole:** {fluid_type}"
                if density_min and density_max:
                    line += f", Density: {density_min}-{density_max} kg/m³"
                lines.append(line)
            lines.append("")
        
        # 7. Geology/Formations
        formations = well_summary.get('formations', [])
        if formations:
            lines.append("### Geology - Formation Tops")
            lines.append("")
            lines.append("| Formation | Top MD (m) | Top TVD (m) | Lithology |")
            lines.append("|-----------|------------|-------------|-----------|")
            
            for fm in formations:
                name = fm.get('formation_name', 'N/A')
                top_md = fm.get('top_md', 'N/A')
                top_tvd = fm.get('top_tvd', 'N/A')
                lithology = fm.get('lithology', 'N/A')
                
                lines.append(f"| {name} | {top_md} | {top_tvd} | {lithology} |")
            
            lines.append("")
        
        # 8. Incidents
        incidents = well_summary.get('incidents', [])
        if incidents:
            lines.append("### Incidents & Problems")
            for incident in incidents:
                date = incident.get('date', 'N/A')
                itype = incident.get('incident_type', 'Unknown')
                desc = incident.get('description', '')
                depth = incident.get('depth_md')
                
                line = f"**{date} - {itype}**"
                if depth:
                    line += f" (at {depth} m MD)"
                lines.append(line)
                if desc:
                    lines.append(f"  {desc}")
                lines.append("")
        
        # Add narrative context if available
        if narrative_context:
            lines.append("### Additional Context")
            lines.append(narrative_context[:500])  # First 500 chars
            lines.append("")
        
        lines.append("---")
        lines.append("*Summary generated from database and vector store*")
        
        return "\n".join(lines)
    
    def _calculate_confidence_from_db(self, well_summary: Dict) -> float:
        """
        Calculate confidence score based on data completeness
        
        Score based on 8 data types (each worth 12.5%):
        1. General Data
        2. Timeline  
        3. Depths
        4. Casing
        5. Cementing
        6. Fluids
        7. Formations
        8. Incidents
        """
        score = 0.0
        
        well_info = well_summary.get('well_info', {})
        
        # 1. General Data (12.5%)
        general_fields = ['well_name', 'license_number', 'operator', 'location', 'rig_name', 'target_formation']
        general_count = sum(1 for field in general_fields if well_info.get(field))
        score += 0.125 * (general_count / len(general_fields))
        
        # 2. Timeline (12.5%)
        timeline_fields = ['spud_date', 'end_of_operations', 'total_days']
        timeline_count = sum(1 for field in timeline_fields if well_info.get(field))
        score += 0.125 * (timeline_count / len(timeline_fields))
        
        # 3. Depths (12.5%)
        depth_fields = ['total_depth_md', 'total_depth_tvd']
        depth_count = sum(1 for field in depth_fields if well_info.get(field))
        score += 0.125 * (depth_count / len(depth_fields))
        
        # 4. Casing (12.5%) - CRITICAL: check for pipe IDs
        casings = well_summary.get('casing_strings', [])
        if casings:
            ids_present = sum(1 for c in casings if c.get('pipe_id_nominal') or c.get('pipe_id_drift'))
            id_rate = ids_present / len(casings) if casings else 0
            score += 0.125 * id_rate
        
        # 5. Cementing (12.5%)
        if well_summary.get('cementing'):
            score += 0.125
        
        # 6. Fluids (12.5%)
        if well_summary.get('drilling_fluids'):
            score += 0.125
        
        # 7. Formations (12.5%)
        if well_summary.get('formations'):
            score += 0.125
        
        # 8. Incidents (12.5%)
        if well_summary.get('incidents'):
            score += 0.125
        
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
