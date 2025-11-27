"""
Universal Geothermal Metadata Extractor
Uses spaCy NER + Regex patterns for comprehensive entity extraction
"""

import logging
import re
import spacy
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniversalGeothermalMetadataExtractor:
    """
    Advanced metadata extractor for geothermal well documents
    
    Extracts:
    - Well names (e.g., ADK-GT-01, LDD-GT-02-S1)
    - Formation names (e.g., Slochteren, Rotliegend)
    - Depths (MD, TVD with units)
    - Pressures (with units)
    - Temperatures (with units)
    - Operators/Companies
    - Dates
    - Equipment IDs
    
    Methods:
    - Regex (fast, rule-based)
    - spaCy NER (context-aware)
    - Domain-specific patterns
    """
    
    def __init__(self, use_spacy: bool = True):
        """
        Initialize metadata extractor
        
        Args:
            use_spacy: Whether to use spaCy NER (slower but more accurate)
        """
        self.use_spacy = use_spacy
        
        # Load spaCy for NER
        if self.use_spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
                logger.info("✓ spaCy loaded for NER")
            except OSError:
                logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
                self.use_spacy = False
        else:
            self.nlp = None
        
        # Compile regex patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for extraction"""
        
        # Well name patterns (Dutch geothermal conventions)
        self.well_patterns = [
            r'\b([A-Z]{2,10}-GT-\d{2}(?:-S\d+)?)\b',  # Standard: XXX-GT-##[-S#]
            r'\b([A-Z]{2,10}-\d{2})\b',  # Simplified: XXX-##
            r'\bWell\s+([A-Z0-9\-]+)\b'  # "Well ABC-123"
        ]
        
        # Formation name patterns (common Dutch geothermal formations)
        self.formation_keywords = [
            'Slochteren', 'Rotliegend', 'Carboniferous', 'Upper Germanic Trias',
            'Lower Germanic Trias', 'Muschelkalk', 'Bunter', 'Zechstein',
            'Delft', 'Werkendam', 'Delfland', 'Schieland', 'Scruff', 'Vlieland'
        ]
        self.formation_pattern = r'\b(' + '|'.join(self.formation_keywords) + r')(?:\s+Formation)?\b'
        
        # Depth patterns (with units)
        self.depth_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:m|meter|metres?)\s+(?:MD|Measured\s+Depth)',  # MD
            r'(\d+(?:\.\d+)?)\s*(?:m|meter|metres?)\s+(?:TVD|True\s+Vertical\s+Depth)',  # TVD
            r'(?:depth|TD|depth\s+of)[\s:]+(\d+(?:\.\d+)?)\s*(?:m|meter|metres?|ft|feet)',
            r'(\d+(?:\.\d+)?)\s*(?:m|meter|metres?|ft|feet)\s+(?:MD|TVD)'
        ]
        
        # Pressure patterns (with units)
        self.pressure_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:bar|psi|kPa|MPa)',
            r'(?:pressure|BHP|THP)[\s:]+(\d+(?:\.\d+)?)\s*(?:bar|psi|kPa|MPa)'
        ]
        
        # Temperature patterns (with units)
        self.temperature_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:°C|°F|degC|degF)',
            r'(?:temperature|temp)[\s:]+(\d+(?:\.\d+)?)\s*(?:°C|°F|degC|degF)'
        ]
        
        # Date patterns
        self.date_patterns = [
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',  # DD/MM/YYYY or MM-DD-YYYY
            r'\b(\d{2,4}[/-]\d{1,2}[/-]\d{1,2})\b',  # YYYY-MM-DD
            r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b'  # Month DD, YYYY
        ]
        
        # Equipment/Casing ID patterns
        self.equipment_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:inch|in|")\s+(?:casing|tubing|pipe)',
            r'(?:casing|tubing|pipe)[\s:]+(\d+(?:\.\d+)?)\s*(?:inch|in|")'
        ]
    
    def extract_metadata(self, text: str, document_id: str = None) -> Dict:
        """
        Extract all metadata from text
        
        Args:
            text: Document text
            document_id: Optional document identifier
            
        Returns:
            Dict with extracted entities
        """
        metadata = {
            'document_id': document_id,
            'well_names': [],
            'formations': [],
            'depths': [],
            'pressures': [],
            'temperatures': [],
            'dates': [],
            'equipment': [],
            'operators': [],
            'extraction_method': 'regex+spacy' if self.use_spacy else 'regex'
        }
        
        # 1. Regex-based extraction (fast)
        metadata['well_names'] = self._extract_well_names(text)
        metadata['formations'] = self._extract_formations(text)
        metadata['depths'] = self._extract_depths(text)
        metadata['pressures'] = self._extract_pressures(text)
        metadata['temperatures'] = self._extract_temperatures(text)
        metadata['dates'] = self._extract_dates(text)
        metadata['equipment'] = self._extract_equipment(text)
        
        # 2. spaCy NER (context-aware, for operators/companies)
        if self.use_spacy and self.nlp:
            spacy_entities = self._extract_with_spacy(text)
            metadata['operators'] = spacy_entities.get('ORG', [])
            
            # Merge with regex results
            metadata['dates'].extend(spacy_entities.get('DATE', []))
            metadata['dates'] = list(set(metadata['dates']))  # Deduplicate
        
        # Log extraction summary
        logger.info(f"Extracted metadata: {len(metadata['well_names'])} wells, "
                   f"{len(metadata['formations'])} formations, "
                   f"{len(metadata['depths'])} depths")
        
        return metadata
    
    def _extract_well_names(self, text: str) -> List[str]:
        """Extract well names using regex patterns"""
        well_names = set()
        
        for pattern in self.well_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            well_names.update(matches)
        
        return sorted(list(well_names))
    
    def _extract_formations(self, text: str) -> List[str]:
        """Extract formation names"""
        formations = set()
        
        matches = re.findall(self.formation_pattern, text, re.IGNORECASE)
        formations.update(matches)
        
        return sorted(list(formations))
    
    def _extract_depths(self, text: str) -> List[Dict]:
        """Extract depth values with units and type (MD/TVD)"""
        depths = []
        
        for pattern in self.depth_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                depth_value = match.group(1)
                full_match = match.group(0)
                
                # Determine depth type
                depth_type = 'MD' if 'MD' in full_match.upper() or 'MEASURED' in full_match.upper() else 'TVD' if 'TVD' in full_match.upper() or 'TRUE' in full_match.upper() else 'unknown'
                
                # Determine unit
                unit = 'm' if 'm' in full_match.lower() or 'meter' in full_match.lower() else 'ft' if 'ft' in full_match.lower() or 'feet' in full_match.lower() else 'm'
                
                depths.append({
                    'value': float(depth_value),
                    'unit': unit,
                    'type': depth_type,
                    'raw_text': full_match
                })
        
        return depths
    
    def _extract_pressures(self, text: str) -> List[Dict]:
        """Extract pressure values with units"""
        pressures = []
        
        for pattern in self.pressure_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                pressure_value = match.group(1)
                full_match = match.group(0)
                
                # Determine unit
                if 'bar' in full_match.lower():
                    unit = 'bar'
                elif 'psi' in full_match.lower():
                    unit = 'psi'
                elif 'kpa' in full_match.lower():
                    unit = 'kPa'
                elif 'mpa' in full_match.lower():
                    unit = 'MPa'
                else:
                    unit = 'bar'  # Default
                
                pressures.append({
                    'value': float(pressure_value),
                    'unit': unit,
                    'raw_text': full_match
                })
        
        return pressures
    
    def _extract_temperatures(self, text: str) -> List[Dict]:
        """Extract temperature values with units"""
        temperatures = []
        
        for pattern in self.temperature_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                temp_value = match.group(1)
                full_match = match.group(0)
                
                # Determine unit
                unit = 'C' if '°C' in full_match or 'degC' in full_match else 'F'
                
                temperatures.append({
                    'value': float(temp_value),
                    'unit': unit,
                    'raw_text': full_match
                })
        
        return temperatures
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates"""
        dates = set()
        
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.update(matches)
        
        return sorted(list(dates))
    
    def _extract_equipment(self, text: str) -> List[Dict]:
        """Extract equipment specifications (casing/tubing sizes)"""
        equipment = []
        
        for pattern in self.equipment_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                size_value = match.group(1)
                full_match = match.group(0)
                
                # Determine equipment type
                equip_type = 'casing' if 'casing' in full_match.lower() else 'tubing' if 'tubing' in full_match.lower() else 'pipe'
                
                equipment.append({
                    'type': equip_type,
                    'size': float(size_value),
                    'unit': 'inch',
                    'raw_text': full_match
                })
        
        return equipment
    
    def _extract_with_spacy(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using spaCy NER"""
        if not self.nlp:
            return {}
        
        # Process text in chunks (spaCy has limits)
        max_length = 1000000  # 1M chars
        if len(text) > max_length:
            text = text[:max_length]
        
        doc = self.nlp(text)
        
        entities = defaultdict(list)
        for ent in doc.ents:
            entities[ent.label_].append(ent.text)
        
        # Deduplicate
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return dict(entities)
    
    def enrich_chunks_with_metadata(self, chunks: List[Dict], 
                                    document_metadata: Dict) -> List[Dict]:
        """
        Enrich chunks with document-level metadata
        
        Args:
            chunks: List of chunk dicts
            document_metadata: Metadata extracted from full document
            
        Returns:
            Enriched chunk list
        """
        for chunk in chunks:
            # Add document-level metadata
            chunk['metadata'] = chunk.get('metadata', {})
            chunk['metadata'].update({
                'well_names': document_metadata.get('well_names', []),
                'formations': document_metadata.get('formations', []),
                'operators': document_metadata.get('operators', [])
            })
            
            # Extract chunk-specific metadata (depths, pressures in this chunk)
            chunk_text = chunk.get('text', '')
            chunk_depths = self._extract_depths(chunk_text)
            chunk_pressures = self._extract_pressures(chunk_text)
            chunk_temps = self._extract_temperatures(chunk_text)
            
            if chunk_depths:
                chunk['metadata']['depths'] = chunk_depths
            if chunk_pressures:
                chunk['metadata']['pressures'] = chunk_pressures
            if chunk_temps:
                chunk['metadata']['temperatures'] = chunk_temps
        
        return chunks


def create_metadata_extractor(config: Dict = None) -> UniversalGeothermalMetadataExtractor:
    """Factory function to create metadata extractor"""
    if config is None:
        config = {}
    
    return UniversalGeothermalMetadataExtractor(
        use_spacy=config.get('use_spacy', True)
    )
