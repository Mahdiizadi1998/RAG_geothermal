"""
Template Selector - Choose best summary template based on available data
Scores templates based on database completeness
"""

import logging
from typing import Dict, Optional, List
from agents.summary_templates import SUMMARY_TEMPLATES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemplateSelectorAgent:
    """
    Select optimal summary template based on available database data
    
    Scoring criteria:
    - Required data present: +10 points each
    - Optional data present: +5 points each
    - Missing required data: disqualify template
    """
    
    def __init__(self, database_manager):
        self.db = database_manager
    
    def select_template(self, well_name: str, user_preference: Optional[str] = None) -> Dict:
        """
        Select best template for well summary
        
        Args:
            well_name: Well name to summarize
            user_preference: User-requested template name (overrides scoring)
            
        Returns:
            Template dict with 'name', 'template', 'sql_queries'
        """
        # If user specified a template, use it
        if user_preference and user_preference in SUMMARY_TEMPLATES:
            logger.info(f"Using user-specified template: {user_preference}")
            return SUMMARY_TEMPLATES[user_preference]
        
        # Get data availability for this well
        available_data = self._check_data_availability(well_name)
        
        logger.debug(f"Data availability for {well_name}: {available_data}")
        
        # Score each template
        scores = {}
        for template_name, template in SUMMARY_TEMPLATES.items():
            score = self._score_template(template, available_data)
            scores[template_name] = score
            logger.debug(f"  {template_name}: {score} points")
        
        # Select template with highest score
        best_template = max(scores, key=scores.get)
        
        if scores[best_template] < 0:
            # All templates disqualified - use basic as fallback
            logger.warning(f"No suitable template found for {well_name}, using basic_completion")
            best_template = 'basic_completion'
        
        logger.info(f"Selected template: {best_template} ({scores[best_template]} points)")
        return SUMMARY_TEMPLATES[best_template]
    
    def _check_data_availability(self, well_name: str) -> Dict[str, bool]:
        """
        Check what data is available for this well
        
        Returns:
            Dict mapping data type to availability (True/False)
        """
        availability = {}
        
        # Check well basic info
        well = self.db.get_well_by_name(well_name)
        availability['well_name'] = well is not None
        
        if not well:
            return availability
        
        well_id = well['well_id']
        
        # Check total depth
        availability['total_depth_md'] = well.get('total_depth_md') is not None
        availability['total_depth_tvd'] = well.get('total_depth_tvd') is not None
        
        # Check operator
        availability['operator'] = bool(well.get('operator'))
        
        # Check location
        availability['location'] = bool(well.get('location'))
        
        # Check dates
        availability['spud_date'] = bool(well.get('spud_date'))
        availability['completion_date'] = bool(well.get('completion_date'))
        
        # Check casing data
        casing_query = """
        SELECT COUNT(*) as count FROM casing_strings WHERE well_id = ?
        """
        casing_count = self.db.query_sql(casing_query, (well_id,))[0]['count']
        availability['casing_strings'] = casing_count > 0
        
        # Check formations
        formation_query = """
        SELECT COUNT(*) as count FROM formations WHERE well_id = ?
        """
        formation_count = self.db.query_sql(formation_query, (well_id,))[0]['count']
        availability['formations'] = formation_count > 0
        
        # Check cementing
        cement_query = """
        SELECT COUNT(*) as count FROM cementing WHERE well_id = ?
        """
        cement_count = self.db.query_sql(cement_query, (well_id,))[0]['count']
        availability['cementing'] = cement_count > 0
        
        # Check operations
        ops_query = """
        SELECT COUNT(*) as count FROM operations WHERE well_id = ?
        """
        ops_count = self.db.query_sql(ops_query, (well_id,))[0]['count']
        availability['operations'] = ops_count > 0
        
        # Check measurements
        measure_query = """
        SELECT COUNT(*) as count FROM measurements WHERE well_id = ?
        """
        measure_count = self.db.query_sql(measure_query, (well_id,))[0]['count']
        availability['measurements'] = measure_count > 0
        
        return availability
    
    def _score_template(self, template: Dict, available_data: Dict[str, bool]) -> int:
        """
        Score a template based on data availability
        
        Returns:
            Score (higher is better, -1 means disqualified)
        """
        score = 0
        
        # Check required data
        required = template.get('required_data', [])
        for req in required:
            if not available_data.get(req, False):
                # Missing required data - disqualify this template
                logger.debug(f"    Template disqualified: missing required '{req}'")
                return -1
            score += 10  # +10 for each required field present
        
        # Check optional data
        optional = template.get('optional_data', [])
        for opt in optional:
            if available_data.get(opt, False):
                score += 5  # +5 for each optional field present
        
        return score
    
    def get_data_completeness_report(self, well_name: str) -> str:
        """Generate human-readable report of data completeness"""
        availability = self._check_data_availability(well_name)
        
        lines = [f"Data completeness for {well_name}:"]
        lines.append(f"  ✓ Basic Info: {'Yes' if availability.get('well_name') else 'No'}")
        lines.append(f"  ✓ Depth Data: {'Yes' if availability.get('total_depth_md') else 'No'}")
        lines.append(f"  ✓ Operator: {'Yes' if availability.get('operator') else 'No'}")
        lines.append(f"  ✓ Location: {'Yes' if availability.get('location') else 'No'}")
        lines.append(f"  ✓ Casing: {'Yes' if availability.get('casing_strings') else 'No'}")
        lines.append(f"  ✓ Formations: {'Yes' if availability.get('formations') else 'No'}")
        lines.append(f"  ✓ Cementing: {'Yes' if availability.get('cementing') else 'No'}")
        lines.append(f"  ✓ Operations: {'Yes' if availability.get('operations') else 'No'}")
        
        return '\n'.join(lines)
