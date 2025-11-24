"""
Missing Data Detection Agent
Detects incomplete or missing data in well extractions and generates clarification questions.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MissingDataIssue:
    """Represents a missing or incomplete data issue"""
    category: str  # 'trajectory', 'casing', 'tubing', 'pvt', 'equipment'
    severity: str  # 'critical', 'important', 'optional'
    description: str
    clarification_question: str


@dataclass
class CompletenessAssessment:
    """Overall data completeness assessment"""
    completeness_score: float  # 0.0 to 1.0
    missing_issues: List[MissingDataIssue]
    has_critical_gaps: bool
    clarification_questions: List[str]
    summary: str


class MissingDataAgent:
    """Detects missing/incomplete data and generates clarification questions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Completeness requirements
        validation_config = config.get('validation', {})
        self.require_trajectory = validation_config.get('require_trajectory', True)
        self.require_casing = validation_config.get('require_casing', True)
        self.require_tubing = validation_config.get('require_tubing', False)
        self.require_pvt = validation_config.get('require_pvt', False)
        
        logger.info(f"Missing data detection: Trajectory={self.require_trajectory}, Casing={self.require_casing}, Tubing={self.require_tubing}, PVT={self.require_pvt}")
    
    def assess_completeness(self, extracted_data: Dict[str, Any]) -> CompletenessAssessment:
        """
        Assess completeness of extracted well data
        
        Args:
            extracted_data: Dict with keys like 'trajectory', 'casing', 'tubing', 'pvt'
            
        Returns:
            CompletenessAssessment with missing issues and clarification questions
        """
        logger.info("Assessing data completeness...")
        
        missing_issues = []
        
        # Check trajectory data
        trajectory_issues = self._check_trajectory(extracted_data.get('trajectory', []))
        missing_issues.extend(trajectory_issues)
        
        # Check casing data
        casing_issues = self._check_casing(extracted_data.get('casing', []))
        missing_issues.extend(casing_issues)
        
        # Check tubing data
        tubing_issues = self._check_tubing(extracted_data.get('tubing', []))
        missing_issues.extend(tubing_issues)
        
        # Check PVT data
        pvt_issues = self._check_pvt(extracted_data.get('pvt', {}))
        missing_issues.extend(pvt_issues)
        
        # Check equipment/completion data
        equipment_issues = self._check_equipment(extracted_data.get('equipment', {}))
        missing_issues.extend(equipment_issues)
        
        # Calculate completeness score
        completeness_score = self._calculate_completeness(missing_issues)
        
        # Check for critical gaps
        critical_issues = [i for i in missing_issues if i.severity == 'critical']
        has_critical_gaps = len(critical_issues) > 0
        
        # Generate clarification questions
        clarification_questions = [issue.clarification_question for issue in missing_issues 
                                   if issue.severity in ['critical', 'important']]
        
        # Generate summary
        summary = self._generate_summary(missing_issues, completeness_score, has_critical_gaps)
        
        logger.info(f"Completeness: {completeness_score*100:.0f}%, {len(critical_issues)} critical gaps, {len(clarification_questions)} questions")
        
        return CompletenessAssessment(
            completeness_score=completeness_score,
            missing_issues=missing_issues,
            has_critical_gaps=has_critical_gaps,
            clarification_questions=clarification_questions,
            summary=summary
        )
    
    def _check_trajectory(self, trajectory: List[Dict]) -> List[MissingDataIssue]:
        """Check trajectory data completeness"""
        issues = []
        
        if not trajectory:
            if self.require_trajectory:
                issues.append(MissingDataIssue(
                    category='trajectory',
                    severity='critical',
                    description="No well trajectory data found",
                    clarification_question="Could you provide the well trajectory survey data including Measured Depth (MD), True Vertical Depth (TVD), and inclination angles?"
                ))
            return issues
        
        # Check required fields
        required_fields = ['MD', 'TVD']
        for i, point in enumerate(trajectory):
            missing_fields = [f for f in required_fields if f not in point or point[f] is None]
            if missing_fields:
                issues.append(MissingDataIssue(
                    category='trajectory',
                    severity='critical',
                    description=f"Trajectory point {i+1} missing: {', '.join(missing_fields)}",
                    clarification_question=f"What are the {' and '.join(missing_fields)} values for trajectory point at index {i+1}?"
                ))
        
        # Check for Pipe ID in trajectory
        if trajectory and 'ID' not in trajectory[0]:
            issues.append(MissingDataIssue(
                category='trajectory',
                severity='important',
                description="Pipe Inner Diameter (ID) not specified in trajectory",
                clarification_question="What are the pipe inner diameters (ID in inches) for each trajectory section?"
            ))
        
        # Check for sufficient data points
        if len(trajectory) < 2:
            issues.append(MissingDataIssue(
                category='trajectory',
                severity='important',
                description=f"Only {len(trajectory)} trajectory point(s) - insufficient for analysis",
                clarification_question="Could you provide additional trajectory survey points? At least 2-3 points are needed for meaningful analysis."
            ))
        
        return issues
    
    def _check_casing(self, casing: List[Dict]) -> List[MissingDataIssue]:
        """Check casing data completeness"""
        issues = []
        
        if not casing:
            if self.require_casing:
                issues.append(MissingDataIssue(
                    category='casing',
                    severity='critical',
                    description="No casing program data found",
                    clarification_question="Could you provide the casing program including depths, outer diameters (OD), and inner diameters (ID) for each casing string?"
                ))
            return issues
        
        # Check required fields for each casing string
        required_fields = ['depth', 'OD', 'ID']
        for i, string in enumerate(casing):
            missing_fields = [f for f in required_fields if f not in string or string[f] is None]
            if missing_fields:
                issues.append(MissingDataIssue(
                    category='casing',
                    severity='important',
                    description=f"Casing string {i+1} missing: {', '.join(missing_fields)}",
                    clarification_question=f"What are the {' and '.join(missing_fields)} for casing string {i+1}?"
                ))
        
        # Check for casing grade/weight
        if casing and 'grade' not in casing[0]:
            issues.append(MissingDataIssue(
                category='casing',
                severity='optional',
                description="Casing grade/weight not specified",
                clarification_question="What are the casing grades and weights (e.g., K-55, N-80, 47 lb/ft)?"
            ))
        
        return issues
    
    def _check_tubing(self, tubing: List[Dict]) -> List[MissingDataIssue]:
        """Check tubing data completeness"""
        issues = []
        
        if not tubing:
            if self.require_tubing:
                issues.append(MissingDataIssue(
                    category='tubing',
                    severity='important',
                    description="No tubing/production string data found",
                    clarification_question="Could you provide the tubing size (OD and ID in inches) and setting depth?"
                ))
            return issues
        
        # Check required fields
        required_fields = ['depth', 'OD', 'ID']
        for i, string in enumerate(tubing):
            missing_fields = [f for f in required_fields if f not in string or string[f] is None]
            if missing_fields:
                issues.append(MissingDataIssue(
                    category='tubing',
                    severity='important',
                    description=f"Tubing string {i+1} missing: {', '.join(missing_fields)}",
                    clarification_question=f"What are the {' and '.join(missing_fields)} for tubing string {i+1}?"
                ))
        
        return issues
    
    def _check_pvt(self, pvt: Dict) -> List[MissingDataIssue]:
        """Check PVT/fluid properties data completeness"""
        issues = []
        
        if not pvt:
            if self.require_pvt:
                issues.append(MissingDataIssue(
                    category='pvt',
                    severity='important',
                    description="No PVT/fluid properties data found",
                    clarification_question="Could you provide fluid properties (density, viscosity, gas-oil ratio, etc.)?"
                ))
            return issues
        
        # Check common PVT properties
        important_props = ['density', 'viscosity']
        missing_props = [p for p in important_props if p not in pvt or pvt[p] is None]
        
        if missing_props:
            issues.append(MissingDataIssue(
                category='pvt',
                severity='important',
                description=f"Missing fluid properties: {', '.join(missing_props)}",
                clarification_question=f"What are the {' and '.join(missing_props)} values for the production fluid?"
            ))
        
        # Check for temperature and pressure
        if 'temperature' not in pvt:
            issues.append(MissingDataIssue(
                category='pvt',
                severity='optional',
                description="Reservoir temperature not specified",
                clarification_question="What is the reservoir temperature (°C or °F)?"
            ))
        
        if 'pressure' not in pvt:
            issues.append(MissingDataIssue(
                category='pvt',
                severity='optional',
                description="Reservoir pressure not specified",
                clarification_question="What is the reservoir pressure (bar or psi)?"
            ))
        
        return issues
    
    def _check_equipment(self, equipment: Dict) -> List[MissingDataIssue]:
        """Check equipment/completion data"""
        issues = []
        
        # Check for completion type
        if 'completion_type' not in equipment:
            issues.append(MissingDataIssue(
                category='equipment',
                severity='optional',
                description="Completion type not specified",
                clarification_question="What type of completion is used (e.g., openhole, slotted liner, perforated)?"
            ))
        
        # Check for artificial lift
        if 'artificial_lift' not in equipment:
            issues.append(MissingDataIssue(
                category='equipment',
                severity='optional',
                description="Artificial lift system not specified",
                clarification_question="Is there an artificial lift system? If yes, what type (ESP, gas lift, beam pump)?"
            ))
        
        return issues
    
    def _calculate_completeness(self, issues: List[MissingDataIssue]) -> float:
        """Calculate overall completeness score"""
        if not issues:
            return 1.0
        
        # Weight by severity
        severity_weights = {
            'critical': 0.3,
            'important': 0.15,
            'optional': 0.05
        }
        
        total_deduction = sum(severity_weights.get(i.severity, 0.1) for i in issues)
        
        completeness = 1.0 - min(total_deduction, 1.0)
        return max(0.0, completeness)
    
    def _generate_summary(self, issues: List[MissingDataIssue], 
                         completeness: float, has_critical: bool) -> str:
        """Generate completeness summary"""
        if not issues:
            return f"✓ Data is complete ({completeness*100:.0f}% completeness)"
        
        critical = [i for i in issues if i.severity == 'critical']
        important = [i for i in issues if i.severity == 'important']
        optional = [i for i in issues if i.severity == 'optional']
        
        summary_parts = [f"Data completeness: {completeness*100:.0f}%"]
        
        if critical:
            summary_parts.append(f"✗ {len(critical)} critical gaps - analysis may not be possible")
        if important:
            summary_parts.append(f"⚠ {len(important)} important gaps - results may be limited")
        if optional:
            summary_parts.append(f"ℹ {len(optional)} optional items missing")
        
        return "\n".join(summary_parts)


def create_agent(config: Dict[str, Any]) -> MissingDataAgent:
    """Factory function to create agent"""
    return MissingDataAgent(config)
