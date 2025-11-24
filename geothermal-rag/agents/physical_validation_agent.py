"""
Physical Validation Agent
Validates well trajectory data against physical constraints and engineering rules.
"""

import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PhysicalViolation:
    """Physical constraint violation"""
    severity: str  # 'error', 'warning'
    violation_type: str
    description: str
    affected_depths: List[float]
    suggestion: str


@dataclass
class PhysicalValidationResult:
    """Result of physical validation"""
    is_valid: bool
    violations: List[PhysicalViolation]
    confidence: float
    summary: str


class PhysicalValidationAgent:
    """Validates trajectory and well data against physical constraints"""
    
    def __init__(self, config: Dict[str, Any], ollama_host: str = "http://localhost:11434"):
        self.config = config
        self.ollama_host = ollama_host
        
        # Model selection
        ollama_config = config.get('ollama', {})
        self.model = ollama_config.get('model_verification', 'llama3.1')
        self.timeout = ollama_config.get('timeout_verification', 420)
        
        # Physical constraints
        validation_config = config.get('validation', {})
        self.max_md = validation_config.get('max_md', 5000.0)  # meters
        self.max_tvd = validation_config.get('max_tvd', 5000.0)  # meters
        self.min_pipe_id = validation_config.get('min_pipe_id', 2.0)  # inches
        self.max_pipe_id = validation_config.get('max_pipe_id', 30.0)  # inches
        
        logger.info(f"Physical validation: MD≤{self.max_md}m, TVD≤{self.max_tvd}m, ID∈[{self.min_pipe_id}, {self.max_pipe_id}] inch")
    
    def validate_trajectory(self, trajectory_data: List[Dict]) -> PhysicalValidationResult:
        """
        Validate well trajectory against physical constraints
        
        Args:
            trajectory_data: List of dicts with keys: MD, TVD, ID (Pipe ID)
            
        Returns:
            PhysicalValidationResult with violations and confidence
        """
        logger.info(f"Validating trajectory with {len(trajectory_data)} points")
        
        violations = []
        
        # Rule 1: MD must be >= TVD (measured depth >= true vertical depth)
        violations.extend(self._check_md_tvd_constraint(trajectory_data))
        
        # Rule 2: Telescoping - deeper pipes must have smaller or equal ID
        violations.extend(self._check_telescoping(trajectory_data))
        
        # Rule 3: Realistic depth ranges
        violations.extend(self._check_depth_ranges(trajectory_data))
        
        # Rule 4: Realistic pipe ID ranges
        violations.extend(self._check_pipe_id_ranges(trajectory_data))
        
        # Rule 5: Monotonic MD/TVD increase
        violations.extend(self._check_monotonic_depths(trajectory_data))
        
        # Use LLM for complex validation if violations found
        if violations:
            llm_assessment = self._llm_validate_complex_cases(trajectory_data, violations)
            if llm_assessment:
                violations.append(llm_assessment)
        
        # Determine overall validity
        errors = [v for v in violations if v.severity == 'error']
        is_valid = len(errors) == 0
        
        # Calculate confidence
        confidence = self._calculate_confidence(trajectory_data, violations)
        
        # Generate summary
        summary = self._generate_summary(violations, is_valid, confidence)
        
        logger.info(f"Validation complete: {'✓ VALID' if is_valid else '✗ INVALID'} ({len(errors)} errors, {len(violations)-len(errors)} warnings)")
        
        return PhysicalValidationResult(
            is_valid=is_valid,
            violations=violations,
            confidence=confidence,
            summary=summary
        )
    
    def _check_md_tvd_constraint(self, data: List[Dict]) -> List[PhysicalViolation]:
        """Check that MD >= TVD for all points"""
        violations = []
        
        for i, point in enumerate(data):
            md = point.get('MD', 0)
            tvd = point.get('TVD', 0)
            
            if md < tvd:
                violations.append(PhysicalViolation(
                    severity='error',
                    violation_type='MD_LESS_THAN_TVD',
                    description=f"Measured Depth ({md:.2f}m) is less than True Vertical Depth ({tvd:.2f}m) at point {i+1}",
                    affected_depths=[md, tvd],
                    suggestion="MD must always be ≥ TVD. Check if depths are swapped or measured incorrectly."
                ))
        
        return violations
    
    def _check_telescoping(self, data: List[Dict]) -> List[PhysicalViolation]:
        """Check that deeper pipes have smaller or equal diameter (telescoping rule)"""
        violations = []
        
        # Sort by MD to check depth progression
        sorted_data = sorted(data, key=lambda x: x.get('MD', 0))
        
        for i in range(len(sorted_data) - 1):
            current = sorted_data[i]
            next_point = sorted_data[i + 1]
            
            current_id = current.get('ID', 0)
            next_id = next_point.get('ID', 0)
            current_md = current.get('MD', 0)
            next_md = next_point.get('MD', 0)
            current_tvd = current.get('TVD', 0)
            next_tvd = next_point.get('TVD', 0)
            
            # Critical rule: deeper pipes must have smaller or equal ID
            if next_id > current_id and next_tvd > current_tvd:
                violations.append(PhysicalViolation(
                    severity='error',
                    violation_type='TELESCOPING_VIOLATION',
                    description=f"Pipe ID increases with depth: {current_id:.2f}\" at {current_tvd:.2f}m TVD → {next_id:.2f}\" at {next_tvd:.2f}m TVD",
                    affected_depths=[current_md, next_md],
                    suggestion="Deeper pipes must have smaller or equal diameter. Check casing/tubing program."
                ))
        
        return violations
    
    def _check_depth_ranges(self, data: List[Dict]) -> List[PhysicalViolation]:
        """Check for unrealistic depth values"""
        violations = []
        
        for i, point in enumerate(data):
            md = point.get('MD', 0)
            tvd = point.get('TVD', 0)
            
            if md > self.max_md:
                violations.append(PhysicalViolation(
                    severity='warning',
                    violation_type='EXCESSIVE_MD',
                    description=f"MD ({md:.2f}m) exceeds typical maximum ({self.max_md}m) at point {i+1}",
                    affected_depths=[md],
                    suggestion="Verify depth units (should be meters) and measurement accuracy."
                ))
            
            if tvd > self.max_tvd:
                violations.append(PhysicalViolation(
                    severity='warning',
                    violation_type='EXCESSIVE_TVD',
                    description=f"TVD ({tvd:.2f}m) exceeds typical maximum ({self.max_tvd}m) at point {i+1}",
                    affected_depths=[tvd],
                    suggestion="Verify depth units (should be meters) and measurement accuracy."
                ))
            
            if md < 0 or tvd < 0:
                violations.append(PhysicalViolation(
                    severity='error',
                    violation_type='NEGATIVE_DEPTH',
                    description=f"Negative depth detected: MD={md:.2f}m, TVD={tvd:.2f}m at point {i+1}",
                    affected_depths=[md, tvd],
                    suggestion="Depths cannot be negative. Check data extraction."
                ))
        
        return violations
    
    def _check_pipe_id_ranges(self, data: List[Dict]) -> List[PhysicalViolation]:
        """Check for unrealistic pipe ID values"""
        violations = []
        
        for i, point in enumerate(data):
            pipe_id = point.get('ID', 0)
            
            if pipe_id < self.min_pipe_id:
                violations.append(PhysicalViolation(
                    severity='warning',
                    violation_type='SMALL_PIPE_ID',
                    description=f"Pipe ID ({pipe_id:.2f}\") is unusually small (< {self.min_pipe_id}\") at point {i+1}",
                    affected_depths=[point.get('MD', 0)],
                    suggestion="Verify units (should be inches) and that this isn't a tubing string."
                ))
            
            if pipe_id > self.max_pipe_id:
                violations.append(PhysicalViolation(
                    severity='warning',
                    violation_type='LARGE_PIPE_ID',
                    description=f"Pipe ID ({pipe_id:.2f}\") is unusually large (> {self.max_pipe_id}\") at point {i+1}",
                    affected_depths=[point.get('MD', 0)],
                    suggestion="Verify units (should be inches) and conductor casing specifications."
                ))
            
            if pipe_id <= 0:
                violations.append(PhysicalViolation(
                    severity='error',
                    violation_type='INVALID_PIPE_ID',
                    description=f"Invalid Pipe ID ({pipe_id:.2f}\") at point {i+1}",
                    affected_depths=[point.get('MD', 0)],
                    suggestion="Pipe ID must be positive. Check data extraction."
                ))
        
        return violations
    
    def _check_monotonic_depths(self, data: List[Dict]) -> List[PhysicalViolation]:
        """Check that depths increase monotonically"""
        violations = []
        
        sorted_data = sorted(enumerate(data), key=lambda x: x[1].get('MD', 0))
        
        for i in range(len(sorted_data) - 1):
            idx1, point1 = sorted_data[i]
            idx2, point2 = sorted_data[i + 1]
            
            md1 = point1.get('MD', 0)
            md2 = point2.get('MD', 0)
            tvd1 = point1.get('TVD', 0)
            tvd2 = point2.get('TVD', 0)
            
            # TVD should not decrease as MD increases
            if tvd2 < tvd1:
                violations.append(PhysicalViolation(
                    severity='warning',
                    violation_type='NON_MONOTONIC_TVD',
                    description=f"TVD decreases from {tvd1:.2f}m to {tvd2:.2f}m as MD increases from {md1:.2f}m to {md2:.2f}m",
                    affected_depths=[md1, md2],
                    suggestion="TVD should increase or stay constant as MD increases. Check trajectory survey."
                ))
        
        return violations
    
    def _llm_validate_complex_cases(self, data: List[Dict], violations: List[PhysicalViolation]) -> PhysicalViolation:
        """Use LLM to assess complex validation scenarios"""
        try:
            # Build data summary
            data_summary = "Trajectory data:\n"
            for i, point in enumerate(data, 1):
                data_summary += f"  Point {i}: MD={point.get('MD', 0):.2f}m, TVD={point.get('TVD', 0):.2f}m, ID={point.get('ID', 0):.2f}\"\n"
            
            # Build violations summary
            violations_summary = "Detected violations:\n"
            for v in violations[:5]:  # Limit to top 5
                violations_summary += f"  - {v.violation_type}: {v.description}\n"
            
            prompt = f"""You are a geothermal well engineering expert. Analyze the following trajectory data and physical constraint violations.

{data_summary}

{violations_summary}

QUESTION: Are these violations likely due to:
A) Data extraction errors (wrong units, swapped columns, parsing issues)
B) Real physical impossibilities that make the data invalid
C) Unusual but potentially valid well design (e.g., sidetrack, complex trajectory)

Provide your assessment in 2-3 sentences, focusing on the most likely root cause and engineering considerations.

Assessment:"""
            
            response = self._call_ollama(prompt)
            
            return PhysicalViolation(
                severity='warning',
                violation_type='COMPLEX_ASSESSMENT',
                description=f"LLM Engineering Assessment: {response}",
                affected_depths=[],
                suggestion="Review data extraction process and verify against source documents."
            )
            
        except Exception as e:
            logger.error(f"LLM validation failed: {str(e)}")
            return None
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API"""
        url = f"{self.ollama_host}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        
        result = response.json()
        return result['response'].strip()
    
    def _calculate_confidence(self, data: List[Dict], violations: List[PhysicalViolation]) -> float:
        """Calculate validation confidence score"""
        if not data:
            return 0.0
        
        # Start with 100% confidence
        confidence = 1.0
        
        # Deduct for violations
        errors = [v for v in violations if v.severity == 'error']
        warnings = [v for v in violations if v.severity == 'warning']
        
        confidence -= len(errors) * 0.3  # -30% per error
        confidence -= len(warnings) * 0.1  # -10% per warning
        
        # Bonus for complete data
        complete_points = sum(1 for p in data if 'MD' in p and 'TVD' in p and 'ID' in p)
        if complete_points == len(data):
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_summary(self, violations: List[PhysicalViolation], 
                         is_valid: bool, confidence: float) -> str:
        """Generate human-readable summary"""
        if is_valid and not violations:
            return f"✓ All physical constraints satisfied (confidence: {confidence*100:.0f}%)"
        
        errors = [v for v in violations if v.severity == 'error']
        warnings = [v for v in violations if v.severity == 'warning']
        
        summary_parts = []
        
        if errors:
            summary_parts.append(f"✗ {len(errors)} critical errors:")
            for err in errors[:3]:  # Show top 3
                summary_parts.append(f"  • {err.description}")
        
        if warnings:
            summary_parts.append(f"⚠ {len(warnings)} warnings:")
            for warn in warnings[:3]:  # Show top 3
                summary_parts.append(f"  • {warn.description}")
        
        summary_parts.append(f"Confidence: {confidence*100:.0f}%")
        
        return "\n".join(summary_parts)


def create_agent(config: Dict[str, Any]) -> PhysicalValidationAgent:
    """Factory function to create agent"""
    ollama_host = config.get('ollama', {}).get('host', 'http://localhost:11434')
    return PhysicalValidationAgent(config, ollama_host)
