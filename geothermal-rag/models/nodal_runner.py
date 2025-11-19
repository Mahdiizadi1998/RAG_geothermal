"""
Nodal Analysis Runner - Executes nodal_analysis.py with extracted trajectory data
"""

import sys
from pathlib import Path
import subprocess
import tempfile
import logging
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NodalAnalysisRunner:
    """
    Run nodal_analysis.py with extracted well trajectory data
    
    Strategy:
    1. Take extracted trajectory in format: [{"MD": x, "TVD": y, "ID": z}, ...]
    2. Inject into nodal_analysis.py by replacing well_trajectory variable
    3. Execute modified script
    4. Capture results and plot
    """
    
    def __init__(self, nodal_analysis_path: str = None):
        """
        Initialize runner
        
        Args:
            nodal_analysis_path: Path to nodal_analysis.py
        """
        if nodal_analysis_path is None:
            nodal_analysis_path = Path(__file__).parent.parent / 'models' / 'nodal_analysis.py'
        
        self.nodal_analysis_path = Path(nodal_analysis_path)
        
        if not self.nodal_analysis_path.exists():
            raise FileNotFoundError(f"nodal_analysis.py not found at {self.nodal_analysis_path}")
    
    def run_with_extracted_data(self, extracted_data: Dict) -> Tuple[bool, str, Optional[str]]:
        """
        Run nodal analysis with extracted well trajectory
        
        Args:
            extracted_data: Dict from ParameterExtractionAgent with 'trajectory' key
            
        Returns:
            Tuple of (success: bool, output: str, plot_path: Optional[str])
        """
        trajectory = extracted_data.get('trajectory', [])
        
        if not trajectory:
            return False, "No trajectory data to analyze", None
        
        try:
            # Read original nodal_analysis.py
            with open(self.nodal_analysis_path, 'r') as f:
                original_code = f.read()
            
            # Generate well_trajectory in exact format
            well_trajectory_code = self._format_trajectory(trajectory)
            
            # Replace well_trajectory in the code
            modified_code = self._inject_trajectory(original_code, well_trajectory_code)
            
            # Inject PVT data if available
            pvt = extracted_data.get('pvt_data', {})
            if pvt:
                modified_code = self._inject_pvt_data(modified_code, pvt)
            
            # Create temporary file with modified code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(modified_code)
                temp_path = temp_file.name
            
            logger.info(f"Running nodal analysis with {len(trajectory)} trajectory points...")
            
            # Execute the modified script
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up
            Path(temp_path).unlink()
            
            if result.returncode == 0:
                output = result.stdout
                logger.info("✓ Nodal analysis completed successfully")
                return True, output, None  # Could save plot and return path
            else:
                error_msg = f"Nodal analysis failed:\n{result.stderr}"
                logger.error(error_msg)
                return False, error_msg, None
                
        except subprocess.TimeoutExpired:
            return False, "Nodal analysis timed out (>30s)", None
        except Exception as e:
            logger.error(f"Nodal analysis error: {str(e)}")
            return False, f"Error: {str(e)}", None
    
    def _format_trajectory(self, trajectory: list) -> str:
        """
        Format trajectory data in exact required format:
        well_trajectory = [
            {"MD": 0.0,    "TVD": 0.0,    "ID": 0.3397},
            ...
        ]
        """
        lines = ["well_trajectory = ["]
        
        for point in trajectory:
            md = point['md']
            tvd = point['tvd']
            pipe_id = point['pipe_id']
            
            # Format with exact spacing
            line = f'    {{"MD": {md:<7.1f}, "TVD": {tvd:<7.1f}, "ID": {pipe_id:.4f}}},'
            
            # Add comment if available
            if 'comment' in point:
                line += f"  # {point['comment']}"
            
            lines.append(line)
        
        lines.append("]")
        
        return "\n".join(lines)
    
    def _inject_trajectory(self, code: str, new_trajectory: str) -> str:
        """
        Replace well_trajectory in code
        
        Strategy:
        - Find "well_trajectory = [" line
        - Find matching closing "]"
        - Replace entire section
        """
        import re
        
        # Pattern to match well_trajectory = [...] including nested brackets
        pattern = r'well_trajectory\s*=\s*\[.*?\n\]'
        
        if not re.search(pattern, code, re.DOTALL):
            # If pattern not found, inject before "# Well segments"
            marker = "# Well segments"
            if marker in code:
                code = code.replace(marker, f"{new_trajectory}\n\n{marker}")
            else:
                # Inject after "# %% Well trajectory" comment
                marker = "# %% Well trajectory"
                if marker in code:
                    code = code.replace(marker, f"{marker}\n{new_trajectory}\n")
                else:
                    # Fallback: inject at beginning
                    code = f"{new_trajectory}\n\n{code}"
            return code
        
        # Replace existing well_trajectory
        modified_code = re.sub(pattern, new_trajectory, code, flags=re.DOTALL)
        
        return modified_code
    
    def _inject_pvt_data(self, code: str, pvt: Dict) -> str:
        """Inject PVT data into code"""
        import re
        
        # Replace rho if available
        if 'density' in pvt:
            code = re.sub(
                r'rho\s*=\s*[\d.]+',
                f'rho = {pvt["density"]}',
                code
            )
        
        # Replace mu if available
        if 'viscosity' in pvt:
            code = re.sub(
                r'mu\s*=\s*[\d.e-]+',
                f'mu = {pvt["viscosity"]}',
                code
            )
        
        return code
    
    def generate_preview_code(self, extracted_data: Dict) -> str:
        """
        Generate preview of code that will be executed
        
        Args:
            extracted_data: Dict from ParameterExtractionAgent
            
        Returns:
            Python code string showing what will be injected
        """
        trajectory = extracted_data.get('trajectory', [])
        pvt = extracted_data.get('pvt_data', {})
        
        code = "# ========================================\n"
        code += "# Extracted Data for Nodal Analysis\n"
        code += "# ========================================\n\n"
        
        if trajectory:
            code += self._format_trajectory(trajectory)
            code += "\n\n"
        else:
            code += "# No trajectory data extracted\n\n"
        
        code += "# Fluid properties\n"
        code += f"rho = {pvt.get('density', 1000.0)}  # kg/m³\n"
        code += f"mu = {pvt.get('viscosity', 0.001)}   # Pa·s\n\n"
        
        code += "# ========================================\n"
        code += f"# Total trajectory points: {len(trajectory)}\n"
        code += "# This data will be injected into nodal_analysis.py\n"
        code += "# ========================================\n"
        
        return code

