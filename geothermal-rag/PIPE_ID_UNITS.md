# Pipe ID Unit Standardization

## Overview
All pipe internal diameter (ID) values throughout the system are now **consistently provided in inches**, which is the standard unit in well engineering and oilfield operations.

## Changes Made (Nov 23, 2025)

### 1. **Parameter Extraction Agent**
- **Before:** Converted pipe ID from inches to meters during extraction
- **After:** Keeps pipe ID in inches throughout extraction
- **Location:** `agents/parameter_extraction_agent.py`
- **Impact:** `extracted_data['trajectory']` now contains `pipe_id` in inches

### 2. **Physical Validation Agent**
- **No Change:** Already expected pipe ID in inches (config: 2.0-30.0 inches)
- **Location:** `agents/physical_validation_agent.py`
- **Validation Ranges:** 
  - Minimum: 2.0 inches
  - Maximum: 30.0 inches

### 3. **Main Application**
- **Before:** Converted pipe_id from meters to inches (multiplied by 39.3701)
- **After:** Uses pipe_id directly in inches (no conversion needed)
- **Location:** `app.py` lines 481, 532
- **Impact:** Display and validation now work directly with inches

### 4. **Nodal Analysis Format**
- **No Change in Final Output:** Still converts to meters for nodal analysis
- **Location:** `agents/parameter_extraction_agent.py` - `format_for_nodal_analysis()`
- **Process:** 
  1. Store pipe_id in inches throughout system
  2. Convert to meters only when generating final nodal_analysis.py format
  3. Conversion: `pipe_id_meters = inches * 0.0254`

### 5. **Test System**
- **Updated:** Test data now uses realistic inch values
- **Location:** `test_system.py`
- **Example Values:**
  - 13.375" (13 3/8" casing)
  - 9.625" (9 5/8" casing)  
  - 7.0" (7" casing)

### 6. **Configuration**
- **Updated:** Removed obsolete `pipe_id_min_mm` and `pipe_id_max_mm`
- **Added:** Clear documentation that all pipe IDs are in inches
- **Location:** `config/config.yaml`

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. PDF Extraction                                               │
│    Pattern Library extracts: "13 3/8" casing"                  │
│    ➜ Parsed to: 13.375 inches                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Parameter Extraction Agent                                   │
│    Casing: {'od': 13.375, 'id': 12.615, ...}                  │
│    ➜ Stored in INCHES (no conversion)                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Trajectory-Casing Merge                                      │
│    trajectory[i]['pipe_id'] = 12.615 inches                    │
│    ➜ Stored in INCHES (no conversion)                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Physical Validation                                          │
│    Validates: 2.0 ≤ pipe_id ≤ 30.0 inches                     │
│    ➜ Works directly with INCHES                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Display to User                                              │
│    "ID: 12.62""                                                 │
│    ➜ Shows INCHES (standard unit)                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Nodal Analysis Format (FINAL OUTPUT ONLY)                   │
│    {"ID": 0.3205}  # meters                                    │
│    ➜ Converted to METERS: 12.615 * 0.0254 = 0.3205m           │
└─────────────────────────────────────────────────────────────────┘
```

## Why Inches?

### Industry Standard
- Oilfield and well engineering use inches for pipe sizes globally
- Standard nomenclature: 13 3/8", 9 5/8", 7", 5 1/2"
- All technical documentation uses inches

### User Expectations
- Geothermal engineers expect to see pipe sizes in inches
- Easier to verify against source documents
- No mental conversion needed

### Validation
- Realistic ranges are well-defined in inches:
  - Small tubing: 2-4 inches
  - Production casing: 5-10 inches
  - Surface casing: 10-20 inches
  - Conductor: 20-30 inches

## Unit Conversion Reference

### Inches ↔ Meters
```python
# Inches to meters
meters = inches * 0.0254

# Meters to inches
inches = meters / 0.0254  # or meters * 39.3701
```

### Common Sizes
| Fractional | Decimal | Meters   |
|------------|---------|----------|
| 13 3/8"    | 13.375" | 0.33975m |
| 9 5/8"     | 9.625"  | 0.24448m |
| 7"         | 7.0"    | 0.17780m |
| 5 1/2"     | 5.5"    | 0.13970m |

## Implementation Details

### Where Conversion Happens

**ONLY in `format_for_nodal_analysis()`:**
```python
def format_for_nodal_analysis(self, extracted_data: Dict) -> str:
    for point in trajectory:
        md = point['md']  # meters
        tvd = point['tvd']  # meters
        pipe_id_inches = point['pipe_id']  # INCHES
        
        # Convert to meters ONLY for nodal analysis output
        pipe_id_meters = self.converter.inches_to_meters(pipe_id_inches)
        
        code += f'{{"MD": {md:.1f}, "TVD": {tvd:.1f}, "ID": {pipe_id_meters:.4f}}},\n'
```

### Where NO Conversion Happens

1. **Pattern Library** - Extracts decimal inches directly
2. **Parameter Extraction** - Stores inches without conversion
3. **Physical Validation** - Validates inches directly
4. **App Display** - Shows inches to user
5. **Storage/Memory** - All internal structures use inches

## Testing

### Test Data (Updated)
```python
trajectory = [
    {'md': 0, 'tvd': 0, 'inclination': 0, 'pipe_id': 13.375},  # 13 3/8"
    {'md': 500, 'tvd': 500, 'inclination': 0, 'pipe_id': 9.625},  # 9 5/8"
    {'md': 1500, 'tvd': 1500, 'inclination': 0, 'pipe_id': 7.0},  # 7"
]
```

### Validation Test
```python
# Physical validation expects inches
assert 2.0 <= pipe_id <= 30.0  # inches
assert validate_trajectory([
    {'MD': 0, 'TVD': 0, 'ID': 13.375}  # inches
]).is_valid == True
```

## Configuration

### config.yaml
```yaml
validation:
  # All pipe IDs in inches
  min_pipe_id: 2.0      # inches (minimum realistic)
  max_pipe_id: 30.0     # inches (maximum realistic)
```

## User Documentation

### UI Info Box
```
Inputs:
- MD, TVD: meters
- Inclination: degrees
- Pipe ID: inches (2-30")
```

### Validation Rules
```
✓ Pipe ID: 2-30 inches
✓ MD ≥ TVD (±1m tolerance)
✓ Inclination: 0-90°
```

## Benefits

1. **Consistency** - Single unit throughout entire system
2. **Industry Standard** - Matches user expectations
3. **Easier Validation** - Realistic ranges are intuitive in inches
4. **Better Display** - Users see familiar units
5. **Fewer Conversions** - Convert only once at final output
6. **Fewer Errors** - Less conversion = less chance for mistakes

## Migration Notes

### Breaking Changes
- **Before:** `trajectory[i]['pipe_id']` was in meters
- **After:** `trajectory[i]['pipe_id']` is in inches

### Compatible Changes
- Final nodal_analysis.py output format unchanged (still meters)
- Physical validation ranges unchanged (always used inches)
- User-facing display units unchanged (always showed inches)

### Testing
```bash
# All tests pass with new inch-based system
python test_system.py
# ✓ Physical validation working
# ✓ Pipe ID validation working
# ✓ All tests completed successfully!
```

## Summary

**Pipe ID units are now standardized to INCHES throughout the entire system**, with conversion to meters happening only at the final step when generating nodal_analysis.py code. This matches industry standards, improves data consistency, and reduces conversion errors.
