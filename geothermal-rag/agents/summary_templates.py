"""
Summary Templates - Structured templates for well summaries
Templates with placeholders filled from database + enriched with narrative context
"""

SUMMARY_TEMPLATES = {
    'basic_completion': {
        'name': 'Basic Completion Summary',
        'description': 'Essential completion information: well name, depths, casing',
        'required_data': ['well_name', 'total_depth_md'],
        'optional_data': ['operator', 'casing_strings'],
        'template': """
The {well_name} well{operator_text} reached a total measured depth of {total_depth_md}m MD{tvd_text}.
{casing_summary}
{completion_date_text}
        """.strip(),
        'sql_queries': {
            'well_info': """
                SELECT well_name, operator, total_depth_md, total_depth_tvd, 
                       completion_date, spud_date, location
                FROM wells WHERE well_name = ?
            """,
            'casing_strings': """
                SELECT outer_diameter, diameter_unit, weight, weight_unit, 
                       grade, bottom_depth_md, source_page
                FROM casing_strings 
                WHERE well_id = (SELECT well_id FROM wells WHERE well_name = ?)
                ORDER BY bottom_depth_md DESC
            """
        }
    },
    
    'detailed_technical': {
        'name': 'Detailed Technical Summary',
        'description': 'Comprehensive: well data, casing, formations, cementing',
        'required_data': ['well_name', 'total_depth_md', 'casing_strings'],
        'optional_data': ['formations', 'cementing', 'operator'],
        'template': """
The {well_name} well{operator_text}{location_text} was drilled to a total measured depth of {total_depth_md}m MD{tvd_text}.

{casing_summary}

{formation_summary}

{cement_summary}

{dates_summary}
        """.strip(),
        'sql_queries': {
            'well_info': """
                SELECT well_name, operator, location, total_depth_md, total_depth_tvd,
                       completion_date, spud_date
                FROM wells WHERE well_name = ?
            """,
            'casing_strings': """
                SELECT casing_type, outer_diameter, diameter_unit, weight, weight_unit,
                       grade, connection_type, bottom_depth_md, shoe_depth_md, source_page
                FROM casing_strings
                WHERE well_id = (SELECT well_id FROM wells WHERE well_name = ?)
                ORDER BY bottom_depth_md DESC
            """,
            'formations': """
                SELECT formation_name, top_md, top_tvd, lithology, age, source_page
                FROM formations
                WHERE well_id = (SELECT well_id FROM wells WHERE well_name = ?)
                ORDER BY top_md
            """,
            'cementing': """
                SELECT stage_number, cement_type, top_of_cement_md, bottom_of_cement_md,
                       volume, density, source_page
                FROM cementing
                WHERE well_id = (SELECT well_id FROM wells WHERE well_name = ?)
                ORDER BY stage_number
            """
        }
    },
    
    'comprehensive': {
        'name': 'Comprehensive Well Report',
        'description': 'Full details: all technical data + narrative context',
        'required_data': ['well_name', 'total_depth_md'],
        'optional_data': ['operator', 'casing_strings', 'formations', 'cementing', 'operations', 'measurements'],
        'template': """
{well_header}

WELL DATA:
{well_data_section}

CASING PROGRAM:
{casing_section}

GEOLOGICAL INFORMATION:
{geology_section}

CEMENTING OPERATIONS:
{cementing_section}

OPERATIONS SUMMARY:
{operations_section}

ADDITIONAL INFORMATION:
{narrative_context}
        """.strip(),
        'sql_queries': {
            'well_info': """
                SELECT * FROM wells WHERE well_name = ?
            """,
            'casing_strings': """
                SELECT * FROM casing_strings
                WHERE well_id = (SELECT well_id FROM wells WHERE well_name = ?)
                ORDER BY bottom_depth_md DESC
            """,
            'formations': """
                SELECT * FROM formations
                WHERE well_id = (SELECT well_id FROM wells WHERE well_name = ?)
                ORDER BY top_md
            """,
            'cementing': """
                SELECT * FROM cementing
                WHERE well_id = (SELECT well_id FROM wells WHERE well_name = ?)
                ORDER BY stage_number
            """,
            'operations': """
                SELECT * FROM operations
                WHERE well_id = (SELECT well_id FROM wells WHERE well_name = ?)
                ORDER BY operation_date
            """,
            'measurements': """
                SELECT * FROM measurements
                WHERE well_id = (SELECT well_id FROM wells WHERE well_name = ?)
                ORDER BY depth_md
            """
        }
    }
}


def format_casing_summary(casing_strings: list, include_source: bool = True) -> str:
    """Format casing data into human-readable summary"""
    if not casing_strings:
        return "No casing information available."
    
    lines = []
    for i, casing in enumerate(casing_strings, 1):
        size = casing.get('outer_diameter')
        size_unit = casing.get('diameter_unit', 'inch')
        weight = casing.get('weight')
        weight_unit = casing.get('weight_unit', 'lb/ft')
        grade = casing.get('grade')
        depth = casing.get('bottom_depth_md')
        source_page = casing.get('source_page')
        
        # Format size (handle fractions)
        if size:
            if size == int(size):
                size_str = f"{int(size)}"
            else:
                # Convert decimal to fraction representation
                size_str = f"{size:.3f}".rstrip('0').rstrip('.')
        else:
            size_str = "unknown"
        
        parts = []
        if size:
            parts.append(f"{size_str} {size_unit}")
        if weight:
            parts.append(f"{weight} {weight_unit}")
        if grade:
            parts.append(grade)
        if depth:
            parts.append(f"set at {depth}m MD")
        
        line = f"- {' '.join(parts)}"
        
        if include_source and source_page:
            line += f" [Source: Page {source_page}]"
        
        lines.append(line)
    
    return '\n'.join(lines)


def format_formation_summary(formations: list, include_source: bool = True) -> str:
    """Format formation tops into human-readable summary"""
    if not formations:
        return "No formation information available."
    
    lines = []
    for formation in formations:
        name = formation.get('formation_name')
        top_md = formation.get('top_md')
        top_tvd = formation.get('top_tvd')
        lithology = formation.get('lithology')
        source_page = formation.get('source_page')
        
        parts = [f"- {name}"]
        if top_md:
            parts.append(f"top at {top_md}m MD")
        if top_tvd and top_tvd != top_md:
            parts.append(f"({top_tvd}m TVD)")
        if lithology:
            parts.append(f"- {lithology}")
        
        line = ' '.join(parts)
        
        if include_source and source_page:
            line += f" [Source: Page {source_page}]"
        
        lines.append(line)
    
    return '\n'.join(lines)


def format_cement_summary(cementing: list, include_source: bool = True) -> str:
    """Format cementing operations into human-readable summary"""
    if not cementing:
        return "No cementing information available."
    
    lines = []
    for cement in cementing:
        stage = cement.get('stage_number')
        cement_type = cement.get('cement_type')
        top = cement.get('top_of_cement_md')
        bottom = cement.get('bottom_of_cement_md')
        volume = cement.get('volume')
        source_page = cement.get('source_page')
        
        parts = [f"- Stage {stage}" if stage else "- Cement job"]
        if cement_type:
            parts.append(f"({cement_type})")
        if top and bottom:
            parts.append(f"from {bottom}m to {top}m MD")
        elif top:
            parts.append(f"to {top}m MD")
        if volume:
            parts.append(f"{volume}m³")
        
        line = ' '.join(parts)
        
        if include_source and source_page:
            line += f" [Source: Page {source_page}]"
        
        lines.append(line)
    
    return '\n'.join(lines)
