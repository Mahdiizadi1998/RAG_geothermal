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
                SELECT well_name, license_number, well_type, operator, location, 
                       coordinate_x, coordinate_y, coordinate_system, rig_name, target_formation,
                       total_depth_md, total_depth_tvd, sidetrack_start_depth_md,
                       completion_date, spud_date, end_of_operations, total_days
                FROM wells WHERE well_name = ?
            """,
            'casing_strings': """
                SELECT casing_type, outer_diameter, diameter_unit, pipe_id_nominal, pipe_id_drift, id_unit,
                       weight, weight_unit, grade, connection_type, 
                       top_depth_md, bottom_depth_md, source_page
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
                SELECT casing_type, outer_diameter, diameter_unit, pipe_id_nominal, pipe_id_drift, id_unit,
                       weight, weight_unit, grade, connection_type, 
                       top_depth_md, bottom_depth_md, shoe_depth_md, source_page
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
                SELECT stage_number, cement_type, lead_volume, lead_density, 
                       tail_volume, tail_density, top_of_cement_md, toc_tvd,
                       bottom_of_cement_md, volume, density, source_page
                FROM cementing
                WHERE well_id = (SELECT well_id FROM wells WHERE well_name = ?)
                ORDER BY stage_number
            """,
            'fluids': """
                SELECT hole_size, hole_size_unit, fluid_type, density_min, density_max, density_unit,
                       depth_interval_from, depth_interval_to, source_page
                FROM drilling_fluids
                WHERE well_id = (SELECT well_id FROM wells WHERE well_name = ?)
                ORDER BY hole_size DESC
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
    """Format casing data into human-readable summary with Pipe IDs"""
    if not casing_strings:
        return "No casing information available."
    
    lines = []
    for i, casing in enumerate(casing_strings, 1):
        casing_type = casing.get('casing_type', 'Casing')
        size = casing.get('outer_diameter')
        size_unit = casing.get('diameter_unit', 'inch')
        id_nominal = casing.get('pipe_id_nominal')
        id_drift = casing.get('pipe_id_drift')
        id_unit = casing.get('id_unit', 'inch')
        weight = casing.get('weight')
        weight_unit = casing.get('weight_unit', 'lb/ft')
        grade = casing.get('grade')
        connection = casing.get('connection_type')
        top_depth = casing.get('top_depth_md')
        depth = casing.get('bottom_depth_md')
        source_page = casing.get('source_page')
        
        # Format size (handle fractions)
        if size:
            if size == int(size):
                size_str = f"{int(size)}"
            else:
                size_str = f"{size:.3f}".rstrip('0').rstrip('.')
        else:
            size_str = "unknown"
        
        parts = [casing_type.capitalize()]
        if size:
            parts.append(f"{size_str} {size_unit} OD")
        
        # Add Pipe IDs (CRITICAL INFO)
        id_parts = []
        if id_nominal:
            id_parts.append(f"ID Nominal: {id_nominal:.3f} {id_unit}")
        if id_drift:
            id_parts.append(f"ID Drift: {id_drift:.3f} {id_unit}")
        if id_parts:
            parts.append(f"({', '.join(id_parts)})")
        
        if weight:
            parts.append(f"{weight} {weight_unit}")
        if grade:
            parts.append(grade)
        if connection:
            parts.append(connection)
        if top_depth and depth:
            parts.append(f"from {top_depth}m to {depth}m MD")
        elif depth:
            parts.append(f"set at {depth}m MD")
        
        line = f"- {' '.join(parts)}"
        
        if include_source and source_page:
            line += f" [Page {source_page}]"
        
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
    """Format cementing operations with Lead/Tail details"""
    if not cementing:
        return "No cementing information available."
    
    lines = []
    for cement in cementing:
        stage = cement.get('stage_number')
        cement_type = cement.get('cement_type')
        lead_vol = cement.get('lead_volume')
        lead_dens = cement.get('lead_density')
        tail_vol = cement.get('tail_volume')
        tail_dens = cement.get('tail_density')
        top = cement.get('top_of_cement_md')
        toc_tvd = cement.get('toc_tvd')
        bottom = cement.get('bottom_of_cement_md')
        volume = cement.get('volume')
        density = cement.get('density')
        source_page = cement.get('source_page')
        
        parts = [f"- Stage {stage}" if stage else "- Cement job"]
        if cement_type:
            parts.append(f"({cement_type})")
        
        # Lead/Tail details (CRITICAL)
        if lead_vol and lead_dens:
            parts.append(f"Lead: {lead_vol}m³ @ {lead_dens}kg/m³")
        if tail_vol and tail_dens:
            parts.append(f"Tail: {tail_vol}m³ @ {tail_dens}kg/m³")
        
        # TOC
        if top:
            toc_str = f"TOC: {top}m MD"
            if toc_tvd:
                toc_str += f" ({toc_tvd}m TVD)"
            parts.append(toc_str)
        
        if bottom:
            parts.append(f"from {bottom}m MD")
        if volume:
            parts.append(f"{volume}m³")
        
        line = ' '.join(parts)
        
        if include_source and source_page:
            line += f" [Source: Page {source_page}]"
        
        lines.append(line)
    
    return '\n'.join(lines)


def format_fluids_summary(fluids: list, include_source: bool = True) -> str:
    """Format drilling fluids data with Hole Size, Type, Density Range"""
    if not fluids:
        return "No drilling fluids information available."
    
    lines = []
    for fluid in fluids:
        hole_size = fluid.get('hole_size')
        hole_unit = fluid.get('hole_size_unit', 'inch')
        fluid_type = fluid.get('fluid_type')
        dens_min = fluid.get('density_min')
        dens_max = fluid.get('density_max')
        dens_unit = fluid.get('density_unit', 'kg/m3')
        depth_from = fluid.get('depth_interval_from')
        depth_to = fluid.get('depth_interval_to')
        source_page = fluid.get('source_page')
        
        parts = []
        if hole_size:
            parts.append(f"- {hole_size} {hole_unit} hole")
        else:
            parts.append("- Drilling fluid")
        
        if fluid_type:
            parts.append(f"({fluid_type})")
        
        # Density range (CRITICAL)
        if dens_min and dens_max:
            if dens_min == dens_max:
                parts.append(f"Density: {dens_min} {dens_unit}")
            else:
                parts.append(f"Density: {dens_min}-{dens_max} {dens_unit}")
        elif dens_min:
            parts.append(f"Density: {dens_min} {dens_unit}")
        
        if depth_from and depth_to:
            parts.append(f"from {depth_from}m to {depth_to}m")
        
        line = ' '.join(parts)
        
        if include_source and source_page:
            line += f" [Page {source_page}]"
        
        lines.append(line)
    
    return '\n'.join(lines)
