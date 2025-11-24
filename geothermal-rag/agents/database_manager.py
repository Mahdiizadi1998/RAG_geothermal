"""
Database Manager - Structured Storage for Well Data
Stores exact numerical data, table contents, and technical specs in SQLite
for precise querying without semantic embedding ambiguity.
"""

import sqlite3
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WellDatabaseManager:
    """
    Manages SQLite database for well technical data
    
    Schema Design:
    - wells: Core well information (name, operator, location, dates)
    - casing_strings: Casing program details (size, weight, grade, depth)
    - cementing: Cement job records (stage, top, bottom, volume)
    - formations: Formation tops (name, MD, TVD)
    - operations: Time-based operations log
    - measurements: Numerical measurements (temperature, pressure, etc.)
    - documents: Source document tracking
    """
    
    def __init__(self, db_path: str = "./well_data.db"):
        """Initialize database connection and create schema"""
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return dict-like rows
        self._create_schema()
        logger.info(f"Initialized well database at {self.db_path}")
    
    def _create_schema(self):
        """Create database schema for well data with automatic migration"""
        cursor = self.conn.cursor()
        
        # Check if migration is needed
        self._migrate_if_needed(cursor)
        
        # Wells table - core information
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS wells (
            well_id INTEGER PRIMARY KEY AUTOINCREMENT,
            well_name TEXT UNIQUE NOT NULL,
            license_number TEXT,
            well_type TEXT,
            operator TEXT,
            location TEXT,
            country TEXT,
            coordinate_x REAL,
            coordinate_y REAL,
            coordinate_system TEXT,
            rig_name TEXT,
            target_formation TEXT,
            spud_date TEXT,
            completion_date TEXT,
            end_of_operations TEXT,
            total_days INTEGER,
            total_depth_md REAL,
            total_depth_tvd REAL,
            sidetrack_start_depth_md REAL,
            status TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Casing strings table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS casing_strings (
            casing_id INTEGER PRIMARY KEY AUTOINCREMENT,
            well_id INTEGER NOT NULL,
            string_number INTEGER,
            casing_type TEXT,
            outer_diameter REAL,
            diameter_unit TEXT DEFAULT 'inch',
            pipe_id_nominal REAL,
            pipe_id_drift REAL,
            id_unit TEXT DEFAULT 'inch',
            weight REAL,
            weight_unit TEXT DEFAULT 'lb/ft',
            grade TEXT,
            connection_type TEXT,
            top_depth_md REAL,
            bottom_depth_md REAL,
            depth_unit TEXT DEFAULT 'm',
            shoe_depth_md REAL,
            shoe_depth_tvd REAL,
            source_page INTEGER,
            source_table TEXT,
            FOREIGN KEY (well_id) REFERENCES wells(well_id)
        )
        """)
        
        # Cementing operations table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS cementing (
            cement_id INTEGER PRIMARY KEY AUTOINCREMENT,
            well_id INTEGER NOT NULL,
            casing_id INTEGER,
            stage_number INTEGER,
            cement_type TEXT,
            lead_volume REAL,
            lead_density REAL,
            tail_volume REAL,
            tail_density REAL,
            top_of_cement_md REAL,
            toc_tvd REAL,
            bottom_of_cement_md REAL,
            volume REAL,
            volume_unit TEXT DEFAULT 'm3',
            density REAL,
            density_unit TEXT DEFAULT 'kg/m3',
            date TEXT,
            source_page INTEGER,
            FOREIGN KEY (well_id) REFERENCES wells(well_id),
            FOREIGN KEY (casing_id) REFERENCES casing_strings(casing_id)
        )
        """)
        
        # Formation tops table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS formations (
            formation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            well_id INTEGER NOT NULL,
            formation_name TEXT,
            top_md REAL,
            top_tvd REAL,
            bottom_md REAL,
            bottom_tvd REAL,
            lithology TEXT,
            age TEXT,
            source_page INTEGER,
            FOREIGN KEY (well_id) REFERENCES wells(well_id)
        )
        """)
        
        # Operations log table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS operations (
            operation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            well_id INTEGER NOT NULL,
            operation_date TEXT,
            operation_type TEXT,
            description TEXT,
            depth_md REAL,
            duration_hours REAL,
            status TEXT,
            source_page INTEGER,
            FOREIGN KEY (well_id) REFERENCES wells(well_id)
        )
        """)
        
        # Measurements table (generic for temperature, pressure, flow, etc.)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS measurements (
            measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
            well_id INTEGER NOT NULL,
            measurement_type TEXT,
            depth_md REAL,
            depth_tvd REAL,
            value REAL,
            unit TEXT,
            measurement_date TEXT,
            source_page INTEGER,
            FOREIGN KEY (well_id) REFERENCES wells(well_id)
        )
        """)
        
        # Drilling fluids table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS drilling_fluids (
            fluid_id INTEGER PRIMARY KEY AUTOINCREMENT,
            well_id INTEGER NOT NULL,
            hole_size REAL,
            hole_size_unit TEXT DEFAULT 'inch',
            fluid_type TEXT,
            density_min REAL,
            density_max REAL,
            density_unit TEXT DEFAULT 'kg/m3',
            depth_interval_from REAL,
            depth_interval_to REAL,
            depth_unit TEXT DEFAULT 'm',
            source_page INTEGER,
            FOREIGN KEY (well_id) REFERENCES wells(well_id)
        )
        """)
        
        # Documents table - track source files
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            document_id INTEGER PRIMARY KEY AUTOINCREMENT,
            well_id INTEGER,
            filename TEXT NOT NULL,
            filepath TEXT,
            document_type TEXT,
            total_pages INTEGER,
            upload_date TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (well_id) REFERENCES wells(well_id)
        )
        """)
        
        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_well_name ON wells(well_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_casing_well ON casing_strings(well_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_formations_well ON formations(well_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fluids_well ON drilling_fluids(well_id)")
        
        self.conn.commit()
        logger.info("Database schema created successfully")
    
    def _migrate_if_needed(self, cursor):
        """Check if old schema exists and migrate to new schema"""
        try:
            # Check if casing_strings table exists and has old schema
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='casing_strings'")
            result = cursor.fetchone()
            
            if result and 'pipe_id_nominal' not in result[0]:
                logger.warning("⚠️  Old database schema detected - applying migration...")
                
                # Add missing columns to casing_strings
                try:
                    cursor.execute("ALTER TABLE casing_strings ADD COLUMN pipe_id_nominal REAL")
                    logger.info("  ✓ Added pipe_id_nominal")
                except sqlite3.OperationalError:
                    pass
                
                try:
                    cursor.execute("ALTER TABLE casing_strings ADD COLUMN pipe_id_drift REAL")
                    logger.info("  ✓ Added pipe_id_drift")
                except sqlite3.OperationalError:
                    pass
                
                try:
                    cursor.execute("ALTER TABLE casing_strings ADD COLUMN id_unit TEXT DEFAULT 'inch'")
                    logger.info("  ✓ Added id_unit")
                except sqlite3.OperationalError:
                    pass
                
                # Check and add wells table columns
                cursor.execute("PRAGMA table_info(wells)")
                wells_cols = [row[1] for row in cursor.fetchall()]
                
                wells_migrations = [
                    ('license_number', 'TEXT'),
                    ('coordinate_x', 'REAL'),
                    ('coordinate_y', 'REAL'),
                    ('coordinate_system', 'TEXT'),
                    ('rig_name', 'TEXT'),
                    ('target_formation', 'TEXT'),
                    ('end_of_operations', 'TEXT'),
                    ('total_days', 'INTEGER'),
                    ('sidetrack_start_depth_md', 'REAL')
                ]
                
                for col_name, col_type in wells_migrations:
                    if col_name not in wells_cols:
                        try:
                            cursor.execute(f"ALTER TABLE wells ADD COLUMN {col_name} {col_type}")
                            logger.info(f"  ✓ Added wells.{col_name}")
                        except sqlite3.OperationalError:
                            pass
                
                # Check cementing table
                cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='cementing'")
                cement_result = cursor.fetchone()
                
                if cement_result and 'lead_volume' not in cement_result[0]:
                    cement_migrations = [
                        ('lead_volume', 'REAL'),
                        ('lead_density', 'REAL'),
                        ('tail_volume', 'REAL'),
                        ('tail_density', 'REAL'),
                        ('toc_tvd', 'REAL')
                    ]
                    
                    for col_name, col_type in cement_migrations:
                        try:
                            cursor.execute(f"ALTER TABLE cementing ADD COLUMN {col_name} {col_type}")
                            logger.info(f"  ✓ Added cementing.{col_name}")
                        except sqlite3.OperationalError:
                            pass
                
                self.conn.commit()
                logger.info("✅ Migration complete!")
                
        except Exception as e:
            logger.error(f"Migration check failed: {str(e)}")
    
    def add_or_get_well(self, well_name: str, **kwargs) -> int:
        """
        Add new well or get existing well_id
        
        Args:
            well_name: Well name (required)
            **kwargs: Optional fields (operator, location, spud_date, etc.)
            
        Returns:
            well_id (int)
        """
        cursor = self.conn.cursor()
        
        # Check if well exists
        cursor.execute("SELECT well_id FROM wells WHERE well_name = ?", (well_name,))
        result = cursor.fetchone()
        
        if result:
            well_id = result['well_id']
            # Update if new information provided
            if kwargs:
                update_fields = ', '.join([f"{k} = ?" for k in kwargs.keys()])
                update_values = list(kwargs.values()) + [well_name]
                cursor.execute(f"UPDATE wells SET {update_fields}, updated_at = CURRENT_TIMESTAMP WHERE well_name = ?", 
                             update_values)
                self.conn.commit()
                logger.debug(f"Updated well {well_name} (ID: {well_id})")
            return well_id
        
        # Insert new well
        fields = ['well_name'] + list(kwargs.keys())
        placeholders = ','.join(['?'] * len(fields))
        values = [well_name] + list(kwargs.values())
        
        cursor.execute(f"INSERT INTO wells ({','.join(fields)}) VALUES ({placeholders})", values)
        self.conn.commit()
        well_id = cursor.lastrowid
        logger.info(f"Added new well: {well_name} (ID: {well_id})")
        return well_id
    
    def add_casing_string(self, well_name: str, casing_data: Dict) -> int:
        """Add casing string to database"""
        well_id = self.add_or_get_well(well_name)
        cursor = self.conn.cursor()
        
        cursor.execute("""
        INSERT INTO casing_strings (
            well_id, string_number, casing_type, outer_diameter, diameter_unit,
            pipe_id_nominal, pipe_id_drift, id_unit,
            weight, weight_unit, grade, connection_type, top_depth_md, bottom_depth_md,
            depth_unit, shoe_depth_md, shoe_depth_tvd, source_page, source_table
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            well_id,
            casing_data.get('string_number'),
            casing_data.get('casing_type'),
            casing_data.get('outer_diameter'),
            casing_data.get('diameter_unit', 'inch'),
            casing_data.get('pipe_id_nominal'),
            casing_data.get('pipe_id_drift'),
            casing_data.get('id_unit', 'inch'),
            casing_data.get('weight'),
            casing_data.get('weight_unit', 'lb/ft'),
            casing_data.get('grade'),
            casing_data.get('connection_type'),
            casing_data.get('top_depth_md'),
            casing_data.get('bottom_depth_md'),
            casing_data.get('depth_unit', 'm'),
            casing_data.get('shoe_depth_md'),
            casing_data.get('shoe_depth_tvd'),
            casing_data.get('source_page'),
            casing_data.get('source_table')
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def add_formation(self, well_name: str, formation_data: Dict) -> int:
        """Add formation top to database"""
        well_id = self.add_or_get_well(well_name)
        cursor = self.conn.cursor()
        
        cursor.execute("""
        INSERT INTO formations (
            well_id, formation_name, top_md, top_tvd, bottom_md, bottom_tvd,
            lithology, age, source_page
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            well_id,
            formation_data.get('formation_name'),
            formation_data.get('top_md'),
            formation_data.get('top_tvd'),
            formation_data.get('bottom_md'),
            formation_data.get('bottom_tvd'),
            formation_data.get('lithology'),
            formation_data.get('age'),
            formation_data.get('source_page')
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def add_cementing_job(self, well_name: str, cement_data: Dict) -> int:
        """Add cementing job to database"""
        well_id = self.add_or_get_well(well_name)
        cursor = self.conn.cursor()
        
        cursor.execute("""
        INSERT INTO cementing (
            well_id, casing_id, stage_number, cement_type, lead_volume, lead_density,
            tail_volume, tail_density, top_of_cement_md, toc_tvd, bottom_of_cement_md,
            volume, volume_unit, density, density_unit, date, source_page
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            well_id,
            cement_data.get('casing_id'),
            cement_data.get('stage_number'),
            cement_data.get('cement_type'),
            cement_data.get('lead_volume'),
            cement_data.get('lead_density'),
            cement_data.get('tail_volume'),
            cement_data.get('tail_density'),
            cement_data.get('top_of_cement_md'),
            cement_data.get('toc_tvd'),
            cement_data.get('bottom_of_cement_md'),
            cement_data.get('volume'),
            cement_data.get('volume_unit', 'm3'),
            cement_data.get('density'),
            cement_data.get('density_unit', 'kg/m3'),
            cement_data.get('date'),
            cement_data.get('source_page')
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def add_drilling_fluid(self, well_name: str, fluid_data: Dict) -> int:
        """Add drilling fluid data to database"""
        well_id = self.add_or_get_well(well_name)
        cursor = self.conn.cursor()
        
        cursor.execute("""
        INSERT INTO drilling_fluids (
            well_id, hole_size, hole_size_unit, fluid_type, density_min, density_max,
            density_unit, depth_interval_from, depth_interval_to, depth_unit, source_page
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            well_id,
            fluid_data.get('hole_size'),
            fluid_data.get('hole_size_unit', 'inch'),
            fluid_data.get('fluid_type'),
            fluid_data.get('density_min'),
            fluid_data.get('density_max'),
            fluid_data.get('density_unit', 'kg/m3'),
            fluid_data.get('depth_interval_from'),
            fluid_data.get('depth_interval_to'),
            fluid_data.get('depth_unit', 'm'),
            fluid_data.get('source_page')
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_well_summary(self, well_name: str) -> Dict:
        """Get comprehensive well summary from database"""
        cursor = self.conn.cursor()
        
        # Get well info
        cursor.execute("SELECT * FROM wells WHERE well_name = ?", (well_name,))
        well = cursor.fetchone()
        
        if not well:
            return None
        
        well_id = well['well_id']
        
        # Get casing program
        cursor.execute("""
        SELECT * FROM casing_strings 
        WHERE well_id = ? 
        ORDER BY bottom_depth_md DESC
        """, (well_id,))
        casings = [dict(row) for row in cursor.fetchall()]
        
        # Get formations
        cursor.execute("""
        SELECT * FROM formations 
        WHERE well_id = ? 
        ORDER BY top_md
        """, (well_id,))
        formations = [dict(row) for row in cursor.fetchall()]
        
        # Get cementing
        cursor.execute("""
        SELECT * FROM cementing 
        WHERE well_id = ? 
        ORDER BY stage_number
        """, (well_id,))
        cementing = [dict(row) for row in cursor.fetchall()]
        
        # Get drilling fluids
        cursor.execute("""
        SELECT * FROM drilling_fluids 
        WHERE well_id = ? 
        ORDER BY hole_size DESC
        """, (well_id,))
        fluids = [dict(row) for row in cursor.fetchall()]
        
        return {
            'well_info': dict(well),
            'casing_strings': casings,
            'formations': formations,
            'cementing': cementing,
            'drilling_fluids': fluids
        }
    
    def query_sql(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute custom SQL query"""
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_well_by_name(self, well_name: str) -> Optional[Dict]:
        """Get well basic info by name"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM wells WHERE well_name = ?", (well_name,))
        result = cursor.fetchone()
        return dict(result) if result else None
    
    def clear_well_data(self, well_name: str):
        """Clear all data for a specific well"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT well_id FROM wells WHERE well_name = ?", (well_name,))
        result = cursor.fetchone()
        
        if not result:
            return
        
        well_id = result['well_id']
        
        # Delete in order (respect foreign keys)
        cursor.execute("DELETE FROM cementing WHERE well_id = ?", (well_id,))
        cursor.execute("DELETE FROM casing_strings WHERE well_id = ?", (well_id,))
        cursor.execute("DELETE FROM formations WHERE well_id = ?", (well_id,))
        cursor.execute("DELETE FROM operations WHERE well_id = ?", (well_id,))
        cursor.execute("DELETE FROM measurements WHERE well_id = ?", (well_id,))
        cursor.execute("DELETE FROM documents WHERE well_id = ?", (well_id,))
        cursor.execute("DELETE FROM wells WHERE well_id = ?", (well_id,))
        
        self.conn.commit()
        logger.info(f"Cleared all data for well: {well_name}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
