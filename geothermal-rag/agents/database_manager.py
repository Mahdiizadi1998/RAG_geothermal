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
        """Create database schema for well data"""
        cursor = self.conn.cursor()
        
        # Wells table - core information
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS wells (
            well_id INTEGER PRIMARY KEY AUTOINCREMENT,
            well_name TEXT UNIQUE NOT NULL,
            operator TEXT,
            location TEXT,
            country TEXT,
            spud_date TEXT,
            completion_date TEXT,
            total_depth_md REAL,
            total_depth_tvd REAL,
            well_type TEXT,
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
            top_of_cement_md REAL,
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
        
        self.conn.commit()
        logger.info("Database schema created successfully")
    
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
            weight, weight_unit, grade, connection_type, top_depth_md, bottom_depth_md,
            depth_unit, shoe_depth_md, shoe_depth_tvd, source_page, source_table
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            well_id,
            casing_data.get('string_number'),
            casing_data.get('casing_type'),
            casing_data.get('outer_diameter'),
            casing_data.get('diameter_unit', 'inch'),
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
    
    def get_well_summary(self, well_name: str) -> Optional[Dict]:
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
        
        return {
            'well_info': dict(well),
            'casing_strings': casings,
            'formations': formations,
            'cementing': cementing
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
