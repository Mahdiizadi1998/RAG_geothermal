"""
Database Migration Script
Migrates existing well_data.db to new schema with additional columns
Run this ONCE to update your database schema
"""

import sqlite3
import os
from pathlib import Path

def migrate_database(db_path='well_data.db'):
    """Add new columns to existing database tables"""
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        print("No migration needed - new database will be created with correct schema")
        return
    
    print(f"🔧 Migrating database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Track successful migrations
    migrations_done = []
    
    try:
        # Add new columns to wells table
        new_wells_columns = [
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
        
        print("\n📋 Migrating wells table...")
        for col_name, col_type in new_wells_columns:
            try:
                cursor.execute(f"ALTER TABLE wells ADD COLUMN {col_name} {col_type}")
                migrations_done.append(f"wells.{col_name}")
                print(f"  ✓ Added column: {col_name}")
            except sqlite3.OperationalError as e:
                if 'duplicate column' in str(e).lower():
                    print(f"  ⊘ Column already exists: {col_name}")
                else:
                    print(f"  ✗ Failed to add {col_name}: {e}")
        
        # Add new columns to casing_strings table
        new_casing_columns = [
            ('pipe_id_nominal', 'REAL'),
            ('pipe_id_drift', 'REAL'),
            ('id_unit', 'TEXT')
        ]
        
        print("\n🔩 Migrating casing_strings table...")
        for col_name, col_type in new_casing_columns:
            try:
                cursor.execute(f"ALTER TABLE casing_strings ADD COLUMN {col_name} {col_type}")
                migrations_done.append(f"casing_strings.{col_name}")
                print(f"  ✓ Added column: {col_name}")
            except sqlite3.OperationalError as e:
                if 'duplicate column' in str(e).lower():
                    print(f"  ⊘ Column already exists: {col_name}")
                else:
                    print(f"  ✗ Failed to add {col_name}: {e}")
        
        # Add new columns to cementing table
        new_cement_columns = [
            ('lead_volume', 'REAL'),
            ('lead_density', 'REAL'),
            ('tail_volume', 'REAL'),
            ('tail_density', 'REAL'),
            ('toc_tvd', 'REAL')
        ]
        
        print("\n🧱 Migrating cementing table...")
        for col_name, col_type in new_cement_columns:
            try:
                cursor.execute(f"ALTER TABLE cementing ADD COLUMN {col_name} {col_type}")
                migrations_done.append(f"cementing.{col_name}")
                print(f"  ✓ Added column: {col_name}")
            except sqlite3.OperationalError as e:
                if 'duplicate column' in str(e).lower():
                    print(f"  ⊘ Column already exists: {col_name}")
                else:
                    print(f"  ✗ Failed to add {col_name}: {e}")
        
        # Create drilling_fluids table if it doesn't exist
        print("\n💧 Creating drilling_fluids table...")
        try:
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
            migrations_done.append("drilling_fluids table")
            print("  ✓ Created drilling_fluids table")
        except sqlite3.OperationalError as e:
            print(f"  ⊘ Table already exists or error: {e}")
        
        # Create index for drilling_fluids
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fluids_well ON drilling_fluids(well_id)")
            print("  ✓ Created index on drilling_fluids")
        except sqlite3.OperationalError:
            pass
        
        conn.commit()
        
        print(f"\n✅ Migration complete! Applied {len(migrations_done)} changes:")
        for change in migrations_done:
            print(f"   • {change}")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        conn.rollback()
        raise
    
    finally:
        conn.close()

if __name__ == "__main__":
    # Run migration
    db_path = Path(__file__).parent / 'well_data.db'
    migrate_database(str(db_path))
    print("\n🎉 Database is now ready for use!")
    print("You can restart the application.")
