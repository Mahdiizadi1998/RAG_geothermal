"""
Recreate Database Script
Deletes old database and creates new one with correct schema
WARNING: This will delete all existing data!
"""

import os
from pathlib import Path

def recreate_database():
    """Delete old database file"""
    db_path = Path(__file__).parent / 'well_data.db'
    
    if db_path.exists():
        print(f"🗑️  Deleting old database: {db_path}")
        os.remove(db_path)
        print("✅ Old database deleted")
    else:
        print("ℹ️  No existing database found")
    
    print("\n✅ Database will be recreated with new schema on next app start")
    print("Run: python app.py")

if __name__ == "__main__":
    response = input("⚠️  WARNING: This will DELETE all existing well data!\nType 'yes' to continue: ")
    if response.lower() == 'yes':
        recreate_database()
    else:
        print("❌ Cancelled")
