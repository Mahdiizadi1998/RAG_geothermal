# IMPORTANT: Fix Database Schema Error

## Problem
You're getting this error:
```
table casing_strings has no column named pipe_id_nominal
```

This means your existing `well_data.db` file has the **old schema** and needs to be updated.

## Solution (CHOOSE ONE)

### Option 1: Migrate Existing Database (Keeps your data)
Run the migration script to add new columns to your existing database:

```bash
cd C:\Users\smada\Documents\geothermal-rag\geothermal-rag
python migrate_database.py
```

This will:
- Add `pipe_id_nominal`, `pipe_id_drift`, `id_unit` to casing_strings table
- Add `license_number`, `coordinate_x`, `coordinate_y`, `rig_name`, etc. to wells table
- Add `lead_volume`, `lead_density`, `tail_volume`, `tail_density` to cementing table
- Create `drilling_fluids` table

### Option 2: Delete and Recreate (Fresh start - RECOMMENDED)
Delete the old database file and let the app create a new one:

```bash
cd C:\Users\smada\Documents\geothermal-rag\geothermal-rag
del well_data.db
```

**OR** using the migration script:

```bash
python migrate_database.py recreate
```

Then restart the app:
```bash
python app.py
```

## What Changed

The database schema was enhanced to store:
1. **Wells table**: License, Coordinates (X/Y), Rig Name, Target Formation, Total Days, Sidetrack Depth
2. **Casing table**: **Pipe ID (Nominal + Drift)** ← This is the critical addition
3. **Cementing table**: Lead/Tail volumes and densities, TOC (MD + TVD)
4. **Drilling Fluids table** (NEW): Hole Size, Fluid Type, Density Range

## After Fixing

Re-upload your PDF and you should see:
- ✅ No column errors
- ✅ Casing with Pipe IDs extracted
- ✅ General well data stored (License, Coordinates, Rig, Target)
- ✅ Complete cementing data (Lead/Tail)
- ✅ Drilling fluids data

## Pipe ID Column Detection

The system now:
- Looks for **exact "Pipe ID"** column name (ignoring units in parentheses)
- Example: "Pipe ID (in)" will be matched as "pipe id"
- Falls back to "Nominal ID" or "Inner Diameter" if "Pipe ID" not found
- Stores in `pipe_id_nominal` column

Drift ID is still detected separately from "Drift" or "Drift ID" columns.
