#!/usr/bin/env python3
"""
Import scraped CSV data into PostgreSQL database with full data cleaning
"""
import os
import sys
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from io import StringIO
import re

load_dotenv()

def clean_and_transform_data(df):
    """Clean and transform raw CSV data to match database schema"""
    
    # Column mapping (CSV to Database)
    column_map = {
        'url': 'url',
        'brand': 'brand',
        'model': 'model',
        'variant': 'variant',
        'title': 'title',
        'price': 'price_raw',
        'description': 'description',
        'details_model_year': 'model_year',
        'details_first_registration': 'first_registration',
        'details_mileage_km': 'mileage_raw',
        'details_fuel_type': 'fuel_type',
        'details_range_km': 'range_km',
        'details_battery_capacity_kwh': 'battery_capacity',
        'details_energy_consumption': 'energy_consumption',
        'details_home_charging_ac': 'home_charging_ac',
        'details_fast_charging_dc': 'fast_charging_dc',
        'details_charging_time_dc_10_80_pct': 'charging_time_dc',
        'details_periodic_tax': 'periodic_tax',
        'details_power_hp_nm': 'power_raw',
        'details_acceleration_0_100': 'acceleration',
        'details_top_speed': 'top_speed',
        'details_towing_capacity': 'towing_capacity',
        'details_color': 'color',
        'model_new_price': 'new_price_raw',
        'model_category': 'category',
        'model_body_type': 'body_type',
        'model_trunk_size': 'trunk_size',
        'model_weight_kg': 'weight',
        'model_width_cm': 'width',
        'model_length_cm': 'length',
        'model_height_cm': 'height',
        'model_load_capacity_kg': 'load_capacity',
        'model_max_towing_with_brake': 'max_towing_weight',
        'model_drive_type': 'drive_type',
        'model_abs_brakes': 'abs_brakes',
        'model_esp': 'esp',
        'model_airbags': 'airbags',
        'model_doors': 'doors',
        'equipment': 'equipment',
        'details_fuel_consumption': 'fuel_consumption',
        'details_co2_emission': 'co2_emission',
        'details_euro_norm': 'euro_norm',
        'details_transmission_type': 'transmission',
        'details_number_of_gears': 'gear_count',
        'model_cylinders': 'cylinders',
        'model_tank_capacity_l': 'tank_capacity',
        'details_production_year': 'production_date'
    }
    
    df = df.rename(columns=column_map)
    
    # Helper to safely cast numeric to Int64
    def to_int64(series):
        ser = pd.to_numeric(series, errors='coerce')
        ser = ser.apply(lambda x: int(round(x)) if pd.notna(x) else pd.NA)
        return ser.astype('Int64')
    
    # Clean price - remove "kr." and dots (thousands separator)
    print("  Cleaning prices...")
    df['price_raw'] = df['price_raw'].astype(str).str.replace(' kr.', '', regex=False).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).str.strip()
    df['price'] = pd.to_numeric(df['price_raw'], errors='coerce')
    
    # Clean new_price
    if 'new_price_raw' in df.columns:
        df['new_price_raw'] = df['new_price_raw'].astype(str).str.replace(' kr.', '', regex=False).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).str.strip()
        df['new_price'] = pd.to_numeric(df['new_price_raw'], errors='coerce')
    else:
        df['new_price'] = None
    
    # Clean mileage
    df['mileage_raw'] = df['mileage_raw'].astype(str).str.replace(' km', '', regex=False).str.replace('.', '', regex=False).str.strip()
    df['mileage'] = to_int64(df['mileage_raw'])
    
    # Year
    df['model_year'] = to_int64(df['model_year'])
    df['year'] = df['model_year']
    
    # Other numeric columns
    df['range_km'] = to_int64(df.get('range_km', pd.Series()))
    df['battery_capacity'] = pd.to_numeric(df.get('battery_capacity', pd.Series()), errors='coerce').round(1)
    df['energy_consumption'] = to_int64(df.get('energy_consumption', pd.Series()))
    df['acceleration'] = pd.to_numeric(df.get('acceleration', pd.Series()), errors='coerce').round(1)
    df['top_speed'] = to_int64(df.get('top_speed', pd.Series()))
    df['towing_capacity'] = to_int64(df.get('towing_capacity', pd.Series()))
    df['trunk_size'] = to_int64(df.get('trunk_size', pd.Series()))
    df['weight'] = to_int64(df.get('weight', pd.Series()))
    df['width'] = to_int64(df.get('width', pd.Series()))
    df['length'] = to_int64(df.get('length', pd.Series()))
    df['height'] = to_int64(df.get('height', pd.Series()))
    df['load_capacity'] = to_int64(df.get('load_capacity', pd.Series()))
    df['max_towing_weight'] = to_int64(df.get('max_towing_weight', pd.Series()))
    df['gear_count'] = to_int64(df.get('gear_count', pd.Series()))
    df['cylinders'] = to_int64(df.get('cylinders', pd.Series()))
    df['tank_capacity'] = to_int64(df.get('tank_capacity', pd.Series()))
    df['airbags'] = to_int64(df.get('airbags', pd.Series()))
    df['doors'] = to_int64(df.get('doors', pd.Series()))
    
    # Extract horsepower and torque from power_raw
    if 'power_raw' in df.columns:
        power_split = df['power_raw'].astype(str).str.extract(r'(\d+)\s*hk[/\s]*(\d+)?\s*nm?', expand=True)
        df['horsepower'] = to_int64(power_split[0])
        df['torque_nm'] = to_int64(power_split[1])
    else:
        df['horsepower'] = None
        df['torque_nm'] = None
    
    # Boolean conversions
    for col in ['abs_brakes', 'esp']:
        if col in df.columns:
            df[col] = df[col].map({'True': True, 'False': False, 'Ja': True, 'Nej': False, 
                                    'ja': True, 'nej': False, True: True, False: False})
    
    # Standardize fuel types
    print("  Standardizing fuel types...")
    df['fuel_type'] = df['fuel_type'].apply(standardize_fuel_type)
    
    # Standardize body types
    body_type_mapping = {
        'SUV': 'SUV', 'Suv': 'SUV', 'Cuv': 'SUV', 'CUV': 'SUV',
        'Sedan': 'Sedan',
        'Stationcar': 'Station Wagon', 'Station wagon': 'Station Wagon', 'St.car': 'Station Wagon',
        'Hatchback': 'Hatchback', 'Halvkombi': 'Hatchback', 'Mikro': 'Hatchback',
        'Coupé': 'Coupe', 'Coupe': 'Coupe',
        'Cabriolet': 'Cabriolet', 'Kabrio': 'Cabriolet',
        'Minibus': 'Van', 'Van': 'Van', 'MPV': 'Van', 'Kassevogn': 'Van', 'Wagon': 'Van',
        'Pickup': 'Pickup', 'Pick-up': 'Pickup', '4x4': 'Pickup'
    }
    if 'body_type' in df.columns:
        df['body_type'] = df['body_type'].map(body_type_mapping).fillna(df['body_type'])
    
    # Standardize transmission
    transmission_mapping = {
        'Automatgear': 'Automatic', 'Automatic': 'Automatic', 'Automatisk': 'Automatic',
        'Manuel': 'Manual', 'Manual': 'Manual', 'Manuell': 'Manual'
    }
    if 'transmission' in df.columns:
        df['transmission'] = df['transmission'].map(transmission_mapping).fillna(df['transmission'])
        # Electric cars get Automatic
        electric_mask = (df['fuel_type'] == 'Electricity')
        df.loc[electric_mask & df['transmission'].isna(), 'transmission'] = 'Automatic'
    
    # Standardize drive type
    drive_type_mapping = {
        'Forhjulstræk': 'Front-Wheel Drive', 'Front-wheel drive': 'Front-Wheel Drive', 'FWD': 'Front-Wheel Drive',
        'Baghjulstræk': 'Rear-Wheel Drive', 'Rear-wheel drive': 'Rear-Wheel Drive', 'RWD': 'Rear-Wheel Drive',
        'Firehjulstræk': 'All-Wheel Drive', 'All-wheel drive': 'All-Wheel Drive', 'AWD': 'All-Wheel Drive', '4WD': 'All-Wheel Drive', '4x4': 'All-Wheel Drive'
    }
    if 'drive_type' in df.columns:
        df['drive_type'] = df['drive_type'].map(drive_type_mapping).fillna(df['drive_type'])
    
    # Fix constraints
    if 'doors' in df.columns:
        df.loc[df['doors'].notna() & ~df['doors'].between(2,5), 'doors'] = 4
    if 'seats' not in df.columns or df['seats'].isna().all():
        df['seats'] = 5
    df.loc[df['seats'].notna() & ~df['seats'].between(2,9), 'seats'] = 5
    
    # Add missing columns
    df['location'] = None
    df['dealer_name'] = None
    df['source_url'] = df['url']
    
    if 'engine_size' not in df.columns:
        df['engine_size'] = None
    
    # Filter out invalid rows
    print("  Filtering valid rows...")
    df = df[(df['brand'].notna()) & (df['model'].notna()) & (df['price'].notna()) & (df['price'] > 0)]
    df = df.drop_duplicates(subset=['url'])
    
    print(f"  ✅ Cleaned {len(df)} valid rows")
    return df

def standardize_fuel_type(fuel):
    """Standardize fuel types to 7 categories"""
    if pd.isna(fuel) or fuel == '' or fuel is None:
        return None
    
    fuel_normalized = str(fuel).strip().lower().replace('-', '').replace(' ', '')
    fuel_original = str(fuel).strip().lower()
    
    if fuel_normalized in ['el', 'electricity', 'electric']:
        return 'Electricity'
    if fuel_normalized in ['benzin', 'petrol', 'gasoline'] and 'hybrid' not in fuel_original:
        return 'Petrol'
    if fuel_normalized == 'diesel' and 'hybrid' not in fuel_original:
        return 'Diesel'
    if 'plugin' in fuel_normalized or 'pluginhybrid' in fuel_normalized:
        return 'Plug-in Hybrid - Diesel' if 'diesel' in fuel_original else 'Plug-in Hybrid - Petrol'
    if 'hybrid' in fuel_original:
        return 'Hybrid - Diesel' if 'diesel' in fuel_original else 'Hybrid - Petrol'
    
    return str(fuel).strip()

def import_csv_to_db(csv_path):
    """Import car data from CSV to database"""
    
    # Database connection
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'database': os.getenv('POSTGRES_DB', 'car_prediction'),
        'user': os.getenv('POSTGRES_USER', 'bpr_user'),
        'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
    }
    
    print(f"📖 Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows")
    
    print("\n🧹 Cleaning and transforming data...")
    df = clean_and_transform_data(df)
    
    if len(df) == 0:
        print("❌ No valid rows to import after cleaning")
        return
    
    # Connect to database
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    
    print("\n🗄️  Importing to database using COPY (fast bulk insert)...")
    
    # Use PostgreSQL COPY for fast bulk insert
    # Prepare columns for database (match your schema exactly)
    db_columns = [
        'url', 'brand', 'model', 'variant', 'title', 'description',
        'price', 'new_price', 'model_year', 'year', 'first_registration', 
        'production_date', 'mileage', 'fuel_type', 'transmission', 'gear_count',
        'cylinders', 'horsepower', 'torque_nm', 'acceleration', 'top_speed',
        'range_km', 'battery_capacity', 'energy_consumption', 'home_charging_ac',
        'fast_charging_dc', 'charging_time_dc', 'fuel_consumption', 'co2_emission',
        'euro_norm', 'tank_capacity', 'body_type', 'weight', 'width', 'length',
        'height', 'trunk_size', 'load_capacity', 'towing_capacity', 'max_towing_weight',
        'drive_type', 'abs_brakes', 'esp', 'airbags', 'doors', 'seats',
        'color', 'category', 'equipment', 'periodic_tax', 'engine_size',
        'source_url', 'location', 'dealer_name'
    ]
    
    # Ensure all columns exist in dataframe
    for col in db_columns:
        if col not in df.columns:
            df[col] = None
    
    # Select only the columns we need in the right order
    df_insert = df[db_columns].copy()
    
    # Convert NaN to None for proper NULL handling
    df_insert = df_insert.where(pd.notna(df_insert), None)
    
    # Convert to CSV string in memory
    buffer = StringIO()
    df_insert.to_csv(buffer, index=False, header=False, sep='|', na_rep='\\N')
    buffer.seek(0)
    
    try:
        # Use COPY for fast bulk insert with ON CONFLICT handling
        # First create temp table
        cur.execute("""
            CREATE TEMP TABLE temp_cars (LIKE cars INCLUDING ALL)
        """)
        
        # Copy data to temp table
        cur.copy_expert(
            sql=f"COPY temp_cars ({', '.join(db_columns)}) FROM STDIN WITH (FORMAT CSV, DELIMITER '|', NULL '\\N')",
            file=buffer
        )
        
        # Insert from temp table with conflict handling
        cur.execute(f"""
            INSERT INTO cars ({', '.join(db_columns)})
            SELECT {', '.join(db_columns)}
            FROM temp_cars
            ON CONFLICT (url) DO NOTHING
        """)
        
        inserted = cur.rowcount
        conn.commit()
        
        print(f"\n✅ Import complete!")
        print(f"   Inserted: {inserted} new cars")
        print(f"   Skipped: {len(df) - inserted} duplicates")
        print(f"   Total: {len(df)}")
        
    except Exception as e:
        conn.rollback()
        print(f"\n❌ Bulk insert failed: {e}")
        print("\nFalling back to row-by-row insert...")
        
        # Fallback to row-by-row
        inserted = 0
        skipped = 0
        errors = []
        
        for idx, row in df_insert.iterrows():
            try:
                values = tuple(row[col] for col in db_columns)
                placeholders = ', '.join(['%s'] * len(db_columns))
                
                cur.execute(f"""
                    INSERT INTO cars ({', '.join(db_columns)})
                    VALUES ({placeholders})
                    ON CONFLICT (url) DO NOTHING
                """, values)
                
                conn.commit()
                
                if cur.rowcount > 0:
                    inserted += 1
                else:
                    skipped += 1
                    
                if (idx + 1) % 100 == 0:
                    print(f"  Progress: {idx + 1}/{len(df)} rows processed...")
                    
            except Exception as e:
                conn.rollback()
                error_msg = f"Row {idx + 1}: {str(e)[:100]}"
                errors.append(error_msg)
                if len(errors) <= 10:
                    print(f"⚠️  {error_msg}")
                skipped += 1
                continue
        
        print(f"\n✅ Row-by-row import complete!")
        print(f"   Inserted: {inserted}")
        print(f"   Skipped: {skipped}")
        print(f"   Total: {len(df)}")
        
        if len(errors) > 10:
            print(f"\n⚠️  {len(errors)} total errors (showing first 10 above)")
    
    finally:
        cur.close()
        conn.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python import_csv_to_db.py <csv_file>")
        print("Example: python import_csv_to_db.py bilbasen_scrape/car_details.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)
    
    import_csv_to_db(csv_path)
