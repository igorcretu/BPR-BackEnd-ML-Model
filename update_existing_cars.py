#!/usr/bin/env python3
"""
Update existing cars in database with missing fields from CSV
"""
import os
import sys
import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv
import re

load_dotenv()


def clean_price(price_str):
    """Clean Danish price format: '38.900 kr.' -> 38900"""
    if pd.isna(price_str) or price_str == '':
        return None
    price_str = str(price_str).replace(' kr.', '').replace('.', '').replace(',', '.').strip()
    try:
        return float(price_str)
    except:
        return None


def clean_numeric(value_str, suffix=''):
    """Clean numeric values with optional suffix like 'km', 'hk'"""
    if pd.isna(value_str) or value_str == '' or value_str == '-':
        return None
    value_str = str(value_str).replace(suffix, '').replace('.', '').replace(',', '.').replace('kg', '').replace('cm', '').strip()
    try:
        return int(float(value_str))
    except:
        return None


def clean_float(value_str):
    """Clean float values like '14,0 kWh' or '-'"""
    if pd.isna(value_str) or value_str == '' or value_str == '-':
        return None
    # Remove units and clean
    value_str = str(value_str).replace('kWh', '').replace('kwh', '').replace(',', '.').strip()
    try:
        return float(value_str)
    except:
        return None


def safe_value(val):
    """Convert pandas NA/NaN to Python None, keep '-' as is"""
    if pd.isna(val):
        return None
    if isinstance(val, (np.integer, np.floating)):
        if np.isnan(val):
            return None
        return int(val) if isinstance(val, np.integer) else float(val)
    if val == '' or str(val).lower() == 'nan':
        return None
    # Keep '-' as is, convert everything else to string
    val_str = str(val).strip()
    if val_str == '':
        return None
    return val_str


def extract_torque(power_str):
    """Extract torque from format like '178 hk/230 nm'"""
    if pd.isna(power_str) or power_str == '':
        return None
    match = re.search(r'(\d+)\s*nm', str(power_str).lower())
    if match:
        return int(match.group(1))
    return None


def extract_engine_size(power_str):
    """Extract engine size from power string if present like '1.6 L'"""
    if pd.isna(power_str) or power_str == '':
        return None
    # Look for patterns like "1.6 L" or "2.0L"
    match = re.search(r'(\d+[.,]\d+)\s*[lL]', str(power_str))
    if match:
        return float(match.group(1).replace(',', '.'))
    return None


def update_existing_cars(csv_path):
    """Update existing cars in database with missing fields"""
    
    # Database connection
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'database': os.getenv('POSTGRES_DB', 'car_prediction'),
        'user': os.getenv('POSTGRES_USER', 'bpr_user'),
        'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
    }
    
    print(f"[INFO] Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[OK] Loaded {len(df)} rows")
    
    print("\n[INFO] Connecting to database...")
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    
    updated = 0
    not_found = 0
    errors = []
    
    for idx, row in df.iterrows():
        try:
            external_id = str(row.get('external_id')) if pd.notna(row.get('external_id')) else None
            if not external_id:
                continue
            
            # Check if car exists
            cur.execute("SELECT id FROM cars WHERE external_id = %s", (external_id,))
            result = cur.fetchone()
            
            if not result:
                not_found += 1
                continue
            
            # Extract all the missing fields
            power_str = row.get('details_power_hp_nm')
            torque_nm = extract_torque(power_str)
            engine_size = extract_engine_size(power_str)
            
            # Performance fields
            acceleration_val = row.get('details_acceleration_0_100')
            if pd.isna(acceleration_val) or acceleration_val == '' or acceleration_val == '-':
                acceleration_val = row.get('attr_acceleration_0_100')
            
            gear_count = clean_numeric(row.get('details_number_of_gears'))
            cylinders = clean_numeric(row.get('model_cylinders'))
            
            # Registration fields
            first_registration = safe_value(row.get('details_first_registration'))
            production_date = safe_value(row.get('details_production_year'))
            
            # Battery/EV fields
            energy_consumption = clean_numeric(row.get('details_energy_consumption'))
            home_charging_ac = safe_value(row.get('details_home_charging_ac'))
            fast_charging_dc = safe_value(row.get('details_fast_charging_dc'))
            charging_time_dc = safe_value(row.get('details_charging_time_dc_10_80_pct'))
            
            # Fuel consumption and emissions
            fuel_consumption = safe_value(row.get('details_fuel_consumption'))
            co2_emission = safe_value(row.get('details_co2_udledning'))
            euro_norm = safe_value(row.get('details_euro_norm'))
            tank_capacity = clean_numeric(row.get('model_tankkapacitet'))
            
            # Other fields
            category = safe_value(row.get('model_category'))
            periodic_tax = safe_value(row.get('details_periodic_tax'))
            
            # Booleans
            airbags = clean_numeric(row.get('model_airbags'))
            
            # Dimensions and capacity
            width = clean_numeric(row.get('model_width_cm'))
            length = clean_numeric(row.get('model_length_cm'))
            height = clean_numeric(row.get('model_height_cm'))
            load_capacity = clean_numeric(row.get('model_load_capacity_kg'))
            
            # Towing capacity
            towing_val = row.get('details_towing_capacity')
            if pd.isna(towing_val) or towing_val == '' or towing_val == '-':
                towing_val = row.get('model_max_towing_with_brake')
            towing_capacity = clean_numeric(towing_val)
            max_towing_weight = clean_numeric(row.get('model_max_towing_with_brake'))
            
            # Location
            seller_city = safe_value(row.get('seller_city'))
            seller_zipcode = safe_value(row.get('seller_zipcode'))
            if seller_city and seller_zipcode:
                location = f"{seller_city}, {seller_zipcode}"
            elif seller_city:
                location = seller_city
            else:
                location = None
            
            # Update the car
            cur.execute("""
                UPDATE cars SET
                    torque_nm = %s,
                    engine_size = %s,
                    gear_count = %s,
                    cylinders = %s,
                    first_registration = %s,
                    production_date = %s,
                    energy_consumption = %s,
                    home_charging_ac = %s,
                    fast_charging_dc = %s,
                    charging_time_dc = %s,
                    fuel_consumption = %s,
                    co2_emission = %s,
                    euro_norm = %s,
                    tank_capacity = %s,
                    category = %s,
                    periodic_tax = %s,
                    airbags = %s,
                    width = %s,
                    length = %s,
                    height = %s,
                    load_capacity = %s,
                    towing_capacity = %s,
                    max_towing_weight = %s,
                    location = %s
                WHERE external_id = %s
            """, (
                torque_nm, engine_size, gear_count, cylinders,
                first_registration, production_date,
                energy_consumption, home_charging_ac, fast_charging_dc, charging_time_dc,
                fuel_consumption, co2_emission, euro_norm, tank_capacity,
                category, periodic_tax, airbags,
                width, length, height, load_capacity,
                towing_capacity, max_towing_weight, location,
                external_id
            ))
            
            if cur.rowcount > 0:
                updated += 1
            
            conn.commit()
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(df)} rows... (updated: {updated})")
                
        except Exception as e:
            conn.rollback()
            error_msg = f"Row {idx + 1} (ID: {external_id}): {str(e)[:80]}"
            errors.append(error_msg)
            if len(errors) <= 10:
                print(f"[WARN] {error_msg}")
    
    cur.close()
    conn.close()
    
    print(f"\n[OK] Update complete!")
    print(f"   Updated: {updated} cars")
    print(f"   Not found: {not_found}")
    print(f"   Total processed: {len(df)}")
    
    if len(errors) > 10:
        print(f"\n[WARN] {len(errors)} total errors (showing first 10 above)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python update_existing_cars.py <csv_file>")
        print("Example: python update_existing_cars.py bilbasen_scrape/car_details.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)
    
    update_existing_cars(csv_path)
