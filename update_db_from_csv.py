#!/usr/bin/env python3
"""
Re-import CSV data to update missing fields in existing cars
"""
import psycopg2
import pandas as pd
import sys
import os
from import_csv_to_db import (
    clean_price, clean_numeric, clean_float, clean_boolean, safe_value,
    standardize_fuel_type, standardize_transmission, standardize_body_type,
    standardize_drive_type, extract_horsepower, extract_torque, extract_engine_size
)

DB_PARAMS = {
    'dbname': 'car_prediction',
    'user': 'bpr_user',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432'
}

def update_existing_cars(csv_path):
    """Update existing cars with missing data from CSV"""
    
    print(f"[INFO] Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[OK] Loaded {len(df)} rows")
    
    print("\n[INFO] Connecting to database...")
    conn = psycopg2.connect(**DB_PARAMS)
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
            
            # Extract all fields
            fuel_type = standardize_fuel_type(row.get('details_fuel_type'))
            
            # Power fields
            power_str = row.get('details_power_hp_nm')
            horsepower = extract_horsepower(power_str)
            if not horsepower:
                horsepower = clean_numeric(row.get('attr_power_hp'))
            torque_nm = extract_torque(power_str)
            engine_size = extract_engine_size(power_str)
            
            # Performance
            acceleration_val = row.get('details_acceleration_0_100')
            if pd.isna(acceleration_val) or acceleration_val == '' or acceleration_val == '-':
                acceleration_val = row.get('attr_acceleration_0_100')
            acceleration = clean_float(acceleration_val)
            
            top_speed_val = row.get('details_top_speed')
            if pd.isna(top_speed_val) or top_speed_val == '' or top_speed_val == '-':
                top_speed_val = row.get('attr_top_speed_kmh')
            top_speed = clean_numeric(top_speed_val)
            
            # Body fields
            gear_count = clean_numeric(row.get('details_number_of_gears'))
            cylinders = clean_numeric(row.get('model_cylinders'))
            
            # Registration
            first_registration = safe_value(row.get('details_first_registration'))
            production_date = safe_value(row.get('details_production_year'))
            
            # EV fields
            energy_consumption = clean_numeric(row.get('details_energy_consumption'))
            home_charging_ac = safe_value(row.get('details_home_charging_ac'))
            fast_charging_dc = safe_value(row.get('details_fast_charging_dc'))
            charging_time_dc = safe_value(row.get('details_charging_time_dc_10_80_pct'))
            
            # Fuel/emissions
            fuel_consumption = safe_value(row.get('details_fuel_consumption'))
            co2_emission = safe_value(row.get('details_co2_udledning'))
            euro_norm = safe_value(row.get('details_euro_norm'))
            tank_capacity = clean_numeric(row.get('model_tankkapacitet'))
            
            # Other
            category = safe_value(row.get('model_category'))
            periodic_tax = safe_value(row.get('details_periodic_tax'))
            airbags = clean_numeric(row.get('model_airbags'))
            
            # Dimensions
            width = clean_numeric(row.get('model_width_cm'))
            length = clean_numeric(row.get('model_length_cm'))
            height = clean_numeric(row.get('model_height_cm'))
            load_capacity = clean_numeric(row.get('model_load_capacity_kg'))
            
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
                    location = %s
                WHERE external_id = %s
            """, (
                torque_nm, engine_size, gear_count, cylinders,
                first_registration, production_date,
                energy_consumption, home_charging_ac, fast_charging_dc, charging_time_dc,
                fuel_consumption, co2_emission, euro_norm, tank_capacity,
                category, periodic_tax, airbags,
                width, length, height, load_capacity,
                location,
                external_id
            ))
            
            if cur.rowcount > 0:
                updated += 1
            
            if (idx + 1) % 500 == 0:
                conn.commit()
                print(f"  Updated {idx + 1}/{len(df)} rows...")
            
        except Exception as e:
            error_msg = f"Row {idx + 1}: {str(e)[:80]}"
            errors.append(error_msg)
            if len(errors) <= 10:
                print(f"[WARN] {error_msg}")
    
    conn.commit()
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
        print("Usage: python update_db_from_csv.py <csv_file>")
        print("Example: python update_db_from_csv.py bilbasen_scrape/car_details.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)
    
    update_existing_cars(csv_path)
