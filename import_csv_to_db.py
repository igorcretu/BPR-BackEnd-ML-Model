#!/usr/bin/env python3
"""
Import scraped CSV data into PostgreSQL database with full data cleaning
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
    value_str = str(value_str).replace(suffix, '').replace('.', '').replace(',', '.').strip()
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


def standardize_fuel_type(fuel):
    """Standardize fuel types to 7 categories"""
    if pd.isna(fuel) or fuel == '':
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
    
    return 'Petrol'  # Default


def standardize_transmission(trans, fuel_type):
    """Standardize transmission types"""
    if pd.isna(trans) or trans == '' or trans == '-':
        # Electric cars get automatic by default
        return 'Automatic' if fuel_type == 'Electricity' else None
    
    trans_lower = str(trans).strip().lower()
    if trans_lower in ['automatgear', 'automatic', 'automatisk', 'auto', 'automat', 'a']:
        return 'Automatic'
    if trans_lower in ['manuel', 'manual', 'manuell', 'manuelt', 'm']:
        return 'Manual'
    
    return 'Manual'  # Default


def standardize_body_type(body):
    """Standardize body types"""
    if pd.isna(body) or body == '':
        return None
    
    mapping = {
        'suv': 'SUV', 'cuv': 'SUV',
        'sedan': 'Sedan',
        'stationcar': 'Station Wagon', 'station wagon': 'Station Wagon', 'st.car': 'Station Wagon',
        'hatchback': 'Hatchback', 'halvkombi': 'Hatchback', 'mikro': 'Hatchback',
        'coupé': 'Coupe', 'coupe': 'Coupe',
        'cabriolet': 'Cabriolet', 'kabrio': 'Cabriolet',
        'minibus': 'Van', 'van': 'Van', 'mpv': 'Van', 'kassevogn': 'Van', 'wagon': 'Van',
        'pickup': 'Pickup', 'pick-up': 'Pickup', '4x4': 'Pickup'
    }
    
    body_lower = str(body).strip().lower()
    return mapping.get(body_lower, str(body).strip())


def standardize_drive_type(drive):
    """Standardize drive types"""
    if pd.isna(drive) or drive == '':
        return None
    
    mapping = {
        'forhjulstræk': 'Front-Wheel Drive', 'front-wheel drive': 'Front-Wheel Drive', 'fwd': 'Front-Wheel Drive',
        'baghjulstræk': 'Rear-Wheel Drive', 'rear-wheel drive': 'Rear-Wheel Drive', 'rwd': 'Rear-Wheel Drive',
        'firehjulstræk': 'All-Wheel Drive', 'all-wheel drive': 'All-Wheel Drive', 
        'awd': 'All-Wheel Drive', '4wd': 'All-Wheel Drive', '4x4': 'All-Wheel Drive'
    }
    
    drive_lower = str(drive).strip().lower()
    return mapping.get(drive_lower, str(drive).strip())


def extract_horsepower(power_str):
    """Extract horsepower from format like '50 hk/-' or '178 hk/230 nm'"""
    if pd.isna(power_str) or power_str == '':
        return None
    match = re.search(r'(\d+)\s*hk', str(power_str))
    if match:
        return int(match.group(1))
    return None


def clean_boolean(value):
    """Convert various boolean representations"""
    if pd.isna(value) or value == '':
        return None
    value_lower = str(value).strip().lower()
    if value_lower in ['ja', 'yes', 'true', '1']:
        return True
    if value_lower in ['nej', 'no', 'false', '0']:
        return False
    return None


def safe_value(val):
    """Convert pandas NA/NaN to Python None"""
    if pd.isna(val):
        return None
    if isinstance(val, (np.integer, np.floating)):
        if np.isnan(val):
            return None
        return int(val) if isinstance(val, np.integer) else float(val)
    if val == '' or str(val).lower() == 'nan':
        return None
    # Convert to string for text fields
    return str(val) if val is not None else None


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
    
    print("\n🧹 Processing data...")
    
    # Process each row
    cars_data = []
    skipped = 0
    
    for idx, row in df.iterrows():
        try:
            # Extract and clean basic info
            url = safe_value(row.get('url'))
            external_id = str(row.get('external_id')) if pd.notna(row.get('external_id')) else None
            brand = safe_value(row.get('brand'))
            model = safe_value(row.get('model'))
            
            # Skip if missing essential fields
            if not url or not brand or not model:
                skipped += 1
                continue
            
            # Clean price
            price = clean_price(row.get('price'))
            if not price or price <= 0:
                skipped += 1
                continue
            
            # Extract all other fields
            variant = safe_value(row.get('variant'))
            title = safe_value(row.get('title'))
            description = safe_value(row.get('description'))
            
            # Standardize fuel type first (needed for transmission)
            fuel_type = standardize_fuel_type(row.get('details_fuel_type'))
            
            # Other details
            new_price = clean_price(row.get('model_new_price'))
            model_year = clean_numeric(row.get('details_model_year'))
            year = model_year
            mileage = clean_numeric(row.get('details_mileage_km'), ' km')
            
            transmission = standardize_transmission(row.get('details_geartype'), fuel_type)
            body_type = standardize_body_type(row.get('model_body_type'))
            drive_type = standardize_drive_type(row.get('model_drive_type'))
            
            horsepower = extract_horsepower(row.get('details_power_hp_nm'))
            if not horsepower:
                horsepower = clean_numeric(row.get('attr_power_hp'))
            
            # Numeric fields
            acceleration = clean_float(row.get('attr_acceleration_0_100'))
            top_speed = clean_numeric(row.get('attr_top_speed_kmh'))
            doors = clean_numeric(row.get('model_doors'))
            if doors and (doors < 2 or doors > 5):
                doors = 4
            
            # Battery/EV fields
            range_km = clean_numeric(row.get('details_range_km'))
            battery_capacity = clean_float(row.get('details_battery_capacity_kwh'))
            
            # Other fields
            color = safe_value(row.get('attr_color'))
            equipment = safe_value(row.get('equipment'))
            
            # Image fields
            image_filename = safe_value(row.get('image_filename'))
            image_path = f"images/{image_filename}" if image_filename else None
            
            # Booleans
            abs_brakes = clean_boolean(row.get('model_abs_brakes'))
            esp = clean_boolean(row.get('model_esp'))
            
            # Dimensions
            weight = clean_numeric(row.get('model_weight_kg'))
            trunk_size = clean_numeric(row.get('model_trunk_size'))
            
            # Build database row
            car_data = {
                'external_id': external_id or url,  # Use URL as fallback
                'url': url,
                'brand': brand,
                'model': model,
                'variant': variant,
                'title': title,
                'description': description,
                'price': price,
                'new_price': new_price,
                'model_year': model_year,
                'year': year,
                'mileage': mileage,
                'fuel_type': fuel_type,
                'transmission': transmission,
                'body_type': body_type,
                'horsepower': horsepower,
                'acceleration': acceleration,
                'top_speed': top_speed,
                'drive_type': drive_type,
                'doors': doors,
                'seats': 5,  # Default
                'color': color,
                'equipment': equipment,
                'abs_brakes': abs_brakes,
                'esp': esp,
                'weight': weight,
                'trunk_size': trunk_size,
                'range_km': range_km,
                'battery_capacity': battery_capacity,
                'image_path': image_path,
                'source_url': url,
                'location': None,
                'dealer_name': safe_value(row.get('seller_name'))
            }
            
            cars_data.append(car_data)
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(df)} rows...")
                
        except Exception as e:
            print(f"⚠️  Error processing row {idx + 1}: {e}")
            skipped += 1
    
    print(f"✅ Processed {len(cars_data)} valid rows (skipped {skipped})")
    
    if len(cars_data) == 0:
        print("❌ No valid rows to import")
        return
    
    # Connect to database and insert
    print("\n🗄️  Importing to database...")
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    
    inserted = 0
    duplicates = 0
    errors = []
    
    for idx, car in enumerate(cars_data):
        try:
            # Check if already exists
            cur.execute("SELECT id FROM cars WHERE external_id = %s", (car['external_id'],))
            if cur.fetchone():
                duplicates += 1
                if (idx + 1) % 500 == 0:
                    print(f"  Imported {idx + 1}/{len(cars_data)} rows...")
                continue
            
            cur.execute("""
                INSERT INTO cars (
                    external_id, url, brand, model, variant, title, description,
                    price, new_price, model_year, year, mileage,
                    fuel_type, transmission, body_type, horsepower,
                    acceleration, top_speed, drive_type, doors, seats,
                    color, equipment, abs_brakes, esp, weight, trunk_size,
                    range_km, battery_capacity, image_path,
                    source_url, location, dealer_name
                ) VALUES (
                    %(external_id)s, %(url)s, %(brand)s, %(model)s, %(variant)s,
                    %(title)s, %(description)s, %(price)s, %(new_price)s,
                    %(model_year)s, %(year)s, %(mileage)s, %(fuel_type)s,
                    %(transmission)s, %(body_type)s, %(horsepower)s,
                    %(acceleration)s, %(top_speed)s, %(drive_type)s,
                    %(doors)s, %(seats)s, %(color)s, %(equipment)s,
                    %(abs_brakes)s, %(esp)s, %(weight)s, %(trunk_size)s,
                    %(range_km)s, %(battery_capacity)s, %(image_path)s,
                    %(source_url)s, %(location)s, %(dealer_name)s
                )
            """, car)
            
            conn.commit()
            
            if cur.rowcount > 0:
                inserted += 1
            else:
                duplicates += 1
            
            if (idx + 1) % 500 == 0:
                print(f"  Imported {idx + 1}/{len(cars_data)} rows...")
                
        except Exception as e:
            conn.rollback()
            error_msg = f"Row {idx + 1} ({car.get('brand')} {car.get('model')}): {str(e)[:80]}"
            errors.append(error_msg)
            if len(errors) <= 10:
                print(f"⚠️  {error_msg}")
    
    cur.close()
    conn.close()
    
    print(f"\n✅ Import complete!")
    print(f"   Inserted: {inserted} new cars")
    print(f"   Duplicates: {duplicates}")
    print(f"   Total processed: {len(cars_data)}")
    
    if len(errors) > 10:
        print(f"\n⚠️  {len(errors)} total errors (showing first 10 above)")


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
