import pandas as pd
import re

def clean_float(value_str):
    """Clean float values like '14,0 kWh' or '10.7 sec' or '-'"""
    if pd.isna(value_str) or value_str == '' or value_str == '-':
        return None
    # Remove units and clean
    value_str = str(value_str).replace('kWh', '').replace('kwh', '').replace('sec', '').replace('s', '').replace(',', '.').strip()
    try:
        return float(value_str)
    except:
        return None

def clean_numeric(value_str):
    """Clean numeric values with optional suffix like 'km', 'hk', 'km/h', 'km/t'"""
    if pd.isna(value_str) or value_str == '' or value_str == '-':
        return None
    # Remove Danish and English units
    value_str = str(value_str).replace('km/t', '').replace('km/h', '').replace('kmh', '').replace('kg', '').replace('cm', '').strip()
    # Remove thousand separators (.) then convert comma to decimal point
    value_str = value_str.replace('.', '').replace(',', '.')
    try:
        return int(float(value_str))
    except:
        return None

# Test with actual data
df = pd.read_csv('bilbasen_scrape/car_details.csv', nrows=5)

print("Testing data extraction for first 5 cars:")
print("=" * 80)

for idx, row in df.iterrows():
    print(f"\nCar {idx + 1}: {row.get('brand')} {row.get('model')}")
    
    # Acceleration
    accel_detail = row.get('details_acceleration_0_100')
    accel_attr = row.get('attr_acceleration_0_100')
    accel_val = accel_detail if pd.notna(accel_detail) and str(accel_detail) != '-' else accel_attr
    accel_cleaned = clean_float(accel_val)
    print(f"  Acceleration: detail='{accel_detail}' attr='{accel_attr}' -> {accel_cleaned}")
    
    # Top Speed
    speed_detail = row.get('details_top_speed')
    speed_attr = row.get('attr_top_speed_kmh')
    speed_val = speed_detail if pd.notna(speed_detail) and str(speed_detail) != '-' else speed_attr
    speed_cleaned = clean_numeric(speed_val)
    print(f"  Top Speed: detail='{speed_detail}' attr='{speed_attr}' -> {speed_cleaned}")
    
    # Horsepower
    power_str = row.get('details_power_hp_nm')
    print(f"  Power String: '{power_str}'")
