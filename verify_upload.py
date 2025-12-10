import pandas as pd
import psycopg2

# Read CSV
df = pd.read_csv('bilbasen_scrape/car_details.csv')

# Get cars with different fuel types
fuel_types = df['details_fuel_type'].dropna().unique()[:5]
sample_cars = []

for fuel in fuel_types:
    car = df[df['details_fuel_type'] == fuel].iloc[0]
    sample_cars.append({
        'external_id': str(car['external_id']),
        'brand': car['brand'],
        'model': car['model'],
        'fuel': car['details_fuel_type']
    })

print("Selected sample cars:")
print("=" * 80)
for car in sample_cars:
    print(f"  {car['brand']} {car['model']} ({car['fuel']}) - ID: {car['external_id']}")

# Connect to database (try different credentials)
try:
    conn = psycopg2.connect('postgresql://postgres:postgres@localhost:5432/car_price_db')
except:
    try:
        conn = psycopg2.connect('postgresql://bpr_user:postgres@localhost:5432/car_prediction')
    except:
        conn = psycopg2.connect('postgresql://bpr_user:postgres@localhost:5432/car_price_db')
cur = conn.cursor()

print("\n\nChecking database values:")
print("=" * 80)

for car in sample_cars:
    external_id = car['external_id']
    
    # Get CSV data
    csv_row = df[df['external_id'] == int(external_id)].iloc[0]
    
    # Get DB data
    cur.execute("""
        SELECT 
            fuel_consumption, co2_emission, euro_norm, periodic_tax, tank_capacity,
            gear_count, cylinders, torque_nm, engine_size, acceleration, top_speed,
            first_registration, production_date, category, airbags,
            width, length, height, load_capacity, towing_capacity, max_towing_weight,
            location, energy_consumption, home_charging_ac, fast_charging_dc, charging_time_dc,
            range_km, battery_capacity
        FROM cars 
        WHERE external_id = %s
    """, (external_id,))
    
    db_row = cur.fetchone()
    
    print(f"\n{car['brand']} {car['model']} ({car['fuel']}) - ID: {external_id}")
    print("-" * 80)
    
    fields = [
        ('fuel_consumption', 'details_fuel_consumption', db_row[0]),
        ('co2_emission', 'details_co2_udledning', db_row[1]),
        ('euro_norm', 'details_euro_norm', db_row[2]),
        ('periodic_tax', 'details_periodic_tax', db_row[3]),
        ('tank_capacity', 'model_tankkapacitet', db_row[4]),
        ('gear_count', 'details_number_of_gears', db_row[5]),
        ('cylinders', 'model_cylinders', db_row[6]),
        ('torque_nm', 'details_power_hp_nm', db_row[7]),
        ('engine_size', 'details_power_hp_nm', db_row[8]),
        ('acceleration', 'details_acceleration_0_100', db_row[9]),
        ('top_speed', 'details_top_speed', db_row[10]),
        ('first_registration', 'details_first_registration', db_row[11]),
        ('production_date', 'details_production_year', db_row[12]),
        ('category', 'model_category', db_row[13]),
        ('airbags', 'model_airbags', db_row[14]),
        ('width', 'model_width_cm', db_row[15]),
        ('length', 'model_length_cm', db_row[16]),
        ('height', 'model_height_cm', db_row[17]),
        ('load_capacity', 'model_load_capacity_kg', db_row[18]),
        ('towing_capacity', 'details_towing_capacity', db_row[19]),
        ('max_towing_weight', 'model_max_towing_with_brake', db_row[20]),
        ('location', 'seller_city', db_row[21]),
        ('energy_consumption', 'details_energy_consumption', db_row[22]),
        ('home_charging_ac', 'details_home_charging_ac', db_row[23]),
        ('fast_charging_dc', 'details_fast_charging_dc', db_row[24]),
        ('charging_time_dc', 'details_charging_time_dc_10_80_pct', db_row[25]),
        ('range_km', 'details_range_km', db_row[26]),
        ('battery_capacity', 'details_battery_capacity_kwh', db_row[27]),
    ]
    
    for db_field, csv_field, db_value in fields:
        csv_value = csv_row.get(csv_field, 'NOT FOUND')
        
        # Status
        if db_value is not None and str(db_value).strip():
            status = "✓"
        elif csv_value and str(csv_value) not in ['nan', '-', '']:
            status = "✗"
        else:
            status = " "
        
        print(f"  {status} {db_field:20s}: CSV='{csv_value}' -> DB='{db_value}'")

cur.close()
conn.close()

print("\n" + "=" * 80)
print("Legend: ✓ = Uploaded correctly, ✗ = Missing in DB but exists in CSV,   = Not in CSV")
