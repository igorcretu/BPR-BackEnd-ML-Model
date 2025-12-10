import pandas as pd

df = pd.read_csv('bilbasen_scrape/car_details.csv')

# Find the car, or use first row if not found
matching_rows = df[df['external_id'] == '6660295']
if len(matching_rows) > 0:
    row = matching_rows.iloc[0]
    print('Sample row data for car 6660295 (Kia Rio):')
else:
    row = df.iloc[0]
    print(f'Sample row data for first car (ID: {row.get("external_id")}):')
print('=' * 60)

columns_to_check = [
    'details_fuel_consumption',
    'details_co2_udledning', 
    'details_euro_norm',
    'details_periodic_tax',
    'model_tankkapacitet',
    'model_length_cm',
    'model_width_cm', 
    'model_height_cm',
    'details_number_of_gears',
    'model_cylinders',
    'model_airbags',
    'model_category',
    'details_first_registration',
    'details_production_year',
    'model_load_capacity_kg',
    'model_max_towing_with_brake',
    'details_towing_capacity',
    'details_power_hp_nm',
    'attr_power_hp',
    'details_top_speed',
    'seller_city',
    'seller_zipcode'
]

for col in columns_to_check:
    val = row.get(col, 'NOT FOUND')
    print(f'{col:40s}: {val}')
