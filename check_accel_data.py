import pandas as pd

df = pd.read_csv('bilbasen_scrape/car_details.csv')

# Find cars with acceleration data
accel_data = df[df['details_acceleration_0_100'].notna() & (df['details_acceleration_0_100'] != '-')].head(5)

print('Cars with acceleration data:')
print('=' * 80)
for idx, row in accel_data.iterrows():
    year = row.get('details_model_year', 'N/A')
    print(f"\n{row['brand']} {row['model']} ({year})")
    print(f"  Acceleration: {row['details_acceleration_0_100']}")
    print(f"  Top Speed: {row['details_top_speed']}")
    print(f"  Power: {row['details_power_hp_nm']}")
