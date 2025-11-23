# CSV to Database Column Mapping

## All 47 CSV Columns from book1.csv  Database Schema

### Core Identification (7 columns)
url  url
brand  brand
model  model
variant  variant
title  title
description  description
price  price (cleaned from kr. format)

### Pricing (1 column)
model_new_price  new_price

### Year/Date Information (3 columns)
details_model_year  model_year, year
details_first_registration  first_registration
details_production_year  production_date

### Basic Specs (4 columns)
details_mileage_km  mileage (cleaned from km format)
details_fuel_type  fuel_type
details_transmission_type  transmission
details_number_of_gears  gear_count

### Engine (1 column)
model_cylinders  cylinders

### Power & Performance (4 columns)
details_power_hp_nm  horsepower, torque_nm (extracted from 'XXX hk/YYY nm')
details_acceleration_0_100  acceleration
details_top_speed  top_speed

### Electric/Hybrid Specs (6 columns)
details_range_km  range_km
details_battery_capacity_kwh  battery_capacity
details_energy_consumption  energy_consumption
details_home_charging_ac  home_charging_ac
details_fast_charging_dc  fast_charging_dc
details_charging_time_dc_10_80_pct  charging_time_dc

### Fuel Consumption & Emissions (4 columns)
details_fuel_consumption  fuel_consumption
details_co2_emission  co2_emission
details_euro_norm  euro_norm
model_tank_capacity_l  tank_capacity

### Physical Dimensions (6 columns)
model_body_type  body_type
model_weight_kg  weight
model_width_cm  width
model_length_cm  length
model_height_cm  height
model_trunk_size  trunk_size

### Capacity & Towing (3 columns)
model_load_capacity_kg  load_capacity
details_towing_capacity  towing_capacity
model_max_towing_with_brake  max_towing_weight

### Drivetrain & Safety (5 columns)
model_drive_type  drive_type
model_abs_brakes  abs_brakes (boolean)
model_esp  esp (boolean)
model_airbags  airbags

### Configuration (2 columns)
model_doors  doors
(seats added as default 5)

### Appearance & Features (4 columns)
details_color  color
model_category  category
equipment  equipment
details_periodic_tax  periodic_tax

### Additional Derived Columns
- engine_size: extracted from variant
- source_url: copy of url
- location: NULL
- dealer_name: NULL
- listing_date: CURRENT_TIMESTAMP
- created_at: CURRENT_TIMESTAMP
- updated_at: CURRENT_TIMESTAMP

## Summary
 All 47 CSV columns mapped to database
 Schema organized by functional sections
 Data cleaning and type conversions applied
 Upload notebook updated to match schema exactly
