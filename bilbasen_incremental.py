#!/usr/bin/env python3
"""
Bilbasen Incremental Scraper - Daily Updates
Scrapes only NEW listings since the last run and imports directly to PostgreSQL.

Usage:
    # Run daily via cron
    0 3 * * * cd /home/pi/bilbasen && python3 bilbasen_incremental.py >> logs/incremental.log 2>&1

    # Or manually
    python3 bilbasen_incremental.py

    # Test mode (limit to 10 new listings)
    python3 bilbasen_incremental.py --test

    # Dry run (scrape but don't insert to DB)
    python3 bilbasen_incremental.py --dry-run
"""

import requests
import json
import re
import time
import os
import sys
import logging
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import random
import argparse
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Base URL for newest listings (sorted by date descending)
    'BASE_URL': 'https://www.bilbasen.dk/brugt/bil',
    'SEARCH_PARAMS': {
        'includeengroscvr': 'true',
        'includeleasing': 'false',
        'sortby': 'date',
        'sortorder': 'desc'
    },
    
    'ITEMS_PER_PAGE': 30,
    'MAX_PAGES': 100,  # Safety limit
    
    # Delays (be respectful)
    'DELAY_BETWEEN_REQUESTS': (1.0, 2.5),
    'DELAY_BETWEEN_DETAILS': (1.0, 2.0),
    
    # Retry settings
    'MAX_RETRIES': 3,
    'RETRY_DELAY_BASE': 5,
    
    # Output
    'LOG_DIR': 'logs',
    'IMAGES_DIR': 'images',
}

# Danish to English translation map
DANISH_TO_ENGLISH = {
    'modelår': 'model_year',
    '1. registrering': 'first_registration',
    '1_registrering': 'first_registration',
    'kilometertal': 'mileage_km',
    'drivmiddel': 'fuel_type',
    'rækkevidde': 'range_km',
    'batterikapacitet': 'battery_capacity_kwh',
    'energiforbrug': 'energy_consumption',
    'hjemmeopladning ac': 'home_charging_ac',
    'hjemmeopladning_ac': 'home_charging_ac',
    'hurtig opladning dc': 'fast_charging_dc',
    'hurtig_opladning_dc': 'fast_charging_dc',
    'opladningstid dc 10-80%': 'charging_time_dc_10_80_pct',
    'opladningstid_dc_1080': 'charging_time_dc_10_80_pct',
    'periodisk afgift': 'periodic_tax',
    'periodisk_afgift': 'periodic_tax',
    'ydelse': 'power_hp_nm',
    'acceleration': 'acceleration_0_100',
    'tophastighed': 'top_speed',
    'trækvægt': 'towing_capacity',
    'farve': 'color',
    'brændstofforbrug': 'fuel_consumption',
    'co2-udledning': 'co2_emission',
    'co2udledning': 'co2_emission',
    'euronorm': 'euro_norm',
    'gearkasse': 'transmission_type',
    'gear': 'number_of_gears',
    'antal gear': 'number_of_gears',
    'produceret': 'production_year',
    'nypris': 'new_price',
    'kategori': 'category',
    'type': 'body_type',
    'bagagerumsstørrelse': 'trunk_size',
    'vægt': 'weight_kg',
    'bredde': 'width_cm',
    'længde': 'length_cm',
    'højde': 'height_cm',
    'lasteevne': 'load_capacity_kg',
    'max. trækvægt m/bremse': 'max_towing_with_brake',
    'max_trækvægt_mbremse': 'max_towing_with_brake',
    'trækhjul': 'drive_type',
    'abs-bremser': 'abs_brakes',
    'absbremser': 'abs_brakes',
    'esp': 'esp',
    'airbags': 'airbags',
    'døre': 'doors',
    'cylindre': 'cylinders',
    'tankstørrelse': 'tank_capacity_l',
    'motorstørrelse': 'engine_size',
}

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    """Setup logging to both file and console."""
    os.makedirs(CONFIG['LOG_DIR'], exist_ok=True)
    
    log_file = os.path.join(CONFIG['LOG_DIR'], f"incremental_{datetime.now().strftime('%Y%m%d')}.log")
    
    logger = logging.getLogger('bilbasen_incremental')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', 
                                     datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# ============================================================================
# DATA CLEANING FUNCTIONS (from import_csv_to_db.py)
# ============================================================================

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
    """Clean numeric values with optional suffix"""
    if pd.isna(value_str) or value_str == '' or value_str == '-':
        return None
    value_str = str(value_str).replace(suffix, '').replace('km/t', '').replace('km/h', '').replace('kmh', '').replace('kg', '').replace('cm', '').strip()
    value_str = value_str.replace('.', '').replace(',', '.')
    try:
        return int(float(value_str))
    except:
        return None

def clean_float(value_str):
    """Clean float values"""
    if pd.isna(value_str) or value_str == '' or value_str == '-':
        return None
    value_str = str(value_str).replace('kWh', '').replace('kwh', '').replace('sek.', '').replace('sek', '').replace('sec', '').replace(',', '.').strip()
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
    
    return 'Petrol'

def standardize_transmission(trans, fuel_type):
    """Standardize transmission types"""
    if pd.isna(trans) or trans == '' or trans == '-':
        return 'Automatic' if fuel_type == 'Electricity' else None
    
    trans_lower = str(trans).strip().lower()
    if trans_lower in ['automatgear', 'automatic', 'automatisk', 'auto', 'automat', 'a']:
        return 'Automatic'
    if trans_lower in ['manuel', 'manual', 'manuell', 'manuelt', 'm']:
        return 'Manual'
    
    return 'Manual'

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

def extract_torque(power_str):
    """Extract torque from format like '178 hk/230 nm'"""
    if pd.isna(power_str) or power_str == '':
        return None
    match = re.search(r'(\d+)\s*nm', str(power_str).lower())
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
    val_str = str(val).strip()
    if val_str == '':
        return None
    return val_str

def translate_key(danish_key: str) -> str:
    """Translate Danish key to English."""
    normalized = danish_key.lower().strip()
    normalized = re.sub(r'[^a-zæøå0-9\s_-]', '', normalized)
    normalized = normalized.replace(' ', '_').replace('-', '_')
    
    if normalized in DANISH_TO_ENGLISH:
        return DANISH_TO_ENGLISH[normalized]
    if danish_key.lower() in DANISH_TO_ENGLISH:
        return DANISH_TO_ENGLISH[danish_key.lower()]
    return normalized

# ============================================================================
# SCRAPER CLASS
# ============================================================================

class IncrementalScraper:
    def __init__(self, logger: logging.Logger, dry_run: bool = False):
        self.logger = logger
        self.dry_run = dry_run
        self.session = requests.Session()
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'car_prediction'),
            'user': os.getenv('POSTGRES_USER', 'bpr_user'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
        }
    
    def get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            'User-Agent': 'Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'da-DK,da;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
    
    def get_db_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.db_config)
    
    def get_known_external_ids(self) -> set:
        """Get all external_ids currently in the database."""
        self.logger.info("Fetching known external_ids from database...")
        
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT external_id FROM cars")
            ids = {str(row[0]) for row in cur.fetchall()}
            cur.close()
            conn.close()
            self.logger.info(f"Found {len(ids)} existing cars in database")
            return ids
        except Exception as e:
            self.logger.error(f"Database error: {e}")
            return set()
    
    def fetch_page(self, url: str) -> Optional[str]:
        """Fetch a page with retry logic."""
        for attempt in range(CONFIG['MAX_RETRIES']):
            try:
                response = self.session.get(url, headers=self.get_headers(), timeout=30)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                delay = CONFIG['RETRY_DELAY_BASE'] * (attempt + 1)
                self.logger.warning(f"Attempt {attempt + 1}/{CONFIG['MAX_RETRIES']} failed: {e}")
                if attempt < CONFIG['MAX_RETRIES'] - 1:
                    time.sleep(delay)
        return None
    
    def build_search_url(self, page: int = 1) -> str:
        """Build search URL sorted by newest."""
        params = CONFIG['SEARCH_PARAMS'].copy()
        if page > 1:
            params['page'] = page
        query_string = '&'.join(f"{k}={v}" for k, v in params.items())
        return f"{CONFIG['BASE_URL']}?{query_string}"
    
    def extract_listings_from_html(self, html_content: str) -> List[Dict]:
        """Extract listings from search page HTML."""
        listings = []
        
        pattern = r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>'
        match = re.search(pattern, html_content, re.DOTALL)
        
        if not match:
            return listings
        
        try:
            data = json.loads(match.group(1))
            queries = data.get('props', {}).get('pageProps', {}).get('dehydratedState', {}).get('queries', [])
            
            for query in queries:
                state_data = query.get('state', {}).get('data', {})
                if 'listings' in state_data:
                    raw_listings = state_data['listings']
                    
                    for item in raw_listings:
                        uri = item.get('uri', '')
                        image_url = ''
                        for media in item.get('media', []):
                            if media.get('mediaType') == 'Picture':
                                image_url = media.get('url', '')
                                break
                        if uri:
                            external_id = uri.rstrip('/').split('/')[-1]
                            listings.append({
                                'uri': uri,
                                'image_url': image_url,
                                'external_id': external_id
                            })
                    break
        except Exception as e:
            self.logger.error(f"Error extracting listings: {e}")
        
        return listings
    
    def extract_car_details(self, html_content: str, car_url: str, listing_image_url: str) -> Optional[Dict]:
        """Extract car details from car page HTML."""
        match = re.search(r'var _props = ({.*?});', html_content, re.DOTALL)
        if not match:
            return None
        
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
        
        listing = data.get('listing', {})
        vehicle = listing.get('vehicle', {})
        tracking = data.get('tracking', {})
        attr = tracking.get('gtm', {}).get('dataLayer', {}).get('a', {}).get('attr', {})
        pulse_obj = tracking.get('pulse', {}).get('pulse', {}).get('object', {})
        
        result = {}
        
        # Basic info
        result['external_id'] = listing.get('externalId')
        result['url'] = listing.get('canonicalUrl', car_url)
        result['brand'] = vehicle.get('make')
        result['model'] = vehicle.get('model')
        result['variant'] = vehicle.get('variant')
        result['title'] = f"{vehicle.get('make', '')} {vehicle.get('model', '')} {vehicle.get('variant', '')}".strip()
        
        # Price
        price_info = listing.get('price', {})
        result['price'] = price_info.get('displayValue', '')
        
        # Description
        result['description'] = listing.get('description', '')
        
        # Vehicle details (translated to English)
        for detail in vehicle.get('details', []):
            danish_name = detail.get('name', '')
            english_name = translate_key(danish_name)
            result[f'details_{english_name}'] = detail.get('displayValue', '')
        
        # Model information (translated to English)
        for info in vehicle.get('modelInformation', []):
            danish_name = info.get('name', '')
            english_name = translate_key(danish_name)
            result[f'model_{english_name}'] = info.get('displayValue', '')
        
        # Equipment
        equipment_list = []
        for eq in vehicle.get('equipment', []):
            if isinstance(eq, dict) and 'value' in eq:
                equipment_list.append(eq['value'])
            elif isinstance(eq, str):
                equipment_list.append(eq)
        result['equipment'] = ';'.join(equipment_list)
        
        # Seller
        seller = listing.get('seller', {})
        result['seller_name'] = seller.get('name')
        result['seller_type'] = seller.get('type')
        address = seller.get('address', {})
        result['seller_city'] = address.get('city')
        result['seller_zipcode'] = address.get('zipCode')
        
        # Dates
        result['listing_date'] = pulse_obj.get('publicationDate', '')
        result['last_updated'] = pulse_obj.get('lastUpdateDate', '')
        result['registration_date'] = attr.get('vehicle_history_registration_date', '')
        
        # Additional attr fields
        result['attr_power_hp'] = attr.get('vehicle_model_effect')
        result['attr_weight_kg'] = attr.get('vehicle_model_weight')
        result['attr_acceleration_0_100'] = attr.get('vehicle_model_acc_0_to_100_kmh')
        result['attr_top_speed_kmh'] = attr.get('vehicle_model_top_speed')
        result['attr_color'] = attr.get('vehicle_color_name')
        
        # Image (from search listing)
        result['image_url'] = listing_image_url
        result['image_filename'] = f"{result['external_id']}.jpg" if result['external_id'] else ''
        
        return result
    
    def process_car_data(self, raw_data: Dict) -> Optional[Dict]:
        """Process raw scraped data into database format (matching import_csv_to_db.py)."""
        try:
            url = safe_value(raw_data.get('url'))
            external_id = str(raw_data.get('external_id')) if raw_data.get('external_id') else None
            brand = safe_value(raw_data.get('brand'))
            model = safe_value(raw_data.get('model'))
            
            if not url or not brand or not model:
                return None
            
            price = clean_price(raw_data.get('price'))
            if not price or price <= 0:
                return None
            
            variant = safe_value(raw_data.get('variant'))
            title = safe_value(raw_data.get('title'))
            description = safe_value(raw_data.get('description'))
            
            # Fuel type
            fuel_type = standardize_fuel_type(raw_data.get('details_fuel_type'))
            
            # Other details
            new_price = clean_price(raw_data.get('model_new_price'))
            model_year = clean_numeric(raw_data.get('details_model_year'))
            year = model_year
            mileage = clean_numeric(raw_data.get('details_mileage_km'), ' km')
            
            transmission = standardize_transmission(raw_data.get('details_geartype'), fuel_type)
            body_type = standardize_body_type(raw_data.get('model_body_type'))
            drive_type = standardize_drive_type(raw_data.get('model_drive_type'))
            
            # Power
            power_str = raw_data.get('details_power_hp_nm')
            horsepower = extract_horsepower(power_str)
            if not horsepower:
                horsepower = clean_numeric(raw_data.get('attr_power_hp'))
            torque_nm = extract_torque(power_str)
            
            # Performance
            acceleration = clean_float(raw_data.get('details_acceleration_0_100')) or clean_float(raw_data.get('attr_acceleration_0_100'))
            top_speed = clean_numeric(raw_data.get('details_top_speed')) or clean_numeric(raw_data.get('attr_top_speed_kmh'))
            
            # Body
            doors = clean_numeric(raw_data.get('model_doors'))
            if doors and (doors < 2 or doors > 5):
                doors = 4
            seats = 5
            gear_count = clean_numeric(raw_data.get('details_number_of_gears'))
            cylinders = clean_numeric(raw_data.get('model_cylinders'))
            
            # Registration
            first_registration = safe_value(raw_data.get('details_first_registration'))
            production_date = safe_value(raw_data.get('details_production_year'))
            
            # EV fields
            range_val = raw_data.get('details_range_km')
            if range_val and range_val != '-':
                range_val = str(range_val).replace('(NEDC)', '').replace('(WLTP)', '').replace('km', '').strip()
            range_km = clean_numeric(range_val)
            
            battery_capacity = clean_float(raw_data.get('details_battery_capacity_kwh'))
            energy_consumption = clean_numeric(raw_data.get('details_energy_consumption'))
            home_charging_ac = safe_value(raw_data.get('details_home_charging_ac'))
            fast_charging_dc = safe_value(raw_data.get('details_fast_charging_dc'))
            charging_time_dc = safe_value(raw_data.get('details_charging_time_dc_10_80_pct'))
            
            # Fuel consumption
            fuel_consumption = safe_value(raw_data.get('details_fuel_consumption'))
            co2_emission = safe_value(raw_data.get('details_co2_emission'))
            euro_norm = safe_value(raw_data.get('details_euro_norm'))
            tank_capacity = clean_numeric(raw_data.get('model_tank_capacity_l'))
            
            # Other
            color = safe_value(raw_data.get('details_color')) or safe_value(raw_data.get('attr_color'))
            equipment = safe_value(raw_data.get('equipment'))
            category = safe_value(raw_data.get('model_category'))
            periodic_tax = safe_value(raw_data.get('details_periodic_tax'))
            
            # Image
            image_filename = safe_value(raw_data.get('image_filename'))
            image_path = f"images/{image_filename}" if image_filename else None
            
            # Booleans
            abs_brakes = clean_boolean(raw_data.get('model_abs_brakes'))
            esp = clean_boolean(raw_data.get('model_esp'))
            airbags = clean_numeric(raw_data.get('model_airbags'))
            
            # Dimensions
            weight = clean_numeric(raw_data.get('model_weight_kg')) or clean_numeric(raw_data.get('attr_weight_kg'))
            width = clean_numeric(raw_data.get('model_width_cm'))
            length = clean_numeric(raw_data.get('model_length_cm'))
            height = clean_numeric(raw_data.get('model_height_cm'))
            trunk_size = clean_numeric(raw_data.get('model_trunk_size'))
            load_capacity = clean_numeric(raw_data.get('model_load_capacity_kg'))
            towing_capacity = clean_numeric(raw_data.get('details_towing_capacity')) or clean_numeric(raw_data.get('model_max_towing_with_brake'))
            max_towing_weight = clean_numeric(raw_data.get('model_max_towing_with_brake'))
            
            # Location
            seller_city = safe_value(raw_data.get('seller_city'))
            seller_zipcode = safe_value(raw_data.get('seller_zipcode'))
            if seller_city and seller_zipcode:
                location = f"{seller_city}, {seller_zipcode}"
            elif seller_city:
                location = seller_city
            else:
                location = None
            
            return {
                'external_id': external_id or url,
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
                'first_registration': first_registration,
                'production_date': production_date,
                'mileage': mileage,
                'fuel_type': fuel_type,
                'transmission': transmission,
                'gear_count': gear_count,
                'cylinders': cylinders,
                'body_type': body_type,
                'horsepower': horsepower,
                'torque_nm': torque_nm,
                'engine_size': None,
                'acceleration': acceleration,
                'top_speed': top_speed,
                'drive_type': drive_type,
                'doors': doors,
                'seats': seats,
                'color': color,
                'category': category,
                'equipment': equipment,
                'abs_brakes': abs_brakes,
                'esp': esp,
                'airbags': airbags,
                'weight': weight,
                'width': width,
                'length': length,
                'height': height,
                'trunk_size': trunk_size,
                'load_capacity': load_capacity,
                'towing_capacity': towing_capacity,
                'max_towing_weight': max_towing_weight,
                'range_km': range_km,
                'battery_capacity': battery_capacity,
                'energy_consumption': energy_consumption,
                'home_charging_ac': home_charging_ac,
                'fast_charging_dc': fast_charging_dc,
                'charging_time_dc': charging_time_dc,
                'fuel_consumption': fuel_consumption,
                'co2_emission': co2_emission,
                'euro_norm': euro_norm,
                'tank_capacity': tank_capacity,
                'periodic_tax': periodic_tax,
                'image_path': image_path,
                'image_downloaded': True if image_path else False,
                'source_url': url,
                'location': location,
                'dealer_name': safe_value(raw_data.get('seller_name'))
            }
        except Exception as e:
            self.logger.error(f"Error processing car data: {e}")
            return None
    
    def insert_car_to_db(self, car_data: Dict) -> bool:
        """Insert a single car into the database."""
        if self.dry_run:
            self.logger.info(f"  [DRY RUN] Would insert: {car_data['brand']} {car_data['model']}")
            return True
        
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            
            # Check if already exists
            cur.execute("SELECT id FROM cars WHERE external_id = %s", (car_data['external_id'],))
            if cur.fetchone():
                cur.close()
                conn.close()
                return False  # Already exists
            
            cur.execute("""
                INSERT INTO cars (
                    external_id, url, brand, model, variant, title, description,
                    price, new_price, model_year, year, first_registration, production_date,
                    mileage, fuel_type, transmission, gear_count, cylinders,
                    body_type, horsepower, torque_nm, engine_size,
                    acceleration, top_speed, drive_type, doors, seats,
                    color, category, equipment, abs_brakes, esp, airbags,
                    weight, width, length, height, trunk_size, load_capacity,
                    towing_capacity, max_towing_weight,
                    range_km, battery_capacity, energy_consumption,
                    home_charging_ac, fast_charging_dc, charging_time_dc,
                    fuel_consumption, co2_emission, euro_norm, tank_capacity,
                    periodic_tax, image_path, image_downloaded,
                    source_url, location, dealer_name
                ) VALUES (
                    %(external_id)s, %(url)s, %(brand)s, %(model)s, %(variant)s,
                    %(title)s, %(description)s, %(price)s, %(new_price)s,
                    %(model_year)s, %(year)s, %(first_registration)s, %(production_date)s,
                    %(mileage)s, %(fuel_type)s, %(transmission)s, %(gear_count)s, %(cylinders)s,
                    %(body_type)s, %(horsepower)s, %(torque_nm)s, %(engine_size)s,
                    %(acceleration)s, %(top_speed)s, %(drive_type)s,
                    %(doors)s, %(seats)s, %(color)s, %(category)s, %(equipment)s,
                    %(abs_brakes)s, %(esp)s, %(airbags)s,
                    %(weight)s, %(width)s, %(length)s, %(height)s, %(trunk_size)s, %(load_capacity)s,
                    %(towing_capacity)s, %(max_towing_weight)s,
                    %(range_km)s, %(battery_capacity)s, %(energy_consumption)s,
                    %(home_charging_ac)s, %(fast_charging_dc)s, %(charging_time_dc)s,
                    %(fuel_consumption)s, %(co2_emission)s, %(euro_norm)s, %(tank_capacity)s,
                    %(periodic_tax)s, %(image_path)s, %(image_downloaded)s,
                    %(source_url)s, %(location)s, %(dealer_name)s
                )
            """, car_data)
            
            conn.commit()
            cur.close()
            conn.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Database insert error: {e}")
            return False
    
    def download_image(self, image_url: str, save_path: str) -> bool:
        """Download image to local file."""
        if not image_url or self.dry_run:
            return False
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            response = self.session.get(image_url, headers=self.get_headers(), timeout=30, stream=True)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception:
            return False
    
    def run(self, max_new_listings: int = None) -> Dict:
        """
        Run the incremental scraper.
        
        Returns dict with stats: {new_count, skipped_count, error_count, duration}
        """
        start_time = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info("INCREMENTAL SCRAPER STARTED")
        self.logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.dry_run:
            self.logger.info("MODE: DRY RUN (no database changes)")
        self.logger.info("=" * 60)
        
        # Get known IDs from database
        known_ids = self.get_known_external_ids()
        
        if not known_ids and not self.dry_run:
            self.logger.warning("No existing cars in database. Run the full scraper first!")
            return {'new_count': 0, 'skipped_count': 0, 'error_count': 0, 'duration': 0}
        
        # Phase 1: Find new listings
        self.logger.info("\n--- Phase 1: Finding new listings ---")
        
        new_listings = []
        page = 1
        consecutive_known = 0
        
        while page <= CONFIG['MAX_PAGES']:
            url = self.build_search_url(page)
            self.logger.info(f"Fetching page {page}...")
            
            html = self.fetch_page(url)
            if not html:
                self.logger.error(f"Failed to fetch page {page}")
                break
            
            listings = self.extract_listings_from_html(html)
            if not listings:
                self.logger.info("No more listings found")
                break
            
            page_new = 0
            page_known = 0
            
            for listing in listings:
                ext_id = listing['external_id']
                
                if ext_id in known_ids:
                    page_known += 1
                    consecutive_known += 1
                else:
                    new_listings.append(listing)
                    page_new += 1
                    consecutive_known = 0
                    
                    if max_new_listings and len(new_listings) >= max_new_listings:
                        self.logger.info(f"Reached max new listings limit ({max_new_listings})")
                        break
            
            self.logger.info(f"  Page {page}: {page_new} new, {page_known} known")
            
            # Stop if we've seen many consecutive known listings (we've caught up)
            if consecutive_known >= CONFIG['ITEMS_PER_PAGE']:
                self.logger.info("Caught up with existing data (full page of known listings)")
                break
            
            if max_new_listings and len(new_listings) >= max_new_listings:
                break
            
            page += 1
            time.sleep(random.uniform(*CONFIG['DELAY_BETWEEN_REQUESTS']))
        
        self.logger.info(f"\nFound {len(new_listings)} new listings")
        
        if not new_listings:
            duration = time.time() - start_time
            self.logger.info(f"\nNo new listings to process. Duration: {duration:.1f}s")
            return {'new_count': 0, 'skipped_count': 0, 'error_count': 0, 'duration': duration}
        
        # Phase 2: Extract details and insert to database
        self.logger.info("\n--- Phase 2: Extracting details and importing ---")
        
        new_count = 0
        skipped_count = 0
        error_count = 0
        
        for idx, listing in enumerate(new_listings):
            self.logger.info(f"[{idx+1}/{len(new_listings)}] Processing {listing['external_id']}...")
            
            # Fetch car page
            html = self.fetch_page(listing['uri'])
            if not html:
                self.logger.warning(f"  Failed to fetch car page")
                error_count += 1
                continue
            
            # Extract details
            raw_data = self.extract_car_details(html, listing['uri'], listing['image_url'])
            if not raw_data:
                self.logger.warning(f"  Failed to extract details")
                error_count += 1
                continue
            
            # Process into database format
            car_data = self.process_car_data(raw_data)
            if not car_data:
                self.logger.warning(f"  Failed to process data (missing required fields)")
                skipped_count += 1
                continue
            
            # Insert to database
            if self.insert_car_to_db(car_data):
                new_count += 1
                self.logger.info(f"  Inserted: {car_data['brand']} {car_data['model']} - {car_data['price']} kr")
                
                # Download image
                if listing['image_url']:
                    img_path = os.path.join(CONFIG['IMAGES_DIR'], f"{listing['external_id']}.jpg")
                    if not os.path.exists(img_path):
                        self.download_image(listing['image_url'], img_path)
            else:
                skipped_count += 1
            
            time.sleep(random.uniform(*CONFIG['DELAY_BETWEEN_DETAILS']))
        
        duration = time.time() - start_time
        
        # Summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("INCREMENTAL SCRAPER COMPLETED")
        self.logger.info(f"New cars inserted: {new_count}")
        self.logger.info(f"Skipped (duplicates/invalid): {skipped_count}")
        self.logger.info(f"Errors: {error_count}")
        self.logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f}m)")
        self.logger.info("=" * 60)
        
        return {
            'new_count': new_count,
            'skipped_count': skipped_count,
            'error_count': error_count,
            'duration': duration
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Bilbasen Incremental Scraper')
    parser.add_argument('--test', action='store_true', help='Test mode (limit to 10 new listings)')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (no database changes)')
    parser.add_argument('--max', type=int, default=None, help='Maximum new listings to process')
    args = parser.parse_args()
    
    logger = setup_logging()
    
    max_listings = 10 if args.test else args.max
    
    scraper = IncrementalScraper(logger, dry_run=args.dry_run)
    result = scraper.run(max_new_listings=max_listings)
    
    # Exit code based on results
    if result['error_count'] > result['new_count']:
        sys.exit(1)  # More errors than successes
    sys.exit(0)


if __name__ == '__main__':
    main()
