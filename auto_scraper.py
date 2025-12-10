#!/usr/bin/env python3
"""
Auto-Scraper for Bilbasen.dk - Incremental Scraping
====================================================

This script performs incremental scraping of new car listings from Bilbasen.dk
by tracking the highest external_id in the database and scraping only new listings.

Features:
- Queries database for highest external_id
- Scrapes newest-first until hitting known listing IDs
- Downloads car details and images
- Updates scraping_logs with statistics
- Supports both incremental and full scraping modes
- Can be run as a cron job (every 2 days recommended)

Usage:
    python auto_scraper.py --mode incremental  # Default
    python auto_scraper.py --mode full         # Full scrape
    python auto_scraper.py --no-images         # Skip image downloads
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os
import sys
import argparse
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
import uuid
import re
import json
from dotenv import load_dotenv
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables
load_dotenv()

# Setup logging
# Determine log file path based on environment
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists('/app/data'):
    log_path = '/app/data/auto_scraper.log'
else:
    log_path = os.path.join(script_dir, '..', 'bilbasen_scrape', 'auto_scraper.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database Configuration
DB_NAME = os.getenv("DB_NAME", "bpr_cars")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# Scraper Configuration
BASE_URL = "https://www.bilbasen.dk/brugt/bil"
PARAMS = {
    "includeengroscvr": "true",
    "includeleasing": "false",
    "sortby": "date",
    "sortorder": "desc"
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml',
    'Accept-Language': 'da,en-US;q=0.9,en;q=0.8',
}

# Output directories - use same as manual scraper
OUTPUT_DIR = 'bilbasen_scrape'
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')

# Threading configuration
MAX_WORKERS = 16  # Parallel workers for detail scraping


# ============================================================================
# Data Cleaning Functions (matching import_csv_to_db.py)
# ============================================================================

def clean_price(price_str):
    """Clean Danish price format: '38.900 kr.' -> 38900"""
    if not price_str or price_str == '' or price_str == '-':
        return None
    price_str = str(price_str).replace(' kr.', '').replace('.', '').replace(',', '.').strip()
    try:
        return float(price_str)
    except:
        return None


def clean_numeric(value_str, suffix=''):
    """Clean numeric values with optional suffix like 'km', 'hk', 'km/h', 'km/t'"""
    if not value_str or value_str == '' or value_str == '-':
        return None
    value_str = str(value_str).replace(suffix, '').replace('km/t', '').replace('km/h', '').replace('kmh', '').replace('kg', '').replace('cm', '').strip()
    value_str = value_str.replace('.', '').replace(',', '.')
    try:
        return int(float(value_str))
    except:
        return None


def clean_float(value_str):
    """Clean float values like '14,0 kWh' or '10.7 sec' or '14,7 sek.' or '-'"""
    if not value_str or value_str == '' or value_str == '-':
        return None
    value_str = str(value_str).replace('kWh', '').replace('kwh', '').replace('sek.', '').replace('sek', '').replace('sec', '').replace(',', '.').strip()
    try:
        return float(value_str)
    except:
        return None


def standardize_fuel_type(fuel):
    """Standardize fuel types to 7 categories"""
    if not fuel or fuel == '':
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
    if not trans or trans == '' or trans == '-':
        return 'Automatic' if fuel_type == 'Electricity' else None
    
    trans_lower = str(trans).strip().lower()
    if trans_lower in ['automatgear', 'automatic', 'automatisk', 'auto', 'automat', 'a']:
        return 'Automatic'
    if trans_lower in ['manuel', 'manual', 'manuell', 'manuelt', 'm']:
        return 'Manual'
    
    return 'Manual'  # Default


def standardize_body_type(body):
    """Standardize body types"""
    if not body or body == '':
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
    if not drive or drive == '':
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
    if not power_str or power_str == '':
        return None
    match = re.search(r'(\d+)\s*hk', str(power_str))
    if match:
        return int(match.group(1))
    return None


def extract_torque(power_str):
    """Extract torque from format like '178 hk/230 nm'"""
    if not power_str or power_str == '':
        return None
    match = re.search(r'(\d+)\s*nm', str(power_str).lower())
    if match:
        return int(match.group(1))
    return None


def extract_engine_size(power_str):
    """Extract engine size from power string if present like '1.6 L'"""
    if not power_str or power_str == '':
        return None
    match = re.search(r'(\d+[.,]\d+)\s*[lL]', str(power_str))
    if match:
        return float(match.group(1).replace(',', '.'))
    return None


def clean_boolean(value):
    """Convert various boolean representations"""
    if not value or value == '':
        return None
    value_lower = str(value).strip().lower()
    if value_lower in ['ja', 'yes', 'true', '1']:
        return True
    if value_lower in ['nej', 'no', 'false', '0']:
        return False
    return None


def safe_value(val):
    """Convert empty/null values to None"""
    if not val or val == '' or val == '-':
        return None
    val_str = str(val).strip()
    if val_str == '' or val_str.lower() == 'nan':
        return None
    return val_str


class AutoScraper:
    def __init__(self, mode='incremental', download_images=True):
        self.mode = mode
        self.download_images = download_images
        # No shared session - threads will create their own
        
        # Statistics
        self.cars_new = 0
        self.cars_updated = 0
        self.images_downloaded = 0
        self.highest_external_id = None
        self.known_max_id = None
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Create output directories
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if download_images:
            os.makedirs(IMAGES_DIR, exist_ok=True)
        
        # Database connection
        self.conn = None
        self.cur = None
        
    def connect_db(self):
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASS,
                host=DB_HOST,
                port=DB_PORT
            )
            self.cur = self.conn.cursor()
            logger.info(f"✅ Connected to database: {DB_NAME}")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def get_max_external_id(self):
        """Get the highest external_id from database"""
        try:
            self.cur.execute("SELECT MAX(external_id::bigint) FROM cars WHERE external_id IS NOT NULL")
            result = self.cur.fetchone()[0]
            if result:
                self.known_max_id = str(result)
                logger.info(f"📊 Highest external_id in database: {self.known_max_id}")
                return self.known_max_id
            else:
                logger.info("📊 No external_ids found in database (empty table)")
                return None
        except Exception as e:
            logger.error(f"❌ Error getting max external_id: {e}")
            return None
    
    def extract_external_id(self, listing_url):
        """Extract external_id from listing URL"""
        match = re.search(r'/(\d+)/?$', listing_url)
        if match:
            return match.group(1)
        return None
    
    def scrape_listing_page(self, page=1):
        """Scrape a single page of listings using __NEXT_DATA__ JSON extraction"""
        params = PARAMS.copy()
        params['page'] = page
        
        try:
            url = f"{BASE_URL}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
            logger.info(f"📄 Scraping page {page}: {url}")
            
            # Create session for this request
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })
            
            response = session.get(url, timeout=30)
            response.raise_for_status()
            
            # Extract JSON data from __NEXT_DATA__ script tag
            pattern = r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>'
            match = re.search(pattern, response.text, re.DOTALL)
            
            if not match:
                logger.warning(f"⚠️ No __NEXT_DATA__ found on page {page}")
                return [], False
            
            data = json.loads(match.group(1))
            queries = data.get('props', {}).get('pageProps', {}).get('dehydratedState', {}).get('queries', [])
            
            extracted_listings = []
            for query in queries:
                state_data = query.get('state', {}).get('data', {})
                if 'listings' in state_data:
                    raw_listings = state_data['listings']
                    
                    for item in raw_listings:
                        try:
                            uri = item.get('uri', '')
                            if not uri:
                                continue
                            
                            external_id = uri.rstrip('/').split('/')[-1]
                            if not external_id:
                                continue
                            
                            # Check if we've reached known listings (incremental mode)
                            if self.mode == 'incremental' and self.known_max_id:
                                if int(external_id) <= int(self.known_max_id):
                                    logger.info(f"🛑 Reached known listing ID {external_id}, stopping incremental scrape")
                                    return extracted_listings, True  # Stop signal
                            
                            # Update highest_external_id
                            if not self.highest_external_id or int(external_id) > int(self.highest_external_id):
                                self.highest_external_id = external_id
                            
                            # Extract image URL from media array
                            image_url = ''
                            for media in item.get('media', []):
                                if media.get('mediaType') == 'Picture':
                                    image_url = media.get('url', '')
                                    break
                            
                            listing_url = f"https://www.bilbasen.dk{uri}" if not uri.startswith('http') else uri
                            
                            extracted_listings.append({
                                'external_id': external_id,
                                'url': listing_url,
                                'title': item.get('heading', ''),
                                'image_url': image_url
                            })
                            
                        except Exception as e:
                            logger.warning(f"⚠️ Error extracting listing: {e}")
                            continue
            
            logger.info(f"✅ Extracted {len(extracted_listings)} listings from page {page}")
            return extracted_listings, False  # No stop signal
            
        except Exception as e:
            logger.error(f"❌ Error scraping page {page}: {e}")
            return [], False
    
    def scrape_listing_details(self, listing_url, external_id):
        """Scrape detailed information from a single listing page"""
        try:
            logger.debug(f"Fetching details for {external_id}")
            
            # Create session per thread
            session = requests.Session()
            session.headers.update(HEADERS)
            
            response = session.get(listing_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            details = {
                'external_id': external_id,
                'url': listing_url,
            }
            
            # Extract brand, model, variant from title
            title_elem = soup.find('h1')
            if title_elem:
                title = title_elem.get_text(strip=True)
                details['title'] = title
                # Try to parse brand and model from title
                parts = title.split()
                if len(parts) >= 2:
                    details['brand'] = parts[0]
                    details['model'] = parts[1]
                    details['variant'] = ' '.join(parts[2:]) if len(parts) > 2 else ''
            
            # Extract price
            price_elem = soup.find('span', class_='price') or soup.find('div', class_='price')
            if price_elem:
                details['price'] = price_elem.get_text(strip=True)
            
            # Extract description
            desc_elem = soup.find('div', class_='description') or soup.find('div', id='description')
            if desc_elem:
                details['description'] = desc_elem.get_text(strip=True)
            
            # Extract specifications table
            spec_tables = soup.find_all('table', class_='specifications') or soup.find_all('dl')
            for table in spec_tables:
                rows = table.find_all(['tr', 'div'])
                for row in rows:
                    try:
                        key_elem = row.find(['th', 'dt'])
                        val_elem = row.find(['td', 'dd'])
                        if key_elem and val_elem:
                            key = key_elem.get_text(strip=True).lower()
                            val = val_elem.get_text(strip=True)
                            
                            # Map to database fields
                            if 'årgang' in key or 'modelår' in key:
                                details['model_year'] = val
                            elif 'km' in key and 'kørt' in key:
                                details['mileage_km'] = val
                            elif 'brændstof' in key:
                                details['fuel_type'] = val
                            elif 'hk' in key or 'effekt' in key:
                                details['power_hp_nm'] = val
                            # Add more mappings as needed
                    except:
                        continue
            
            # Extract image
            img_elem = soup.find('img', class_='main-image') or soup.find('img', {'data-src': True})
            if img_elem:
                details['image_url'] = img_elem.get('src') or img_elem.get('data-src')
            
            return details
            
        except Exception as e:
            logger.error(f"❌ Error scraping details for {external_id}: {e}")
            return None
    
    def download_image(self, image_url, external_id):
        """Download car image"""
        if not self.download_images or not image_url:
            return None
        
        try:
            image_filename = f"{external_id}.jpg"
            image_path = os.path.join(IMAGES_DIR, image_filename)
            
            # Skip if already downloaded
            if os.path.exists(image_path):
                return image_filename
            
            # Create session for this thread
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            response = session.get(image_url, timeout=30)
            response.raise_for_status()
            
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            with self.lock:
                self.images_downloaded += 1
            logger.debug(f"📥 Downloaded image: {image_filename}")
            return image_filename
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to download image for {external_id}: {e}")
            return None
    
    def _process_listing(self, listing):
        """Process a single listing (scrape details, download image, upsert to DB). Thread-safe method."""
        try:
            # Get detailed information
            details = self.scrape_listing_details(listing['url'], listing['external_id'])
            
            if details:
                # Download image
                if listing.get('image_url'):
                    image_filename = self.download_image(listing['image_url'], listing['external_id'])
                    if image_filename:
                        details['image_filename'] = image_filename
                        details['image_path'] = f"images/{image_filename}"
                        details['image_downloaded'] = True
                
                # Upsert to database
                self.upsert_car(details)
            
            # Small delay to be respectful
            time.sleep(random.uniform(0.3, 0.8))
            
        except Exception as e:
            logger.error(f"❌ Error processing listing {listing['external_id']}: {e}")
            raise
    
    def check_car_exists(self, external_id):
        """Check if car exists in database"""
        try:
            self.cur.execute("SELECT id FROM cars WHERE external_id = %s", (external_id,))
            result = self.cur.fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"❌ Error checking car existence: {e}")
            return None
    
    def upsert_car(self, raw_data):
        """Insert or update car in database with full data cleaning (matches import_csv_to_db.py)"""
        try:
            # Check if car exists
            external_id = raw_data.get('external_id')
            car_id = self.check_car_exists(external_id)
            
            if car_id:
                with self.lock:
                    self.cars_updated += 1
                logger.debug(f"🔄 Updating car {external_id}")
            else:
                with self.lock:
                    self.cars_new += 1
                logger.debug(f"➕ Inserting new car {external_id}")
            
            # Clean and standardize data (matching import_csv_to_db.py logic)
            price = clean_price(raw_data.get('price'))
            if not price or price <= 0:
                logger.warning(f"⚠️ Skipping {external_id}: invalid price")
                return False
            
            # Extract basic info
            url = raw_data.get('url')
            brand = safe_value(raw_data.get('brand'))
            model = safe_value(raw_data.get('model'))
            variant = safe_value(raw_data.get('variant'))
            title = safe_value(raw_data.get('title'))
            description = safe_value(raw_data.get('description'))
            
            # Standardize fuel type first (needed for transmission default)
            fuel_type = standardize_fuel_type(raw_data.get('fuel_type'))
            
            # Other details
            new_price = clean_price(raw_data.get('new_price'))
            model_year = clean_numeric(raw_data.get('model_year'))
            year = model_year
            mileage = clean_numeric(raw_data.get('mileage_km'), ' km')
            
            transmission = standardize_transmission(raw_data.get('geartype') or raw_data.get('transmission'), fuel_type)
            body_type = standardize_body_type(raw_data.get('body_type'))
            drive_type = standardize_drive_type(raw_data.get('drive_type'))
            
            # Power fields
            power_str = raw_data.get('power_hp_nm')
            horsepower = extract_horsepower(power_str)
            torque_nm = extract_torque(power_str)
            engine_size = extract_engine_size(power_str)
            
            # Performance
            acceleration = clean_float(raw_data.get('acceleration_0_100'))
            top_speed = clean_numeric(raw_data.get('top_speed'))
            
            # Body fields
            doors = clean_numeric(raw_data.get('doors'))
            if doors and (doors < 2 or doors > 5):
                doors = 4
            seats = 5  # Default
            gear_count = clean_numeric(raw_data.get('gear_count'))
            cylinders = clean_numeric(raw_data.get('cylinders'))
            
            # Registration
            first_registration = safe_value(raw_data.get('first_registration'))
            production_date = safe_value(raw_data.get('production_date'))
            
            # EV/Battery fields
            range_km = clean_numeric(raw_data.get('range_km'))
            battery_capacity = clean_float(raw_data.get('battery_capacity_kwh'))
            energy_consumption = clean_numeric(raw_data.get('energy_consumption'))
            home_charging_ac = safe_value(raw_data.get('home_charging_ac'))
            fast_charging_dc = safe_value(raw_data.get('fast_charging_dc'))
            charging_time_dc = safe_value(raw_data.get('charging_time_dc'))
            
            # Fuel consumption and emissions
            fuel_consumption = safe_value(raw_data.get('fuel_consumption'))
            co2_emission = safe_value(raw_data.get('co2_emission'))
            euro_norm = safe_value(raw_data.get('euro_norm'))
            tank_capacity = clean_numeric(raw_data.get('tank_capacity'))
            
            # Other fields
            color = safe_value(raw_data.get('color'))
            equipment = safe_value(raw_data.get('equipment'))
            category = safe_value(raw_data.get('category'))
            periodic_tax = safe_value(raw_data.get('periodic_tax'))
            
            # Image fields
            image_filename = raw_data.get('image_filename')
            # Save path matching manual scraper format
            image_path = f"images/{image_filename}" if image_filename else None
            image_downloaded = bool(image_filename)
            
            # Booleans
            abs_brakes = clean_boolean(raw_data.get('abs_brakes'))
            esp = clean_boolean(raw_data.get('esp'))
            airbags = clean_numeric(raw_data.get('airbags'))
            
            # Dimensions
            weight = clean_numeric(raw_data.get('weight'))
            width = clean_numeric(raw_data.get('width'))
            length = clean_numeric(raw_data.get('length'))
            height = clean_numeric(raw_data.get('height'))
            trunk_size = clean_numeric(raw_data.get('trunk_size'))
            load_capacity = clean_numeric(raw_data.get('load_capacity'))
            towing_capacity = clean_numeric(raw_data.get('towing_capacity'))
            max_towing_weight = clean_numeric(raw_data.get('max_towing_weight'))
            
            # Location
            seller_city = safe_value(raw_data.get('seller_city'))
            seller_zipcode = safe_value(raw_data.get('seller_zipcode'))
            if seller_city and seller_zipcode:
                location = f"{seller_city}, {seller_zipcode}"
            elif seller_city:
                location = seller_city
            else:
                location = None
            
            dealer_name = safe_value(raw_data.get('seller_name') or raw_data.get('dealer_name'))
            
            # Execute INSERT (same as import_csv_to_db.py)
            self.cur.execute("""
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
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s
                )
                ON CONFLICT (external_id)
                DO UPDATE SET
                    url = EXCLUDED.url,
                    brand = EXCLUDED.brand,
                    model = EXCLUDED.model,
                    variant = EXCLUDED.variant,
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    price = EXCLUDED.price,
                    new_price = EXCLUDED.new_price,
                    model_year = EXCLUDED.model_year,
                    year = EXCLUDED.year,
                    first_registration = EXCLUDED.first_registration,
                    production_date = EXCLUDED.production_date,
                    mileage = EXCLUDED.mileage,
                    fuel_type = EXCLUDED.fuel_type,
                    transmission = EXCLUDED.transmission,
                    gear_count = EXCLUDED.gear_count,
                    cylinders = EXCLUDED.cylinders,
                    body_type = EXCLUDED.body_type,
                    horsepower = EXCLUDED.horsepower,
                    torque_nm = EXCLUDED.torque_nm,
                    engine_size = EXCLUDED.engine_size,
                    acceleration = EXCLUDED.acceleration,
                    top_speed = EXCLUDED.top_speed,
                    drive_type = EXCLUDED.drive_type,
                    doors = EXCLUDED.doors,
                    seats = EXCLUDED.seats,
                    color = EXCLUDED.color,
                    category = EXCLUDED.category,
                    equipment = EXCLUDED.equipment,
                    abs_brakes = EXCLUDED.abs_brakes,
                    esp = EXCLUDED.esp,
                    airbags = EXCLUDED.airbags,
                    weight = EXCLUDED.weight,
                    width = EXCLUDED.width,
                    length = EXCLUDED.length,
                    height = EXCLUDED.height,
                    trunk_size = EXCLUDED.trunk_size,
                    load_capacity = EXCLUDED.load_capacity,
                    towing_capacity = EXCLUDED.towing_capacity,
                    max_towing_weight = EXCLUDED.max_towing_weight,
                    range_km = EXCLUDED.range_km,
                    battery_capacity = EXCLUDED.battery_capacity,
                    energy_consumption = EXCLUDED.energy_consumption,
                    home_charging_ac = EXCLUDED.home_charging_ac,
                    fast_charging_dc = EXCLUDED.fast_charging_dc,
                    charging_time_dc = EXCLUDED.charging_time_dc,
                    fuel_consumption = EXCLUDED.fuel_consumption,
                    co2_emission = EXCLUDED.co2_emission,
                    euro_norm = EXCLUDED.euro_norm,
                    tank_capacity = EXCLUDED.tank_capacity,
                    periodic_tax = EXCLUDED.periodic_tax,
                    image_path = EXCLUDED.image_path,
                    image_downloaded = EXCLUDED.image_downloaded,
                    location = EXCLUDED.location,
                    dealer_name = EXCLUDED.dealer_name,
                    updated_at = NOW()
            """, (
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
                url, location, dealer_name
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error upserting car {external_id}: {e}")
            self.conn.rollback()
            return False
    
    def log_scraping_run(self, success=True, error_message=None):
        """Log scraping run to database"""
        try:
            log_data = {
                'id': str(uuid.uuid4()),
                'source_name': 'bilbasen_auto_scraper',
                'cars_scraped': self.cars_new + self.cars_updated,
                'highest_external_id': self.highest_external_id,
                'scraping_mode': self.mode,
                'cars_new': self.cars_new,
                'cars_updated': self.cars_updated,
                'images_downloaded': self.images_downloaded,
                'success': success,
                'error_message': error_message,
                'started_at': self.start_time,
                'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            columns = list(log_data.keys())
            values = [log_data[col] for col in columns]
            placeholders = ', '.join(['%s'] * len(columns))
            
            query = f"""
                INSERT INTO scraping_logs ({', '.join(columns)})
                VALUES ({placeholders})
            """
            
            self.cur.execute(query, values)
            self.conn.commit()
            logger.info("✅ Scraping run logged to database")
            
        except Exception as e:
            logger.error(f"❌ Error logging scraping run: {e}")
    
    def run(self):
        """Main scraping loop"""
        self.start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"🚀 Starting auto-scraper in {self.mode} mode")
        
        # Connect to database
        if not self.connect_db():
            return False
        
        # Get max external_id if incremental mode
        if self.mode == 'incremental':
            self.get_max_external_id()
        
        try:
            page = 1
            total_listings = []
            stop_scraping = False
            
            while not stop_scraping:
                listings, stop_signal = self.scrape_listing_page(page)
                
                if not listings:
                    logger.info("📭 No more listings found")
                    break
                
                total_listings.extend(listings)
                
                if stop_signal:
                    stop_scraping = True
                    break
                
                # Scrape details and download images in parallel
                logger.info(f"🔄 Processing {len(listings)} listings with {MAX_WORKERS} workers")
                
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    future_to_listing = {}
                    for listing in listings:
                        future = executor.submit(self._process_listing, listing)
                        future_to_listing[future] = listing
                    
                    for future in as_completed(future_to_listing):
                        listing = future_to_listing[future]
                        try:
                            future.result()  # Will raise exception if processing failed
                        except Exception as e:
                            logger.error(f"❌ Error processing {listing['external_id']}: {e}")
                
                # Move to next page
                page += 1
                time.sleep(random.uniform(2, 4))
                
                # Safety limit
                if page > 100:
                    logger.warning("⚠️ Reached page limit (100), stopping")
                    break
            
            # Log results
            logger.info(f"\\n{'='*60}")
            logger.info("📊 Scraping Summary")
            logger.info(f"{'='*60}")
            logger.info(f"Mode: {self.mode}")
            logger.info(f"Pages scraped: {page}")
            logger.info(f"Total listings found: {len(total_listings)}")
            logger.info(f"New cars: {self.cars_new}")
            logger.info(f"Updated cars: {self.cars_updated}")
            logger.info(f"Images downloaded: {self.images_downloaded}")
            logger.info(f"Highest external_id: {self.highest_external_id}")
            logger.info(f"{'='*60}")
            
            # Log to database
            self.log_scraping_run(success=True)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Scraping failed: {e}")
            self.log_scraping_run(success=False, error_message=str(e))
            return False
        
        finally:
            if self.conn:
                self.cur.close()
                self.conn.close()
                logger.info("🔒 Database connection closed")


def main():
    parser = argparse.ArgumentParser(description='Bilbasen Auto-Scraper (Multi-threaded)')
    parser.add_argument('--mode', choices=['incremental', 'full'], default='incremental',
                        help='Scraping mode (default: incremental)')
    parser.add_argument('--no-images', action='store_true',
                        help='Skip downloading images')
    # Update global MAX_WORKERS if specified
    global MAX_WORKERS
    
    parser.add_argument('--workers', type=int, default=MAX_WORKERS,
                        help=f'Number of parallel workers (default: {MAX_WORKERS})')
    
    args = parser.parse_args()
    
    MAX_WORKERS = args.workers
    
    logger.info(f"🚀 Starting with {MAX_WORKERS} parallel workers")
    
    scraper = AutoScraper(
        mode=args.mode,
        download_images=not args.no_images
    )
    
    success = scraper.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
