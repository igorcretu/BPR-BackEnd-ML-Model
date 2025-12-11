#!/usr/bin/env python3
"""
Bilbasen Incremental Scraper v3
Scrapes new listings from bilbasen.dk, sorted by newest first.
Extracts data exactly like the full scraper, imports like import_csv_to_db.py
"""

import os
import sys
import re
import json
import time
import random
import logging
import argparse
import requests
import psycopg2
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'SORTED_URL': 'https://www.bilbasen.dk/brugt/bil?includeengroscvr=true&includeleasing=false&sortby=date&sortorder=desc',
    'BASE_URL': 'https://www.bilbasen.dk/brugt/bil',
    
    'ITEMS_PER_PAGE': 30,
    'MAX_PAGES': 200,
    
    'DELAY_BETWEEN_PAGES': (0.5, 1.0),
    'DELAY_BETWEEN_CARS': (0.2, 0.5),
    
    'MAX_WORKERS': 12,
    
    'MAX_RETRIES': 3,
    'RETRY_DELAY_BASE': 5,
    
    'KNOWN_ID_STOP_THRESHOLD': 60,
    
    'COOKIE_FILE': 'bilbasen_cookies.json',
    
    'LOG_DIR': 'logs',
    'IMAGES_DIR': 'images',
}

# Danish to English translation map (from full scraper)
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
    'geartype': 'geartype',
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
    'tankkapacitet': 'tank_capacity_l',
    'motorstørrelse': 'engine_size',
}

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    os.makedirs(CONFIG['LOG_DIR'], exist_ok=True)
    
    # Create timestamped log file for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(CONFIG['LOG_DIR'], f"scraper_{timestamp}.log")
    
    # Detailed formatter with timestamp
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # File handler - keeps everything
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler - only important messages
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    logger = logging.getLogger('incremental_scraper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log the file location
    logger.info(f"Full logs being written to: {log_file}")
    
    return logger

# ============================================================================
# HELPER FUNCTIONS (from full scraper)
# ============================================================================

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

def get_headers() -> Dict[str, str]:
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'da-DK,da;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }

# ============================================================================
# CLEANING FUNCTIONS (from import_csv_to_db.py)
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
    """Clean float values like '14,0 kWh'"""
    if pd.isna(value_str) or value_str == '' or value_str == '-':
        return None
    value_str = str(value_str).replace('kWh', '').replace('kwh', '').replace('sek.', '').replace('sek', '').replace('sec', '').replace(',', '.').strip()
    try:
        return float(value_str)
    except:
        return None

def standardize_fuel_type(fuel):
    """Standardize fuel types"""
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

def extract_engine_size(power_str):
    """Extract engine size from power string"""
    if pd.isna(power_str) or power_str == '':
        return None
    match = re.search(r'(\d+[.,]\d+)\s*[lL]', str(power_str))
    if match:
        return float(match.group(1).replace(',', '.'))
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

# ============================================================================
# THREAD-SAFE STATS
# ============================================================================

class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.new_count = 0
        self.skipped_count = 0
        self.error_count = 0
        self.image_downloaded = 0
        self.image_failed = 0
    
    def increment(self, field: str, value: int = 1):
        with self.lock:
            setattr(self, field, getattr(self, field) + value)

# ============================================================================
# SCRAPER CLASS
# ============================================================================

class IncrementalScraper:
    def __init__(self, logger: logging.Logger, dry_run: bool = False):
        self.logger = logger
        self.dry_run = dry_run
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'car_prediction'),
            'user': os.getenv('POSTGRES_USER', 'bpr_user'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
        }
        self.cookies = {}
        self.stats = Stats()
        self.lock = threading.Lock()
        self.load_cookies()
    
    def load_cookies(self):
        if os.path.exists(CONFIG['COOKIE_FILE']):
            try:
                with open(CONFIG['COOKIE_FILE'], 'r') as f:
                    self.cookies = json.load(f)
                self.logger.info(f"Loaded cookies from {CONFIG['COOKIE_FILE']}")
            except Exception as e:
                self.logger.warning(f"Failed to load cookies: {e}")
    
    def save_cookies(self):
        try:
            with open(CONFIG['COOKIE_FILE'], 'w') as f:
                json.dump(self.cookies, f)
        except Exception as e:
            self.logger.warning(f"Failed to save cookies: {e}")
    
    def refresh_cookies_with_playwright(self) -> bool:
        self.logger.info("Refreshing cookies with Playwright...")
        try:
            from playwright.sync_api import sync_playwright
            
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context()
                page = context.new_page()
                page.goto(CONFIG['SORTED_URL'], wait_until='networkidle')
                cookies = context.cookies()
                self.cookies = {c['name']: c['value'] for c in cookies}
                browser.close()
            
            self.save_cookies()
            self.logger.info(f"Got {len(self.cookies)} cookies via Playwright")
            return True
        except ImportError:
            self.logger.error("Playwright not installed")
            return False
        except Exception as e:
            self.logger.error(f"Playwright error: {e}")
            return False
    
    def get_db_connection(self):
        return psycopg2.connect(**self.db_config)
    
    def get_known_external_ids(self) -> set:
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
    
    def update_scraping_log(self, cars_scraped: int, cars_new: int, cars_updated: int, images_downloaded: int):
        """Update the most recent ScrapingLog entry with final statistics"""
        self.logger.info("=" * 60)
        self.logger.info("UPDATING SCRAPING LOG IN DATABASE")
        self.logger.info(f"Cars scraped: {cars_scraped}")
        self.logger.info(f"Cars new: {cars_new}")
        self.logger.info(f"Cars updated: {cars_updated}")
        self.logger.info(f"Images downloaded: {images_downloaded}")
        
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            
            # First, check what logs exist
            cur.execute("""
                SELECT id, source_name, success, completed_at, started_at 
                FROM scraping_logs 
                WHERE source_name = 'bilbasen' 
                ORDER BY started_at DESC 
                LIMIT 3
            """)
            existing_logs = cur.fetchall()
            self.logger.debug(f"Found {len(existing_logs)} recent scraping logs:")
            for log in existing_logs:
                self.logger.debug(f"  ID: {log[0]}, Success: {log[2]}, Completed: {log[3]}, Started: {log[4]}")
            
            # Find the most recent incomplete log entry
            cur.execute("""
                SELECT id 
                FROM scraping_logs 
                WHERE source_name = 'bilbasen' 
                  AND success = TRUE 
                  AND completed_at IS NULL
                ORDER BY started_at DESC
                LIMIT 1
            """)
            
            log_to_update = cur.fetchone()
            
            if not log_to_update:
                self.logger.warning("⚠️  No ScrapingLog entry found to update (no incomplete success=TRUE logs)")
                cur.close()
                conn.close()
                self.logger.info("=" * 60)
                return
            
            log_id = log_to_update[0]
            self.logger.debug(f"Found log to update: {log_id}")
            
            # Update the found log entry
            cur.execute("""
                UPDATE scraping_logs 
                SET cars_scraped = %s,
                    cars_new = %s,
                    cars_updated = %s,
                    images_downloaded = %s,
                    completed_at = NOW()
                WHERE id = %s
                RETURNING id, cars_scraped, cars_new, images_downloaded
            """, (cars_scraped, cars_new, cars_updated, images_downloaded, log_id))
            
            result = cur.fetchone()
            conn.commit()
            
            if result:
                self.logger.info(f"✅ Successfully updated ScrapingLog ID: {result[0]}")
                self.logger.info(f"   Final stats: {result[1]} scraped, {result[2]} new, {result[3]} images")
            else:
                self.logger.error("❌ Update returned no result - this should not happen!")
            
            cur.close()
            conn.close()
            self.logger.info("=" * 60)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to update ScrapingLog: {type(e).__name__}: {e}")
            self.logger.error("=" * 60)
    
    def fetch_page(self, url: str, use_cookies: bool = False) -> Optional[str]:
        """Fetch a page - create new session per request for thread safety"""
        session = requests.Session()
        for attempt in range(CONFIG['MAX_RETRIES']):
            try:
                cookies = self.cookies if use_cookies else {}
                response = session.get(url, headers=get_headers(), cookies=cookies, timeout=30)
                
                if response.status_code == 202:
                    if attempt == 0 and use_cookies:
                        self.logger.warning("WAF challenge detected, refreshing cookies...")
                        if self.refresh_cookies_with_playwright():
                            continue
                    return None
                
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                delay = CONFIG['RETRY_DELAY_BASE'] * (attempt + 1)
                if attempt < CONFIG['MAX_RETRIES'] - 1:
                    time.sleep(delay)
        return None
    
    def build_sorted_url(self, page: int = 1) -> str:
        url = CONFIG['SORTED_URL']
        if page > 1:
            url += f"&page={page}"
        return url
    
    def extract_listings_from_html(self, html_content: str) -> List[Dict]:
        """Extract listings from search page - uses JSON-LD ItemList"""
        listings = []
        pattern = r'<script type="application/ld\+json">\s*(\{[^<]*"@type"\s*:\s*"ItemList"[^<]*\})\s*</script>'
        match = re.search(pattern, html_content, re.DOTALL)
        
        if match:
            try:
                data = json.loads(match.group(1))
                for item in data.get('itemListElement', []):
                    url = item.get('url', '')
                    if url:
                        external_id = url.rstrip('/').split('/')[-1]
                        images = item.get('image', [])
                        image_url = ''
                        if images and isinstance(images, list):
                            for img in images:
                                if img and 'billeder.bilbasen.dk' in img:
                                    image_url = img
                                    break
                        listings.append({
                            'uri': url,
                            'image_url': image_url,
                            'external_id': external_id
                        })
            except json.JSONDecodeError:
                pass
        return listings
    
    def scrape_new_listings(self, known_ids: set) -> List[Dict]:
        """Scrape sorted listings until we hit known IDs"""
        new_listings = []
        page = 1
        consecutive_known = 0
        
        while page <= CONFIG['MAX_PAGES']:
            url = self.build_sorted_url(page)
            self.logger.info(f"Fetching page {page}...")
            
            html = self.fetch_page(url, use_cookies=True)
            if not html:
                self.logger.warning(f"Failed to fetch page {page}")
                break
            
            listings = self.extract_listings_from_html(html)
            if not listings:
                self.logger.info("No more listings found")
                break
            
            page_new = 0
            page_known = 0
            
            for listing in listings:
                if listing['external_id'] not in known_ids:
                    new_listings.append(listing)
                    page_new += 1
                    consecutive_known = 0
                else:
                    page_known += 1
                    consecutive_known += 1
            
            self.logger.info(f"  Page {page}: {page_new} new, {page_known} known (consecutive: {consecutive_known})")
            
            if consecutive_known >= CONFIG['KNOWN_ID_STOP_THRESHOLD']:
                self.logger.info(f"Hit {consecutive_known} consecutive known IDs - caught up!")
                break
            
            if len(listings) < CONFIG['ITEMS_PER_PAGE']:
                break
            
            page += 1
            time.sleep(random.uniform(*CONFIG['DELAY_BETWEEN_PAGES']))
        
        return new_listings
    
    def extract_car_details(self, html_content: str, car_url: str, listing_image_url: str) -> Optional[Dict]:
        """Extract car details from car page HTML - EXACTLY like full scraper"""
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
        
        # Vehicle details (like details_model_year, details_mileage_km, etc)
        for detail in vehicle.get('details', []):
            danish_name = detail.get('name', '')
            english_name = translate_key(danish_name)
            result[f'details_{english_name}'] = detail.get('displayValue', '')
        
        # Model information (like model_new_price, model_body_type, etc)
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
        if 'vehicle_history_kilometers_driven' in attr:
            result['mileage_km_numeric'] = attr.get('vehicle_history_kilometers_driven')
        
        attr_mappings = {
            'vehicle_model_year': 'attr_model_year',
            'vehicle_model_fuel_type': 'attr_fuel_type',
            'vehicle_model_gear_type': 'attr_gear_type',
            'vehicle_model_weight': 'attr_weight_kg',
            'vehicle_model_effect': 'attr_power_hp',
            'vehicle_model_acc_0_to_100_kmh': 'attr_acceleration_0_100',
            'vehicle_model_top_speed': 'attr_top_speed_kmh',
            'vehicle_color_name': 'attr_color',
        }
        
        for attr_key, result_key in attr_mappings.items():
            if attr_key in attr:
                result[result_key] = attr[attr_key]
        
        # Image
        result['image_url'] = listing_image_url
        result['image_filename'] = f"{result['external_id']}.jpg" if result['external_id'] else ''
        
        return result
    
    def process_car_to_db_format(self, row: Dict) -> Optional[Dict]:
        """Convert raw scraped data to database format - EXACTLY like import_csv_to_db.py"""
        try:
            url = safe_value(row.get('url'))
            external_id = str(row.get('external_id')) if row.get('external_id') else None
            brand = safe_value(row.get('brand'))
            model = safe_value(row.get('model'))
            
            if not url or not brand or not model:
                return None
            
            price = clean_price(row.get('price'))
            if not price or price <= 0:
                return None
            
            variant = safe_value(row.get('variant'))
            title = safe_value(row.get('title'))
            description = safe_value(row.get('description'))
            
            # Fuel type first (needed for transmission)
            fuel_type = standardize_fuel_type(row.get('details_fuel_type'))
            
            new_price = clean_price(row.get('model_new_price'))
            model_year = clean_numeric(row.get('details_model_year'))
            year = model_year
            mileage = clean_numeric(row.get('details_mileage_km'), ' km')
            
            transmission = standardize_transmission(row.get('details_geartype') or row.get('details_transmission_type'), fuel_type)
            body_type = standardize_body_type(row.get('model_body_type'))
            drive_type = standardize_drive_type(row.get('model_drive_type'))
            
            # Power
            power_str = row.get('details_power_hp_nm')
            horsepower = extract_horsepower(power_str)
            if not horsepower:
                horsepower = clean_numeric(row.get('attr_power_hp'))
            torque_nm = extract_torque(power_str)
            engine_size = extract_engine_size(power_str)
            
            # Performance
            acceleration_val = row.get('details_acceleration_0_100')
            if not acceleration_val or acceleration_val == '-':
                acceleration_val = row.get('attr_acceleration_0_100')
            acceleration = clean_float(acceleration_val)
            
            top_speed_val = row.get('details_top_speed')
            if not top_speed_val or top_speed_val == '-':
                top_speed_val = row.get('attr_top_speed_kmh')
            top_speed = clean_numeric(top_speed_val)
            
            # Body
            doors = clean_numeric(row.get('model_doors'))
            if doors and (doors < 2 or doors > 5):
                doors = 4
            seats = 5
            gear_count = clean_numeric(row.get('details_number_of_gears'))
            cylinders = clean_numeric(row.get('model_cylinders'))
            
            # Registration
            first_registration = safe_value(row.get('details_first_registration'))
            production_date = safe_value(row.get('details_production_year'))
            
            # EV fields
            range_val = row.get('details_range_km')
            if range_val and range_val != '-':
                range_val = str(range_val).replace('(NEDC)', '').replace('(WLTP)', '').replace('km', '').strip()
            range_km = clean_numeric(range_val)
            
            battery_capacity = clean_float(row.get('details_battery_capacity_kwh'))
            energy_consumption = clean_numeric(row.get('details_energy_consumption'))
            home_charging_ac = safe_value(row.get('details_home_charging_ac'))
            fast_charging_dc = safe_value(row.get('details_fast_charging_dc'))
            charging_time_dc = safe_value(row.get('details_charging_time_dc_10_80_pct'))
            
            # Fuel/emissions
            fuel_consumption = safe_value(row.get('details_fuel_consumption'))
            co2_emission = safe_value(row.get('details_co2_emission') or row.get('details_co2_udledning'))
            euro_norm = safe_value(row.get('details_euro_norm'))
            
            # Tank
            tank_val = row.get('model_tank_capacity_l') or row.get('model_tankkapacitet')
            if tank_val and tank_val != '-':
                tank_val = str(tank_val).replace('l', '').replace('L', '').strip()
            tank_capacity = clean_numeric(tank_val)
            
            # Color
            color_val = row.get('details_color')
            if not color_val or color_val == '-':
                color_val = row.get('attr_color')
            color = safe_value(color_val)
            
            equipment = safe_value(row.get('equipment'))
            category = safe_value(row.get('model_category'))
            periodic_tax = safe_value(row.get('details_periodic_tax'))
            
            # Booleans
            abs_brakes = clean_boolean(row.get('model_abs_brakes'))
            esp = clean_boolean(row.get('model_esp'))
            airbags = clean_numeric(row.get('model_airbags'))
            
            # Dimensions
            weight_val = row.get('model_weight_kg')
            if not weight_val or weight_val == '-':
                weight_val = row.get('attr_weight_kg')
            weight = clean_numeric(weight_val)
            
            width = clean_numeric(row.get('model_width_cm'))
            length = clean_numeric(row.get('model_length_cm'))
            height = clean_numeric(row.get('model_height_cm'))
            trunk_size = clean_numeric(row.get('model_trunk_size'))
            load_capacity = clean_numeric(row.get('model_load_capacity_kg'))
            
            towing_val = row.get('details_towing_capacity')
            if not towing_val or towing_val == '-':
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
            
            # Image - will be set after download
            image_filename = row.get('image_filename')
            image_path = f"images/{image_filename}" if image_filename else None
            
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
                'engine_size': engine_size,
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
                'image_downloaded': False,
                'source_url': url,
                'location': location,
                'dealer_name': safe_value(row.get('seller_name')),
                'image_url': row.get('image_url', '')
            }
        except Exception as e:
            return None
    
    def download_image(self, image_url: str, save_path: str) -> bool:
        """Download image to local file"""
        if not image_url:
            return False
        try:
            session = requests.Session()
            session.headers.update(get_headers())
            response = session.get(image_url, timeout=30, stream=True)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except:
            return False
    
    def process_single_car(self, listing: Dict, images_dir: str) -> Optional[Dict]:
        """Process a single car - fetch, extract, download image"""
        car_url = listing['uri']
        image_url = listing.get('image_url', '')
        external_id = listing.get('external_id', '')
        
        # Fetch page
        html = self.fetch_page(car_url, use_cookies=False)
        if not html:
            self.stats.increment('error_count')
            return None
        
        # Extract raw details (like full scraper)
        raw_details = self.extract_car_details(html, car_url, image_url)
        if not raw_details:
            self.stats.increment('error_count')
            return None
        
        # Convert to database format (like import script)
        car_data = self.process_car_to_db_format(raw_details)
        if not car_data:
            self.stats.increment('skipped_count')
            return None
        
        # Download image
        if image_url and external_id:
            img_path = os.path.join(images_dir, f"{external_id}.jpg")
            if not os.path.exists(img_path):
                if self.download_image(image_url, img_path):
                    car_data['image_downloaded'] = True
                    self.stats.increment('image_downloaded')
                else:
                    car_data['image_downloaded'] = False
                    self.stats.increment('image_failed')
            else:
                car_data['image_downloaded'] = True
        
        # Small delay
        time.sleep(random.uniform(*CONFIG['DELAY_BETWEEN_CARS']))
        
        return car_data
    
    def insert_car_to_db(self, car_data: Dict) -> bool:
        """Insert a single car to database"""
        if self.dry_run:
            return True
        
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            
            # Check if exists
            cur.execute("SELECT id FROM cars WHERE external_id = %s", (car_data['external_id'],))
            if cur.fetchone():
                cur.close()
                conn.close()
                return False
            
            # Remove image_url before insert (not a DB column)
            insert_data = {k: v for k, v in car_data.items() if k != 'image_url'}
            
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
            """, insert_data)
            
            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            self.logger.error(f"DB insert error: {e}")
            return False
    
    def run(self, max_new_listings: int = None) -> Dict:
        """Run the incremental scraper"""
        start_time = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info("INCREMENTAL SCRAPER v3")
        self.logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Workers: {CONFIG['MAX_WORKERS']}")
        if self.dry_run:
            self.logger.info("MODE: DRY RUN")
        self.logger.info("=" * 60)
        
        # Get known IDs
        known_ids = self.get_known_external_ids()
        
        if not known_ids and not self.dry_run:
            self.logger.warning("No existing cars. Run full scraper first!")
            return {'new_count': 0, 'skipped_count': 0, 'error_count': 0, 'duration': 0}
        
        # Phase 1: Find new listings
        self.logger.info("\n--- Phase 1: Finding new listings ---")
        new_listings = self.scrape_new_listings(known_ids)
        
        self.logger.info(f"\nTotal new listings: {len(new_listings)}")
        
        if max_new_listings and len(new_listings) > max_new_listings:
            new_listings = new_listings[:max_new_listings]
            self.logger.info(f"Limited to: {len(new_listings)}")
        
        if not new_listings:
            duration = time.time() - start_time
            self.logger.info(f"\nNo new listings. Duration: {duration:.1f}s")
            return {'new_count': 0, 'skipped_count': 0, 'error_count': 0, 'duration': duration}
        
        # Phase 2: Extract details with threading
        self.logger.info(f"\n--- Phase 2: Extracting details ({CONFIG['MAX_WORKERS']} threads) ---")
        
        images_dir = CONFIG['IMAGES_DIR']
        os.makedirs(images_dir, exist_ok=True)
        
        processed_cars = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=CONFIG['MAX_WORKERS']) as executor:
            future_to_listing = {
                executor.submit(self.process_single_car, listing, images_dir): listing
                for listing in new_listings
            }
            
            for future in as_completed(future_to_listing):
                listing = future_to_listing[future]
                completed += 1
                
                try:
                    car_data = future.result()
                    if car_data:
                        processed_cars.append(car_data)
                    
                    if completed % 20 == 0 or completed == len(new_listings):
                        self.logger.info(f"Progress: {completed}/{len(new_listings)} ({len(processed_cars)} valid)")
                except Exception as e:
                    self.logger.error(f"Error: {e}")
                    self.stats.increment('error_count')
        
        # Phase 3: Insert to database
        self.logger.info(f"\n--- Phase 3: Inserting {len(processed_cars)} cars ---")
        
        inserted = 0
        for car_data in processed_cars:
            if self.insert_car_to_db(car_data):
                inserted += 1
        
        self.stats.new_count = inserted
        
        duration = time.time() - start_time
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("COMPLETED")
        self.logger.info(f"New cars inserted: {self.stats.new_count}")
        self.logger.info(f"Skipped (invalid): {self.stats.skipped_count}")
        self.logger.info(f"Errors: {self.stats.error_count}")
        self.logger.info(f"Images downloaded: {self.stats.image_downloaded}")
        self.logger.info(f"Images failed: {self.stats.image_failed}")
        self.logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f}m)")
        if duration > 0:
            self.logger.info(f"Speed: {len(new_listings)/duration:.1f} listings/sec")
        self.logger.info("=" * 60)
        
        # Update ScrapingLog in database with final statistics
        if not self.dry_run:
            self.update_scraping_log(
                cars_scraped=len(new_listings),
                cars_new=self.stats.new_count,
                cars_updated=0,  # Incremental scraper only adds new cars
                images_downloaded=self.stats.image_downloaded
            )
        
        return {
            'new_count': self.stats.new_count,
            'skipped_count': self.stats.skipped_count,
            'error_count': self.stats.error_count,
            'duration': duration
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Bilbasen Incremental Scraper v3')
    parser.add_argument('--test', action='store_true', help='Test mode (10 listings)')
    parser.add_argument('--dry-run', action='store_true', help='No database changes')
    parser.add_argument('--max', type=int, default=None, help='Max listings')
    parser.add_argument('--workers', type=int, default=None, help='Worker threads')
    parser.add_argument('--refresh-cookies', action='store_true', help='Force refresh cookies')
    args = parser.parse_args()
    
    if args.workers:
        CONFIG['MAX_WORKERS'] = args.workers
    
    logger = setup_logging()
    scraper = IncrementalScraper(logger, dry_run=args.dry_run)
    
    if args.refresh_cookies:
        logger.info("Refreshing cookies...")
        if not scraper.refresh_cookies_with_playwright():
            sys.exit(1)
    
    max_listings = 10 if args.test else args.max
    result = scraper.run(max_new_listings=max_listings)
    
    sys.exit(1 if result['error_count'] > result['new_count'] else 0)


if __name__ == '__main__':
    main()
