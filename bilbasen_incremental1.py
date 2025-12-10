#!/usr/bin/env python3
"""
Bilbasen Incremental Scraper v2 - Multi-threaded
Scrapes new listings from bilbasen.dk, sorted by newest first.
Uses cookies to bypass WAF protection, multi-threading for speed.
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
    # Base URL with sorting (newest first)
    'SORTED_URL': 'https://www.bilbasen.dk/brugt/bil?includeengroscvr=true&includeleasing=false&sortby=date&sortorder=desc',
    'BASE_URL': 'https://www.bilbasen.dk/brugt/bil',
    
    'ITEMS_PER_PAGE': 30,
    'MAX_PAGES': 200,  # Safety limit
    
    # Delays for search pages (be respectful)
    'DELAY_BETWEEN_PAGES': (0.5, 1.0),
    
    # Multi-threading
    'MAX_WORKERS': 8,  # Parallel detail extraction threads
    
    # Retry settings
    'MAX_RETRIES': 3,
    'RETRY_DELAY_BASE': 5,
    
    # Stop when we hit this many consecutive known IDs
    'KNOWN_ID_STOP_THRESHOLD': 60,
    
    # Cookie file for WAF token
    'COOKIE_FILE': 'bilbasen_cookies.json',
    
    # Output
    'LOG_DIR': 'logs',
    'IMAGES_DIR': 'images',
}

# ============================================================================
# DATA MAPPINGS (same as import script)
# ============================================================================

FUEL_TYPE_MAP = {
    'benzin': 'petrol', 'petrol': 'petrol', 'gasoline': 'petrol',
    'diesel': 'diesel',
    'el': 'electric', 'electric': 'electric', 'elektrisk': 'electric',
    'hybrid': 'hybrid', 'benzin/hybrid': 'hybrid', 'diesel/hybrid': 'hybrid',
    'plugin-hybrid': 'plugin_hybrid', 'plug-in hybrid': 'plugin_hybrid', 'plug-in-hybrid': 'plugin_hybrid',
    'benzin/plugin-hybrid': 'plugin_hybrid', 'diesel/plugin-hybrid': 'plugin_hybrid',
    'brint': 'hydrogen', 'hydrogen': 'hydrogen',
    'gas': 'gas', 'lpg': 'gas', 'cng': 'gas', 'naturgas': 'gas',
    'benzin/gas': 'gas', 'benzin/naturgas': 'gas',
}

TRANSMISSION_MAP = {
    'automatisk': 'automatic', 'automatic': 'automatic', 'auto': 'automatic', 'aut': 'automatic', 'aut.': 'automatic',
    'manuel': 'manual', 'manual': 'manual', 'man': 'manual', 'man.': 'manual',
    'cvt': 'cvt', 'variabel': 'cvt', 'trinløs': 'cvt',
    'semi-automatisk': 'semi_automatic', 'semi-automatic': 'semi_automatic', 'semi': 'semi_automatic',
    'dsg': 'automatic', 's-tronic': 'automatic', 'tiptronic': 'automatic',
    'dct': 'automatic', 'pdk': 'automatic', 'steptronic': 'automatic',
}

BODY_TYPE_MAP = {
    'sedan': 'sedan', 'berlina': 'sedan', 'saloon': 'sedan',
    'stationcar': 'station_wagon', 'station wagon': 'station_wagon', 'stcar': 'station_wagon', 'kombi': 'station_wagon', 'st.car': 'station_wagon', 'touring': 'station_wagon', 'avant': 'station_wagon', 'variant': 'station_wagon', 'sportswagon': 'station_wagon', 'sports tourer': 'station_wagon',
    'hatchback': 'hatchback', 'hatch': 'hatchback', 'coupe/hatchback': 'hatchback', '3-dørs': 'hatchback', '5-dørs': 'hatchback',
    'suv': 'suv', 'offroad': 'suv', 'crossover': 'suv', 'cuv': 'suv', 'terrænbil': 'suv', 'all terrain': 'suv',
    'coupé': 'coupe', 'coupe': 'coupe', 'coupã©': 'coupe',
    'cabriolet': 'convertible', 'convertible': 'convertible', 'cab': 'convertible', 'roadster': 'convertible', 'spider': 'convertible', 'spyder': 'convertible',
    'van': 'van', 'minibus': 'van', 'bus': 'van',
    'mpv': 'mpv', 'mini mpv': 'mpv', 'kompakt mpv': 'mpv',
    'pickup': 'pickup', 'pick-up': 'pickup',
    'kassevogn': 'van', 'varevogn': 'van', 'panel van': 'van',
}

DANISH_TO_ENGLISH = {
    'modelår': 'model_year', 'modelaar': 'model_year', 'model_aar': 'model_year',
    'første_reg': 'first_registration', 'første_registrering': 'first_registration', '1_registrering': 'first_registration', 'førstegangsregistrering': 'first_registration',
    'kilometertal': 'mileage', 'km': 'mileage', 'kilometer': 'mileage', 'km_stand': 'mileage',
    'brændstof': 'fuel_type', 'braendstof': 'fuel_type', 'brændstoftype': 'fuel_type',
    'gearkasse': 'transmission', 'gear': 'transmission', 'geartype': 'transmission',
    'karrosseri': 'body_type', 'karosseri': 'body_type', 'biltype': 'body_type', 'model_type': 'body_type',
    'hestekræfter': 'horsepower', 'hk': 'horsepower', 'hestekrafter': 'horsepower', 'effekt_hk': 'horsepower',
    'motor_størrelse': 'engine_size', 'motor_storrelse': 'engine_size', 'motorstørrelse': 'engine_size', 'ccm': 'engine_size', 'slagvolumen': 'engine_size',
    'døre': 'doors', 'doere': 'doors', 'antal_døre': 'doors',
    'farve': 'color', 'udvendig_farve': 'color', 'lakfarve': 'color',
    'pris': 'price', 'kontantpris': 'price', 'salgspris': 'price',
    'nypris': 'new_price', 'original_pris': 'new_price', 'ny_pris': 'new_price',
    'beskrivelse': 'description', 'sælger_beskrivelse': 'description',
    'titel': 'title', 'overskrift': 'title', 'annonce_titel': 'title',
    'topfart': 'top_speed', 'top_fart': 'top_speed', 'max_hastighed': 'top_speed',
    'acceleration': 'acceleration', '0_100_km_t': 'acceleration', 'acc_0_100': 'acceleration',
    'co2_udledning': 'co2_emission', 'co2': 'co2_emission', 'co2_emission': 'co2_emission', 'co2_g_km': 'co2_emission',
    'vægt': 'weight', 'vaegt': 'weight', 'egenvægt': 'weight', 'totalvægt': 'weight', 'tjenestevægt': 'weight',
    'træk': 'drive_type', 'traek': 'drive_type', 'drivhjul': 'drive_type', 'hjultræk': 'drive_type',
    'sæder': 'seats', 'saeder': 'seats', 'antal_sæder': 'seats', 'passagerer': 'seats',
    'anhængertræk': 'towing_capacity', 'anhaengertraek': 'towing_capacity', 'trailer_vægt': 'towing_capacity',
    'brændstofforbrug': 'fuel_consumption', 'braendstofforbrug': 'fuel_consumption', 'km_l': 'fuel_consumption', 'forbrug': 'fuel_consumption',
    'rækkevidde': 'range_km', 'raekkevidde': 'range_km', 'range': 'range_km', 'elektrisk_rækkevidde': 'range_km',
    'batterikapacitet': 'battery_capacity', 'batteri_kwh': 'battery_capacity', 'batteri': 'battery_capacity',
    'opladningstid': 'charging_time_dc', 'ladetid': 'charging_time_dc',
    'hjemmeoplader_ac': 'home_charging_ac', 'ac_opladning': 'home_charging_ac',
    'hurtigoplader_dc': 'fast_charging_dc', 'dc_opladning': 'fast_charging_dc',
    'energiforbrug': 'energy_consumption', 'el_forbrug': 'energy_consumption', 'kwh_100km': 'energy_consumption',
    'cylindre': 'cylinders', 'antal_cylindre': 'cylinders',
    'antal_gear': 'gear_count', 'gearantal': 'gear_count',
    'ejerafgift': 'periodic_tax', 'grøn_ejerafgift': 'periodic_tax', 'årlig_afgift': 'periodic_tax',
    'euronorm': 'euro_norm', 'euro_norm': 'euro_norm', 'emissionsnorm': 'euro_norm',
    'tankkapacitet': 'tank_capacity', 'tank_størrelse': 'tank_capacity', 'tank': 'tank_capacity',
    'abs_bremser': 'abs_brakes', 'abs': 'abs_brakes',
    'esp': 'esp', 'stabilitetskontrol': 'esp',
    'airbags': 'airbags', 'antal_airbags': 'airbags',
    'bredde': 'width', 'bredde_cm': 'width',
    'længde': 'length', 'laengde': 'length', 'længde_cm': 'length',
    'højde': 'height', 'hoejde': 'height', 'højde_cm': 'height',
    'bagagerum': 'trunk_size', 'bagagerums_størrelse': 'trunk_size', 'lastrum': 'trunk_size',
    'lasteevne': 'load_capacity', 'nyttelast': 'load_capacity',
    'sælger': 'seller_name', 'forhandler': 'seller_name', 'dealer': 'seller_name',
    'postnummer': 'seller_zipcode', 'postnr': 'seller_zipcode',
    'by': 'seller_city', 'lokation': 'seller_city',
    'sidst_opdateret': 'last_updated', 'opdateret': 'last_updated', 'senest_opdateret': 'last_updated',
}

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    """Setup logging to both file and console."""
    os.makedirs(CONFIG['LOG_DIR'], exist_ok=True)
    
    log_file = os.path.join(CONFIG['LOG_DIR'], f"incremental_{datetime.now().strftime('%Y%m%d')}.log")
    
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger('incremental_scraper')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# ============================================================================
# CLEANING FUNCTIONS
# ============================================================================

def clean_price(price_val):
    if pd.isna(price_val) or price_val == '':
        return None
    price_str = str(price_val)
    price_str = re.sub(r'[^\d,.]', '', price_str)
    price_str = price_str.replace(',', '.')
    if '.' in price_str:
        parts = price_str.rsplit('.', 1)
        if len(parts) == 2 and len(parts[1]) <= 2:
            price_str = parts[0].replace('.', '') + '.' + parts[1]
        else:
            price_str = price_str.replace('.', '')
    try:
        return float(price_str)
    except ValueError:
        return None

def clean_numeric(val):
    if pd.isna(val) or val == '' or val is None:
        return None
    val_str = str(val).strip()
    val_str = re.sub(r'[^\d,.\-]', '', val_str)
    val_str = val_str.replace(',', '.')
    if val_str.count('.') > 1:
        parts = val_str.split('.')
        val_str = ''.join(parts[:-1]) + '.' + parts[-1]
    try:
        num = float(val_str)
        if num == int(num):
            return int(num)
        return num
    except ValueError:
        return None

def standardize_fuel_type(fuel):
    if pd.isna(fuel) or fuel == '':
        return None
    fuel_lower = str(fuel).strip().lower()
    return FUEL_TYPE_MAP.get(fuel_lower, fuel_lower)

def standardize_transmission(trans):
    if pd.isna(trans) or trans == '':
        return None
    trans_lower = str(trans).strip().lower()
    return TRANSMISSION_MAP.get(trans_lower, trans_lower)

def standardize_body_type(body):
    if pd.isna(body) or body == '':
        return None
    body_lower = str(body).strip().lower()
    return BODY_TYPE_MAP.get(body_lower, body_lower)

def clean_mileage(mileage_val):
    if pd.isna(mileage_val) or mileage_val == '':
        return None
    mileage_str = re.sub(r'[^\d]', '', str(mileage_val))
    if mileage_str:
        val = int(mileage_str)
        if val > 10000000:
            val = val // 1000
        return val
    return None

def clean_year(year_val):
    if pd.isna(year_val) or year_val == '':
        return None
    year_str = str(year_val).strip()
    match = re.search(r'(19|20)\d{2}', year_str)
    if match:
        year = int(match.group())
        if 1900 <= year <= datetime.now().year + 1:
            return year
    return None

def extract_horsepower(power_str):
    if pd.isna(power_str) or power_str == '':
        return None
    match = re.search(r'(\d+)\s*hk', str(power_str).lower())
    if match:
        return int(match.group(1))
    match = re.search(r'(\d+)\s*(?:hp|ps|kw)', str(power_str).lower())
    if match:
        val = int(match.group(1))
        if 'kw' in str(power_str).lower():
            val = int(val * 1.36)
        return val
    return clean_numeric(power_str)

def extract_torque(power_str):
    if pd.isna(power_str) or power_str == '':
        return None
    match = re.search(r'(\d+)\s*nm', str(power_str).lower())
    if match:
        return int(match.group(1))
    return None

def clean_boolean(value):
    if pd.isna(value) or value == '':
        return None
    value_lower = str(value).strip().lower()
    if value_lower in ['ja', 'yes', 'true', '1']:
        return True
    if value_lower in ['nej', 'no', 'false', '0']:
        return False
    return None

def safe_value(val):
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
    normalized = danish_key.lower().strip()
    normalized = re.sub(r'[^a-zæøå0-9\s_-]', '', normalized)
    normalized = normalized.replace(' ', '_').replace('-', '_')
    if normalized in DANISH_TO_ENGLISH:
        return DANISH_TO_ENGLISH[normalized]
    if danish_key.lower() in DANISH_TO_ENGLISH:
        return DANISH_TO_ENGLISH[danish_key.lower()]
    return normalized

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
        self.session = requests.Session()
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'car_prediction'),
            'user': os.getenv('POSTGRES_USER', 'bpr_user'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
        }
        self.cookies = {}
        self.stats = Stats()
        self.load_cookies()
    
    def get_headers(self) -> Dict[str, str]:
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'da-DK,da;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
    
    def load_cookies(self):
        cookie_path = CONFIG['COOKIE_FILE']
        if os.path.exists(cookie_path):
            try:
                with open(cookie_path, 'r') as f:
                    self.cookies = json.load(f)
                self.logger.info(f"Loaded cookies from {cookie_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load cookies: {e}")
                self.cookies = {}
    
    def save_cookies(self):
        cookie_path = CONFIG['COOKIE_FILE']
        try:
            with open(cookie_path, 'w') as f:
                json.dump(self.cookies, f)
            self.logger.info(f"Saved cookies to {cookie_path}")
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
            self.logger.error("Playwright not installed. Run: pip install playwright && playwright install chromium")
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
    
    def fetch_page(self, url: str, use_cookies: bool = True) -> Optional[str]:
        for attempt in range(CONFIG['MAX_RETRIES']):
            try:
                cookies = self.cookies if use_cookies else {}
                response = self.session.get(url, headers=self.get_headers(), cookies=cookies, timeout=30)
                
                if response.status_code == 202:
                    if attempt == 0:
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
        listings = []
        pattern = r'<script type="application/ld\+json">\s*(\{[^<]*"@type"\s*:\s*"ItemList"[^<]*\})\s*</script>'
        match = re.search(pattern, html_content, re.DOTALL)
        
        if match:
            try:
                data = json.loads(match.group(1))
                items = data.get('itemListElement', [])
                
                for item in items:
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
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parse error: {e}")
        
        return listings
    
    def scrape_new_listings(self, known_ids: set) -> List[Dict]:
        new_listings = []
        page = 1
        consecutive_known = 0
        
        while page <= CONFIG['MAX_PAGES']:
            url = self.build_sorted_url(page)
            self.logger.info(f"Fetching page {page}...")
            
            html = self.fetch_page(url)
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
    
    def extract_car_details(self, html_content: str, url: str, image_url: str = '') -> Optional[Dict]:
        """Extract car details from individual car page."""
        try:
            data = {'url': url, 'image_url': image_url}
            
            # Extract from var _props = {...} (current bilbasen format)
            props_pattern = r'var\s+_props\s*=\s*(\{.+?\});\s*(?:var|</script>)'
            props_match = re.search(props_pattern, html_content, re.DOTALL)
            
            if props_match:
                try:
                    props_data = json.loads(props_match.group(1))
                    listing = props_data.get('listing', {})
                    
                    if listing:
                        # Basic info
                        data['external_id'] = str(listing.get('externalId', ''))
                        
                        # Vehicle info
                        vehicle = listing.get('vehicle', {})
                        data['brand'] = vehicle.get('make')
                        data['model'] = vehicle.get('model')
                        data['variant'] = vehicle.get('variant')
                        data['model_year'] = vehicle.get('modelYear')
                        data['color'] = vehicle.get('color')
                        
                        # Price
                        price_info = listing.get('price', {})
                        price_str = price_info.get('displayValue', '')
                        data['price'] = clean_price(price_str)
                        
                        # Description
                        data['description'] = listing.get('description')
                        data['title'] = f"{vehicle.get('make', '')} {vehicle.get('model', '')} {vehicle.get('variant', '')}".strip()
                        
                        # Properties/specs
                        props_list = listing.get('props', [])
                        if isinstance(props_list, list):
                            for prop in props_list:
                                name = prop.get('name', '')
                                value = prop.get('value') or prop.get('displayValue', '')
                                english_key = translate_key(name)
                                data[f'attr_{english_key}'] = value
                        
                        # Details section
                        details = listing.get('details', {})
                        if isinstance(details, dict):
                            for key, value in details.items():
                                english_key = translate_key(key)
                                data[f'details_{english_key}'] = value
                        
                        # Specs from vehicle
                        specs = vehicle.get('specs', {})
                        if isinstance(specs, dict):
                            for key, value in specs.items():
                                english_key = translate_key(key)
                                data[f'spec_{english_key}'] = value
                        
                        # Equipment
                        equipment_list = listing.get('equipment', [])
                        if isinstance(equipment_list, list):
                            data['equipment'] = ', '.join(str(e.get('name', e)) if isinstance(e, dict) else str(e) for e in equipment_list)
                        
                        # Seller info
                        seller = listing.get('seller', {})
                        if seller:
                            data['seller_name'] = seller.get('name')
                            location = seller.get('location', {})
                            if isinstance(location, dict):
                                data['seller_city'] = location.get('city')
                                data['seller_zipcode'] = location.get('zipCode')
                        
                        # Media/images
                        media = listing.get('media', [])
                        if media and isinstance(media, list) and not data.get('image_url'):
                            for m in media:
                                if isinstance(m, dict) and m.get('url'):
                                    data['image_url'] = m.get('url')
                                    break
                        
                        self.logger.debug(f"Extracted from _props: {data.get('brand')} {data.get('model')}")
                        return data
                        
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse _props JSON: {e}")
            
            # Fallback: try old __NEXT_DATA__ format
            next_data_pattern = r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>'
            next_match = re.search(next_data_pattern, html_content, re.DOTALL)
            
            if next_match:
                try:
                    next_data = json.loads(next_match.group(1))
                    page_props = next_data.get('props', {}).get('pageProps', {})
                    listing_data = page_props.get('listing', {})
                    
                    if listing_data:
                        data['brand'] = listing_data.get('make') or listing_data.get('brand')
                        data['model'] = listing_data.get('model')
                        data['variant'] = listing_data.get('variant')
                        data['price'] = listing_data.get('price')
                        data['description'] = listing_data.get('description')
                        
                        props = listing_data.get('props', {})
                        for key, value in props.items():
                            english_key = translate_key(key)
                            data[f'attr_{english_key}'] = value
                        
                        return data
                except json.JSONDecodeError:
                    pass
            
            self.logger.warning(f"Could not extract data from page: {url}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting car details: {e}")
            return None
    
    def process_car_data(self, raw_data: Dict) -> Optional[Dict]:
        """Process raw data into database format."""
        try:
            url = raw_data.get('url', '')
            # Use external_id from _props if available, otherwise extract from URL
            external_id = raw_data.get('external_id') or (url.rstrip('/').split('/')[-1] if url else None)
            
            brand = safe_value(raw_data.get('brand'))
            model = safe_value(raw_data.get('model'))
            
            if not brand:
                title = raw_data.get('title', '')
                if title:
                    parts = title.split()
                    if len(parts) >= 1:
                        brand = parts[0]
                    if len(parts) >= 2:
                        model = parts[1]
            
            if not brand or not external_id:
                return None
            
            variant = safe_value(raw_data.get('variant') or raw_data.get('model_variant'))
            title = safe_value(raw_data.get('title'))
            description = safe_value(raw_data.get('description'))
            
            price = clean_price(raw_data.get('price') or raw_data.get('model_price'))
            new_price = clean_price(raw_data.get('new_price') or raw_data.get('model_new_price') or raw_data.get('attr_new_price'))
            
            model_year = clean_year(raw_data.get('model_year') or raw_data.get('attr_model_year') or raw_data.get('model_model_year'))
            year = model_year
            
            first_reg = safe_value(raw_data.get('first_registration') or raw_data.get('model_first_registration') or raw_data.get('attr_first_registration'))
            production_date = safe_value(raw_data.get('production_date') or raw_data.get('model_production_date'))
            
            mileage = clean_mileage(raw_data.get('mileage') or raw_data.get('model_mileage') or raw_data.get('attr_mileage') or raw_data.get('model_km'))
            
            fuel_type = standardize_fuel_type(raw_data.get('fuel_type') or raw_data.get('model_fuel_type') or raw_data.get('attr_fuel_type'))
            transmission = standardize_transmission(raw_data.get('transmission') or raw_data.get('model_transmission') or raw_data.get('attr_transmission') or raw_data.get('model_gear'))
            gear_count = clean_numeric(raw_data.get('gear_count') or raw_data.get('model_gear_count'))
            cylinders = clean_numeric(raw_data.get('cylinders') or raw_data.get('model_cylinders'))
            
            body_type = standardize_body_type(raw_data.get('body_type') or raw_data.get('model_body_type') or raw_data.get('attr_body_type') or raw_data.get('model_model_type'))
            
            power_str = raw_data.get('horsepower') or raw_data.get('model_horsepower') or raw_data.get('attr_horsepower') or raw_data.get('model_effect_hk') or raw_data.get('model_hk')
            horsepower = extract_horsepower(power_str)
            torque_nm = extract_torque(power_str)
            
            acceleration = clean_numeric(raw_data.get('acceleration') or raw_data.get('model_acceleration') or raw_data.get('model_0_100'))
            top_speed = clean_numeric(raw_data.get('top_speed') or raw_data.get('model_top_speed') or raw_data.get('model_topfart'))
            
            drive_type = safe_value(raw_data.get('drive_type') or raw_data.get('model_drive_type') or raw_data.get('attr_drive_type') or raw_data.get('model_traek'))
            
            doors = clean_numeric(raw_data.get('doors') or raw_data.get('model_doors') or raw_data.get('attr_doors'))
            seats = clean_numeric(raw_data.get('seats') or raw_data.get('model_seats') or raw_data.get('attr_seats'))
            
            color = safe_value(raw_data.get('color') or raw_data.get('model_color') or raw_data.get('attr_color'))
            category = safe_value(raw_data.get('category') or raw_data.get('model_category'))
            
            equipment_raw = raw_data.get('equipment') or raw_data.get('model_equipment') or raw_data.get('attr_equipment')
            if isinstance(equipment_raw, list):
                equipment = ', '.join(str(e) for e in equipment_raw)
            else:
                equipment = safe_value(equipment_raw)
            
            range_km = clean_numeric(raw_data.get('range_km') or raw_data.get('model_range') or raw_data.get('model_range_km') or raw_data.get('model_raekkevide'))
            battery_capacity = clean_numeric(raw_data.get('battery_capacity') or raw_data.get('model_battery_capacity') or raw_data.get('model_batteri_kapacitet'))
            energy_consumption = clean_numeric(raw_data.get('energy_consumption') or raw_data.get('model_energy_consumption') or raw_data.get('model_energiforbrug'))
            home_charging_ac = clean_numeric(raw_data.get('home_charging_ac') or raw_data.get('model_home_charging_ac'))
            fast_charging_dc = clean_numeric(raw_data.get('fast_charging_dc') or raw_data.get('model_fast_charging_dc'))
            charging_time_dc = clean_numeric(raw_data.get('charging_time_dc') or raw_data.get('model_charging_time'))
            
            fuel_consumption = clean_numeric(raw_data.get('fuel_consumption') or raw_data.get('model_fuel_consumption') or raw_data.get('model_forbrug'))
            co2_emission = clean_numeric(raw_data.get('co2_emission') or raw_data.get('model_co2') or raw_data.get('model_co2_emission'))
            euro_norm = safe_value(raw_data.get('euro_norm') or raw_data.get('model_euro_norm'))
            tank_capacity = clean_numeric(raw_data.get('tank_capacity') or raw_data.get('model_tank_capacity') or raw_data.get('model_tank'))
            
            periodic_tax = clean_numeric(raw_data.get('periodic_tax') or raw_data.get('model_ejerafgift') or raw_data.get('attr_periodic_tax'))
            
            image_url = raw_data.get('image_url', '')
            image_filename = f"{external_id}.jpg" if external_id and image_url else None
            image_path = f"images/{image_filename}" if image_filename else None
            
            abs_brakes = clean_boolean(raw_data.get('model_abs_brakes'))
            esp = clean_boolean(raw_data.get('model_esp'))
            airbags = clean_numeric(raw_data.get('model_airbags'))
            
            weight = clean_numeric(raw_data.get('model_weight_kg')) or clean_numeric(raw_data.get('attr_weight_kg'))
            width = clean_numeric(raw_data.get('model_width_cm'))
            length = clean_numeric(raw_data.get('model_length_cm'))
            height = clean_numeric(raw_data.get('model_height_cm'))
            trunk_size = clean_numeric(raw_data.get('model_trunk_size'))
            load_capacity = clean_numeric(raw_data.get('model_load_capacity_kg'))
            towing_capacity = clean_numeric(raw_data.get('details_towing_capacity')) or clean_numeric(raw_data.get('model_max_towing_with_brake'))
            max_towing_weight = clean_numeric(raw_data.get('model_max_towing_with_brake'))
            
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
                'first_registration': first_reg,
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
                'image_url': image_url,
                'image_downloaded': False,  # Will be set to True only after successful download
                'source_url': url,
                'location': location,
                'dealer_name': safe_value(raw_data.get('seller_name'))
            }
        except Exception as e:
            self.logger.error(f"Error processing car data: {e}")
            return None
    
    def download_image(self, image_url: str, save_path: str) -> bool:
        """Download image to local file."""
        if not image_url or self.dry_run:
            return False
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            response = requests.get(image_url, headers=self.get_headers(), timeout=30, stream=True)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            return False
    
    def process_single_listing(self, listing: Dict) -> Optional[Dict]:
        """Process a single listing - fetch details, download image, return car_data."""
        try:
            # Fetch car page
            html = self.fetch_page(listing['uri'], use_cookies=False)
            if not html:
                self.stats.increment('error_count')
                return None
            
            # Extract details
            raw_data = self.extract_car_details(html, listing['uri'], listing['image_url'])
            if not raw_data:
                self.stats.increment('error_count')
                return None
            
            # Process into database format
            car_data = self.process_car_data(raw_data)
            if not car_data:
                self.stats.increment('skipped_count')
                return None
            
            # Download image BEFORE setting flag
            if listing['image_url']:
                img_path = os.path.join(CONFIG['IMAGES_DIR'], f"{listing['external_id']}.jpg")
                if self.download_image(listing['image_url'], img_path):
                    car_data['image_downloaded'] = True
                    self.stats.increment('image_downloaded')
                else:
                    car_data['image_downloaded'] = False
                    self.stats.increment('image_failed')
            
            return car_data
            
        except Exception as e:
            self.stats.increment('error_count')
            return None
    
    def insert_cars_batch(self, cars: List[Dict]) -> int:
        """Insert multiple cars to database in a single transaction."""
        if self.dry_run or not cars:
            return len(cars)
        
        inserted = 0
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            
            for car_data in cars:
                try:
                    # Check if already exists
                    cur.execute("SELECT id FROM cars WHERE external_id = %s", (car_data['external_id'],))
                    if cur.fetchone():
                        continue
                    
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
                    inserted += 1
                except Exception as e:
                    self.logger.warning(f"Insert error for {car_data.get('external_id')}: {e}")
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Batch insert error: {e}")
        
        return inserted
    
    def run(self, max_new_listings: int = None) -> Dict:
        """Run the incremental scraper with multi-threading."""
        start_time = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info("INCREMENTAL SCRAPER v2 (Multi-threaded)")
        self.logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Workers: {CONFIG['MAX_WORKERS']}")
        if self.dry_run:
            self.logger.info("MODE: DRY RUN (no database changes)")
        self.logger.info("=" * 60)
        
        # Get known IDs from database
        known_ids = self.get_known_external_ids()
        
        if not known_ids and not self.dry_run:
            self.logger.warning("No existing cars in database. Run the full scraper first!")
            return {'new_count': 0, 'skipped_count': 0, 'error_count': 0, 'duration': 0}
        
        # Phase 1: Find new listings
        self.logger.info("\n--- Phase 1: Finding new listings (sorted by newest) ---")
        
        new_listings = self.scrape_new_listings(known_ids)
        
        self.logger.info(f"\nTotal new listings found: {len(new_listings)}")
        
        if max_new_listings and len(new_listings) > max_new_listings:
            new_listings = new_listings[:max_new_listings]
            self.logger.info(f"Limited to: {len(new_listings)}")
        
        if not new_listings:
            duration = time.time() - start_time
            self.logger.info(f"\nNo new listings to process. Duration: {duration:.1f}s")
            return {'new_count': 0, 'skipped_count': 0, 'error_count': 0, 'duration': duration}
        
        # Phase 2: Extract details with multi-threading
        self.logger.info(f"\n--- Phase 2: Extracting details ({CONFIG['MAX_WORKERS']} threads) ---")
        
        os.makedirs(CONFIG['IMAGES_DIR'], exist_ok=True)
        
        processed_cars = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=CONFIG['MAX_WORKERS']) as executor:
            future_to_listing = {
                executor.submit(self.process_single_listing, listing): listing
                for listing in new_listings
            }
            
            for future in as_completed(future_to_listing):
                listing = future_to_listing[future]
                completed += 1
                
                try:
                    car_data = future.result()
                    if car_data:
                        processed_cars.append(car_data)
                        if completed % 10 == 0 or completed == len(new_listings):
                            self.logger.info(f"Progress: {completed}/{len(new_listings)} ({len(processed_cars)} successful)")
                except Exception as e:
                    self.logger.error(f"Error processing {listing['external_id']}: {e}")
                    self.stats.increment('error_count')
        
        # Phase 3: Batch insert to database
        self.logger.info(f"\n--- Phase 3: Inserting {len(processed_cars)} cars to database ---")
        
        # Insert in batches of 100
        batch_size = 100
        total_inserted = 0
        
        for i in range(0, len(processed_cars), batch_size):
            batch = processed_cars[i:i + batch_size]
            inserted = self.insert_cars_batch(batch)
            total_inserted += inserted
            self.logger.info(f"  Batch {i // batch_size + 1}: inserted {inserted}/{len(batch)}")
        
        self.stats.new_count = total_inserted
        
        duration = time.time() - start_time
        
        # Summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("INCREMENTAL SCRAPER COMPLETED")
        self.logger.info(f"New cars inserted: {self.stats.new_count}")
        self.logger.info(f"Skipped (invalid): {self.stats.skipped_count}")
        self.logger.info(f"Errors: {self.stats.error_count}")
        self.logger.info(f"Images downloaded: {self.stats.image_downloaded}")
        self.logger.info(f"Images failed: {self.stats.image_failed}")
        self.logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f}m)")
        if duration > 0:
            self.logger.info(f"Speed: {len(new_listings)/duration:.1f} listings/sec")
        self.logger.info("=" * 60)
        
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
    parser = argparse.ArgumentParser(description='Bilbasen Incremental Scraper v2')
    parser.add_argument('--test', action='store_true', help='Test mode (limit to 10 new listings)')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (no database changes)')
    parser.add_argument('--max', type=int, default=None, help='Maximum new listings to process')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker threads')
    parser.add_argument('--refresh-cookies', action='store_true', help='Force refresh WAF cookies')
    args = parser.parse_args()
    
    if args.workers:
        CONFIG['MAX_WORKERS'] = args.workers
    
    logger = setup_logging()
    
    scraper = IncrementalScraper(logger, dry_run=args.dry_run)
    
    if args.refresh_cookies:
        logger.info("Forcing cookie refresh...")
        if scraper.refresh_cookies_with_playwright():
            logger.info("Cookies refreshed successfully!")
        else:
            logger.error("Failed to refresh cookies")
            sys.exit(1)
    
    max_listings = 10 if args.test else args.max
    
    result = scraper.run(max_new_listings=max_listings)
    
    if result['error_count'] > result['new_count']:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
