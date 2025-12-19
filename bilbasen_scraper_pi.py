#!/usr/bin/env python3
"""
Bilbasen Web Scraper - Raspberry Pi 5 Version
Designed to run in a screen session for multi-day scraping.

Usage:
    # Start a screen session
    screen -S bilbasen
    
    # Run the scraper
    python3 bilbasen_scraper_pi.py
    
    # Detach from screen: Ctrl+A, then D
    # Reattach: screen -r bilbasen
    
    # Check progress from another terminal:
    tail -f bilbasen_scrape/scraper.log
"""

import requests
import json
import re
import time
import os
import csv
import sys
import logging
import signal
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'FUEL_TYPES': {
        1: 'benzin',
        2: 'diesel',
        3: 'electric',
        6: 'hybrid_benzin',
        8: 'hybrid_diesel',
        11: 'plugin_benzin',
        12: 'plugin_diesel'
    },
    
    'OLD_CAR_FUEL_TYPES': [1, 2],
    'NEW_CAR_FUEL_TYPES': [1, 2, 3, 6, 8, 11, 12],
    
    'PRICE_RANGES': [
        (0, 25000),
        (25001, 50000),
        (50001, 75000),
        (75001, 100000),
        (100001, 150000),
        (150001, 200000),
        (200001, 300000),
        (300001, 500000),
        (500001, 1000000),
        (1000001, 5000000)
    ],
    
    'CURRENT_YEAR': 2025,
    'ITEMS_PER_PAGE': 30,
    'MAX_PAGES': 100,
    
    # Delays (seconds) - reduced for multi-threaded scraping
    'DELAY_BETWEEN_REQUESTS': (0.3, 0.8),
    'DELAY_BETWEEN_COMBOS': (0.5, 1.5),
    'DELAY_BETWEEN_CARS': (0.2, 0.6),
    
    # Threading
    'MAX_WORKERS_PHASE1': 8,  # Parallel workers for link scraping
    'MAX_WORKERS_PHASE2': 12,  # Parallel workers for detail scraping
    
    # Checkpoints
    'CHECKPOINT_INTERVAL_COMBOS': 10,
    'CHECKPOINT_INTERVAL_CARS': 50,  # Increased for better performance
    
    # Retry settings
    'MAX_RETRIES': 5,
    'RETRY_DELAY_BASE': 10,
    
    # Output
    'OUTPUT_DIR': 'bilbasen_scrape',
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
# LOGGING SETUP
# ============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging to both file and console."""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, 'scraper.log')
    
    # Create logger
    logger = logging.getLogger('bilbasen')
    logger.setLevel(logging.INFO)
    
    # File handler (detailed)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', 
                                     datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    
    # Console handler (less verbose)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# ============================================================================
# HELPER FUNCTIONS
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
    """Get request headers."""
    return {
        'User-Agent': 'Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'da-DK,da;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

def generate_year_ranges() -> List[Tuple[int, int]]:
    """Generate year ranges for filtering."""
    ranges = []
    for start_year in range(1980, 2000, 5):
        end_year = min(start_year + 4, 1999)
        ranges.append((start_year, end_year))
    for year in range(2000, CONFIG['CURRENT_YEAR'] + 1):
        ranges.append((year, year))
    return ranges

def get_fuel_types_for_year(year_from: int) -> List[int]:
    """Get applicable fuel types based on year."""
    if year_from < 2008:
        return CONFIG['OLD_CAR_FUEL_TYPES']
    return CONFIG['NEW_CAR_FUEL_TYPES']

def generate_all_filter_combinations() -> List[Dict]:
    """Generate all filter combinations."""
    combinations = []
    year_ranges = generate_year_ranges()
    for year_from, year_to in year_ranges:
        fuel_types = get_fuel_types_for_year(year_from)
        for fuel in fuel_types:
            for price_from, price_to in CONFIG['PRICE_RANGES']:
                combinations.append({
                    'year_from': year_from,
                    'year_to': year_to,
                    'price_from': price_from,
                    'price_to': price_to,
                    'fuel': fuel,
                    'fuel_name': CONFIG['FUEL_TYPES'][fuel]
                })
    return combinations

def build_url(fuel: int, year_from: int, year_to: int, 
              price_from: int, price_to: int, page: int = 1) -> str:
    """Build bilbasen search URL."""
    base_url = "https://www.bilbasen.dk/brugt/bil"
    params = {
        'fuel': fuel,
        'includeengroscvr': 'true',
        'includeleasing': 'false',
        'pricefrom': price_from,
        'priceto': price_to,
        'yearfrom': year_from,
        'yearto': year_to
    }
    if page > 1:
        params['page'] = page
    query_string = '&'.join(f"{k}={v}" for k, v in params.items())
    return f"{base_url}?{query_string}"

def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def estimate_remaining_time(completed: int, total: int, elapsed_seconds: float) -> str:
    """Estimate remaining time."""
    if completed == 0:
        return "calculating..."
    rate = elapsed_seconds / completed
    remaining = (total - completed) * rate
    return format_duration(remaining)

# ============================================================================
# SCRAPER CLASS
# ============================================================================

class BilbasenScraper:
    def __init__(self, output_dir: str, logger: logging.Logger):
        self.output_dir = output_dir
        self.logger = logger
        self.start_time = None
        self.stop_requested = False
        self.lock = threading.Lock()  # For thread-safe operations
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.warning(f"Received signal {signum}, requesting graceful shutdown...")
        self.stop_requested = True
    
    def fetch_page(self, url: str) -> Optional[str]:
        """Fetch a page with retry logic. Thread-safe with individual sessions."""
        # Create a session per thread for thread safety
        session = requests.Session()
        for attempt in range(CONFIG['MAX_RETRIES']):
            try:
                response = session.get(url, headers=get_headers(), timeout=30)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                delay = CONFIG['RETRY_DELAY_BASE'] * (attempt + 1)
                if attempt < CONFIG['MAX_RETRIES'] - 1:
                    time.sleep(delay)
        return None
    
    def extract_listings_from_html(self, html_content: str) -> Tuple[List[Dict], int]:
        """Extract listings from search page HTML."""
        listings = []
        total_count = 0
        
        pattern = r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>'
        match = re.search(pattern, html_content, re.DOTALL)
        
        if not match:
            return listings, total_count
        
        try:
            data = json.loads(match.group(1))
            queries = data.get('props', {}).get('pageProps', {}).get('dehydratedState', {}).get('queries', [])
            
            for query in queries:
                state_data = query.get('state', {}).get('data', {})
                if 'listings' in state_data:
                    raw_listings = state_data['listings']
                    pagination = state_data.get('pagination', {})
                    total_count = pagination.get('totalCount', 0)
                    
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
        
        return listings, total_count
    
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
        
        result['external_id'] = listing.get('externalId')
        result['url'] = listing.get('canonicalUrl', car_url)
        result['brand'] = vehicle.get('make')
        result['model'] = vehicle.get('model')
        result['variant'] = vehicle.get('variant')
        result['title'] = f"{vehicle.get('make', '')} {vehicle.get('model', '')} {vehicle.get('variant', '')}".strip()
        
        price_info = listing.get('price', {})
        result['price'] = price_info.get('displayValue', '')
        
        result['description'] = listing.get('description', '')
        
        # Vehicle details
        for detail in vehicle.get('details', []):
            danish_name = detail.get('name', '')
            english_name = translate_key(danish_name)
            result[f'details_{english_name}'] = detail.get('displayValue', '')
        
        # Model information
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
        
        # Image (from Phase 1)
        result['image_url'] = listing_image_url
        result['image_filename'] = f"{result['external_id']}.jpg" if result['external_id'] else ''
        
        return result
    
    def download_image(self, image_url: str, save_path: str) -> bool:
        """Download image to local file."""
        if not image_url:
            return False
        try:
            # Create session for this thread
            session = requests.Session()
            session.headers.update(get_headers())
            
            response = session.get(image_url, timeout=30, stream=True)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception:
            return False
    
    # ========================================================================
    # PHASE 1: Links Scraping
    # ========================================================================
    
    def save_links_checkpoint(self, completed_combos: List[int], 
                               all_listings: List[Dict], seen_uris: set):
        """Save Phase 1 checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, 'links_checkpoint.json')
        checkpoint_data = {
            'completed_combos': completed_combos,
            'seen_uris': list(seen_uris),
            'timestamp': datetime.now().isoformat()
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)
        
        # Save listings CSV
        csv_path = os.path.join(self.output_dir, 'listings.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['uri', 'image_url', 'external_id'])
            writer.writeheader()
            writer.writerows(all_listings)
    
    def load_links_checkpoint(self) -> Tuple[List[int], set, List[Dict]]:
        """Load Phase 1 checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, 'links_checkpoint.json')
        csv_path = os.path.join(self.output_dir, 'listings.csv')
        
        completed_combos = []
        seen_uris = set()
        existing_listings = []
        
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
                completed_combos = data.get('completed_combos', [])
                seen_uris = set(data.get('seen_uris', []))
            self.logger.info(f"Loaded checkpoint: {len(completed_combos)} combos, {len(seen_uris)} URIs")
        
        if os.path.exists(csv_path):
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_listings = list(reader)
            self.logger.info(f"Loaded {len(existing_listings)} existing listings")
        
        return completed_combos, seen_uris, existing_listings
    
    def scrape_filter_combination(self, filter_combo: Dict, seen_uris: set) -> List[Dict]:
        """Scrape all pages for a filter combination."""
        all_listings = []
        page = 1
        
        while page <= CONFIG['MAX_PAGES']:
            if self.stop_requested:
                break
                
            url = build_url(
                fuel=filter_combo['fuel'],
                year_from=filter_combo['year_from'],
                year_to=filter_combo['year_to'],
                price_from=filter_combo['price_from'],
                price_to=filter_combo['price_to'],
                page=page
            )
            
            html = self.fetch_page(url)
            if not html:
                break
            
            listings, total_count = self.extract_listings_from_html(html)
            
            if page == 1:
                if total_count > 0:
                    total_pages = min((total_count + CONFIG['ITEMS_PER_PAGE'] - 1) // CONFIG['ITEMS_PER_PAGE'], 
                                     CONFIG['MAX_PAGES'])
                    self.logger.debug(f"  Found {total_count} cars, {total_pages} pages")
                elif not listings:
                    break
            
            if not listings:
                break
            
            new_listings = []
            for listing in listings:
                if listing['uri'] not in seen_uris:
                    seen_uris.add(listing['uri'])
                    new_listings.append(listing)
            
            all_listings.extend(new_listings)
            
            if len(listings) < CONFIG['ITEMS_PER_PAGE']:
                break
            
            page += 1
            time.sleep(random.uniform(*CONFIG['DELAY_BETWEEN_REQUESTS']))
        
        return all_listings
    
    def run_phase1(self) -> List[Dict]:
        """Run Phase 1: Links scraping with multi-threading."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 1: Scraping listing URLs (Multi-threaded)")
        self.logger.info("=" * 60)
        
        self.start_time = time.time()
        
        completed_combos, seen_uris, all_listings = self.load_links_checkpoint()
        all_combos = generate_all_filter_combinations()
        
        total = len(all_combos)
        completed = len(completed_combos)
        
        self.logger.info(f"Total combinations: {total}")
        self.logger.info(f"Already completed: {completed}")
        self.logger.info(f"Remaining: {total - completed}")
        self.logger.info(f"Current listings: {len(all_listings)}")
        self.logger.info(f"Workers: {CONFIG['MAX_WORKERS_PHASE1']}")
        self.logger.info("-" * 60)
        
        try:
            with ThreadPoolExecutor(max_workers=CONFIG['MAX_WORKERS_PHASE1']) as executor:
                # Submit all pending combinations
                future_to_combo = {}
                for idx, combo in enumerate(all_combos):
                    if idx not in completed_combos:
                        future = executor.submit(self.scrape_filter_combination, combo, seen_uris)
                        future_to_combo[future] = (idx, combo)
                
                # Process completed tasks
                last_checkpoint = completed
                for future in as_completed(future_to_combo):
                    if self.stop_requested:
                        self.logger.warning("Stop requested, cancelling pending tasks...")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    
                    idx, combo = future_to_combo[future]
                    
                    try:
                        new_listings = future.result()
                        
                        # Thread-safe update of shared state
                        with self.lock:
                            all_listings.extend(new_listings)
                            completed_combos.append(idx)
                            current_completed = len(completed_combos)
                            current_total = len(all_listings)
                        
                        elapsed = time.time() - self.start_time
                        remaining_est = estimate_remaining_time(current_completed - completed, 
                                                                total - completed, elapsed)
                        
                        self.logger.info(f"[{idx+1}/{total}] Year: {combo['year_from']}-{combo['year_to']}, "
                                       f"Price: {combo['price_from']:,}-{combo['price_to']:,}, "
                                       f"Fuel: {combo['fuel_name']} | ETA: {remaining_est}")
                        self.logger.info(f"  -> New: {len(new_listings)}, Total: {current_total}")
                        
                        # Save checkpoint only when enough NEW combos completed since last checkpoint
                        if current_completed - last_checkpoint >= CONFIG['CHECKPOINT_INTERVAL_COMBOS']:
                            with self.lock:
                                self.save_links_checkpoint(completed_combos, all_listings, seen_uris)
                                self.logger.info(f"  -> Checkpoint saved")
                                last_checkpoint = current_completed
                        
                        # Slight delay between combo completions
                        time.sleep(random.uniform(*CONFIG['DELAY_BETWEEN_COMBOS']))
                        
                    except Exception as e:
                        self.logger.error(f"Error processing combo {idx+1}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error in Phase 1: {e}")
        finally:
            with self.lock:
                self.save_links_checkpoint(completed_combos, all_listings, seen_uris)
        
        elapsed = time.time() - self.start_time
        self.logger.info("=" * 60)
        self.logger.info(f"PHASE 1 COMPLETE")
        self.logger.info(f"Total listings: {len(all_listings)}")
        self.logger.info(f"Time elapsed: {format_duration(elapsed)}")
        self.logger.info("=" * 60)
        
        return all_listings
    
    # ========================================================================
    # PHASE 2: Car Details Scraping
    # ========================================================================
    
    def save_details_checkpoint(self, processed_ids: List[str], all_details: List[Dict]):
        """Save Phase 2 checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, 'details_checkpoint.json')
        checkpoint_data = {
            'processed_ids': processed_ids,
            'timestamp': datetime.now().isoformat()
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)
        
        self.save_car_details_to_csv(all_details)
    
    def load_details_checkpoint(self) -> Tuple[List[str], List[Dict]]:
        """Load Phase 2 checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, 'details_checkpoint.json')
        csv_path = os.path.join(self.output_dir, 'car_details.csv')
        
        processed_ids = []
        existing_details = []
        
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
                processed_ids = data.get('processed_ids', [])
            self.logger.info(f"Loaded checkpoint: {len(processed_ids)} cars processed")
        
        if os.path.exists(csv_path):
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_details = list(reader)
            self.logger.info(f"Loaded {len(existing_details)} existing car details")
        
        return processed_ids, existing_details
    
    def save_car_details_to_csv(self, details: List[Dict]):
        """Save car details to CSV."""
        if not details:
            return
        
        csv_path = os.path.join(self.output_dir, 'car_details.csv')
        
        priority_columns = [
            'external_id', 'url', 'brand', 'model', 'variant', 'title', 'price',
            'description', 'equipment',
            'details_model_year', 'details_first_registration', 'details_mileage_km',
            'details_fuel_type', 'details_range_km', 'details_battery_capacity_kwh',
            'details_energy_consumption', 'details_home_charging_ac', 'details_fast_charging_dc',
            'details_charging_time_dc_10_80_pct', 'details_periodic_tax', 'details_power_hp_nm',
            'details_acceleration_0_100', 'details_top_speed', 'details_towing_capacity',
            'details_color', 'details_fuel_consumption', 'details_co2_emission',
            'details_euro_norm', 'details_transmission_type', 'details_number_of_gears',
            'model_new_price', 'model_category', 'model_body_type', 'model_trunk_size',
            'model_weight_kg', 'model_width_cm', 'model_length_cm', 'model_height_cm',
            'model_load_capacity_kg', 'model_max_towing_with_brake', 'model_drive_type',
            'model_abs_brakes', 'model_esp', 'model_airbags', 'model_doors',
            'seller_name', 'seller_type', 'seller_city', 'seller_zipcode',
            'listing_date', 'last_updated', 'registration_date',
            'image_url', 'image_filename'
        ]
        
        all_keys = set()
        for d in details:
            all_keys.update(d.keys())
        
        fieldnames = [col for col in priority_columns if col in all_keys]
        remaining = sorted([col for col in all_keys if col not in priority_columns])
        fieldnames.extend(remaining)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(details)
    
    def process_single_car(self, listing: Dict, images_dir: str, download_images: bool) -> Optional[Dict]:
        """Process a single car listing. Thread-safe method."""
        car_url = listing['uri']
        image_url = listing.get('image_url', '')
        external_id = listing.get('external_id', '')
        
        html = self.fetch_page(car_url)
        if not html:
            self.logger.warning(f"  Failed to fetch {external_id}")
            return None
        
        details = self.extract_car_details(html, car_url, image_url)
        
        # Download image if requested
        if details and download_images and image_url:
            img_path = os.path.join(images_dir, f"{external_id}.jpg")
            if not os.path.exists(img_path):
                self.download_image(image_url, img_path)
        
        # Small delay to be respectful
        time.sleep(random.uniform(*CONFIG['DELAY_BETWEEN_CARS']))
        
        return details
    
    def run_phase2(self, download_images: bool = True) -> List[Dict]:
        """Run Phase 2: Car details scraping with multi-threading."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 2: Extracting car details (Multi-threaded)")
        self.logger.info("=" * 60)
        
        self.start_time = time.time()
        
        # Load listings from Phase 1
        listings_csv = os.path.join(self.output_dir, 'listings.csv')
        if not os.path.exists(listings_csv):
            self.logger.error("listings.csv not found. Run Phase 1 first.")
            return []
        
        with open(listings_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            listings = list(reader)
        
        self.logger.info(f"Total listings: {len(listings)}")
        
        # Create images directory
        images_dir = os.path.join(self.output_dir, 'images')
        if download_images:
            os.makedirs(images_dir, exist_ok=True)
        
        # Load checkpoint
        processed_ids, all_details = self.load_details_checkpoint()
        processed_set = set(processed_ids)
        
        remaining = [l for l in listings if l.get('external_id') not in processed_set]
        total = len(remaining)
        
        self.logger.info(f"Already processed: {len(processed_ids)}")
        self.logger.info(f"Remaining: {total}")
        self.logger.info(f"Workers: {CONFIG['MAX_WORKERS_PHASE2']}")
        self.logger.info("-" * 60)
        
        try:
            with ThreadPoolExecutor(max_workers=CONFIG['MAX_WORKERS_PHASE2']) as executor:
                # Submit all pending listings
                future_to_listing = {}
                for listing in remaining:
                    future = executor.submit(self.process_single_car, listing, images_dir, download_images)
                    future_to_listing[future] = listing
                
                # Process completed tasks
                completed_count = 0
                last_checkpoint = 0
                for future in as_completed(future_to_listing):
                    if self.stop_requested:
                        self.logger.warning("Stop requested, cancelling pending tasks...")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    
                    listing = future_to_listing[future]
                    external_id = listing.get('external_id', '')
                    
                    try:
                        details = future.result()
                        
                        # Thread-safe update
                        with self.lock:
                            if details:
                                all_details.append(details)
                            processed_ids.append(external_id)
                            completed_count += 1
                            current_total = len(all_details)
                        
                        elapsed = time.time() - self.start_time
                        remaining_est = estimate_remaining_time(completed_count, total, elapsed)
                        
                        if completed_count % 100 == 0 or completed_count == 1:
                            self.logger.info(f"[{completed_count}/{total}] Processing... | ETA: {remaining_est}")
                        
                        # Save checkpoint only when we've processed enough NEW cars since last checkpoint
                        if completed_count - last_checkpoint >= CONFIG['CHECKPOINT_INTERVAL_CARS']:
                            with self.lock:
                                self.save_details_checkpoint(processed_ids, all_details)
                                self.logger.info(f"  -> Checkpoint saved ({current_total} cars)")
                                last_checkpoint = completed_count
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {external_id}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error in Phase 2: {e}")
        finally:
            with self.lock:
                self.save_details_checkpoint(processed_ids, all_details)
        
        elapsed = time.time() - self.start_time
        self.logger.info("=" * 60)
        self.logger.info(f"PHASE 2 COMPLETE")
        self.logger.info(f"Total car details: {len(all_details)}")
        self.logger.info(f"Time elapsed: {format_duration(elapsed)}")
        self.logger.info("=" * 60)
        
        return all_details

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Bilbasen Web Scraper')
    parser.add_argument('--output', '-o', default=CONFIG['OUTPUT_DIR'],
                        help='Output directory (default: bilbasen_scrape)')
    parser.add_argument('--phase', '-p', type=int, choices=[1, 2],
                        help='Run only phase 1 or 2 (default: both)')
    parser.add_argument('--no-images', action='store_true',
                        help='Skip downloading images')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode (limited data)')
    args = parser.parse_args()
    
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logging(output_dir)
    
    logger.info("=" * 60)
    logger.info("BILBASEN WEB SCRAPER")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Download images: {not args.no_images}")
    logger.info("=" * 60)
    
    scraper = BilbasenScraper(output_dir, logger)
    
    if args.test:
        # Test mode - just scrape a few items
        logger.info("TEST MODE - Limited scraping")
        CONFIG['MAX_PAGES'] = 2
    
    try:
        if args.phase is None or args.phase == 1:
            scraper.run_phase1()
        
        if args.phase is None or args.phase == 2:
            scraper.run_phase2(download_images=not args.no_images)
        
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    
    logger.info("=" * 60)
    logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()
