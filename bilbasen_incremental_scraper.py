#!/usr/bin/env python3
"""
Bilbasen Incremental Scraper
Scrapes newest listings first, stopping when hitting known IDs.

Usage:
    python3 bilbasen_incremental_scraper.py
    python3 bilbasen_incremental_scraper.py --no-images
"""

import requests
import json
import re
import time
import os
import csv
import logging
import signal
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'OUTPUT_DIR': 'bilbasen_scrape',
    'MAX_PAGES': 100,
    'ITEMS_PER_PAGE': 30,
    
    # Delays (seconds)
    'DELAY_BETWEEN_REQUESTS': (0.5, 1.0),
    'DELAY_BETWEEN_CARS': (0.3, 0.7),
    
    # Threading
    'MAX_WORKERS': 12,
    
    # Checkpoints
    'CHECKPOINT_INTERVAL': 50,
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
# HELPER FUNCTIONS
# ============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'incremental_scraper.log')
    
    logger = logging.getLogger('bilbasen_incremental')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', 
                                     datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

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

def get_highest_id_from_csv(csv_path: str) -> int:
    """Get the highest external_id from existing CSV."""
    if not os.path.exists(csv_path):
        return 0
    
    max_id = 0
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    external_id = int(row.get('external_id', 0))
                    if external_id > max_id:
                        max_id = external_id
                except (ValueError, TypeError):
                    continue
    except Exception:
        pass
    
    return max_id

# ============================================================================
# SCRAPER CLASS
# ============================================================================

class IncrementalScraper:
    def __init__(self, output_dir: str, logger: logging.Logger, known_max_id: int):
        self.output_dir = output_dir
        self.logger = logger
        self.known_max_id = known_max_id
        self.stop_requested = False
        self.lock = threading.Lock()
        self.new_listings = []
        self.processed_count = 0
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.warning(f"Received signal {signum}, requesting graceful shutdown...")
        self.stop_requested = True
    
    def fetch_page(self, url: str) -> Optional[str]:
        """Fetch a page with retry logic."""
        session = requests.Session()
        for attempt in range(3):
            try:
                response = session.get(url, headers=get_headers(), timeout=30)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                else:
                    self.logger.error(f"Failed to fetch page after 3 attempts: {e}")
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
                            try:
                                external_id_int = int(external_id)
                            except ValueError:
                                continue
                                
                            listings.append({
                                'uri': uri,
                                'image_url': image_url,
                                'external_id': external_id,
                                'external_id_int': external_id_int
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
        
        # Image
        result['image_url'] = listing_image_url
        result['image_filename'] = f"{result['external_id']}.jpg" if result['external_id'] else ''
        
        return result
    
    def download_image(self, image_url: str, save_path: str) -> bool:
        """Download image to local file."""
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
        except Exception:
            return False
    
    def process_single_car(self, listing: Dict, images_dir: str, download_images: bool) -> Optional[Dict]:
        """Process a single car listing."""
        car_url = f"https://www.bilbasen.dk{listing['uri']}"
        image_url = listing.get('image_url', '')
        external_id = listing.get('external_id', '')
        
        html = self.fetch_page(car_url)
        if not html:
            return None
        
        details = self.extract_car_details(html, car_url, image_url)
        
        # Download image if requested
        if details and download_images and image_url:
            img_path = os.path.join(images_dir, f"{external_id}.jpg")
            if not os.path.exists(img_path):
                self.download_image(image_url, img_path)
        
        time.sleep(random.uniform(*CONFIG['DELAY_BETWEEN_CARS']))
        
        return details
    
    def save_to_csv(self, details: List[Dict]):
        """Append new details to CSV."""
        if not details:
            return
        
        csv_path = os.path.join(self.output_dir, 'car_details.csv')
        
        # Priority columns
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
        
        # Determine fieldnames
        all_keys = set()
        for d in details:
            all_keys.update(d.keys())
        
        file_exists = os.path.exists(csv_path)
        if file_exists:
            # Read existing fieldnames
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_fieldnames = reader.fieldnames or []
                all_keys.update(existing_fieldnames)
        
        fieldnames = [col for col in priority_columns if col in all_keys]
        remaining = sorted([col for col in all_keys if col not in priority_columns])
        fieldnames.extend(remaining)
        
        # Append to CSV
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
            writer.writerows(details)
    
    def run(self, download_images: bool = True) -> int:
        """Run incremental scraper."""
        self.logger.info("=" * 60)
        self.logger.info("INCREMENTAL SCRAPER - Newest First")
        self.logger.info("=" * 60)
        self.logger.info(f"Known max ID: {self.known_max_id}")
        self.logger.info(f"Workers: {CONFIG['MAX_WORKERS']}")
        self.logger.info("-" * 60)
        
        # Create images directory
        images_dir = os.path.join(self.output_dir, 'images')
        if download_images:
            os.makedirs(images_dir, exist_ok=True)
        
        # Scrape pages newest-first
        page = 1
        all_new_listings = []
        stop_scraping = False
        
        while page <= CONFIG['MAX_PAGES'] and not stop_scraping and not self.stop_requested:
            url = f"https://www.bilbasen.dk/brugt/bil?includeengroscvr=true&includeleasing=false&sortby=date&sortorder=desc&page={page}"
            
            self.logger.info(f"Fetching page {page}...")
            html = self.fetch_page(url)
            if not html:
                self.logger.error(f"Failed to fetch page {page}")
                break
            
            listings, total_count = self.extract_listings_from_html(html)
            
            if not listings:
                self.logger.info("No listings found, stopping")
                break
            
            # Check which listings are new
            new_on_page = []
            for listing in listings:
                if listing['external_id_int'] <= self.known_max_id:
                    self.logger.info(f"Reached known ID {listing['external_id']}, stopping")
                    stop_scraping = True
                    break
                new_on_page.append(listing)
            
            if new_on_page:
                all_new_listings.extend(new_on_page)
                self.logger.info(f"Page {page}: {len(new_on_page)} new listings")
            
            if stop_scraping or len(listings) < CONFIG['ITEMS_PER_PAGE']:
                break
            
            page += 1
            time.sleep(random.uniform(*CONFIG['DELAY_BETWEEN_REQUESTS']))
        
        if not all_new_listings:
            self.logger.info("No new listings found!")
            return 0
        
        self.logger.info(f"Total new listings: {len(all_new_listings)}")
        self.logger.info("-" * 60)
        
        # Process new listings in parallel
        try:
            with ThreadPoolExecutor(max_workers=CONFIG['MAX_WORKERS']) as executor:
                future_to_listing = {}
                for listing in all_new_listings:
                    future = executor.submit(self.process_single_car, listing, images_dir, download_images)
                    future_to_listing[future] = listing
                
                completed_count = 0
                last_checkpoint = 0
                for future in as_completed(future_to_listing):
                    if self.stop_requested:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    
                    listing = future_to_listing[future]
                    external_id = listing.get('external_id', '')
                    
                    try:
                        details = future.result()
                        
                        with self.lock:
                            if details:
                                self.new_listings.append(details)
                            completed_count += 1
                        
                        if completed_count % 10 == 0:
                            self.logger.info(f"Processed {completed_count}/{len(all_new_listings)}")
                        
                        # Save checkpoint
                        if completed_count - last_checkpoint >= CONFIG['CHECKPOINT_INTERVAL']:
                            with self.lock:
                                self.save_to_csv(self.new_listings)
                                self.new_listings = []  # Clear after saving
                                self.logger.info(f"Checkpoint saved ({completed_count} processed)")
                                last_checkpoint = completed_count
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {external_id}: {e}")
                
        finally:
            # Save any remaining
            with self.lock:
                if self.new_listings:
                    self.save_to_csv(self.new_listings)
        
        self.logger.info("=" * 60)
        self.logger.info(f"COMPLETE - Added {len(all_new_listings)} new cars")
        self.logger.info("=" * 60)
        
        return len(all_new_listings)

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Bilbasen Incremental Scraper')
    parser.add_argument('--output', '-o', default=CONFIG['OUTPUT_DIR'],
                        help='Output directory (default: bilbasen_scrape)')
    parser.add_argument('--no-images', action='store_true',
                        help='Skip downloading images')
    args = parser.parse_args()
    
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logging(output_dir)
    
    # Get highest ID from existing CSV
    csv_path = os.path.join(output_dir, 'car_details.csv')
    known_max_id = get_highest_id_from_csv(csv_path)
    
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    scraper = IncrementalScraper(output_dir, logger, known_max_id)
    
    try:
        count = scraper.run(download_images=not args.no_images)
        logger.info(f"Successfully added {count} new cars")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    
    logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
