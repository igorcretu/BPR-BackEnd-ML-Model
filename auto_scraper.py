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
from dotenv import load_dotenv
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_scraper.log'),
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

# Output directories
OUTPUT_DIR = 'bilbasen_scrape_auto'
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')

# Threading configuration
MAX_WORKERS = 16  # Parallel workers for detail scraping


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
        """Scrape a single page of listings"""
        params = PARAMS.copy()
        params['page'] = page
        
        try:
            url = f"{BASE_URL}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
            logger.info(f"📄 Scraping page {page}: {url}")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all listing cards
            listings = soup.find_all('article', class_='listing')
            
            if not listings:
                # Try alternative selectors
                listings = soup.find_all('div', class_='listing-container')
            
            extracted_listings = []
            for listing in listings:
                try:
                    # Extract listing URL
                    link = listing.find('a', href=True)
                    if not link:
                        continue
                    
                    listing_url = link['href']
                    if not listing_url.startswith('http'):
                        listing_url = f"https://www.bilbasen.dk{listing_url}"
                    
                    # Extract external_id
                    external_id = self.extract_external_id(listing_url)
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
                    
                    # Extract basic info
                    title = listing.find('h2') or listing.find('h3')
                    title_text = title.get_text(strip=True) if title else ''
                    
                    # Extract image URL
                    img = listing.find('img')
                    image_url = img.get('src') or img.get('data-src') if img else None
                    
                    extracted_listings.append({
                        'external_id': external_id,
                        'url': listing_url,
                        'title': title_text,
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
            
            response = self.session.get(image_url, timeout=30)
            response.raise_for_status()
            
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
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
                        details['image_path'] = f"bilbasen_scrape_auto/images/{image_filename}"
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
    
    def upsert_car(self, car_data):
        """Insert or update car in database"""
        try:
            # Check if car exists
            car_id = self.check_car_exists(car_data.get('external_id'))
            
            if car_id:
                # Update existing car
                with self.lock:
                    self.cars_updated += 1
                logger.debug(f"🔄 Updating car {car_data.get('external_id')}")
            else:
                # Insert new car
                car_data['id'] = str(uuid.uuid4())
                with self.lock:
                    self.cars_new += 1
                logger.debug(f"➕ Inserting new car {car_data.get('external_id')}")
            
            # Build UPSERT query (simplified version - full implementation would include all fields)
            columns = list(car_data.keys())
            values = [car_data[col] for col in columns]
            
            placeholders = ', '.join(['%s'] * len(columns))
            columns_str = ', '.join(columns)
            update_str = ', '.join([f"{col} = EXCLUDED.{col}" for col in columns if col not in ['id', 'external_id']])
            
            query = f"""
                INSERT INTO cars ({columns_str})
                VALUES ({placeholders})
                ON CONFLICT (external_id)
                DO UPDATE SET {update_str}, updated_at = NOW()
            """
            
            self.cur.execute(query, values)
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error upserting car: {e}")
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
    parser.add_argument('--workers', type=int, default=MAX_WORKERS,
                        help=f'Number of parallel workers (default: {MAX_WORKERS})')
    
    args = parser.parse_args()
    
    # Update global MAX_WORKERS if specified
    global MAX_WORKERS
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
