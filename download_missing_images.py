#!/usr/bin/env python3
"""
Multi-threaded script to download missing images for cars in car_details.csv
"""

import os
import csv
import requests
import logging
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configuration
CSV_FILE = os.path.join(os.path.dirname(__file__), 'bilbasen_scrape', 'car_details.csv')
IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'bilbasen_scrape', 'images')
MAX_WORKERS = 16  # Number of parallel download threads
DELAY_BETWEEN_DOWNLOADS = (0.1, 0.3)  # Random delay in seconds

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Thread-safe counters
class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.total = 0
        self.already_exists = 0
        self.downloaded = 0
        self.failed = 0
        self.no_url = 0

stats = Stats()

def download_image(external_id, image_url):
    """Download a single car image"""
    if not image_url or image_url == '' or image_url.lower() == 'nan':
        with stats.lock:
            stats.no_url += 1
        return {'external_id': external_id, 'status': 'no_url'}
    
    image_filename = f"{external_id}.jpg"
    image_path = os.path.join(IMAGES_DIR, image_filename)
    
    # Check if already exists
    if os.path.exists(image_path):
        with stats.lock:
            stats.already_exists += 1
        return {'external_id': external_id, 'status': 'exists', 'path': image_path}
    
    # Download image
    try:
        # Create session for this thread
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        response = session.get(image_url, timeout=30)
        response.raise_for_status()
        
        # Save image
        with open(image_path, 'wb') as f:
            f.write(response.content)
        
        with stats.lock:
            stats.downloaded += 1
        
        logger.info(f"✅ Downloaded: {external_id}")
        
        # Small delay to be respectful
        time.sleep(random.uniform(*DELAY_BETWEEN_DOWNLOADS))
        
        return {'external_id': external_id, 'status': 'downloaded', 'path': image_path}
        
    except requests.exceptions.RequestException as e:
        with stats.lock:
            stats.failed += 1
        logger.warning(f"❌ Failed to download {external_id}: {e}")
        return {'external_id': external_id, 'status': 'failed', 'error': str(e)}
    
    except Exception as e:
        with stats.lock:
            stats.failed += 1
        logger.error(f"❌ Error downloading {external_id}: {e}")
        return {'external_id': external_id, 'status': 'error', 'error': str(e)}

def main():
    logger.info("="*60)
    logger.info("MISSING IMAGE DOWNLOADER (Multi-threaded)")
    logger.info("="*60)
    
    # Create images directory if it doesn't exist
    os.makedirs(IMAGES_DIR, exist_ok=True)
    logger.info(f"Images directory: {IMAGES_DIR}")
    
    # Check if CSV exists
    if not os.path.exists(CSV_FILE):
        logger.error(f"❌ CSV file not found: {CSV_FILE}")
        return
    
    logger.info(f"Reading cars from: {CSV_FILE}")
    
    # Read CSV and collect cars that need images
    cars_to_process = []
    
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            external_id = row.get('external_id')
            image_url = row.get('image_url')
            
            if external_id:
                cars_to_process.append({
                    'external_id': external_id,
                    'image_url': image_url
                })
    
    stats.total = len(cars_to_process)
    logger.info(f"Found {stats.total} cars in CSV")
    logger.info(f"Using {MAX_WORKERS} worker threads")
    logger.info("")
    
    # Process images with thread pool
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all download tasks
        future_to_car = {
            executor.submit(download_image, car['external_id'], car['image_url']): car
            for car in cars_to_process
        }
        
        # Process completed downloads
        completed = 0
        for future in as_completed(future_to_car):
            car = future_to_car[future]
            try:
                result = future.result()
                completed += 1
                
                # Progress update every 50 cars
                if completed % 50 == 0:
                    logger.info(f"Progress: {completed}/{stats.total} ({completed*100//stats.total}%)")
                    
            except Exception as e:
                logger.error(f"❌ Unexpected error processing {car['external_id']}: {e}")
                with stats.lock:
                    stats.failed += 1
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Print summary
    logger.info("")
    logger.info("="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Total cars in CSV:     {stats.total}")
    logger.info(f"Already exist:         {stats.already_exists}")
    logger.info(f"Successfully downloaded: {stats.downloaded}")
    logger.info(f"No image URL:          {stats.no_url}")
    logger.info(f"Failed:                {stats.failed}")
    logger.info(f"Duration:              {duration:.2f} seconds")
    logger.info(f"Average speed:         {stats.total/duration:.2f} cars/second")
    logger.info("="*60)
    
    if stats.downloaded > 0:
        logger.info(f"✅ Successfully downloaded {stats.downloaded} new images!")
    
    if stats.failed > 0:
        logger.warning(f"⚠️  {stats.failed} downloads failed")
    
    logger.info("")
    logger.info("✅ Done!")

if __name__ == "__main__":
    main()
