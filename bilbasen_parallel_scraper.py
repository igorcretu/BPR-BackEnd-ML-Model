#!/usr/bin/env python3
"""
Bilbasen High-Speed Parallel Scraper
With explicit permission from bilbasen.dk
"""

import concurrent.futures
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import os
import re

# ===== CONFIGURATION =====
NUM_WORKERS = 8
BATCH_SIZE = 1000
OUTPUT_DIR = "scraped_data"
REQUEST_DELAY = 0.5  # With permission, can be aggressive

def setup_driver():
    """Setup headless Chrome driver."""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(15)
    return driver

def extract_car_details(soup, url):
    """Extract comprehensive car details."""
    car_data = {'url': url, 'scraped_at': datetime.now().isoformat()}
    
    try:
        # Basic info
        title = soup.find('h1')
        car_data['title'] = title.get_text(strip=True) if title else ''
        
        # Price
        price_elem = soup.find(text=re.compile(r'\d+[.,]\d+\s*kr', re.I))
        car_data['price'] = price_elem.strip() if price_elem else ''
        
        # Description
        desc = soup.find('div', class_=re.compile(r'description', re.I))
        if desc:
            car_data['beskrivelse'] = re.sub(r'\s+', ' ', desc.get_text(separator=' ', strip=True))
        else:
            car_data['beskrivelse'] = ''
        
        # Extract all dt/dd pairs (Details)
        for dt, dd in zip(soup.find_all('dt'), soup.find_all('dd')):
            key = dt.get_text(strip=True).replace(':', '').replace(' ', '_').lower()
            value = dd.get_text(strip=True)
            car_data[f'detail_{key}'] = value
        
        # Equipment list
        equipment = []
        for li in soup.find_all('li'):
            text = li.get_text(strip=True)
            if text and len(text) < 100:
                equipment.append(text)
        car_data['udstyr'] = ' | '.join(equipment) if equipment else ''
        
    except Exception as e:
        car_data['extraction_error'] = str(e)
    
    return car_data

def scrape_single_car(url, worker_id):
    """Scrape one car (called by parallel workers)."""
    driver = setup_driver()
    try:
        driver.get(url)
        time.sleep(REQUEST_DELAY)
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        return extract_car_details(soup, url)
        
    except Exception as e:
        return {'url': url, 'error': str(e)}
    finally:
        driver.quit()

def scrape_batch_parallel(urls, batch_num, num_workers):
    """Scrape batch with parallel workers."""
    print(f"\n{'='*70}")
    print(f"BATCH {batch_num}: {len(urls)} cars | Workers: {num_workers}")
    print(f"{'='*70}")
    
    start = time.time()
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(scrape_single_car, url, i % num_workers): url 
                  for i, url in enumerate(urls)}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(future.result())
            if i % 100 == 0:
                elapsed = time.time() - start
                rate = i / elapsed
                eta = (len(urls) - i) / rate if rate > 0 else 0
                print(f"  {i}/{len(urls)} | {rate:.1f} cars/s | ETA: {eta/60:.0f}m")
    
    elapsed = time.time() - start
    print(f"✅ Batch done in {elapsed/60:.1f}m ({len(results)/elapsed:.1f} cars/s)")
    
    return results

def main(car_urls_file='bilbasen_car_urls.csv', num_workers=8, batch_size=1000):
    """Main execution."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load URLs
    print("Loading car URLs...")
    df_urls = pd.read_csv(car_urls_file)
    urls = df_urls['url'].tolist()
    
    print(f"\n{'='*70}")
    print(f"🚀 BILBASEN HIGH-SPEED SCRAPER")
    print(f"{'='*70}")
    print(f"Total cars: {len(urls)}")
    print(f"Workers: {num_workers}")
    print(f"Batch size: {batch_size}")
    print(f"Est. time: {len(urls)/(num_workers*2.5)/60:.0f} minutes")
    print(f"{'='*70}\n")
    
    all_results = []
    start_time = datetime.now()
    
    # Process in batches
    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i+batch_size]
        batch_num = i//batch_size + 1
        
        batch_results = scrape_batch_parallel(batch_urls, batch_num, num_workers)
        all_results.extend(batch_results)
        
        # Save progress
        df_progress = pd.DataFrame(all_results)
        progress_file = f"{OUTPUT_DIR}/progress_{len(all_results)}.csv"
        df_progress.to_csv(progress_file, index=False)
        print(f"💾 Saved: {progress_file}")
    
    # Final save
    df_final = pd.DataFrame(all_results)
    final_file = f"{OUTPUT_DIR}/bilbasen_complete_{len(all_results)}_cars.csv"
    df_final.to_csv(final_file, index=False)
    
    # Summary
    elapsed = datetime.now() - start_time
    print(f"\n{'='*70}")
    print(f"🎉 COMPLETE!")
    print(f"{'='*70}")
    print(f"Cars: {len(all_results)}")
    print(f"Time: {elapsed}")
    print(f"Rate: {len(all_results)/elapsed.total_seconds():.2f} cars/s")
    print(f"File: {final_file}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    # Customize here
    main(
        car_urls_file='bilbasen_car_urls.csv',
        num_workers=8,        # Adjust based on your CPU
        batch_size=1000
    )
