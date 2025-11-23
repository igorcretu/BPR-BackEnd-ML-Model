#!/usr/bin/env python3
"""
Bilbasen.dk Car Detail Scraper - Raspberry Pi 5 Optimized
Scrapes detailed car information from bilbasen.dk URLs

Author: Igor Cretu, Group 26
Date: November 2025
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
import json
from datetime import datetime
from urllib.parse import urljoin
import os
import sys
import argparse
from typing import Dict, List, Optional

# Selenium imports (headless Chrome for Raspberry Pi)
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️  Selenium not available - Install with: pip install selenium")
    print("    Also install chromium-driver: sudo apt install chromium-chromedriver")


# ==================== CONFIGURATION ====================

BASE_URL = "https://www.bilbasen.dk"

# Output settings
DEFAULT_OUTPUT_CSV = "bilbasen_detailed_cars.csv"
PROGRESS_SAVE_INTERVAL = 10  # Save progress every N cars

# Scraping settings
REQUEST_DELAY = 2.0  # Seconds between requests (be respectful!)
MAX_RETRIES = 3
TIMEOUT = 20

# Browser headers
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'da,en-US;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
}

# Danish to English column name translations
COLUMN_TRANSLATIONS = {
    # Original Danish column names (detaljer_* prefix)
    'detaljer_modelår': 'details_model_year',
    'detaljer_1._registrering': 'details_first_registration',
    'detaljer_kilometertal': 'details_mileage_km',
    'detaljer_drivmiddel': 'details_fuel_type',
    'detaljer_rækkevidde': 'details_range_km',
    'detaljer_batterikapacitet': 'details_battery_capacity_kwh',
    'detaljer_energiforbrug': 'details_energy_consumption',
    'detaljer_hjemmeopladning_ac': 'details_home_charging_ac',
    'detaljer_hurtig_opladning_dc': 'details_fast_charging_dc',
    'detaljer_opladningstid_dc_10-80%': 'details_charging_time_dc_10_80_pct',
    'detaljer_brændstofforbrug': 'details_fuel_consumption',
    'detaljer_co2_udledning': 'details_co2_emission',
    'detaljer_euronorm': 'details_euro_norm',
    'detaljer_periodisk_afgift': 'details_periodic_tax',
    'detaljer_ydelse': 'details_power_hp_nm',
    'detaljer_acceleration': 'details_acceleration_0_100',
    'detaljer_tophastighed': 'details_top_speed',
    'detaljer_geartype': 'details_transmission_type',
    'detaljer_antal_gear': 'details_number_of_gears',
    'detaljer_trækvægt': 'details_towing_capacity',
    'detaljer_farve': 'details_color',
    'detaljer_produceret': 'details_production_year',
    
    # Model information (model_* prefix)
    'model_nypris': 'model_new_price',
    'model_kategori': 'model_category',
    'model_type': 'model_body_type',
    'model_bagagerumsstørrelse': 'model_trunk_size',
    'model_vægt': 'model_weight_kg',
    'model_bredde': 'model_width_cm',
    'model_længde': 'model_length_cm',
    'model_højde': 'model_height_cm',
    'model_lasteevne': 'model_load_capacity_kg',
    'model_max._trækvægt_m/bremse': 'model_max_towing_with_brake',
    'model_trækhjul': 'model_drive_type',
    'model_cylindre': 'model_cylinders',
    'model_abs-bremser': 'model_abs_brakes',
    'model_esp': 'model_esp',
    'model_airbags': 'model_airbags',
    'model_tankkapacitet': 'model_tank_capacity_l',
    'model_døre': 'model_doors',
    
    # Basic fields
    'title': 'title',
    'price': 'price',
    'beskrivelse': 'description',
    'udstyr': 'equipment',
    'url': 'url',
    'mærke': 'brand',
    'model': 'model',
    'variant': 'variant',
}


# ==================== SELENIUM SETUP ====================

def setup_driver() -> webdriver.Chrome:
    """
    Initialize Selenium WebDriver for Raspberry Pi.
    Optimized for headless Chrome on Raspberry Pi 5.
    """
    if not SELENIUM_AVAILABLE:
        raise ImportError("Selenium not available. Install with: pip install selenium")
    
    options = Options()
    
    # Headless mode (no GUI) - essential for Raspberry Pi
    options.add_argument('--headless=new')
    
    # Raspberry Pi specific optimizations
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')  # GPU acceleration not needed
    options.add_argument('--disable-software-rasterizer')
    
    # Memory optimization
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-blink-features=AutomationControlled')
    
    # Set user agent
    options.add_argument(f'user-agent={HEADERS["User-Agent"]}')
    options.add_argument('--window-size=1920,1080')
    
    # Disable unnecessary features to save resources
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_experimental_option('prefs', {
        'profile.managed_default_content_settings.images': 2,  # Don't load images
    })
    
    # For Raspberry Pi, use system's chromium-chromedriver
    try:
        # Try using system chromium-chromedriver first
        service = Service('/usr/bin/chromedriver')
        driver = webdriver.Chrome(service=service, options=options)
    except:
        # Fallback to default
        driver = webdriver.Chrome(options=options)
    
    driver.set_page_load_timeout(TIMEOUT)
    
    return driver


# ==================== EXTRACTION FUNCTIONS ====================

def extract_breadcrumb_info(soup) -> Dict[str, str]:
    """
    Extract brand (mærke), model, and variant from breadcrumb navigation.
    
    Example breadcrumb: Peugeot > 308 > 1,5 BlueHDi 130 Allure Grand SW 5d
    Returns: {'mærke': 'Peugeot', 'model': '308', 'variant': '1,5 BlueHDi 130 Allure Grand SW 5d'}
    """
    breadcrumb_info = {'mærke': '', 'model': '', 'variant': ''}
    
    try:
        # Look for breadcrumb navigation
        breadcrumbs = soup.find('nav', {'aria-label': 'breadcrumb'}) or \
                     soup.find('ol', {'class': re.compile(r'breadcrumb', re.I)})
        
        if breadcrumbs:
            # Find all breadcrumb links
            links = breadcrumbs.find_all('a', href=True)
            
            # Typically structure is: Home > Brand > Model > Variant
            # We want the last 3 (Brand, Model, Variant)
            if len(links) >= 3:
                # Brand (mærke) - e.g., "Peugeot"
                brand_link = links[-3]
                breadcrumb_info['mærke'] = brand_link.get_text(strip=True)
                
                # Model - e.g., "308"
                model_link = links[-2]
                breadcrumb_info['model'] = model_link.get_text(strip=True)
            
            # Variant is usually not a link, but a span or text after the last link
            # Look for element with aria-label="variant" or similar
            variant_elem = breadcrumbs.find(attrs={'aria-label': 'variant'}) or \
                          breadcrumbs.find('span', {'class': re.compile(r'variant', re.I)})
            
            if variant_elem:
                breadcrumb_info['variant'] = variant_elem.get_text(strip=True)
            else:
                # Try to get the last list item that's not a link
                all_items = breadcrumbs.find_all(['li', 'span'])
                for item in reversed(all_items):
                    text = item.get_text(strip=True)
                    # Check if this item doesn't have a link inside it
                    if text and not item.find('a') and text not in [breadcrumb_info['mærke'], breadcrumb_info['model']]:
                        breadcrumb_info['variant'] = text
                        break
        
        # Fallback: try to extract from URL if breadcrumb parsing failed
        if not breadcrumb_info['mærke'] or not breadcrumb_info['model']:
            # URL pattern: /brugt/bil/[brand]/[model]/[variant]/[id]
            meta_url = soup.find('meta', property='og:url')
            if meta_url and meta_url.get('content'):
                url = meta_url['content']
                url_parts = url.rstrip('/').split('/')
                if len(url_parts) >= 6:
                    breadcrumb_info['mærke'] = url_parts[4] if not breadcrumb_info['mærke'] else breadcrumb_info['mærke']
                    breadcrumb_info['model'] = url_parts[5] if not breadcrumb_info['model'] else breadcrumb_info['model']
                    if len(url_parts) >= 7 and not breadcrumb_info['variant']:
                        breadcrumb_info['variant'] = url_parts[6].replace('-', ' ')
                        
    except Exception as e:
        print(f"    Warning: Could not extract breadcrumb info: {e}")
    
    return breadcrumb_info


def extract_car_details(soup, url: str) -> Dict[str, str]:
    """
    Extract all detailed information from a car listing page.
    Returns a dictionary with all car details.
    """
    car_data = {'url': url}
    
    try:
        # === BREADCRUMB INFO (Brand, Model, Variant) ===
        breadcrumb_info = extract_breadcrumb_info(soup)
        car_data.update(breadcrumb_info)
        
        # === BASIC INFORMATION ===
        
        # Title/Model
        title = soup.find('h1')
        car_data['title'] = title.get_text(strip=True) if title else ''
        
        # Price - look for price patterns
        price_patterns = [
            soup.find('span', string=re.compile(r'\d+[.,]\d+\s*kr', re.I)),
            soup.find('div', string=re.compile(r'\d+[.,]\d+\s*kr', re.I)),
            soup.find(attrs={'data-testid': re.compile(r'price', re.I)})
        ]
        for price in price_patterns:
            if price:
                car_data['price'] = price.get_text(strip=True)
                break
        else:
            car_data['price'] = ''
        
        # === DESCRIPTION (Beskrivelse) ===
        description_selectors = [
            soup.find('div', {'class': re.compile(r'description', re.I)}),
            soup.find('section', {'class': re.compile(r'description', re.I)}),
            soup.find('div', {'data-testid': 'description'}),
        ]
        
        description_text = ""
        for desc_elem in description_selectors:
            if desc_elem:
                description_text = desc_elem.get_text(separator=' ', strip=True)
                description_text = re.sub(r'\s+', ' ', description_text)
                break
        
        car_data['beskrivelse'] = description_text
        
        # === DETALJER (Details) Section ===
        details_section = soup.find('section', string=re.compile(r'Detaljer', re.I)) or \
                         soup.find('div', string=re.compile(r'Detaljer', re.I)) or \
                         soup.find('h2', string=re.compile(r'Detaljer', re.I))
        
        if details_section:
            details_container = details_section.find_parent()
            
            # Look for dl/dt/dd structure (definition lists)
            dts = details_container.find_all('dt') if details_container else []
            dds = details_container.find_all('dd') if details_container else []
            
            for dt, dd in zip(dts, dds):
                key = dt.get_text(strip=True).replace(':', '').replace(' ', '_').lower()
                value = dd.get_text(strip=True)
                car_data[f'detaljer_{key}'] = value
            
            # Also look for table rows
            rows = details_container.find_all('tr') if details_container else []
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    key = cells[0].get_text(strip=True).replace(':', '').replace(' ', '_').lower()
                    value = cells[1].get_text(strip=True)
                    car_data[f'detaljer_{key}'] = value
        
        # === GENERELLE MODELOPLYSNINGER (General Model Info) ===
        model_info_section = soup.find('section', string=re.compile(r'Generelle modeloplysninger', re.I)) or \
                           soup.find('div', string=re.compile(r'Generelle modeloplysninger', re.I)) or \
                           soup.find('h2', string=re.compile(r'Generelle modeloplysninger', re.I))
        
        if model_info_section:
            model_container = model_info_section.find_parent()
            
            # Extract key-value pairs
            dts = model_container.find_all('dt') if model_container else []
            dds = model_container.find_all('dd') if model_container else []
            
            for dt, dd in zip(dts, dds):
                key = dt.get_text(strip=True).replace(':', '').replace(' ', '_').lower()
                value = dd.get_text(strip=True)
                car_data[f'model_{key}'] = value
            
            rows = model_container.find_all('tr') if model_container else []
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    key = cells[0].get_text(strip=True).replace(':', '').replace(' ', '_').lower()
                    value = cells[1].get_text(strip=True)
                    car_data[f'model_{key}'] = value
        
        # === UDSTYR OG TILBEHØR (Equipment and Accessories) ===
        equipment_section = soup.find('section', string=re.compile(r'Udstyr og tilbehør', re.I)) or \
                          soup.find('div', string=re.compile(r'Udstyr og tilbehør', re.I)) or \
                          soup.find('h2', string=re.compile(r'Udstyr og tilbehør', re.I))
        
        equipment_list = []
        if equipment_section:
            equipment_container = equipment_section.find_parent()
            
            # Look for list items
            items = equipment_container.find_all('li') if equipment_container else []
            for item in items:
                text = item.get_text(strip=True)
                if text:
                    equipment_list.append(text)
            
            # Also look for divs with equipment
            if not equipment_list:
                divs = equipment_container.find_all('div') if equipment_container else []
                for div in divs:
                    text = div.get_text(strip=True)
                    if text and len(text) < 200:  # Reasonable length for equipment item
                        equipment_list.append(text)
        
        # Store equipment as pipe-separated string for easy parsing
        car_data['udstyr'] = ' | '.join(equipment_list) if equipment_list else ''
        
    except Exception as e:
        print(f"    Error extracting details: {e}")
        car_data['error'] = str(e)
    
    return car_data


def scrape_car_page(driver, url: str, attempt: int = 1) -> Dict[str, str]:
    """Scrape detailed information from a single car page with retry logic."""
    try:
        driver.get(url)
        time.sleep(2)  # Wait for page to load
        
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        
        car_data = extract_car_details(soup, url)
        return car_data
        
    except Exception as e:
        if attempt < MAX_RETRIES:
            print(f"    Retry {attempt}/{MAX_RETRIES}...")
            time.sleep(REQUEST_DELAY * 2)
            return scrape_car_page(driver, url, attempt + 1)
        else:
            print(f"    Failed after {MAX_RETRIES} attempts: {e}")
            return {'url': url, 'error': str(e)}


# ==================== MAIN SCRAPING FUNCTION ====================

def scrape_cars_from_urls(
    car_urls: List[str],
    output_csv: str = DEFAULT_OUTPUT_CSV,
    max_cars: Optional[int] = None,
    resume: bool = True
) -> pd.DataFrame:
    """
    Scrape detailed information for multiple cars from their URLs.
    
    Args:
        car_urls: List of bilbasen.dk car URLs to scrape
        output_csv: Output CSV file path
        max_cars: Maximum number of cars to scrape (None = all)
        resume: Whether to resume from existing progress file
        
    Returns:
        DataFrame with all scraped car data
    """
    print("\n" + "="*70)
    print("🚗 Bilbasen.dk Car Detail Scraper")
    print("="*70)
    print(f"📊 URLs to scrape: {len(car_urls)}")
    print(f"💾 Output file: {output_csv}")
    print(f"⏱️  Request delay: {REQUEST_DELAY}s")
    print(f"🔄 Max retries: {MAX_RETRIES}")
    print("="*70 + "\n")
    
    # Check for existing progress
    all_cars = []
    start_index = 0
    
    if resume:
        # Look for latest progress file
        progress_files = sorted([f for f in os.listdir('.') if f.startswith('progress_') and f.endswith('_cars.csv')])
        if progress_files:
            latest_progress = progress_files[-1]
            print(f"📂 Found progress file: {latest_progress}")
            
            try:
                df_progress = pd.read_csv(latest_progress)
                all_cars = df_progress.to_dict('records')
                start_index = len(all_cars)
                print(f"✅ Resuming from car #{start_index + 1}\n")
            except Exception as e:
                print(f"⚠️  Could not load progress file: {e}")
                print("   Starting from scratch...\n")
    
    # Limit number of cars if specified
    if max_cars:
        car_urls = car_urls[:max_cars]
    
    # Initialize Selenium driver
    print("🌐 Initializing browser...")
    driver = setup_driver()
    print("✅ Browser ready\n")
    
    try:
        # Scrape each URL
        for i, url in enumerate(car_urls[start_index:], start=start_index + 1):
            # Extract short identifier from URL for display
            url_identifier = url.split('/')[-2][:40] if '/' in url else url[:40]
            
            print(f"[{i}/{len(car_urls)}] Scraping: {url_identifier}...", end=" ", flush=True)
            
            car_data = scrape_car_page(driver, url)
            all_cars.append(car_data)
            
            print("✓")
            
            # Save progress every N cars
            if i % PROGRESS_SAVE_INTERVAL == 0:
                df_temp = pd.DataFrame(all_cars)
                progress_file = f'progress_{i}_cars.csv'
                df_temp.to_csv(progress_file, index=False)
                print(f"  💾 Progress saved: {i} cars → {progress_file}")
            
            # Respectful delay between requests
            time.sleep(REQUEST_DELAY)
        
        print(f"\n✅ Successfully scraped {len(all_cars)} cars!")
        
        # Create final DataFrame
        df = pd.DataFrame(all_cars)
        
        # Translate column names to English
        df.rename(columns=COLUMN_TRANSLATIONS, inplace=True)
        
        # Save final results
        df.to_csv(output_csv, index=False)
        print(f"💾 Final data saved to: {output_csv}")
        
        # Print summary statistics
        print("\n" + "="*70)
        print("📊 SCRAPING SUMMARY")
        print("="*70)
        print(f"Total cars scraped: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        print(f"Errors encountered: {df['error'].notna().sum() if 'error' in df.columns else 0}")
        
        # Show sample of column names
        print("\n📋 Sample columns (first 10):")
        for col in list(df.columns)[:10]:
            print(f"  • {col}")
        
        if len(df.columns) > 10:
            print(f"  ... and {len(df.columns) - 10} more columns")
        
        print("="*70 + "\n")
        
        return df
        
    finally:
        driver.quit()
        print("🔒 Browser closed")


# ==================== UTILITY FUNCTIONS ====================

def load_urls_from_csv(csv_file: str, url_column: str = 'url') -> List[str]:
    """Load car URLs from a CSV file."""
    try:
        df = pd.read_csv(csv_file)
        
        if url_column not in df.columns:
            print(f"❌ Column '{url_column}' not found in CSV")
            print(f"   Available columns: {', '.join(df.columns)}")
            return []
        
        urls = df[url_column].dropna().tolist()
        print(f"✅ Loaded {len(urls)} URLs from {csv_file}")
        return urls
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return []


def load_urls_from_file(txt_file: str) -> List[str]:
    """Load car URLs from a text file (one URL per line)."""
    try:
        with open(txt_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and line.strip().startswith('http')]
        print(f"✅ Loaded {len(urls)} URLs from {txt_file}")
        return urls
    except Exception as e:
        print(f"❌ Error loading text file: {e}")
        return []


# ==================== MAIN ENTRY POINT ====================

def main():
    """Main entry point for the scraper."""
    parser = argparse.ArgumentParser(
        description='Scrape detailed car information from bilbasen.dk URLs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape from CSV file with URLs
  python bilbasen_detail_scraper.py --csv car_urls.csv
  
  # Scrape from text file with URLs (one per line)
  python bilbasen_detail_scraper.py --txt urls.txt
  
  # Scrape specific URLs
  python bilbasen_detail_scraper.py --urls "URL1" "URL2" "URL3"
  
  # Limit number of cars and specify output
  python bilbasen_detail_scraper.py --csv car_urls.csv --max-cars 100 --output results.csv
  
  # Don't resume from progress (start fresh)
  python bilbasen_detail_scraper.py --csv car_urls.csv --no-resume
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--csv', type=str, help='CSV file containing URLs')
    input_group.add_argument('--txt', type=str, help='Text file containing URLs (one per line)')
    input_group.add_argument('--urls', nargs='+', help='List of URLs to scrape')
    
    # Optional arguments
    parser.add_argument('--output', '-o', type=str, default=DEFAULT_OUTPUT_CSV,
                       help=f'Output CSV file (default: {DEFAULT_OUTPUT_CSV})')
    parser.add_argument('--max-cars', '-m', type=int, default=None,
                       help='Maximum number of cars to scrape (default: all)')
    parser.add_argument('--url-column', type=str, default='url',
                       help='Column name containing URLs in CSV (default: url)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start from scratch instead of resuming from progress')
    parser.add_argument('--delay', type=float, default=REQUEST_DELAY,
                       help=f'Delay between requests in seconds (default: {REQUEST_DELAY})')
    
    args = parser.parse_args()
    
    # Update delay if specified (modify global)
    if args.delay != REQUEST_DELAY:
        globals()['REQUEST_DELAY'] = args.delay
    
    # Check Selenium availability
    if not SELENIUM_AVAILABLE:
        print("❌ Selenium is not installed!")
        print("   Install with: pip install selenium")
        print("   On Raspberry Pi, also install: sudo apt install chromium-chromedriver")
        sys.exit(1)
    
    # Load URLs based on input method
    car_urls = []
    
    if args.csv:
        car_urls = load_urls_from_csv(args.csv, args.url_column)
    elif args.txt:
        car_urls = load_urls_from_file(args.txt)
    elif args.urls:
        car_urls = args.urls
        print(f"✅ Using {len(car_urls)} URLs from command line")
    
    if not car_urls:
        print("❌ No URLs to scrape!")
        sys.exit(1)
    
    # Run scraper
    try:
        df = scrape_cars_from_urls(
            car_urls=car_urls,
            output_csv=args.output,
            max_cars=args.max_cars,
            resume=not args.no_resume
        )
        
        print("✅ Scraping completed successfully!")
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Scraping interrupted by user")
        print("   Progress has been saved. Run again to resume.")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
