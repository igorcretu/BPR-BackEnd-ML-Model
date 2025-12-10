#!/usr/bin/env python3
"""
Fix image_downloaded flags in database to match actual file existence
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor

# Configuration
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'car_prediction'),
    'user': os.getenv('DB_USER', 'bpr_user'),
    'password': os.getenv('DB_PASS', 'bpr_password'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
}

IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'bilbasen_scrape', 'images')

def main():
    print(f"Connecting to database at {DB_CONFIG['host']}:{DB_CONFIG['port']}...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # Get all cars with image_downloaded=true
    print("Fetching cars marked as having images...")
    cur.execute("""
        SELECT id, external_id, image_path, image_downloaded 
        FROM cars 
        WHERE image_downloaded = true
    """)
    
    cars = cur.fetchall()
    print(f"Found {len(cars)} cars marked with image_downloaded=true")
    
    fixed_count = 0
    missing_count = 0
    
    for car in cars:
        car_id = car['id']
        external_id = car['external_id']
        image_path = car['image_path']
        
        if not image_path:
            # No image path but marked as downloaded
            print(f"❌ Car {external_id}: No image_path but image_downloaded=true")
            cur.execute("""
                UPDATE cars 
                SET image_downloaded = false, image_path = NULL
                WHERE id = %s
            """, (car_id,))
            fixed_count += 1
            continue
        
        # Extract filename from path
        filename = os.path.basename(image_path)
        
        # Check if file exists
        file_path = os.path.join(IMAGES_DIR, filename)
        
        if not os.path.exists(file_path):
            # File doesn't exist but marked as downloaded
            print(f"❌ Car {external_id}: Image file missing at {file_path}")
            cur.execute("""
                UPDATE cars 
                SET image_downloaded = false
                WHERE id = %s
            """, (car_id,))
            fixed_count += 1
            missing_count += 1
        else:
            # File exists and flag is correct
            pass
    
    conn.commit()
    
    print(f"\n✅ Fixed {fixed_count} incorrect image_downloaded flags")
    print(f"📊 {missing_count} cars have missing image files")
    
    # Now check cars marked as NOT having images
    print("\nChecking cars marked as NOT having images...")
    cur.execute("""
        SELECT id, external_id, image_path, image_downloaded 
        FROM cars 
        WHERE image_downloaded = false OR image_downloaded IS NULL
    """)
    
    cars_no_images = cur.fetchall()
    print(f"Found {len(cars_no_images)} cars marked without images")
    
    found_count = 0
    for car in cars_no_images:
        car_id = car['id']
        external_id = car['external_id']
        
        # Check if file actually exists
        filename = f"{external_id}.jpg"
        file_path = os.path.join(IMAGES_DIR, filename)
        
        if os.path.exists(file_path):
            # File exists but not marked
            print(f"✅ Car {external_id}: Found image file, updating database")
            cur.execute("""
                UPDATE cars 
                SET image_path = %s, image_downloaded = true
                WHERE id = %s
            """, (f"images/{filename}", car_id))
            found_count += 1
    
    conn.commit()
    
    print(f"\n✅ Updated {found_count} cars that had images but weren't marked")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Fixed incorrect flags: {fixed_count}")
    print(f"Missing image files: {missing_count}")
    print(f"Found unmarked images: {found_count}")
    
    cur.close()
    conn.close()
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
