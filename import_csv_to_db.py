#!/usr/bin/env python3
"""
Import scraped CSV data into PostgreSQL database
"""
import os
import sys
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import uuid

load_dotenv()

def import_csv_to_db(csv_path):
    """Import car data from CSV to database"""
    
    # Database connection
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'database': os.getenv('POSTGRES_DB', 'car_prediction'),
        'user': os.getenv('POSTGRES_USER', 'bpr_user'),
        'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
    }
    
    print(f"📖 Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows")
    
    # Connect to database
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    
    print("🗄️  Importing to database...")
    
    inserted = 0
    skipped = 0
    
    for _, row in df.iterrows():
        try:
            # Generate UUID if not present
            car_id = str(uuid.uuid4())
            
            # Insert car (adjust columns based on your CSV structure)
            cur.execute("""
                INSERT INTO cars (
                    id, brand, model, variant, title, description, price, 
                    model_year, year, mileage, fuel_type, transmission, 
                    body_type, horsepower, drive_type, doors, seats, color,
                    external_id, source_url, location, dealer_name
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (external_id) DO NOTHING
            """, (
                car_id,
                row.get('brand'),
                row.get('model'),
                row.get('variant'),
                row.get('title'),
                row.get('description'),
                row.get('price'),
                row.get('model_year'),
                row.get('year'),
                row.get('mileage'),
                row.get('fuel_type'),
                row.get('transmission'),
                row.get('body_type'),
                row.get('horsepower'),
                row.get('drive_type'),
                row.get('doors'),
                row.get('seats'),
                row.get('color'),
                row.get('external_id') or str(uuid.uuid4()),
                row.get('source_url'),
                row.get('location'),
                row.get('dealer_name')
            ))
            
            if cur.rowcount > 0:
                inserted += 1
            else:
                skipped += 1
                
        except Exception as e:
            print(f"⚠️  Error on row {_}: {e}")
            skipped += 1
            continue
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"\n✅ Import complete!")
    print(f"   Inserted: {inserted}")
    print(f"   Skipped: {skipped}")
    print(f"   Total: {len(df)}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python import_csv_to_db.py <csv_file>")
        print("Example: python import_csv_to_db.py bilbasen_scrape/car_details.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)
    
    import_csv_to_db(csv_path)
