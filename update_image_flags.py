#!/usr/bin/env python3
"""
Update image_downloaded flag for all cars that have an image_path
"""
import psycopg2
import os

db_config = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'database': os.getenv('POSTGRES_DB', 'car_prediction'),
    'user': os.getenv('POSTGRES_USER', 'bpr_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
}

print("🔧 Updating image_downloaded flag...")
print("=" * 80)

conn = psycopg2.connect(**db_config)
cur = conn.cursor()

# Check current state
cur.execute("SELECT COUNT(*) FROM cars WHERE image_path IS NOT NULL AND image_path != ''")
with_image_path = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM cars WHERE image_downloaded = true")
with_image_downloaded_true = cur.fetchone()[0]

print(f"\nBefore update:")
print(f"  Cars with image_path: {with_image_path}")
print(f"  Cars with image_downloaded=true: {with_image_downloaded_true}")

# Update all cars that have an image_path to have image_downloaded=true
cur.execute("""
    UPDATE cars 
    SET image_downloaded = true 
    WHERE image_path IS NOT NULL 
    AND image_path != ''
    AND image_downloaded = false
""")

rows_updated = cur.rowcount
conn.commit()

# Check after update
cur.execute("SELECT COUNT(*) FROM cars WHERE image_downloaded = true")
with_image_downloaded_true_after = cur.fetchone()[0]

print(f"\n✅ Updated {rows_updated} rows")
print(f"\nAfter update:")
print(f"  Cars with image_downloaded=true: {with_image_downloaded_true_after}")

# Show sample of updated records
cur.execute("""
    SELECT external_id, brand, model, image_path, image_downloaded 
    FROM cars 
    WHERE image_path IS NOT NULL 
    LIMIT 10
""")
samples = cur.fetchall()

print(f"\nSample of updated records:")
print("-" * 80)
for s in samples:
    print(f"{s[0]:<10} {s[1]:<15} {s[2]:<20} {s[3]:<25} downloaded={s[4]}")

cur.close()
conn.close()

print("\n" + "=" * 80)
print("✅ Done! All cars with image_path now have image_downloaded=true")
