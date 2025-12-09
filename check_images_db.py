#!/usr/bin/env python3
import psycopg2
import os

db_config = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'database': os.getenv('POSTGRES_DB', 'car_prediction'),
    'user': os.getenv('POSTGRES_USER', 'bpr_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
}

conn = psycopg2.connect(**db_config)
cur = conn.cursor()

print("Checking image_path in database...")
print("=" * 80)

cur.execute("SELECT external_id, brand, model, image_path FROM cars ORDER BY id LIMIT 20")
rows = cur.fetchall()

print("\nFirst 20 cars:")
for r in rows:
    img = r[3] if r[3] else "NULL"
    print(f"{r[0]:<10} {r[1]:<15} {r[2]:<20} image_path: {img}")

print("\n" + "=" * 80)
cur.execute("SELECT COUNT(*) FROM cars WHERE image_path IS NOT NULL AND image_path != ''")
with_images = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM cars")
total = cur.fetchone()[0]

print(f"\nTotal cars in database: {total}")
print(f"Cars with image_path: {with_images}")
print(f"Cars without image_path: {total - with_images}")

if with_images > 0:
    cur.execute("SELECT image_path FROM cars WHERE image_path IS NOT NULL AND image_path != '' LIMIT 5")
    samples = cur.fetchall()
    print("\nSample image_path values:")
    for s in samples:
        print(f"  {s[0]}")

cur.close()
conn.close()
