import psycopg2

conn = psycopg2.connect(
    dbname='car_price_prediction',
    user='postgres',
    password='postgres',
    host='localhost',
    port='5432'
)

cur = conn.cursor()
cur.execute("SELECT external_id, brand, model, image_path FROM cars LIMIT 20")
rows = cur.fetchall()

print("First 20 cars with image_path:")
print("-" * 80)
for r in rows:
    print(f"{r[0]} - {r[1]} {r[2]} - image_path: '{r[3]}'")

print("\n" + "=" * 80)
cur.execute("SELECT COUNT(*) FROM cars WHERE image_path IS NOT NULL")
with_images = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM cars")
total = cur.fetchone()[0]
print(f"\nTotal cars: {total}")
print(f"Cars with image_path: {with_images}")
print(f"Cars without image_path: {total - with_images}")

cur.close()
conn.close()
