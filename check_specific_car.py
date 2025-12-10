import psycopg2
from psycopg2.extras import RealDictCursor
import os

# Database connection parameters
DB_PARAMS = {
    'dbname': os.getenv('DB_NAME', 'car_price_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'your_password'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
}

def check_car_image(car_id):
    """Check image details for a specific car"""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get car details
        cur.execute("""
            SELECT id, brand, model, image_path, image_downloaded
            FROM cars
            WHERE id = %s
        """, (car_id,))
        
        car = cur.fetchone()
        
        if car:
            print(f"\nCar ID: {car['id']}")
            print(f"Brand: {car['brand']}")
            print(f"Model: {car['model']}")
            print(f"Image Path: {car['image_path']}")
            print(f"Image Downloaded: {car['image_downloaded']}")
            
            # Check if image file exists
            if car['image_path']:
                # Check multiple possible locations
                possible_paths = [
                    f"bilbasen_scrape/{car['image_path']}",
                    car['image_path']
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        print(f"\n✓ Image file found at: {path}")
                        print(f"  File size: {os.path.getsize(path)} bytes")
                        break
                else:
                    print(f"\n✗ Image file NOT found in any of these locations:")
                    for path in possible_paths:
                        print(f"  - {path}")
        else:
            print(f"\nCar with ID {car_id} not found in database")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check the specific car from the error
    car_id = "33e6d46f-f8b3-497d-88ac-21b34ca05d23"
    check_car_image(car_id)
    
    # Also check a few random cars
    print("\n" + "="*60)
    print("Checking overall image statistics:")
    print("="*60)
    
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT 
                COUNT(*) as total_cars,
                COUNT(image_path) as cars_with_path,
                SUM(CASE WHEN image_downloaded THEN 1 ELSE 0 END) as cars_with_downloaded_flag
            FROM cars
        """)
        
        stats = cur.fetchone()
        print(f"\nTotal cars: {stats['total_cars']}")
        print(f"Cars with image_path: {stats['cars_with_path']}")
        print(f"Cars with image_downloaded=true: {stats['cars_with_downloaded_flag']}")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Error getting stats: {e}")
