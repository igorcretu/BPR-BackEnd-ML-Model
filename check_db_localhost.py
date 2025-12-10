import psycopg2
from psycopg2.extras import RealDictCursor

# Database connection parameters for localhost
DB_PARAMS = {
    'dbname': 'car_prediction',
    'user': 'bpr_user',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432'
}

def check_database():
    """Check database state for images"""
    try:
        print("Connecting to database...")
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check overall statistics
        print("\n" + "="*80)
        print("DATABASE IMAGE STATISTICS")
        print("="*80)
        
        cur.execute("""
            SELECT 
                COUNT(*) as total_cars,
                COUNT(image_path) as cars_with_path,
                COUNT(CASE WHEN image_path IS NOT NULL AND image_path != '' THEN 1 END) as cars_with_non_empty_path,
                COUNT(CASE WHEN image_downloaded = true THEN 1 END) as cars_with_downloaded_true,
                COUNT(CASE WHEN image_downloaded = false THEN 1 END) as cars_with_downloaded_false
            FROM cars
        """)
        
        stats = cur.fetchone()
        print(f"\nTotal cars: {stats['total_cars']}")
        print(f"Cars with image_path: {stats['cars_with_path']}")
        print(f"Cars with non-empty image_path: {stats['cars_with_non_empty_path']}")
        print(f"Cars with image_downloaded=TRUE: {stats['cars_with_downloaded_true']}")
        print(f"Cars with image_downloaded=FALSE: {stats['cars_with_downloaded_false']}")
        
        # Check the specific car from the error
        print("\n" + "="*80)
        print("CHECKING SPECIFIC CAR FROM ERROR")
        print("="*80)
        
        car_id = "33e6d46f-f8b3-497d-88ac-21b34ca05d23"
        cur.execute("""
            SELECT id, brand, model, year, image_path, image_downloaded
            FROM cars
            WHERE id = %s
        """, (car_id,))
        
        car = cur.fetchone()
        
        if car:
            print(f"\nCar ID: {car['id']}")
            print(f"Brand: {car['brand']}")
            print(f"Model: {car['model']}")
            print(f"Year: {car['year']}")
            print(f"Image Path: {car['image_path']}")
            print(f"Image Downloaded: {car['image_downloaded']}")
        else:
            print(f"\n❌ Car with ID {car_id} NOT FOUND in database")
        
        # Show sample of cars with images
        print("\n" + "="*80)
        print("SAMPLE OF CARS WITH IMAGE DATA")
        print("="*80)
        
        cur.execute("""
            SELECT id, brand, model, image_path, image_downloaded
            FROM cars
            WHERE image_path IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 10
        """)
        
        samples = cur.fetchall()
        print(f"\nShowing {len(samples)} recent cars with image_path:")
        for i, car in enumerate(samples, 1):
            print(f"\n{i}. {car['brand']} {car['model']}")
            print(f"   ID: {car['id']}")
            print(f"   Image Path: {car['image_path']}")
            print(f"   Downloaded: {car['image_downloaded']}")
        
        cur.close()
        conn.close()
        
        print("\n" + "="*80)
        print("✅ Database check complete")
        print("="*80)
        
    except psycopg2.OperationalError as e:
        print(f"\n❌ Database connection error: {e}")
        print("\nPossible issues:")
        print("  - PostgreSQL is not running on localhost:5432")
        print("  - Database 'car_prediction' doesn't exist")
        print("  - User 'bpr_user' doesn't have access")
        print("  - Password is incorrect")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    check_database()
