# Car Price Prediction ML Model

Machine Learning model for predicting Danish car prices based on bilbasen.dk market data.

## 📋 Overview

This repository contains the complete ML pipeline for car price prediction:
- Web scraping scripts for data collection from bilbasen.dk
- Data cleaning and preprocessing notebooks
- Model training with multiple algorithms (XGBoost, CatBoost, LightGBM, Random Forest)
- Database integration for production deployment

## 🗂️ Project Structure

```
ML_Model/
├── bilbasen_parallel_scraper.py    # Parallel web scraper for car listings
├── bilbasen_detail_scraper.py      # Detailed car information scraper
├── upload.ipynb                    # Data cleaning & PostgreSQL upload pipeline
├── car_price_prediction_improved.ipynb  # Model training & evaluation
├── ss.ipynb                        # Supplementary analysis notebook
├── CSV_DATABASE_MAPPING.md         # Data schema documentation
├── model_results/                  # Trained models & metadata
│   ├── feature_statistics.json
│   ├── model_comparison_results.csv
│   └── model_metadata.json
├── scraped_data/                   # Raw scraped data
│   └── progress_*.csv
└── catboost_info/                  # CatBoost training logs
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.12+
PostgreSQL 16+
```

### Installation

```bash
# Clone repository
git clone https://github.com/igorcretu/BPR-BackEnd-ML-Model.git
cd BPR-BackEnd-ML-Model

# Install dependencies
pip install -r requirements.txt
```

### Required Python Packages

```
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
catboost>=1.2.0
lightgbm>=4.0.0
psycopg2-binary>=2.9.0
jupyter>=1.0.0
beautifulsoup4>=4.12.0
requests>=2.31.0
selenium>=4.15.0
```

## 📊 Data Pipeline

### 1. Data Collection (Scraping)

```bash
# Scrape car listings in parallel
python bilbasen_parallel_scraper.py

# Extract detailed car specifications
python bilbasen_detail_scraper.py
```

**Data Sources:**
- bilbasen.dk (Danish car marketplace)
- ~28,000+ car listings
- 50+ features per car

### 2. Data Cleaning & Upload

Run `upload.ipynb` notebook to:
- Standardize categorical values (fuel types, body types, transmissions)
- Handle missing data
- Normalize Danish to English labels
- Upload to PostgreSQL database

**Key Standardizations:**
- **Fuel Types:** 7 categories (Electricity, Petrol, Diesel, Hybrid - Petrol, Hybrid - Diesel, Plug-in Hybrid - Petrol, Plug-in Hybrid - Diesel)
- **Body Types:** 9 categories (Hatchback, SUV, Station Wagon, Sedan, Van, Cabriolet, Coupe, Pickup)
- **Transmissions:** 3 categories (Automatic, Manual, Semi-Automatic)
- **Drive Types:** 4 categories (Front-Wheel Drive, Rear-Wheel Drive, All-Wheel Drive, 4WD)

### 3. Model Training

Run `car_price_prediction_improved.ipynb` to:
- Feature engineering (age, mileage_per_year, power_to_weight)
- Train multiple models (XGBoost, CatBoost, LightGBM, Random Forest)
- Compare performance metrics (R², MAE, RMSE)
- Export best model to `../API/models/`

**Model Performance:**
- **Best Model:** XGBoost / CatBoost
- **R² Score:** ~0.85-0.90
- **MAE:** ~30,000-40,000 DKK
- **Confidence:** 70-95%

## 🔧 Database Schema

### PostgreSQL Connection

```python
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="car_prediction",
    user="postgres",
    password="your_password"
)
```

### Main Table: `cars`

See `CSV_DATABASE_MAPPING.md` for complete schema details.

**Key Columns:**
- `id` (UUID): Primary key
- `brand`, `model`, `variant`: Car identification
- `year`, `mileage`, `price`: Core metrics
- `fuel_type`, `transmission`, `body_type`: Categorical features
- `horsepower`, `engine_size`, `torque_nm`: Performance specs
- Electric vehicle specific: `range_km`, `battery_capacity`, `energy_consumption`
- Dimensions: `weight`, `length`, `width`, `height`, `trunk_size`

## 📈 Features Used for Prediction

### Numeric Features
- `age` (current_year - year)
- `mileage_numeric`
- `horsepower`
- `torque_nm`
- `engine_size`
- `weight`
- `trunk_size`
- `top_speed`
- `doors`, `seats`
- `range_km` (for electric)
- `battery_capacity` (for electric)
- `mileage_per_year` (derived)
- `power_to_weight` (derived)

### Categorical Features (Label Encoded)
- `brand`
- `fuel_type_en`
- `transmission_en`
- `body_type_en`
- `drive_type_en`
- `color`

### Boolean Features
- `is_electric`
- `is_hybrid`
- `is_automatic`
- `is_premium` (brand-based)

## 🎯 Model Integration

The trained model is deployed in the backend API:

```
BackEnd/API/models/
├── best_model_xgboost.pkl
├── feature_scaler.pkl
├── label_encoders.pkl
├── model_metadata.json
└── feature_statistics.json
```

**API Endpoint:** `POST /api/predict`

## 📝 Data Quality Rules

### Fuel Type Standardization
```python
# Danish → English mapping
'El' → 'Electricity'
'Benzin' → 'Petrol'
'Diesel' → 'Diesel'
'Hybrid Benzin' → 'Hybrid - Petrol'
'Plug-in hybrid Benzin' → 'Plug-in Hybrid - Petrol'
```

### Body Type Standardization
```python
# Danish → English mapping
'Halvkombi' → 'Hatchback'
'St.car' → 'Station Wagon'
'Mikro' → 'Hatchback'
'Kassevogn' → 'Van'
'Cabriolet' → 'Cabriolet'
```

### Electric Car Rules
- If `fuel_type == 'Electricity'` and `transmission` is null → set to `'Automatic'`
- All 11,301 electric cars have automatic transmission

### Mileage Handling
- New cars (mileage = null or N/A) → set to `0`
- Used for realistic predictions on brand new vehicles

## 🔄 Update Pipeline

To update the database with new scraped data:

1. Run scrapers to collect new listings
2. Open `upload.ipynb`
3. Execute cells in order:
   - Load and clean new data
   - Standardize categorical values
   - Run `bulk_update_changed_records_only()` function
   - Verify fuel types count: `SELECT fuel_type, COUNT(*) FROM cars GROUP BY fuel_type`

**Optimized Update:** Only updates records that have changed (comparison-based)

## 📊 Model Comparison Results

Models are evaluated on:
- R² Score (variance explained)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Training time
- Inference speed

Results saved in `model_results/model_comparison_results.csv`

## 🐛 Known Issues & Fixes

### Issue: 400 Bad Request on /api/predict
**Cause:** Predictor mappings don't recognize standardized database values  
**Fix:** Update `predictor.py` fuel_type_mapping, body_type_mapping, drive_type_mapping

### Issue: N/A Mileage Causing Errors
**Cause:** Prediction requires mileage as integer  
**Fix:** Set mileage=0 for new cars in API endpoint validation

### Issue: Database shows 10 fuel types instead of 7
**Cause:** Some records had variants (Electric, Hybrid, Plugin-Hybrid)  
**Fix:** Run direct SQL UPDATE to fix inconsistent records

## 📄 License

This project is part of a Bachelor's degree thesis.

## 👥 Authors

Igor Cretu - Bachelor Project 2025

## 🔗 Related Repositories

- [Backend API](https://github.com/igorcretu/BPR-BackEnd-API)
- [Frontend](https://github.com/igorcretu/BPR-FrontEnd)

## 📞 Contact

For questions about the ML model or data pipeline, please open an issue in this repository.
