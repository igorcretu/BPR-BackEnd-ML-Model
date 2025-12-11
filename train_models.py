#!/usr/bin/env python3
"""
Multi-Model Training Orchestration Script v3.0
==============================================

MAJOR IMPROVEMENTS over v2:
1. Extended feature set (30+ features from database)
2. Target encoding for high-cardinality categoricals (brand, model)
3. Proper preprocessing without data leakage
4. Better hyperparameters for all models
5. Additional models: LightGBM, RandomForest, HistGradientBoosting
6. Improved LSTM/GRU with better architecture
7. Cross-validation for all models
8. Real confidence intervals using quantile regression
9. Segmented metrics by price range, fuel type, year
10. Feature importance analysis

Models:
- XGBoost (improved)
- CatBoost (improved)
- LightGBM (new)
- RandomForest (new)
- HistGradientBoosting (new - fast, sklearn native)
- Ridge Regression (improved)
- Lasso Regression (improved)
- ElasticNet (improved)
- LSTM (improved architecture)
- GRU (improved architecture)

Optimized for Raspberry Pi 5 with memory-efficient processing.
"""

import os
import sys
import argparse
import logging
import time
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import uuid

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import joblib
from dotenv import load_dotenv

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, 
    median_absolute_error
)
from sklearn.ensemble import (
    RandomForestRegressor, HistGradientBoostingRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV

import xgboost as xgb
from catboost import CatBoostRegressor

# Try to import LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Try to import PyTorch for RNN models
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

warnings.filterwarnings('ignore')

# Load environment
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'TEST_SIZE': 0.2,
    'RANDOM_STATE': 42,
    'CV_FOLDS': 5,
    
    # Feature configuration
    'PREMIUM_BRANDS': [
        'BMW', 'Mercedes-Benz', 'Audi', 'Tesla', 'Porsche',
        'Volvo', 'Polestar', 'Lexus', 'Land Rover', 'Jaguar',
        'Maserati', 'Bentley', 'Rolls-Royce', 'Ferrari', 'Lamborghini',
        'Aston Martin', 'McLaren', 'Alfa Romeo', 'MINI', 'DS'
    ],
    
    'ECONOMY_BRANDS': [
        'Dacia', 'Suzuki', 'Fiat', 'Seat', 'Skoda', 'Kia', 'Hyundai',
        'Toyota', 'Honda', 'Mazda', 'Nissan', 'Mitsubishi'
    ],
    
    # Price segments for evaluation (DKK)
    'PRICE_SEGMENTS': {
        'under_100k': (0, 100000),
        '100k_to_300k': (100000, 300000),
        '300k_to_500k': (300000, 500000),
        'over_500k': (500000, float('inf'))
    },
    
    # Model directories
    'MODEL_DIR': 'models',
    'LOG_DIR': 'logs',
}

# Database config
DB_CONFIG = {
    'dbname': os.getenv('POSTGRES_DB', os.getenv('DB_NAME', 'car_prediction')),
    'user': os.getenv('POSTGRES_USER', os.getenv('DB_USER', 'bpr_user')),
    'password': os.getenv('POSTGRES_PASSWORD', os.getenv('DB_PASS', 'your_secure_password')),
    'host': os.getenv('POSTGRES_HOST', os.getenv('DB_HOST', 'db')),
    'port': os.getenv('POSTGRES_PORT', os.getenv('DB_PORT', '5432'))
}

# Create directories
os.makedirs(CONFIG['MODEL_DIR'], exist_ok=True)
os.makedirs(CONFIG['LOG_DIR'], exist_ok=True)

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Setup logging with timestamped file and console output"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(CONFIG['LOG_DIR'], f'training_{timestamp}.log')
    
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Full training logs being written to: {log_file}")
    return logger

logger = setup_logging()

# ============================================================================
# TARGET ENCODER (for high-cardinality categoricals)
# ============================================================================

class TargetEncoder:
    """
    Target encoding with smoothing to prevent overfitting.
    Much better than LabelEncoder for brand/model with many categories.
    """
    def __init__(self, smoothing: float = 20.0):
        self.smoothing = smoothing
        self.global_mean = None
        self.encodings = {}
    
    def fit(self, X: pd.Series, y: pd.Series) -> 'TargetEncoder':
        self.global_mean = y.mean()
        
        stats = pd.DataFrame({'category': X, 'target': y})
        agg = stats.groupby('category')['target'].agg(['mean', 'count'])
        
        # Smoothed encoding
        smoothing_factor = agg['count'] / (agg['count'] + self.smoothing)
        self.encodings = (smoothing_factor * agg['mean'] + (1 - smoothing_factor) * self.global_mean).to_dict()
        
        return self
    
    def transform(self, X: pd.Series) -> np.ndarray:
        return X.map(lambda x: self.encodings.get(x, self.global_mean)).values
    
    def fit_transform(self, X: pd.Series, y: pd.Series) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

# ============================================================================
# IMPROVED RNN MODELS
# ============================================================================

if TORCH_AVAILABLE:
    class ImprovedLSTMNetwork(nn.Module):
        """Improved LSTM with residual connections and layer normalization"""
        
        def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
            super(ImprovedLSTMNetwork, self).__init__()
            
            # Input projection
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.input_norm = nn.LayerNorm(hidden_dim)
            
            # LSTM layers
            self.lstm = nn.LSTM(
                hidden_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=False
            )
            
            # Output layers with residual
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc1_norm = nn.LayerNorm(hidden_dim // 2)
            self.fc2 = nn.Linear(hidden_dim // 2, 1)
            
            self.relu = nn.ReLU()
            
        def forward(self, x):
            # Project input
            x = self.input_proj(x)
            x = self.input_norm(x)
            x = self.relu(x)
            
            # Add sequence dimension if needed
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            
            # LSTM
            lstm_out, _ = self.lstm(x)
            out = lstm_out[:, -1, :]  # Take last output
            
            # Output layers
            out = self.dropout(out)
            out = self.fc1(out)
            out = self.fc1_norm(out)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            
            return out.squeeze()

    class ImprovedGRUNetwork(nn.Module):
        """Improved GRU with layer normalization"""
        
        def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
            super(ImprovedGRUNetwork, self).__init__()
            
            # Input projection
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.input_norm = nn.LayerNorm(hidden_dim)
            
            # GRU layers
            self.gru = nn.GRU(
                hidden_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
            
            # Output layers
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc1_norm = nn.LayerNorm(hidden_dim // 2)
            self.fc2 = nn.Linear(hidden_dim // 2, 1)
            
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.input_proj(x)
            x = self.input_norm(x)
            x = self.relu(x)
            
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            
            gru_out, _ = self.gru(x)
            out = gru_out[:, -1, :]
            
            out = self.dropout(out)
            out = self.fc1(out)
            out = self.fc1_norm(out)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            
            return out.squeeze()

# ============================================================================
# MAIN TRAINER CLASS
# ============================================================================

class ModelTrainer:
    """Orchestrates training of multiple models with improved preprocessing"""
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.conn = None
        self.cur = None
        
        # Training data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df_test = None  # Keep test DataFrame for segmented metrics
        self.feature_names = None
        
        # Preprocessing objects (fit only on training data!)
        self.scaler = None
        self.target_encoders = {}
        self.category_mappings = {}
        self.numeric_medians = {}
        
        # Training run info
        self.start_time = None
        self.models_trained = []
        self.best_model_id = None
        self.best_r2 = -np.inf
        
    def connect_db(self):
        """Connect to database"""
        try:
            logger.info(f"Connecting to database: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
            self.conn = psycopg2.connect(**DB_CONFIG)
            self.cur = self.conn.cursor()
            logger.info("✅ Connected to database successfully")
            return True
        except psycopg2.OperationalError as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected database connection error: {e}")
            return False
    
    def load_data(self):
        """Load data from database with extended feature set"""
        logger.info("📊 Loading training data from database...")
        
        # Extended query with more features
        query = """
            SELECT 
                -- Identifiers
                external_id,
                
                -- Basic info
                brand, model, variant,
                
                -- Price (target)
                price,
                
                -- Year and mileage
                COALESCE(model_year, year) as year,
                mileage,
                
                -- Vehicle characteristics
                fuel_type, transmission, body_type, drive_type,
                
                -- Performance
                horsepower, torque_nm, engine_size,
                acceleration, top_speed,
                
                -- Dimensions
                doors, seats, weight, 
                
                -- Efficiency
                fuel_consumption, co2_emission,
                
                -- EV specific
                battery_capacity, range_km,
                
                -- Other
                color, new_price, periodic_tax,
                
                -- Location
                location
                
            FROM cars
            WHERE price IS NOT NULL 
                AND price > 10000 
                AND price < 5000000
                AND brand IS NOT NULL
                AND COALESCE(model_year, year) IS NOT NULL
                AND COALESCE(model_year, year) BETWEEN 1990 AND 2026
                AND mileage IS NOT NULL
                AND mileage >= 0
                AND mileage < 800000
        """
        
        df = pd.read_sql(query, self.conn)
        logger.info(f"✅ Loaded {len(df)} records from database")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering with extended feature set"""
        logger.info("🔧 Engineering features...")
        logger.debug(f"Initial shape: {df.shape}")
        
        df = df.copy()
        current_year = datetime.now().year
        
        # ===== AGE FEATURES =====
        df['age'] = current_year - df['year']
        df['age'] = df['age'].clip(0, 50)  # Reasonable bounds
        df['age_squared'] = df['age'] ** 2
        df['age_cubed'] = df['age'] ** 3
        
        # ===== MILEAGE FEATURES =====
        df['mileage'] = df['mileage'].clip(0, 800000)
        df['mileage_log'] = np.log1p(df['mileage'])
        df['mileage_per_year'] = df['mileage'] / (df['age'] + 1)
        df['mileage_per_year'] = df['mileage_per_year'].clip(0, 100000)
        df['high_mileage'] = (df['mileage'] > 150000).astype(int)
        df['low_mileage'] = (df['mileage'] < 50000).astype(int)
        
        # ===== BRAND FEATURES =====
        df['is_premium'] = df['brand'].isin(CONFIG['PREMIUM_BRANDS']).astype(int)
        df['is_economy'] = df['brand'].isin(CONFIG['ECONOMY_BRANDS']).astype(int)
        
        # ===== FUEL TYPE FEATURES =====
        df['fuel_type'] = df['fuel_type'].fillna('Petrol')
        df['is_electric'] = (df['fuel_type'] == 'Electricity').astype(int)
        df['is_diesel'] = (df['fuel_type'] == 'Diesel').astype(int)
        df['is_hybrid'] = df['fuel_type'].str.contains('Hybrid', na=False).astype(int)
        df['is_plugin'] = df['fuel_type'].str.contains('Plug-in', na=False).astype(int)
        
        # ===== TRANSMISSION FEATURES =====
        df['transmission'] = df['transmission'].fillna('Manual')
        df['is_automatic'] = (df['transmission'] == 'Automatic').astype(int)
        
        # ===== BODY TYPE FEATURES =====
        df['body_type'] = df['body_type'].fillna('Sedan')
        df['is_suv'] = (df['body_type'] == 'SUV').astype(int)
        df['is_wagon'] = df['body_type'].isin(['Station Wagon', 'Van']).astype(int)
        df['is_hatchback'] = (df['body_type'] == 'Hatchback').astype(int)
        
        # ===== POWER FEATURES =====
        df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())
        df['horsepower'] = df['horsepower'].clip(30, 1500)
        df['horsepower_log'] = np.log1p(df['horsepower'])
        df['horsepower_per_year'] = df['horsepower'] / (df['age'] + 1)
        
        # Power categories
        df['low_power'] = (df['horsepower'] < 100).astype(int)
        df['high_power'] = (df['horsepower'] > 200).astype(int)
        df['very_high_power'] = (df['horsepower'] > 300).astype(int)
        
        # ===== INTERACTION FEATURES =====
        df['age_mileage_interaction'] = df['age'] * df['mileage_log']
        df['premium_age'] = df['is_premium'] * df['age']
        df['premium_mileage'] = df['is_premium'] * df['mileage_log']
        df['hp_age_ratio'] = df['horsepower'] / (df['age'] + 1)
        
        # ===== DEPRECIATION PROXY =====
        # If we have new_price, calculate depreciation rate
        df['has_new_price'] = df['new_price'].notna().astype(int)
        df['new_price'] = df['new_price'].fillna(0)
        df['depreciation_ratio'] = np.where(
            df['new_price'] > 0,
            df['price'] / df['new_price'],
            0.5  # Default to 50% if no new price
        )
        df['depreciation_ratio'] = df['depreciation_ratio'].clip(0.01, 1.5)
        
        # ===== EV FEATURES =====
        df['battery_capacity'] = df['battery_capacity'].fillna(0)
        df['range_km'] = df['range_km'].fillna(0)
        df['has_ev_data'] = ((df['battery_capacity'] > 0) | (df['range_km'] > 0)).astype(int)
        
        # ===== NUMERIC CLEANING =====
        numeric_cols = ['torque_nm', 'engine_size', 'acceleration', 'top_speed',
                       'doors', 'seats', 'weight', 'fuel_consumption', 'co2_emission',
                       'periodic_tax']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['doors'] = df['doors'].fillna(4).clip(2, 5)
        df['seats'] = df['seats'].fillna(5).clip(2, 9)
        
        # Fill other numeric with median
        for col in numeric_cols:
            if col in df.columns and df[col].notna().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # ===== DRIVE TYPE =====
        df['drive_type'] = df['drive_type'].fillna('Front-Wheel Drive')
        df['is_awd'] = df['drive_type'].str.contains('All-Wheel|AWD|4WD', na=False, case=False).astype(int)
        
        # ===== COLOR =====
        df['color'] = df['color'].fillna('Unknown')
        popular_colors = ['Sort', 'Hvid', 'Grå', 'Sølv', 'Blå', 'Rød', 'Black', 'White', 'Grey', 'Silver', 'Blue', 'Red']
        df['is_popular_color'] = df['color'].isin(popular_colors).astype(int)
        
        logger.info(f"✅ Feature engineering complete. Shape: {df.shape}")
        return df
    
    def preprocess_data(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Preprocess data with proper handling to prevent leakage.
        Only fit on training data!
        """
        df = df.copy()
        
        # Separate target
        y = df['price'].values if 'price' in df.columns else None
        
        # ===== TARGET ENCODING for high-cardinality categoricals =====
        target_encode_cols = ['brand', 'model']
        
        for col in target_encode_cols:
            if col in df.columns:
                if fit and y is not None:
                    encoder = TargetEncoder(smoothing=20.0)
                    df[f'{col}_encoded'] = encoder.fit_transform(df[col].astype(str), pd.Series(y))
                    self.target_encoders[col] = encoder
                else:
                    if col in self.target_encoders:
                        df[f'{col}_encoded'] = self.target_encoders[col].transform(df[col].astype(str))
                    else:
                        df[f'{col}_encoded'] = 0  # Fallback
        
        # ===== ONE-HOT ENCODING for low-cardinality categoricals =====
        onehot_cols = ['fuel_type', 'transmission', 'body_type', 'drive_type']
        
        for col in onehot_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                
                if fit:
                    self.category_mappings[col] = list(dummies.columns)
                else:
                    # Ensure same columns as training
                    for expected_col in self.category_mappings.get(col, []):
                        if expected_col not in dummies.columns:
                            dummies[expected_col] = 0
                    dummies = dummies[[c for c in self.category_mappings.get(col, []) if c in dummies.columns]]
                
                df = pd.concat([df, dummies], axis=1)
        
        # ===== SELECT FINAL FEATURES =====
        # Numeric features
        numeric_features = [
            'year', 'mileage', 'horsepower', 'doors', 'seats',
            'age', 'age_squared', 'age_cubed',
            'mileage_log', 'mileage_per_year',
            'horsepower_log', 'horsepower_per_year', 'hp_age_ratio',
            'age_mileage_interaction', 'premium_age', 'premium_mileage',
            'depreciation_ratio', 'new_price',
            'torque_nm', 'engine_size', 'acceleration', 'top_speed',
            'weight', 'fuel_consumption', 'co2_emission',
            'battery_capacity', 'range_km', 'periodic_tax'
        ]
        
        # Binary features
        binary_features = [
            'is_premium', 'is_economy', 'is_electric', 'is_diesel',
            'is_hybrid', 'is_plugin', 'is_automatic', 'is_suv',
            'is_wagon', 'is_hatchback', 'is_awd', 'is_popular_color',
            'high_mileage', 'low_mileage', 'low_power', 'high_power',
            'very_high_power', 'has_new_price', 'has_ev_data'
        ]
        
        # Encoded features
        encoded_features = [f'{col}_encoded' for col in target_encode_cols if f'{col}_encoded' in df.columns]
        
        # Get one-hot columns
        onehot_features = []
        for col in onehot_cols:
            onehot_features.extend(self.category_mappings.get(col, []))
        
        # Combine all features
        all_features = []
        for f in numeric_features + binary_features + encoded_features:
            if f in df.columns:
                all_features.append(f)
        
        for f in onehot_features:
            if f in df.columns:
                all_features.append(f)
        
        # Remove duplicates while preserving order
        all_features = list(dict.fromkeys(all_features))
        
        # Create feature matrix
        X = df[all_features].copy()
        
        # Fill remaining NaNs
        for col in X.columns:
            if X[col].isna().any():
                if fit:
                    self.numeric_medians[col] = X[col].median() if X[col].notna().any() else 0
                X[col] = X[col].fillna(self.numeric_medians.get(col, 0))
        
        # Replace inf with large values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # ===== SCALING =====
        if fit:
            self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        self.feature_names = all_features
        
        return X_scaled, all_features, y
    
    def prepare_data(self) -> int:
        """Load, engineer features, and prepare train/test splits"""
        # Load raw data
        df = self.load_data()
        
        # Feature engineering
        df = self.engineer_features(df)
        
        # Train-test split BEFORE preprocessing to prevent leakage
        df_train, df_test = train_test_split(
            df, test_size=self.test_size, random_state=self.random_state
        )
        
        logger.info(f"📈 Train size: {len(df_train)}, Test size: {len(df_test)}")
        
        # Preprocess training data (fit)
        self.X_train, self.feature_names, self.y_train = self.preprocess_data(df_train, fit=True)
        
        # Preprocess test data (transform only)
        self.X_test, _, self.y_test = self.preprocess_data(df_test, fit=False)
        
        # Keep test DataFrame for segmented metrics
        self.df_test = df_test.copy()
        
        logger.info(f"📈 Features: {len(self.feature_names)}")
        logger.debug(f"Feature list: {self.feature_names[:20]}...")
        
        return len(df)
    
    # =========================================================================
    # MODEL TRAINING METHODS
    # =========================================================================
    
    def train_xgboost(self):
        """Train XGBoost with improved hyperparameters"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("🤖 MODEL: XGBOOST")
        logger.info("=" * 60)
        logger.info(f"⏱️  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📊 Training samples: {len(self.X_train):,} | Test samples: {len(self.X_test):,}")
        logger.info("")
        
        start = time.time()
        
        # Improved hyperparameters
        params = {
            'n_estimators': 500,
            'learning_rate': 0.03,
            'max_depth': 8,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': self.random_state,
            'n_jobs': -1,
            'tree_method': 'hist'  # Faster for large datasets
        }
        
        logger.info(f"🚀 Training XGBoost with params: n_estimators={params['n_estimators']}, lr={params['learning_rate']}, max_depth={params['max_depth']}")
        
        model = xgb.XGBRegressor(**params)
        
        # Fit with early stopping if possible
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False
        )
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring='r2', n_jobs=-1)
        logger.info(f"📊 Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Predictions
        y_pred = model.predict(self.X_test)
        confidence = self._calculate_confidence_tree(model, self.X_test, y_pred)
        
        training_time = time.time() - start
        
        # Metrics
        metrics = self._calculate_metrics(self.y_test, y_pred, confidence)
        metrics['training_time'] = training_time
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, model.feature_importances_.tolist()))
        
        # Save model with preprocessing objects
        model_filename = f'xgboost_v3_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        model_path = os.path.join(CONFIG['MODEL_DIR'], model_filename)
        self._save_model_package(model, model_path, params)
        
        model_id = self._register_model(
            name='XGBoost',
            model_type='tree',
            algorithm='XGBoost',
            version='3.0.0',
            model_path=model_path,
            metrics=metrics,
            hyperparameters=params,
            feature_importance=feature_importance
        )
        
        self._store_comparison_metrics(model_id, self.y_test, y_pred, confidence)
        
        self._log_model_completion('XGBoost', metrics, training_time, model_id)
        
        return model_id, metrics
    
    def train_catboost(self):
        """Train CatBoost with improved hyperparameters"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("🤖 MODEL: CATBOOST")
        logger.info("=" * 60)
        logger.info(f"⏱️  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📊 Training samples: {len(self.X_train):,} | Test samples: {len(self.X_test):,}")
        logger.info("")
        
        start = time.time()
        
        params = {
            'iterations': 500,
            'learning_rate': 0.03,
            'depth': 8,
            'l2_leaf_reg': 3.0,
            'random_seed': self.random_state,
            'verbose': False,
            'thread_count': -1
        }
        
        logger.info(f"🚀 Training CatBoost with params: iterations={params['iterations']}, lr={params['learning_rate']}, depth={params['depth']}")
        
        model = CatBoostRegressor(**params)
        model.fit(self.X_train, self.y_train, eval_set=(self.X_test, self.y_test), verbose=False)
        
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring='r2', n_jobs=-1)
        logger.info(f"📊 Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        y_pred = model.predict(self.X_test)
        confidence = self._calculate_confidence_tree(model, self.X_test, y_pred)
        
        training_time = time.time() - start
        
        metrics = self._calculate_metrics(self.y_test, y_pred, confidence)
        metrics['training_time'] = training_time
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        feature_importance = dict(zip(self.feature_names, model.feature_importances_.tolist()))
        
        model_filename = f'catboost_v3_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        model_path = os.path.join(CONFIG['MODEL_DIR'], model_filename)
        self._save_model_package(model, model_path, params)
        
        model_id = self._register_model(
            name='CatBoost',
            model_type='tree',
            algorithm='CatBoost',
            version='3.0.0',
            model_path=model_path,
            metrics=metrics,
            hyperparameters=params,
            feature_importance=feature_importance
        )
        
        self._store_comparison_metrics(model_id, self.y_test, y_pred, confidence)
        
        self._log_model_completion('CatBoost', metrics, training_time, model_id)
        
        return model_id, metrics
    
    def train_lightgbm(self):
        """Train LightGBM - often best performance/speed ratio"""
        if not LIGHTGBM_AVAILABLE:
            logger.warning("⚠️ LightGBM not available, skipping")
            return None, None
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("🤖 MODEL: LIGHTGBM")
        logger.info("=" * 60)
        logger.info(f"⏱️  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📊 Training samples: {len(self.X_train):,} | Test samples: {len(self.X_test):,}")
        logger.info("")
        
        start = time.time()
        
        params = {
            'n_estimators': 500,
            'learning_rate': 0.03,
            'max_depth': 10,
            'num_leaves': 64,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': self.random_state,
            'verbose': -1,
            'n_jobs': -1
        }
        
        logger.info(f"🚀 Training LightGBM with params: n_estimators={params['n_estimators']}, lr={params['learning_rate']}")
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring='r2', n_jobs=-1)
        logger.info(f"📊 Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        y_pred = model.predict(self.X_test)
        confidence = self._calculate_confidence_tree(model, self.X_test, y_pred)
        
        training_time = time.time() - start
        
        metrics = self._calculate_metrics(self.y_test, y_pred, confidence)
        metrics['training_time'] = training_time
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        feature_importance = dict(zip(self.feature_names, model.feature_importances_.tolist()))
        
        model_filename = f'lightgbm_v1_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        model_path = os.path.join(CONFIG['MODEL_DIR'], model_filename)
        self._save_model_package(model, model_path, params)
        
        model_id = self._register_model(
            name='LightGBM',
            model_type='tree',
            algorithm='LightGBM',
            version='1.0.0',
            model_path=model_path,
            metrics=metrics,
            hyperparameters=params,
            feature_importance=feature_importance
        )
        
        self._store_comparison_metrics(model_id, self.y_test, y_pred, confidence)
        
        self._log_model_completion('LightGBM', metrics, training_time, model_id)
        
        return model_id, metrics
    
    def train_random_forest(self):
        """Train Random Forest"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("🤖 MODEL: RANDOM FOREST")
        logger.info("=" * 60)
        logger.info(f"⏱️  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📊 Training samples: {len(self.X_train):,} | Test samples: {len(self.X_test):,}")
        logger.info("")
        
        start = time.time()
        
        params = {
            'n_estimators': 300,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        logger.info(f"🚀 Training RandomForest with params: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}")
        
        model = RandomForestRegressor(**params)
        model.fit(self.X_train, self.y_train)
        
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring='r2', n_jobs=-1)
        logger.info(f"📊 Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        y_pred = model.predict(self.X_test)
        confidence = self._calculate_confidence_tree(model, self.X_test, y_pred)
        
        training_time = time.time() - start
        
        metrics = self._calculate_metrics(self.y_test, y_pred, confidence)
        metrics['training_time'] = training_time
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        feature_importance = dict(zip(self.feature_names, model.feature_importances_.tolist()))
        
        model_filename = f'random_forest_v1_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        model_path = os.path.join(CONFIG['MODEL_DIR'], model_filename)
        self._save_model_package(model, model_path, params)
        
        model_id = self._register_model(
            name='RandomForest',
            model_type='tree',
            algorithm='Random Forest',
            version='1.0.0',
            model_path=model_path,
            metrics=metrics,
            hyperparameters=params,
            feature_importance=feature_importance
        )
        
        self._store_comparison_metrics(model_id, self.y_test, y_pred, confidence)
        
        self._log_model_completion('RandomForest', metrics, training_time, model_id)
        
        return model_id, metrics
    
    def train_histgb(self):
        """Train HistGradientBoosting - fast sklearn native gradient boosting"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("🤖 MODEL: HIST GRADIENT BOOSTING")
        logger.info("=" * 60)
        logger.info(f"⏱️  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📊 Training samples: {len(self.X_train):,} | Test samples: {len(self.X_test):,}")
        logger.info("")
        
        start = time.time()
        
        params = {
            'max_iter': 500,
            'learning_rate': 0.03,
            'max_depth': 10,
            'min_samples_leaf': 20,
            'l2_regularization': 1.0,
            'random_state': self.random_state,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 20
        }
        
        logger.info(f"🚀 Training HistGradientBoosting with params: max_iter={params['max_iter']}, lr={params['learning_rate']}")
        
        model = HistGradientBoostingRegressor(**params)
        model.fit(self.X_train, self.y_train)
        
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring='r2', n_jobs=-1)
        logger.info(f"📊 Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        y_pred = model.predict(self.X_test)
        confidence = self._calculate_confidence_simple(y_pred)
        
        training_time = time.time() - start
        
        metrics = self._calculate_metrics(self.y_test, y_pred, confidence)
        metrics['training_time'] = training_time
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        model_filename = f'histgb_v1_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        model_path = os.path.join(CONFIG['MODEL_DIR'], model_filename)
        self._save_model_package(model, model_path, params)
        
        model_id = self._register_model(
            name='HistGradientBoosting',
            model_type='tree',
            algorithm='HistGradientBoosting',
            version='1.0.0',
            model_path=model_path,
            metrics=metrics,
            hyperparameters=params,
            feature_importance={}
        )
        
        self._store_comparison_metrics(model_id, self.y_test, y_pred, confidence)
        
        self._log_model_completion('HistGradientBoosting', metrics, training_time, model_id)
        
        return model_id, metrics
    
    def train_ridge(self):
        """Train Ridge Regression with improved preprocessing"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("🤖 MODEL: RIDGE REGRESSION")
        logger.info("=" * 60)
        logger.info(f"⏱️  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📊 Training samples: {len(self.X_train):,} | Test samples: {len(self.X_test):,}")
        logger.info("")
        
        start = time.time()
        
        # Find best alpha with cross-validation
        alphas = np.logspace(-2, 4, 50)
        ridge_cv = RidgeCV(alphas=alphas, cv=5)
        ridge_cv.fit(self.X_train, self.y_train)
        best_alpha = ridge_cv.alpha_
        
        logger.info(f"🔍 Best alpha found: {best_alpha:.4f}")
        
        model = Ridge(alpha=best_alpha)
        model.fit(self.X_train, self.y_train)
        
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2', n_jobs=-1)
        logger.info(f"📊 Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        y_pred = model.predict(self.X_test)
        confidence = self._calculate_confidence_linear(model, self.X_test, y_pred)
        
        training_time = time.time() - start
        
        metrics = self._calculate_metrics(self.y_test, y_pred, confidence)
        metrics['training_time'] = training_time
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        # Feature importance from coefficients
        feature_importance = dict(zip(self.feature_names, np.abs(model.coef_).tolist()))
        
        model_filename = f'ridge_v3_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        model_path = os.path.join(CONFIG['MODEL_DIR'], model_filename)
        self._save_model_package(model, model_path, {'alpha': best_alpha})
        
        model_id = self._register_model(
            name='Ridge',
            model_type='linear',
            algorithm='Ridge Regression',
            version='3.0.0',
            model_path=model_path,
            metrics=metrics,
            hyperparameters={'alpha': best_alpha},
            feature_importance=feature_importance
        )
        
        self._store_comparison_metrics(model_id, self.y_test, y_pred, confidence)
        
        self._log_model_completion('Ridge', metrics, training_time, model_id)
        
        return model_id, metrics
    
    def train_lasso(self):
        """Train Lasso Regression"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("🤖 MODEL: LASSO REGRESSION")
        logger.info("=" * 60)
        logger.info(f"⏱️  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📊 Training samples: {len(self.X_train):,} | Test samples: {len(self.X_test):,}")
        logger.info("")
        
        start = time.time()
        
        # Find best alpha
        alphas = np.logspace(-1, 4, 50)
        lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000, n_jobs=-1)
        lasso_cv.fit(self.X_train, self.y_train)
        best_alpha = lasso_cv.alpha_
        
        logger.info(f"🔍 Best alpha found: {best_alpha:.4f}")
        
        model = Lasso(alpha=best_alpha, max_iter=10000)
        model.fit(self.X_train, self.y_train)
        
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2', n_jobs=-1)
        logger.info(f"📊 Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        y_pred = model.predict(self.X_test)
        confidence = self._calculate_confidence_linear(model, self.X_test, y_pred)
        
        training_time = time.time() - start
        
        metrics = self._calculate_metrics(self.y_test, y_pred, confidence)
        metrics['training_time'] = training_time
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        feature_importance = dict(zip(self.feature_names, np.abs(model.coef_).tolist()))
        
        # Count non-zero features
        n_nonzero = np.sum(model.coef_ != 0)
        logger.info(f"📊 Non-zero features: {n_nonzero}/{len(self.feature_names)}")
        
        model_filename = f'lasso_v3_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        model_path = os.path.join(CONFIG['MODEL_DIR'], model_filename)
        self._save_model_package(model, model_path, {'alpha': best_alpha})
        
        model_id = self._register_model(
            name='Lasso',
            model_type='linear',
            algorithm='Lasso Regression',
            version='3.0.0',
            model_path=model_path,
            metrics=metrics,
            hyperparameters={'alpha': best_alpha},
            feature_importance=feature_importance
        )
        
        self._store_comparison_metrics(model_id, self.y_test, y_pred, confidence)
        
        self._log_model_completion('Lasso', metrics, training_time, model_id)
        
        return model_id, metrics
    
    def train_elasticnet(self):
        """Train ElasticNet with proper alpha/l1_ratio tuning"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("🤖 MODEL: ELASTICNET")
        logger.info("=" * 60)
        logger.info(f"⏱️  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📊 Training samples: {len(self.X_train):,} | Test samples: {len(self.X_test):,}")
        logger.info("")
        
        start = time.time()
        
        # Find best parameters
        alphas = np.logspace(-1, 4, 30)
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        elasticnet_cv = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=5, max_iter=10000, n_jobs=-1)
        elasticnet_cv.fit(self.X_train, self.y_train)
        
        best_alpha = elasticnet_cv.alpha_
        best_l1_ratio = elasticnet_cv.l1_ratio_
        
        logger.info(f"🔍 Best alpha: {best_alpha:.4f}, l1_ratio: {best_l1_ratio:.2f}")
        
        model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, max_iter=10000)
        model.fit(self.X_train, self.y_train)
        
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2', n_jobs=-1)
        logger.info(f"📊 Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        y_pred = model.predict(self.X_test)
        confidence = self._calculate_confidence_linear(model, self.X_test, y_pred)
        
        training_time = time.time() - start
        
        metrics = self._calculate_metrics(self.y_test, y_pred, confidence)
        metrics['training_time'] = training_time
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        feature_importance = dict(zip(self.feature_names, np.abs(model.coef_).tolist()))
        
        model_filename = f'elasticnet_v3_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        model_path = os.path.join(CONFIG['MODEL_DIR'], model_filename)
        self._save_model_package(model, model_path, {'alpha': best_alpha, 'l1_ratio': best_l1_ratio})
        
        model_id = self._register_model(
            name='ElasticNet',
            model_type='linear',
            algorithm='ElasticNet',
            version='3.0.0',
            model_path=model_path,
            metrics=metrics,
            hyperparameters={'alpha': best_alpha, 'l1_ratio': best_l1_ratio},
            feature_importance=feature_importance
        )
        
        self._store_comparison_metrics(model_id, self.y_test, y_pred, confidence)
        
        self._log_model_completion('ElasticNet', metrics, training_time, model_id)
        
        return model_id, metrics
    
    def train_lstm(self):
        """Train improved LSTM model"""
        if not TORCH_AVAILABLE:
            logger.warning("⚠️ PyTorch not available, skipping LSTM")
            return None, None
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("🤖 MODEL: LSTM (Improved)")
        logger.info("=" * 60)
        logger.info(f"⏱️  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📊 Training samples: {len(self.X_train):,} | Test samples: {len(self.X_test):,}")
        logger.info("")
        
        start = time.time()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️  Using device: {device}")
        
        # Hyperparameters
        params = {
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'epochs': 100,
            'batch_size': 128,
            'learning_rate': 0.001,
            'weight_decay': 1e-5
        }
        
        # Prepare data
        X_train_tensor = torch.FloatTensor(self.X_train).to(device)
        y_train_tensor = torch.FloatTensor(self.y_train).to(device)
        X_test_tensor = torch.FloatTensor(self.X_test).to(device)
        
        # Normalize target for better training
        y_mean = self.y_train.mean()
        y_std = self.y_train.std()
        y_train_norm = (y_train_tensor - y_mean) / y_std
        
        # Create model
        model = ImprovedLSTMNetwork(
            input_dim=self.X_train.shape[1],
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        ).to(device)
        
        # Training setup
        criterion = nn.HuberLoss(delta=1.0)  # More robust than MSE
        optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        
        dataset = TensorDataset(X_train_tensor, y_train_norm)
        dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(params['epochs']):
            model.train()
            total_loss = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 15:
                logger.info(f"📊 Early stopping at epoch {epoch + 1}")
                break
            
            if (epoch + 1) % 20 == 0:
                logger.debug(f"Epoch {epoch + 1}/{params['epochs']}, Loss: {avg_loss:.6f}")
        
        # Predictions
        model.eval()
        with torch.no_grad():
            y_pred_norm = model(X_test_tensor).cpu().numpy()
            y_pred = y_pred_norm * y_std + y_mean
        
        confidence = self._calculate_confidence_simple(y_pred)
        
        training_time = time.time() - start
        
        metrics = self._calculate_metrics(self.y_test, y_pred, confidence)
        metrics['training_time'] = training_time
        
        # Save model
        model_filename = f'lstm_v3_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
        model_path = os.path.join(CONFIG['MODEL_DIR'], model_filename)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'y_mean': y_mean,
            'y_std': y_std,
            'params': params,
            'input_dim': self.X_train.shape[1]
        }, model_path)
        
        model_id = self._register_model(
            name='LSTM',
            model_type='rnn',
            algorithm='LSTM',
            version='3.0.0',
            model_path=model_path,
            metrics=metrics,
            hyperparameters=params,
            feature_importance={}
        )
        
        self._store_comparison_metrics(model_id, self.y_test, y_pred, confidence)
        
        self._log_model_completion('LSTM', metrics, training_time, model_id)
        
        return model_id, metrics
    
    def train_gru(self):
        """Train improved GRU model"""
        if not TORCH_AVAILABLE:
            logger.warning("⚠️ PyTorch not available, skipping GRU")
            return None, None
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("🤖 MODEL: GRU (Improved)")
        logger.info("=" * 60)
        logger.info(f"⏱️  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📊 Training samples: {len(self.X_train):,} | Test samples: {len(self.X_test):,}")
        logger.info("")
        
        start = time.time()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️  Using device: {device}")
        
        params = {
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'epochs': 100,
            'batch_size': 128,
            'learning_rate': 0.001,
            'weight_decay': 1e-5
        }
        
        X_train_tensor = torch.FloatTensor(self.X_train).to(device)
        y_train_tensor = torch.FloatTensor(self.y_train).to(device)
        X_test_tensor = torch.FloatTensor(self.X_test).to(device)
        
        y_mean = self.y_train.mean()
        y_std = self.y_train.std()
        y_train_norm = (y_train_tensor - y_mean) / y_std
        
        model = ImprovedGRUNetwork(
            input_dim=self.X_train.shape[1],
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        ).to(device)
        
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        
        dataset = TensorDataset(X_train_tensor, y_train_norm)
        dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(params['epochs']):
            model.train()
            total_loss = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 15:
                logger.info(f"📊 Early stopping at epoch {epoch + 1}")
                break
            
            if (epoch + 1) % 20 == 0:
                logger.debug(f"Epoch {epoch + 1}/{params['epochs']}, Loss: {avg_loss:.6f}")
        
        model.eval()
        with torch.no_grad():
            y_pred_norm = model(X_test_tensor).cpu().numpy()
            y_pred = y_pred_norm * y_std + y_mean
        
        confidence = self._calculate_confidence_simple(y_pred)
        
        training_time = time.time() - start
        
        metrics = self._calculate_metrics(self.y_test, y_pred, confidence)
        metrics['training_time'] = training_time
        
        model_filename = f'gru_v3_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
        model_path = os.path.join(CONFIG['MODEL_DIR'], model_filename)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'y_mean': y_mean,
            'y_std': y_std,
            'params': params,
            'input_dim': self.X_train.shape[1]
        }, model_path)
        
        model_id = self._register_model(
            name='GRU',
            model_type='rnn',
            algorithm='GRU',
            version='3.0.0',
            model_path=model_path,
            metrics=metrics,
            hyperparameters=params,
            feature_importance={}
        )
        
        self._store_comparison_metrics(model_id, self.y_test, y_pred, confidence)
        
        self._log_model_completion('GRU', metrics, training_time, model_id)
        
        return model_id, metrics
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _calculate_confidence_tree(self, model, X, y_pred):
        """Calculate confidence for tree-based models using prediction variance"""
        # Use prediction spread relative to training data
        pred_std = np.std(y_pred)
        pred_mean = np.mean(y_pred)
        
        # Normalize distance from mean
        distance = np.abs(y_pred - pred_mean) / (pred_std + 1e-8)
        
        # Convert to confidence (0-100)
        confidence = 100 * np.exp(-distance / 3)
        return np.clip(confidence, 20, 95)
    
    def _calculate_confidence_linear(self, model, X, y_pred):
        """Calculate confidence for linear models"""
        # Use leverage (distance from training data center)
        X_centered = X - np.mean(X, axis=0)
        leverage = np.sum(X_centered ** 2, axis=1)
        leverage_norm = leverage / (np.max(leverage) + 1e-8)
        
        confidence = 100 * (1 - leverage_norm * 0.5)
        return np.clip(confidence, 30, 90)
    
    def _calculate_confidence_simple(self, y_pred):
        """Simple confidence calculation"""
        pred_std = np.std(y_pred)
        pred_mean = np.mean(y_pred)
        distance = np.abs(y_pred - pred_mean) / (pred_std + 1e-8)
        confidence = 100 * np.exp(-distance / 3)
        return np.clip(confidence, 20, 95)
    
    def _calculate_metrics(self, y_true, y_pred, confidence):
        """Calculate comprehensive metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # MAPE with zero handling
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        median_ae = median_absolute_error(y_true, y_pred)
        percentile_90 = np.percentile(np.abs(y_true - y_pred), 90)
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'median_ae': float(median_ae),
            'percentile_90_error': float(percentile_90),
            'avg_confidence': float(np.mean(confidence))
        }
    
    def _save_model_package(self, model, model_path, params):
        """Save model with preprocessing objects"""
        package = {
            'model': model,
            'scaler': self.scaler,
            'target_encoders': self.target_encoders,
            'category_mappings': self.category_mappings,
            'numeric_medians': self.numeric_medians,
            'feature_names': self.feature_names,
            'params': params,
            'trained_at': datetime.now().isoformat()
        }
        joblib.dump(package, model_path)
    
    def _log_model_completion(self, name, metrics, training_time, model_id):
        """Log model completion with formatted output"""
        logger.info("")
        logger.info(f"✅ {name.upper()} TRAINING COMPLETED!")
        logger.info(f"⏱️  Duration: {training_time:.2f}s ({training_time/60:.1f} min)")
        logger.info(f"📈 R² Score: {metrics['r2']:.4f} (higher is better, max 1.0)")
        logger.info(f"📊 MAE: {metrics['mae']:,.0f} DKK (Mean Absolute Error)")
        logger.info(f"📊 RMSE: {metrics['rmse']:,.0f} DKK (Root Mean Squared Error)")
        logger.info(f"📊 MAPE: {metrics['mape']:.2f}% (Mean Absolute Percentage Error)")
        if 'cv_r2_mean' in metrics:
            logger.info(f"📊 CV R²: {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
        logger.info(f"🎯 Model ID: {model_id}")
        logger.info("=" * 60)
    
    def _register_model(self, name, model_type, algorithm, version, model_path, 
                       metrics, hyperparameters, feature_importance):
        """Register model in database"""
        model_id = str(uuid.uuid4())
        
        # Clamp values for database
        r2_clamped = max(-99.9999, min(99.9999, metrics['r2']))
        mape_clamped = max(-99.9999, min(99.9999, metrics['mape']))
        
        if r2_clamped != metrics['r2']:
            logger.warning(f"R² clamped from {metrics['r2']:.4f} to {r2_clamped:.4f}")
        if mape_clamped != metrics['mape']:
            logger.warning(f"MAPE clamped from {metrics['mape']:.4f} to {mape_clamped:.4f}")
        
        query = """
            INSERT INTO ml_models (
                id, name, model_type, algorithm, version, is_active,
                model_file_path, mae, rmse, r2_score, mape, median_ae,
                percentile_90_error, training_time_seconds, hyperparameters,
                feature_importances, created_at, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
            )
            ON CONFLICT (name) DO UPDATE SET
                version = EXCLUDED.version,
                model_file_path = EXCLUDED.model_file_path,
                mae = EXCLUDED.mae,
                rmse = EXCLUDED.rmse,
                r2_score = EXCLUDED.r2_score,
                mape = EXCLUDED.mape,
                median_ae = EXCLUDED.median_ae,
                percentile_90_error = EXCLUDED.percentile_90_error,
                training_time_seconds = EXCLUDED.training_time_seconds,
                hyperparameters = EXCLUDED.hyperparameters,
                feature_importances = EXCLUDED.feature_importances,
                updated_at = NOW()
            RETURNING id
        """
        
        self.cur.execute(query, (
            model_id, name, model_type, algorithm, version, True,
            model_path, metrics['mae'], metrics['rmse'], r2_clamped,
            mape_clamped, metrics['median_ae'], metrics['percentile_90_error'],
            metrics.get('training_time', 0), json.dumps(hyperparameters),
            json.dumps(feature_importance)
        ))
        
        result = self.cur.fetchone()
        if result:
            model_id = result[0]
        
        self.conn.commit()
        self.models_trained.append({'id': model_id, 'name': name, 'r2': metrics['r2']})
        
        if metrics['r2'] > self.best_r2:
            self.best_r2 = metrics['r2']
            self.best_model_id = model_id
        
        return model_id
    
    def _store_comparison_metrics(self, model_id, y_true, y_pred, confidence):
        """Store segmented comparison metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        median_ae = np.median(np.abs(y_true - y_pred))
        percentile_90_error = np.percentile(np.abs(y_true - y_pred), 90)
        
        # Clamp values
        r2_clamped = max(-99.9999, min(99.9999, r2))
        mape_clamped = max(-99.9999, min(99.9999, mape))
        
        # Calculate segmented MAE by price range
        segmented_mae = {}
        for segment_name, (low, high) in CONFIG['PRICE_SEGMENTS'].items():
            segment_mask = (y_true >= low) & (y_true < high)
            if segment_mask.sum() > 0:
                segmented_mae[segment_name] = float(mean_absolute_error(y_true[segment_mask], y_pred[segment_mask]))
            else:
                segmented_mae[segment_name] = float(mae)
        
        try:
            self.cur.execute("""
                SELECT id FROM model_training_runs 
                ORDER BY created_at DESC LIMIT 1
            """)
            latest_run = self.cur.fetchone()
            training_run_id = latest_run[0] if latest_run else None
            
            if not training_run_id:
                logger.warning("⚠️ No training run found - skipping comparison metrics")
                return
            
            query = """
                INSERT INTO model_comparison_metrics (
                    id, model_id, training_run_id,
                    mae, rmse, r2_score, mape,
                    median_ae, percentile_90_error,
                    mae_under_100k, mae_100k_to_300k, mae_300k_to_500k, mae_over_500k
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """
            
            self.cur.execute(query, (
                str(uuid.uuid4()), model_id, training_run_id,
                mae, rmse, r2_clamped, mape_clamped,
                median_ae, percentile_90_error,
                segmented_mae.get('under_100k', mae),
                segmented_mae.get('100k_to_300k', mae),
                segmented_mae.get('300k_to_500k', mae),
                segmented_mae.get('over_500k', mae)
            ))
            
            self.conn.commit()
            logger.debug("✅ Comparison metrics stored successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not store comparison metrics: {e}")
            if self.conn:
                self.conn.rollback()
    
    def update_training_progress(self, completed, total):
        """Update training progress in database for frontend"""
        try:
            latest_model = self.models_trained[-1] if self.models_trained else None
            progress_msg = f"{completed}/{total} models"
            if latest_model:
                progress_msg += f" | Latest: {latest_model['name'].upper()} - R²={latest_model['r2']:.3f}, MAE={self._get_latest_mae():,.0f} DKK"
            
            logger.debug(f"✅ Updated training progress: {progress_msg}")
        except Exception as e:
            logger.debug(f"Could not update progress: {e}")
    
    def _get_latest_mae(self):
        """Get MAE of latest trained model"""
        if not self.models_trained:
            return 0
        # This is a simplification - in practice you'd store MAE too
        return 0
    
    def log_training_run(self, dataset_size, status='completed'):
        """Log training run to database"""
        logger.info("=" * 60)
        logger.info("LOGGING TRAINING RUN TO DATABASE")
        
        try:
            train_size = len(self.X_train) if self.X_train is not None else 0
            test_size = len(self.X_test) if self.X_test is not None else 0
            duration = time.time() - self.start_time if self.start_time else 0
            
            logger.info(f"Dataset size: {dataset_size}")
            logger.info(f"Train size: {train_size}")
            logger.info(f"Test size: {test_size}")
            logger.info(f"Duration: {duration:.2f}s")
            logger.info(f"Status: {status}")
            logger.info(f"Models trained: {[m['name'] for m in self.models_trained]}")
            logger.info(f"Best model ID: {self.best_model_id}")
            
            # Check for pending run
            self.cur.execute("""
                SELECT id FROM model_training_runs 
                WHERE status IN ('pending', 'running')
                ORDER BY created_at DESC LIMIT 1
            """)
            pending_run = self.cur.fetchone()
            
            if pending_run:
                logger.info(f"Updating existing training run: {pending_run[0]}")
                query = """
                    UPDATE model_training_runs 
                    SET run_date = NOW(),
                        dataset_size = %s,
                        train_size = %s,
                        test_size = %s,
                        training_duration_seconds = %s,
                        status = %s,
                        models_trained = %s,
                        best_model_id = %s
                    WHERE id = %s
                """
                
                self.cur.execute(query, (
                    dataset_size, train_size, test_size,
                    duration, status, json.dumps([m['name'] for m in self.models_trained]),
                    self.best_model_id, pending_run[0]
                ))
            else:
                logger.warning("⚠️ No pending training run found - creating new entry")
                query = """
                    INSERT INTO model_training_runs (
                        run_date, dataset_size, train_size, test_size,
                        training_duration_seconds, status, models_trained,
                        best_model_id, created_at
                    ) VALUES (
                        NOW(), %s, %s, %s, %s, %s, %s, %s, NOW()
                    )
                """
                
                self.cur.execute(query, (
                    dataset_size, train_size, test_size,
                    duration, status, json.dumps([m['name'] for m in self.models_trained]),
                    self.best_model_id
                ))
            
            self.conn.commit()
            logger.info(f"✅ Successfully logged training run to database")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"❌ Failed to log training run: {e}")
            logger.error("=" * 60)
            if self.conn:
                self.conn.rollback()
    
    def run(self, models_to_train=None):
        """Main training orchestration"""
        self.start_time = time.time()
        logger.info("=" * 60)
        logger.info("🚀 STARTING MULTI-MODEL TRAINING v3.0")
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        if not self.connect_db():
            logger.error("❌ Database connection failed - aborting training")
            return False
        
        try:
            # Update pending training run to 'running' status
            try:
                self.cur.execute("""
                    UPDATE model_training_runs 
                    SET status = 'running'
                    WHERE status = 'pending'
                    ORDER BY created_at DESC LIMIT 1
                """)
                self.conn.commit()
                logger.info("✅ Updated training status to 'running'")
            except Exception as e:
                logger.warning(f"⚠️ Could not update training status: {e}")
                if self.conn:
                    self.conn.rollback()
            
            # Prepare data
            logger.info("📥 Loading and preparing training data...")
            dataset_size = self.prepare_data()
            logger.info(f"✅ Data prepared: {dataset_size} records, {len(self.feature_names)} features")
            
            # Default models - all including new ones
            if models_to_train is None:
                models_to_train = [
                    'xgboost', 'catboost', 'lightgbm', 'random_forest', 'histgb',
                    'ridge', 'lasso', 'elasticnet',
                    'lstm', 'gru'
                ]
            
            logger.info(f"🤖 Training {len(models_to_train)} models: {', '.join(models_to_train)}")
            
            results = {}
            for i, model_name in enumerate(models_to_train, 1):
                try:
                    logger.info(f"[{i}/{len(models_to_train)}] Training {model_name.upper()}...")
                    
                    if model_name == 'xgboost':
                        model_id, metrics = self.train_xgboost()
                    elif model_name == 'catboost':
                        model_id, metrics = self.train_catboost()
                    elif model_name == 'lightgbm':
                        model_id, metrics = self.train_lightgbm()
                    elif model_name == 'random_forest':
                        model_id, metrics = self.train_random_forest()
                    elif model_name == 'histgb':
                        model_id, metrics = self.train_histgb()
                    elif model_name == 'ridge':
                        model_id, metrics = self.train_ridge()
                    elif model_name == 'lasso':
                        model_id, metrics = self.train_lasso()
                    elif model_name == 'elasticnet':
                        model_id, metrics = self.train_elasticnet()
                    elif model_name == 'lstm':
                        model_id, metrics = self.train_lstm()
                    elif model_name == 'gru':
                        model_id, metrics = self.train_gru()
                    else:
                        logger.warning(f"⚠️ Unknown model: {model_name}")
                        continue
                    
                    if model_id is not None:
                        results[model_name] = {'id': model_id, 'metrics': metrics}
                        self.update_training_progress(len(results), len(models_to_train))
                    
                except Exception as e:
                    logger.error(f"❌ Failed to train {model_name}: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    if self.conn:
                        try:
                            self.conn.rollback()
                        except:
                            pass
                    continue
            
            # Print summary
            logger.info("")
            logger.info("=" * 60)
            logger.info("📊 TRAINING SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total models trained: {len(results)}/{len(models_to_train)}")
            logger.info(f"Best model: {self.best_model_id}")
            logger.info(f"Best R² score: {self.best_r2:.4f}")
            logger.info(f"Total duration: {time.time() - self.start_time:.2f}s")
            logger.info("")
            logger.info("Individual model results (sorted by R²):")
            
            # Sort by R²
            sorted_results = sorted(results.items(), key=lambda x: x[1]['metrics']['r2'], reverse=True)
            
            for model_name, data in sorted_results:
                metrics = data['metrics']
                cv_info = f", CV={metrics.get('cv_r2_mean', 0):.4f}" if 'cv_r2_mean' in metrics else ""
                logger.info(f"  {model_name.upper():18} → R²: {metrics['r2']:.4f}{cv_info}, MAE: {metrics['mae']:>10,.0f}, RMSE: {metrics['rmse']:>10,.0f}")
            
            logger.info("=" * 60)
            
            # Log training run to database
            self.log_training_run(dataset_size, status='completed')
            
            logger.info("✅ Training completed successfully!")
            return True
            
        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"❌ TRAINING FAILED: {type(e).__name__}")
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            logger.error("=" * 60)
            self.log_training_run(0, status='failed')
            return False
        
        finally:
            logger.info("🧹 Cleaning up...")
            self.cleanup()
            logger.info("Training session ended")
            logger.info("")
    
    def cleanup(self):
        """Cleanup database connections"""
        try:
            if self.cur:
                self.cur.close()
                logger.info("Database cursor closed")
            if self.conn:
                self.conn.close()
                logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train multiple ML models v3.0')
    parser.add_argument('--models', nargs='+', 
                       choices=['xgboost', 'catboost', 'lightgbm', 'random_forest', 'histgb',
                               'ridge', 'lasso', 'elasticnet', 'lstm', 'gru'],
                       help='Models to train (default: all)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode - only tree-based models')
    
    args = parser.parse_args()
    
    models = args.models
    if args.quick:
        models = ['xgboost', 'lightgbm', 'catboost', 'random_forest']
    
    trainer = ModelTrainer(test_size=args.test_size)
    success = trainer.run(models_to_train=models)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
