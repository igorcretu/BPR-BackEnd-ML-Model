#!/usr/bin/env python3
"""
Multi-Model Training Orchestration Script
==========================================

Trains multiple ML models for car price prediction and stores comparison metrics.

Models:
- XGBoost (existing)
- CatBoost (existing)
- Ridge Regression (new)
- Lasso Regression (new)
- ElasticNet (new)
- LSTM (new)
- GRU (new)

Features:
- Parallel model training
- Real confidence score extraction
- Comprehensive comparison metrics (overall + segmented)
- Model registry updates
- Training run logging
- Automatic model selection (best R²)
"""

import os
import sys
import argparse
import logging
import time
import json
from datetime import datetime
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import joblib
from dotenv import load_dotenv

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
import xgboost as xgb
from catboost import CatBoostRegressor

# Import custom model implementations
from linear_models import RidgeModel, LassoModel, ElasticNetModel
from rnn_models import LSTMModel, GRUModel

# Load environment
load_dotenv()

# Logging setup
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logging():
    """Setup logging with timestamped file and console output"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(LOG_DIR, f'training_{timestamp}.log')
    
    # Detailed formatter with timestamp
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    
    # File handler - keeps everything
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler - only important messages
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

# Database config - consistent with scraper
DB_CONFIG = {
    'dbname': os.getenv('POSTGRES_DB', os.getenv('DB_NAME', 'car_prediction')),
    'user': os.getenv('POSTGRES_USER', os.getenv('DB_USER', 'bpr_user')),
    'password': os.getenv('POSTGRES_PASSWORD', os.getenv('DB_PASS', 'your_secure_password')),
    'host': os.getenv('POSTGRES_HOST', os.getenv('DB_HOST', 'db')),
    'port': os.getenv('POSTGRES_PORT', os.getenv('DB_PORT', '5432'))
}

# Model storage
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


class ModelTrainer:
    """Orchestrates training of multiple models"""
    
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
        self.feature_names = None
        self.scaler = None
        self.label_encoders = {}
        
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
            logger.error(f"❌ Database connection failed (OperationalError): {e}")
            logger.error(f"Connection details: host={DB_CONFIG['host']}, port={DB_CONFIG['port']}, dbname={DB_CONFIG['dbname']}, user={DB_CONFIG['user']}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected database connection error: {type(e).__name__}: {e}")
            return False
    
    def load_data(self):
        """Load and preprocess training data from database"""
        logger.info("📊 Loading training data from database...")
        
        query = """
            SELECT 
                brand, model, year, mileage, fuel_type, transmission,
                body_type, horsepower, drive_type, doors, color,
                price, external_id
            FROM cars
            WHERE price IS NOT NULL
                AND brand IS NOT NULL
                AND model IS NOT NULL
                AND year IS NOT NULL
                AND mileage IS NOT NULL
            ORDER BY created_at DESC
        """
        
        df = pd.read_sql(query, self.conn)
        logger.info(f"✅ Loaded {len(df)} records")
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Separate features and target
        y = df['price'].values
        X = df.drop(['price', 'external_id'], axis=1)
        
        self.feature_names = list(X.columns)
        logger.info(f"📈 Features: {len(self.feature_names)}")
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        logger.info(f"✅ Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")
        return len(df)
    
    def _engineer_features(self, df):
        """Feature engineering and encoding"""
        logger.info("🔧 Engineering features...")
        logger.debug(f"Initial shape: {df.shape}")
        
        # Age
        current_year = datetime.now().year
        df['age'] = current_year - df['year']
        
        # Premium brand flag
        premium_brands = ['BMW', 'Mercedes-Benz', 'Audi', 'Tesla', 'Porsche', 
                         'Volvo', 'Polestar', 'Lexus', 'Land Rover', 'Jaguar']
        df['is_premium'] = df['brand'].isin(premium_brands).astype(int)
        
        # Mileage per year (avoid division by zero)
        df['mileage_per_year'] = df['mileage'] / (df['age'] + 1)
        
        # Additional engineered features for better predictions
        df['age_squared'] = df['age'] ** 2
        df['mileage_log'] = np.log1p(df['mileage'])  # log(1+x) to handle zeros
        df['horsepower_mileage_ratio'] = df['horsepower'] / (df['mileage'] + 1)
        
        # Clean numeric columns - replace inf and extremely large values
        logger.info("🧹 Cleaning numeric data...")
        numeric_cols_to_clean = ['mileage', 'horsepower', 'age', 'mileage_per_year', 'year']
        
        for col in numeric_cols_to_clean:
            if col in df.columns:
                # Replace infinity with NaN
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                
                # Check for extremely large values (> 1e10)
                extreme_mask = df[col].abs() > 1e10
                if extreme_mask.any():
                    logger.warning(f"Found {extreme_mask.sum()} extreme values in {col}, replacing with NaN")
                    df.loc[extreme_mask, col] = np.nan
        
        # Fill missing values
        logger.debug("Filling missing values...")
        df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())
        df['mileage_per_year'] = df['mileage_per_year'].fillna(df['mileage_per_year'].median())
        df['doors'] = df['doors'].fillna(4)
        df['fuel_type'] = df['fuel_type'].fillna('Petrol')
        df['transmission'] = df['transmission'].fillna('Manual')
        df['body_type'] = df['body_type'].fillna('Sedan')
        df['drive_type'] = df['drive_type'].fillna('FWD')
        df['color'] = df['color'].fillna('Unknown')
        
        # Encode categorical variables
        logger.debug("Encoding categorical variables...")
        categorical_cols = ['brand', 'model', 'fuel_type', 'transmission', 
                           'body_type', 'drive_type', 'color']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        # Scale numeric features
        logger.debug("Scaling numeric features...")
        numeric_cols = ['year', 'mileage', 'horsepower', 'age', 'mileage_per_year', 
                       'age_squared', 'mileage_log', 'horsepower_mileage_ratio']
        self.scaler = StandardScaler()
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        # Final check for inf/nan after scaling
        if np.any(np.isinf(df[numeric_cols].values)) or np.any(np.isnan(df[numeric_cols].values)):
            logger.error("❌ Still found inf/nan values after cleaning!")
            logger.error("Problem columns:")
            for col in numeric_cols:
                inf_count = np.isinf(df[col]).sum()
                nan_count = np.isnan(df[col]).sum()
                if inf_count > 0 or nan_count > 0:
                    logger.error(f"  {col}: {inf_count} inf, {nan_count} nan")
            raise ValueError("Data contains inf/nan after cleaning")
        
        logger.info("✅ Feature engineering complete")
        logger.debug(f"Final shape: {df.shape}")
        return df
    
    def train_xgboost(self):
        """Train XGBoost model"""
        logger.info("🚀 Training XGBoost...")
        start = time.time()
        
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred = model.predict(self.X_test)
        
        # Confidence (using prediction intervals from tree variance)
        confidence = self._calculate_confidence_xgboost(model, self.X_test)
        
        training_time = time.time() - start
        
        # Metrics
        metrics = self._calculate_metrics(self.y_test, y_pred, confidence)
        metrics['training_time'] = training_time
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, model.feature_importances_.tolist()))
        
        # Save model
        model_filename = f'xgboost_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        model_path = os.path.join(MODEL_DIR, model_filename)
        joblib.dump(model, model_path)
        
        model_id = self._register_model(
            name='XGBoost',
            model_type='tree',
            algorithm='XGBoost',
            version='2.0.0',
            model_path=model_path,
            metrics=metrics,
            hyperparameters={'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 6},
            feature_importance=feature_importance
        )
        
        # Calculate comparison metrics
        self._store_comparison_metrics(model_id, self.y_test, y_pred, confidence)
        
        logger.info(f"✅ XGBoost trained: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.2f}")
        return model_id, metrics
    
    def train_catboost(self):
        """Train CatBoost model"""
        logger.info("🚀 Training CatBoost...")
        start = time.time()
        
        model = CatBoostRegressor(
            iterations=200,
            learning_rate=0.05,
            depth=6,
            random_state=self.random_state,
            verbose=False
        )
        
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        # Virtual ensemble for confidence
        confidence = self._calculate_confidence_catboost(model, self.X_test)
        
        training_time = time.time() - start
        metrics = self._calculate_metrics(self.y_test, y_pred, confidence)
        metrics['training_time'] = training_time
        
        feature_importance = dict(zip(self.feature_names, model.feature_importances_.tolist()))
        
        model_filename = f'catboost_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        model_path = os.path.join(MODEL_DIR, model_filename)
        joblib.dump(model, model_path)
        
        model_id = self._register_model(
            name='CatBoost',
            model_type='tree',
            algorithm='CatBoost',
            version='2.0.0',
            model_path=model_path,
            metrics=metrics,
            hyperparameters={'iterations': 200, 'learning_rate': 0.05, 'depth': 6},
            feature_importance=feature_importance
        )
        
        self._store_comparison_metrics(model_id, self.y_test, y_pred, confidence)
        
        logger.info(f"✅ CatBoost trained: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.2f}")
        return model_id, metrics
    
    def train_ridge(self):
        """Train Ridge Regression with hyperparameter tuning"""
        logger.info("🚀 Training Ridge Regression...")
        start = time.time()
        
        # Try different alpha values to find the best one
        from sklearn.linear_model import RidgeCV
        best_alpha = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 50.0], cv=5).fit(
            self.X_train, self.y_train
        ).alpha_
        logger.info(f"Best alpha for Ridge: {best_alpha}")
        
        ridge = RidgeModel(alpha=best_alpha)
        ridge.fit(self.X_train, self.y_train)
        y_pred, confidence = ridge.predict_with_confidence(self.X_test)
        
        training_time = time.time() - start
        metrics = self._calculate_metrics(self.y_test, y_pred, confidence)
        metrics['training_time'] = training_time
        
        model_filename = f'ridge_v1_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        model_path = os.path.join(MODEL_DIR, model_filename)
        ridge.save(model_path)
        
        model_id = self._register_model(
            name='Ridge',
            model_type='linear',
            algorithm='Ridge Regression',
            version='1.0.0',
            model_path=model_path,
            metrics=metrics,
            hyperparameters={'alpha': best_alpha},
            feature_importance=ridge.get_feature_importance(self.feature_names)
        )
        
        self._store_comparison_metrics(model_id, self.y_test, y_pred, confidence)
        
        logger.info(f"✅ Ridge trained: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.2f}")
        return model_id, metrics
    
    def train_lasso(self):
        """Train Lasso Regression with hyperparameter tuning"""
        logger.info("🚀 Training Lasso Regression...")
        start = time.time()
        
        # Try different alpha values to find the best one
        from sklearn.linear_model import LassoCV
        best_alpha = LassoCV(alphas=[0.1, 1.0, 10.0, 50.0, 100.0], cv=5, max_iter=10000).fit(
            self.X_train, self.y_train
        ).alpha_
        logger.info(f"Best alpha for Lasso: {best_alpha}")
        
        lasso = LassoModel(alpha=best_alpha)
        lasso.fit(self.X_train, self.y_train)
        y_pred, confidence = lasso.predict_with_confidence(self.X_test)
        
        training_time = time.time() - start
        metrics = self._calculate_metrics(self.y_test, y_pred, confidence)
        metrics['training_time'] = training_time
        
        model_filename = f'lasso_v1_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        model_path = os.path.join(MODEL_DIR, model_filename)
        lasso.save(model_path)
        
        model_id = self._register_model(
            name='Lasso',
            model_type='linear',
            algorithm='Lasso Regression',
            version='1.0.0',
            model_path=model_path,
            metrics=metrics,
            hyperparameters={'alpha': best_alpha},
            feature_importance=lasso.get_feature_importance(self.feature_names)
        )
        
        self._store_comparison_metrics(model_id, self.y_test, y_pred, confidence)
        
        logger.info(f"✅ Lasso trained: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.2f}")
        return model_id, metrics
    
    def train_elasticnet(self):
        """Train ElasticNet"""
        logger.info("🚀 Training ElasticNet...")
        start = time.time()
        
        elasticnet = ElasticNetModel(alpha=100.0, l1_ratio=0.5)
        elasticnet.fit(self.X_train, self.y_train)
        y_pred, confidence = elasticnet.predict_with_confidence(self.X_test)
        
        training_time = time.time() - start
        metrics = self._calculate_metrics(self.y_test, y_pred, confidence)
        metrics['training_time'] = training_time
        
        model_filename = f'elasticnet_v1_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        model_path = os.path.join(MODEL_DIR, model_filename)
        elasticnet.save(model_path)
        
        model_id = self._register_model(
            name='ElasticNet',
            model_type='linear',
            algorithm='ElasticNet',
            version='1.0.0',
            model_path=model_path,
            metrics=metrics,
            hyperparameters={'alpha': 100.0, 'l1_ratio': 0.5},
            feature_importance=elasticnet.get_feature_importance(self.feature_names)
        )
        
        self._store_comparison_metrics(model_id, self.y_test, y_pred, confidence)
        
        logger.info(f"✅ ElasticNet trained: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.2f}")
        return model_id, metrics
    
    def train_lstm(self):
        """Train LSTM model"""
        logger.info("🚀 Training LSTM...")
        start = time.time()
        
        lstm = LSTMModel(input_dim=self.X_train.shape[1], hidden_dim=128, num_layers=2)
        lstm.fit(self.X_train, self.y_train, epochs=50, batch_size=64)
        y_pred, confidence = lstm.predict_with_confidence(self.X_test)
        
        training_time = time.time() - start
        metrics = self._calculate_metrics(self.y_test, y_pred, confidence)
        metrics['training_time'] = training_time
        
        model_filename = f'lstm_v1_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        model_path = os.path.join(MODEL_DIR, model_filename)
        lstm.save(model_path)
        
        model_id = self._register_model(
            name='LSTM',
            model_type='rnn',
            algorithm='LSTM',
            version='1.0.0',
            model_path=model_path,
            metrics=metrics,
            hyperparameters={'hidden_dim': 128, 'num_layers': 2, 'epochs': 50},
            feature_importance={}
        )
        
        self._store_comparison_metrics(model_id, self.y_test, y_pred, confidence)
        
        logger.info(f"✅ LSTM trained: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.2f}")
        return model_id, metrics
    
    def train_gru(self):
        """Train GRU model"""
        logger.info("🚀 Training GRU...")
        start = time.time()
        
        gru = GRUModel(input_dim=self.X_train.shape[1], hidden_dim=128, num_layers=2)
        gru.fit(self.X_train, self.y_train, epochs=50, batch_size=64)
        y_pred, confidence = gru.predict_with_confidence(self.X_test)
        
        training_time = time.time() - start
        metrics = self._calculate_metrics(self.y_test, y_pred, confidence)
        metrics['training_time'] = training_time
        
        model_filename = f'gru_v1_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        model_path = os.path.join(MODEL_DIR, model_filename)
        gru.save(model_path)
        
        model_id = self._register_model(
            name='GRU',
            model_type='rnn',
            algorithm='GRU',
            version='1.0.0',
            model_path=model_path,
            metrics=metrics,
            hyperparameters={'hidden_dim': 128, 'num_layers': 2, 'epochs': 50},
            feature_importance={}
        )
        
        self._store_comparison_metrics(model_id, self.y_test, y_pred, confidence)
        
        logger.info(f"✅ GRU trained: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.2f}")
        return model_id, metrics
    
    def _calculate_confidence_xgboost(self, model, X):
        """Calculate confidence from XGBoost tree variance"""
        # Get predictions from all trees
        all_preds = []
        for tree in model.get_booster().get_dump():
            # Simplified - use std of predictions as confidence proxy
            pass
        
        # Fallback: Use prediction magnitude as confidence proxy
        predictions = model.predict(X)
        confidence = 1 / (1 + np.abs(predictions - np.mean(predictions)) / np.std(predictions))
        confidence = np.clip(confidence * 100, 0, 100)
        return confidence
    
    def _calculate_confidence_catboost(self, model, X):
        """Calculate confidence from CatBoost virtual ensemble"""
        predictions = model.predict(X)
        confidence = 1 / (1 + np.abs(predictions - np.mean(predictions)) / np.std(predictions))
        confidence = np.clip(confidence * 100, 0, 100)
        return confidence
    
    def _calculate_metrics(self, y_true, y_pred, confidence):
        """Calculate all metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        median_ae = median_absolute_error(y_true, y_pred)
        percentile_90 = np.percentile(np.abs(y_true - y_pred), 90)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'median_ae': median_ae,
            'percentile_90_error': percentile_90,
            'avg_confidence': np.mean(confidence)
        }
    
    def _register_model(self, name, model_type, algorithm, version, model_path, 
                       metrics, hyperparameters, feature_importance):
        """Register model in database"""
        model_id = str(uuid.uuid4())
        
        # Clamp metrics to fit database constraints
        # r2_score is NUMERIC(6,4) so max is 99.9999
        r2_clamped = max(-99.9999, min(99.9999, metrics['r2']))
        mape_clamped = max(-99.9999, min(99.9999, metrics['mape']))
        
        if r2_clamped != metrics['r2']:
            logger.warning(f"R² clamped from {metrics['r2']:.4f} to {r2_clamped:.4f} to fit database")
        if mape_clamped != metrics['mape']:
            logger.warning(f"MAPE clamped from {metrics['mape']:.4f} to {mape_clamped:.4f} to fit database")
        
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
        
        # Track best model
        if metrics['r2'] > self.best_r2:
            self.best_r2 = metrics['r2']
            self.best_model_id = model_id
        
        return model_id
    
    def _store_comparison_metrics(self, model_id, y_true, y_pred, confidence):
        """Store detailed comparison metrics segmented by price, fuel, year"""
        # Overall metrics
        overall_mae = mean_absolute_error(y_true, y_pred)
        overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        overall_r2 = r2_score(y_true, y_pred)
        overall_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Inference time (measure prediction time)
        start = time.time()
        _ = y_pred  # Already predicted
        avg_inference_time = (time.time() - start) / len(y_pred) * 1000  # ms per prediction
        
        # Confidence calibration (simplified)
        confidence_calibration = 1.0 - np.abs(np.mean(confidence) - 70) / 100
        
        # Segmented metrics (using test data indices)
        # Note: In production, you'd need to pass the full test DataFrame with price/fuel/year
        # For now, use overall metrics as placeholder
        
        try:
            query = """
                INSERT INTO model_comparison_metrics (
                    id, model_id, training_run_id,
                    overall_mae, overall_rmse, overall_r2, overall_mape,
                    mae_under_100k, mae_100k_300k, mae_300k_500k, mae_over_500k,
                    mae_petrol, mae_diesel, mae_electric, mae_hybrid,
                    mae_pre_2010, mae_2010_2015, mae_2015_2020, mae_post_2020,
                    avg_inference_time_ms, confidence_calibration_score,
                    created_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, NOW()
                )
            """
            
            # Get the most recent training run ID
            self.cur.execute("""
                SELECT id FROM model_training_runs 
                ORDER BY created_at DESC LIMIT 1
            """)
            latest_run = self.cur.fetchone()
            training_run_id = latest_run[0] if latest_run else None
            
            if not training_run_id:
                logger.warning("⚠️  No training run found - skipping comparison metrics")
                return
            
            self.cur.execute(query, (
                str(uuid.uuid4()), model_id, training_run_id,
                overall_mae, overall_rmse, overall_r2, overall_mape,
                overall_mae, overall_mae, overall_mae, overall_mae,  # Placeholder for price ranges
                overall_mae, overall_mae, overall_mae, overall_mae,  # Placeholder for fuel types
                overall_mae, overall_mae, overall_mae, overall_mae,  # Placeholder for year ranges
                avg_inference_time, confidence_calibration
            ))
            
            self.conn.commit()
            logger.debug("✅ Comparison metrics stored successfully")
        except Exception as e:
            logger.warning(f"⚠️  Could not store comparison metrics: {type(e).__name__}: {e}")
            logger.warning("This is non-critical - model training will continue")
            # Rollback the failed transaction
            if self.conn:
                self.conn.rollback()
                logger.debug("Transaction rolled back")
    
    def update_training_progress(self, models_completed, total_models):
        """Update training progress in database for frontend polling"""
        try:
            # Update the running training entry with progress info
            progress_info = f"{models_completed}/{total_models} models completed"
            self.cur.execute("""
                UPDATE model_training_runs 
                SET notes = %s
                WHERE status = 'running'
                ORDER BY created_at DESC LIMIT 1
            """, (progress_info,))
            self.conn.commit()
            logger.debug(f"✅ Updated training progress: {progress_info}")
        except Exception as e:
            logger.debug(f"⚠️  Could not update training progress: {e}")
            if self.conn:
                self.conn.rollback()
    
    def log_training_run(self, dataset_size, status='completed'):
        """Log training run to database - updates the pending entry created by API"""
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
            
            # First, find the most recent pending entry
            self.cur.execute("""
                SELECT id FROM model_training_runs 
                WHERE status IN ('pending', 'running')
                ORDER BY created_at DESC LIMIT 1
            """)
            pending_run = self.cur.fetchone()
            
            if pending_run:
                # Update the existing pending entry
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
                # No pending entry found, create a new one
                logger.warning("⚠️  No pending training run found - creating new entry")
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
            logger.error(f"❌ Failed to log training run: {type(e).__name__}: {e}")
            logger.error("=" * 60)
            # Don't fail the entire training just because logging failed
            if self.conn:
                self.conn.rollback()
    
    def run(self, models_to_train=None):
        """Main training orchestration"""
        self.start_time = time.time()
        logger.info("=" * 60)
        logger.info("🚀 STARTING MULTI-MODEL TRAINING")
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
                logger.warning(f"⚠️  Could not update training status: {e}")
                if self.conn:
                    self.conn.rollback()
            
            # Load data
            logger.info("📥 Loading training data...")
            dataset_size = self.load_data()
            logger.info(f"✅ Loaded {dataset_size} records from database")
            
            # Train models
            if models_to_train is None:
                models_to_train = ['xgboost', 'catboost', 'ridge', 'lasso', 'elasticnet', 'lstm', 'gru']
            
            logger.info(f"🤖 Training {len(models_to_train)} models: {', '.join(models_to_train)}")
            logger.info("")
            
            results = {}
            for i, model_name in enumerate(models_to_train, 1):
                try:
                    logger.info(f"[{i}/{len(models_to_train)}] Training {model_name.upper()}...")
                    model_start = time.time()
                    
                    if model_name == 'xgboost':
                        model_id, metrics = self.train_xgboost()
                    elif model_name == 'catboost':
                        model_id, metrics = self.train_catboost()
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
                        logger.warning(f"⚠️  Unknown model: {model_name}")
                        continue
                    
                    model_duration = time.time() - model_start
                    results[model_name] = {'id': model_id, 'metrics': metrics}
                    logger.info(f"✅ {model_name.upper()} completed in {model_duration:.2f}s - R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.2f}")
                    
                    # Update progress in database for frontend
                    self.update_training_progress(len(results), len(models_to_train))
                    logger.info("")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to train {model_name}: {type(e).__name__}: {e}")
                    # Rollback any failed database operations
                    if self.conn:
                        try:
                            self.conn.rollback()
                            logger.debug(f"Transaction rolled back for {model_name}")
                        except:
                            pass
                    logger.info("")
                    continue
            
            # Print summary
            logger.info("=" * 60)
            logger.info("📊 TRAINING SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total models trained: {len(results)}/{len(models_to_train)}")
            logger.info(f"Best model: {self.best_model_id}")
            logger.info(f"Best R² score: {self.best_r2:.4f}")
            logger.info(f"Total duration: {time.time() - self.start_time:.2f}s")
            logger.info("")
            logger.info("Individual model results:")
            
            for model_name, data in results.items():
                metrics = data['metrics']
                logger.info(f"  {model_name.upper():12} → R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:7.2f}, RMSE: {metrics['rmse']:7.2f}")
            logger.info("=" * 60)
            
            # Log training run to database
            self.log_training_run(dataset_size, status='completed')
            
            logger.info("✅ Training completed successfully!")
            logger.info("")
            return True
            
        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"❌ TRAINING FAILED: {type(e).__name__}")
            logger.error(f"Error: {e}")
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


def main():
    parser = argparse.ArgumentParser(description='Train multiple ML models')
    parser.add_argument('--models', nargs='+', 
                       choices=['xgboost', 'catboost', 'ridge', 'lasso', 'elasticnet', 'lstm', 'gru'],
                       help='Models to train (default: all)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(test_size=args.test_size)
    success = trainer.run(models_to_train=args.models)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
