"""
Linear Regression Models for Car Price Prediction
=================================================

Implements Ridge, Lasso, and ElasticNet regression with confidence estimation.
"""

import numpy as np
import joblib
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler


class BaseLinearModel:
    """Base class for linear models with confidence estimation"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X, y):
        """Fit the model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # Calculate residuals for confidence estimation
        y_pred = self.model.predict(X_scaled)
        self.residuals_std = np.std(y - y_pred)
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_with_confidence(self, X):
        """
        Make predictions with confidence scores
        
        Confidence is based on:
        - Distance from training data mean (leverage)
        - Model residuals (prediction uncertainty)
        
        Returns:
            predictions: array of predicted prices
            confidence: array of confidence scores (0-100)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Calculate leverage (distance from training mean)
        leverage = np.sum((X_scaled - np.mean(X_scaled, axis=0))**2, axis=1)
        leverage = leverage / np.max(leverage)  # Normalize to [0, 1]
        
        # Calculate prediction intervals (simplified)
        # Standard error of prediction
        se_pred = self.residuals_std * np.sqrt(1 + leverage)
        
        # Convert to confidence score (0-100)
        # Higher confidence = lower standard error
        max_se = 3 * self.residuals_std  # 3 std as max uncertainty
        confidence = (1 - np.clip(se_pred / max_se, 0, 1)) * 100
        
        return predictions, confidence
    
    def get_prediction_interval(self, X, confidence_level=0.95):
        """
        Get prediction interval for given confidence level
        
        Args:
            X: features
            confidence_level: confidence level (e.g., 0.95 for 95% interval)
            
        Returns:
            predictions, lower_bound, upper_bound
        """
        from scipy import stats
        
        predictions, _ = self.predict_with_confidence(X)
        X_scaled = self.scaler.transform(X)
        
        # Calculate leverage
        leverage = np.sum((X_scaled - np.mean(X_scaled, axis=0))**2, axis=1)
        leverage = leverage / np.max(leverage)
        
        # Standard error of prediction
        se_pred = self.residuals_std * np.sqrt(1 + leverage)
        
        # t-value for confidence level
        t_value = stats.t.ppf((1 + confidence_level) / 2, df=len(X) - 1)
        
        # Prediction interval
        margin = t_value * se_pred
        lower_bound = predictions - margin
        upper_bound = predictions + margin
        
        return predictions, lower_bound, upper_bound
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from coefficients"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Use absolute coefficient values as importance
        importance = np.abs(self.model.coef_)
        
        # Normalize to sum to 1
        importance = importance / np.sum(importance)
        
        return dict(zip(feature_names, importance.tolist()))
    
    def save(self, filepath):
        """Save model to file"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'residuals_std': self.residuals_std,
            'is_fitted': self.is_fitted
        }, filepath)
    
    def load(self, filepath):
        """Load model from file"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.residuals_std = data['residuals_std']
        self.is_fitted = data['is_fitted']
        return self


class RidgeModel(BaseLinearModel):
    """
    Ridge Regression (L2 regularization)
    
    Good for:
    - Handling multicollinearity
    - Stable predictions
    - All features contribute
    """
    
    def __init__(self, alpha=1.0):
        super().__init__()
        self.model = Ridge(alpha=alpha, random_state=42)
        self.alpha = alpha


class LassoModel(BaseLinearModel):
    """
    Lasso Regression (L1 regularization)
    
    Good for:
    - Feature selection
    - Sparse models
    - Interpretability
    """
    
    def __init__(self, alpha=1.0):
        super().__init__()
        self.model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
        self.alpha = alpha
    
    def get_selected_features(self, feature_names, threshold=0.0001):
        """Get features selected by Lasso (non-zero coefficients)"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        coef = np.abs(self.model.coef_)
        selected = [(name, coef[i]) for i, name in enumerate(feature_names) if coef[i] > threshold]
        selected.sort(key=lambda x: x[1], reverse=True)
        
        return selected


class ElasticNetModel(BaseLinearModel):
    """
    ElasticNet (L1 + L2 regularization)
    
    Good for:
    - Balance between Ridge and Lasso
    - Feature selection with stability
    - Correlated features
    """
    
    def __init__(self, alpha=1.0, l1_ratio=0.5):
        super().__init__()
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=10000)
        self.alpha = alpha
        self.l1_ratio = l1_ratio


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(1000) * 0.5 + 100
    
    feature_names = [f'feature_{i}' for i in range(10)]
    
    # Train Ridge
    print("Training Ridge Regression...")
    ridge = RidgeModel(alpha=1.0)
    ridge.fit(X, y)
    
    # Predictions with confidence
    X_test = np.random.randn(10, 10)
    predictions, confidence = ridge.predict_with_confidence(X_test)
    
    print(f"\nSample predictions:")
    for i in range(5):
        print(f"  Prediction: {predictions[i]:.2f}, Confidence: {confidence[i]:.1f}%")
    
    # Feature importance
    importance = ridge.get_feature_importance(feature_names)
    print(f"\nTop 5 important features:")
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for name, imp in sorted_features[:5]:
        print(f"  {name}: {imp:.4f}")
    
    # Prediction intervals
    preds, lower, upper = ridge.get_prediction_interval(X_test[:5])
    print(f"\n95% Prediction Intervals:")
    for i in range(5):
        print(f"  {preds[i]:.2f} [{lower[i]:.2f}, {upper[i]:.2f}]")
    
    # Train Lasso
    print("\n" + "="*60)
    print("Training Lasso Regression...")
    lasso = LassoModel(alpha=0.1)
    lasso.fit(X, y)
    
    predictions, confidence = lasso.predict_with_confidence(X_test)
    print(f"\nSample predictions:")
    for i in range(5):
        print(f"  Prediction: {predictions[i]:.2f}, Confidence: {confidence[i]:.1f}%")
    
    # Selected features
    selected = lasso.get_selected_features(feature_names)
    print(f"\nSelected features (non-zero coefficients): {len(selected)}")
    for name, coef in selected[:5]:
        print(f"  {name}: {coef:.4f}")
    
    # Train ElasticNet
    print("\n" + "="*60)
    print("Training ElasticNet...")
    elasticnet = ElasticNetModel(alpha=0.1, l1_ratio=0.5)
    elasticnet.fit(X, y)
    
    predictions, confidence = elasticnet.predict_with_confidence(X_test)
    print(f"\nSample predictions:")
    for i in range(5):
        print(f"  Prediction: {predictions[i]:.2f}, Confidence: {confidence[i]:.1f}%")
    
    print("\n✅ All models trained successfully!")
