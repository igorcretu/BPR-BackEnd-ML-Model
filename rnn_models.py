"""
Recurrent Neural Network Models for Car Price Prediction
========================================================

Implements LSTM and GRU models with confidence estimation using dropout variance.
"""

import numpy as np
import joblib
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - RNN models will not work")


if TORCH_AVAILABLE:
    class LSTMNetwork(nn.Module):
        """LSTM Neural Network"""
        
        def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
            super(LSTMNetwork, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim, 1)
            
        def forward(self, x):
            # LSTM expects (batch, seq_len, features)
            # We treat each feature as a sequence of length 1
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # Add sequence dimension
            
            lstm_out, _ = self.lstm(x)
            lstm_out = self.dropout(lstm_out[:, -1, :])  # Take last output
            out = self.fc(lstm_out)
            return out.squeeze()


    class GRUNetwork(nn.Module):
        """GRU Neural Network"""
        
        def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
            super(GRUNetwork, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim, 1)
            
        def forward(self, x):
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            
            gru_out, _ = self.gru(x)
            gru_out = self.dropout(gru_out[:, -1, :])
            out = self.fc(gru_out)
            return out.squeeze()


class BaseRNNModel:
    """Base class for RNN models"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for RNN models")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_fitted = False
        
        # For normalization
        self.y_mean = None
        self.y_std = None
        
    def _normalize_target(self, y):
        """Normalize target variable"""
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        return (y - self.y_mean) / self.y_std
    
    def _denormalize_target(self, y_normalized):
        """Denormalize target variable"""
        return y_normalized * self.y_std + self.y_mean
    
    def fit(self, X, y, epochs=50, batch_size=64, lr=0.001, verbose=True):
        """Train the model"""
        # Normalize target
        y_normalized = self._normalize_target(y)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y_normalized).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions_normalized = self.model(X_tensor).cpu().numpy()
            predictions = self._denormalize_target(predictions_normalized)
        
        return predictions
    
    def predict_with_confidence(self, X, n_samples=20):
        """
        Make predictions with confidence using Monte Carlo Dropout
        
        Confidence is estimated by:
        1. Running multiple predictions with dropout enabled
        2. Calculating variance of predictions
        3. Converting variance to confidence score
        
        Args:
            X: features
            n_samples: number of Monte Carlo samples
            
        Returns:
            predictions: mean predictions
            confidence: confidence scores (0-100)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Enable dropout during inference
        self.model.train()  # Keep dropout active
        
        predictions_samples = []
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            for _ in range(n_samples):
                pred_normalized = self.model(X_tensor).cpu().numpy()
                pred = self._denormalize_target(pred_normalized)
                predictions_samples.append(pred)
        
        # Back to eval mode
        self.model.eval()
        
        # Calculate mean and std
        predictions_samples = np.array(predictions_samples)
        predictions = np.mean(predictions_samples, axis=0)
        std_predictions = np.std(predictions_samples, axis=0)
        
        # Convert std to confidence score
        # Lower std = higher confidence
        # Normalize by typical price std (e.g., 50000 DKK)
        typical_std = 50000  # Adjust based on your data
        relative_std = std_predictions / typical_std
        confidence = (1 - np.clip(relative_std, 0, 1)) * 100
        
        return predictions, confidence
    
    def get_prediction_interval(self, X, n_samples=100, confidence_level=0.95):
        """
        Get prediction interval using Monte Carlo Dropout
        
        Args:
            X: features
            n_samples: number of Monte Carlo samples
            confidence_level: confidence level (e.g., 0.95 for 95% interval)
            
        Returns:
            predictions, lower_bound, upper_bound
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Enable dropout
        self.model.train()
        
        predictions_samples = []
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            for _ in range(n_samples):
                pred_normalized = self.model(X_tensor).cpu().numpy()
                pred = self._denormalize_target(pred_normalized)
                predictions_samples.append(pred)
        
        self.model.eval()
        
        # Calculate percentiles
        predictions_samples = np.array(predictions_samples)
        predictions = np.mean(predictions_samples, axis=0)
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(predictions_samples, lower_percentile, axis=0)
        upper_bound = np.percentile(predictions_samples, upper_percentile, axis=0)
        
        return predictions, lower_bound, upper_bound
    
    def save(self, filepath):
        """Save model to file"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'y_mean': self.y_mean,
            'y_std': self.y_std,
            'is_fitted': self.is_fitted
        }, filepath)
    
    def load(self, filepath):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.input_dim = checkpoint['input_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.num_layers = checkpoint['num_layers']
        self.dropout = checkpoint['dropout']
        self.y_mean = checkpoint['y_mean']
        self.y_std = checkpoint['y_std']
        self.is_fitted = checkpoint['is_fitted']
        
        # Recreate model
        self._create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        
        return self


class LSTMModel(BaseRNNModel):
    """
    LSTM Model for Car Price Prediction
    
    Good for:
    - Capturing long-term dependencies
    - Sequential patterns in data
    - Time-series features
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__(input_dim, hidden_dim, num_layers, dropout)
        self._create_model()
    
    def _create_model(self):
        """Create LSTM network"""
        self.model = LSTMNetwork(self.input_dim, self.hidden_dim, 
                                self.num_layers, self.dropout)


class GRUModel(BaseRNNModel):
    """
    GRU Model for Car Price Prediction
    
    Good for:
    - Faster training than LSTM
    - Similar performance to LSTM
    - Fewer parameters
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__(input_dim, hidden_dim, num_layers, dropout)
        self._create_model()
    
    def _create_model(self):
        """Create GRU network"""
        self.model = GRUNetwork(self.input_dim, self.hidden_dim,
                               self.num_layers, self.dropout)


# Mock classes for when PyTorch is not available
if not TORCH_AVAILABLE:
    class LSTMModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LSTM model")
    
    class GRUModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for GRU model")


# Example usage
if __name__ == "__main__" and TORCH_AVAILABLE:
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 10).astype(np.float32)
    y = (3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(1000) * 0.5 + 100).astype(np.float32)
    
    # Train LSTM
    print("Training LSTM...")
    lstm = LSTMModel(input_dim=10, hidden_dim=64, num_layers=2)
    lstm.fit(X, y, epochs=30, batch_size=32, verbose=True)
    
    # Predictions with confidence
    X_test = np.random.randn(10, 10).astype(np.float32)
    predictions, confidence = lstm.predict_with_confidence(X_test, n_samples=20)
    
    print(f"\nLSTM Sample predictions:")
    for i in range(5):
        print(f"  Prediction: {predictions[i]:.2f}, Confidence: {confidence[i]:.1f}%")
    
    # Prediction intervals
    preds, lower, upper = lstm.get_prediction_interval(X_test[:5], n_samples=50)
    print(f"\n95% Prediction Intervals:")
    for i in range(5):
        width = upper[i] - lower[i]
        print(f"  {preds[i]:.2f} [{lower[i]:.2f}, {upper[i]:.2f}] (width: {width:.2f})")
    
    # Train GRU
    print("\n" + "="*60)
    print("Training GRU...")
    gru = GRUModel(input_dim=10, hidden_dim=64, num_layers=2)
    gru.fit(X, y, epochs=30, batch_size=32, verbose=True)
    
    predictions, confidence = gru.predict_with_confidence(X_test, n_samples=20)
    
    print(f"\nGRU Sample predictions:")
    for i in range(5):
        print(f"  Prediction: {predictions[i]:.2f}, Confidence: {confidence[i]:.1f}%")
    
    print("\n✅ All RNN models trained successfully!")
elif __name__ == "__main__":
    print("⚠️ PyTorch not available. Install with: pip install torch")
