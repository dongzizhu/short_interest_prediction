"""
Models and training module for the iterative agent-based feature selection system.

This module contains the LSTM model architecture, training logic, and evaluation functions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from typing import Dict, Tuple, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from .config import ModelConfig
    from .utils import validate_data_shape, calculate_improvement_metrics
except ImportError:
    from config import ModelConfig
    from utils import validate_data_shape, calculate_improvement_metrics


class EnhancedLSTMTimeSeries(nn.Module):
    """Enhanced LSTM model for time series prediction."""
    
    def __init__(self, input_size: int =97, hidden_size: int = 64, num_layers: int = 3, 
                 output_size: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class SVMModel:
    """SVM model for time series prediction with feature importance calculation."""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale', 
                 epsilon: float = 0.1, max_iter: int = 1000):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.model = None
        self.scaler = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMModel':
        """Fit the SVM model to the data."""
        # Reshape data for SVM (flatten time series dimension)
        if X.ndim == 3:
            # Flatten from (N, lookback_window, features) to (N, lookback_window * features)
            X_reshaped = X.reshape(X.shape[0], -1)
        else:
            X_reshaped = X
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_reshaped)
        
        # Fit SVM
        self.model = SVR(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            epsilon=self.epsilon,
            max_iter=self.max_iter
        )
        self.model.fit(X_scaled, y.ravel())
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Reshape data for SVM (flatten time series dimension)
        if X.ndim == 3:
            # Flatten from (N, lookback_window, features) to (N, lookback_window * features)
            X_reshaped = X.reshape(X.shape[0], -1)
        else:
            X_reshaped = X
        
        # Scale features
        X_scaled = self.scaler.transform(X_reshaped)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        return predictions.reshape(-1, 1)
    
    def get_feature_importance(self, X: np.ndarray, y: np.ndarray, 
                              feature_names: List[str]) -> Tuple[Dict[str, Any], np.ndarray]:
        """Calculate feature importance using permutation method."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating feature importance")
        
        # Get baseline performance
        baseline_pred = self.predict(X)
        baseline_mse = np.mean((y.ravel() - baseline_pred.ravel()) ** 2)
        
        # Reshape data for permutation (same as in fit/predict)
        if X.ndim == 3:
            # Flatten from (N, lookback_window, features) to (N, lookback_window * features)
            X_reshaped = X.reshape(X.shape[0], -1)
            # Create expanded feature names for time series data
            n_timesteps = X.shape[1]
            n_features_per_timestep = X.shape[2]
            expanded_feature_names = []
            for t in range(n_timesteps):
                for f in range(n_features_per_timestep):
                    expanded_feature_names.append(f"{feature_names[f]}_t{t}")
        else:
            X_reshaped = X
            expanded_feature_names = feature_names
        
        feature_importance = {}
        importance_scores = []
        
        # Calculate importance for each feature
        for i in range(X_reshaped.shape[1]):
            # Create permuted data
            X_permuted = X_reshaped.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # For SVM, we can work directly with the flattened data
            # No need to reshape back since SVM expects flattened input
            X_permuted_scaled = self.scaler.transform(X_permuted)
            
            # Calculate performance with permuted feature
            perm_pred = self.model.predict(X_permuted_scaled).reshape(-1, 1)
            perm_mse = np.mean((y.ravel() - perm_pred.ravel()) ** 2)
            
            # Importance = increase in MSE when feature is permuted
            importance = perm_mse - baseline_mse
            importance_scores.append(importance)
            
            # Determine significance based on importance magnitude
            max_importance = max(importance_scores) if importance_scores else 1.0
            relative_importance = importance / max_importance if max_importance > 0 else 0
            
            feature_importance[expanded_feature_names[i]] = {
                'importance': importance,
                'relative_importance': relative_importance,
                'significant': relative_importance > 0.1,  # Top 10% of features
                'highly_significant': relative_importance > 0.2,  # Top 5% of features
                'rank': 0  # Will be updated after all features are processed
            }
        
        # Rank features by importance
        sorted_indices = np.argsort(importance_scores)[::-1]  # Descending order
        for rank, idx in enumerate(sorted_indices):
            feature_importance[expanded_feature_names[idx]]['rank'] = rank + 1
        
        return feature_importance, np.array(importance_scores)


class ModelTrainer:
    """Model training and evaluation class."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def train_and_evaluate_model(self, X_train: np.ndarray, X_test: np.ndarray, 
                               y_train: np.ndarray, y_test: np.ndarray, 
                               prev_log_test: np.ndarray, 
                               model_name: str = "Model", 
                               epochs: Optional[int] = None,
                               model_type: str = "lstm") -> Dict[str, Any]:
        """
        Train and evaluate a model (LSTM or SVM), returning performance metrics and feature statistics.
        
        Parameters:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        prev_log_test: Previous log values for test set (for level reconstruction)
        model_name: Name for the model
        epochs: Number of training epochs (uses config default if None, only for LSTM)
        model_type: Type of model to use ("lstm" or "svm")
        
        Returns:
        Dictionary containing performance metrics, model, and feature importance
        """
        print(f"\n{'='*50}")
        print(f"Training {model_name} ({model_type.upper()})")
        print(f"{'='*50}")
        
        if model_type == "lstm":
            return self._train_lstm_model(X_train, X_test, y_train, y_test, prev_log_test, 
                                        model_name, epochs)
        elif model_type == "svm":
            return self._train_svm_model(X_train, X_test, y_train, y_test, prev_log_test, 
                                       model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Use 'lstm' or 'svm'.")
    
    def _train_lstm_model(self, X_train: np.ndarray, X_test: np.ndarray, 
                         y_train: np.ndarray, y_test: np.ndarray, 
                         prev_log_test: np.ndarray, model_name: str, 
                         epochs: int) -> Dict[str, Any]:
        """Train and evaluate LSTM model."""
        # Scale inputs
        scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        scaler.fit(X_train_reshaped)
        
        X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
        
        # Convert to tensors
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32)
        
        # Create data loaders
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), 
                                 batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), 
                                batch_size=self.config.batch_size)
        
        # Initialize model
        model = EnhancedLSTMTimeSeries(
            input_size=X_train.shape[-1], 
            hidden_size=self.config.hidden_size, 
            num_layers=self.config.num_layers, 
            output_size=self.config.output_size,
            dropout=self.config.dropout
        ).to(self.device)
        
        criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate, 
                              weight_decay=self.config.weight_decay)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # Early stopping
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}")
            
            if patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Evaluation
        model.eval()
        pred_logret = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(self.device)
                pred_logret.append(model(xb).cpu().numpy())
        pred_logret = np.concatenate(pred_logret, axis=0).ravel()
        
        # Reconstruct levels
        y_pred_levels = np.exp(prev_log_test + pred_logret)
        y_true_levels = np.exp(prev_log_test + y_test.ravel())
        
        # Calculate metrics
        mae = np.mean(np.abs(y_pred_levels - y_true_levels))
        rmse = np.sqrt(np.mean((y_pred_levels - y_true_levels)**2))
        mape = np.mean(np.abs((y_true_levels - y_pred_levels) / y_true_levels)) * 100
        
        print(f"\n{model_name} Performance:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        # Calculate DL-based feature importance
        print(f"\n📊 Calculating DL-based feature importance...")
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[-1])]
        
        # Use permutation importance as default (more robust)
        feature_stats, importance_scores = self._calculate_dl_feature_importance(
            model, X_train_scaled, y_train, feature_names, method=self.config.importance_method
        )

        # Print feature importance summary
        significant_features = [name for name, stats in feature_stats.items() if stats['significant']]
        highly_significant = [name for name, stats in feature_stats.items() if stats['highly_significant']]
        
        print(f"📈 DL-Based Feature Importance Analysis:")
        print(f"   • Total features: {len(feature_stats)}")
        print(f"   • Important features (relative importance larger than 0.3): {len(significant_features)}")
        print(f"   • Highly important features (relative importance larger than 0.5): {len(highly_significant)}")
        
        # Show top 5 most important features with their importance scores
        sorted_features = sorted(feature_stats.items(), key=lambda x: x[1]['importance'], reverse=True)
        print(f"\n🔍 TOP 5 MOST IMPORTANT FEATURES:")
        for i, (name, stats) in enumerate(sorted_features[:5]):
            print(f"   {i+1}. {name}: importance={stats['importance']:.4f}, rank={stats['rank']}")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'predictions': y_pred_levels,
            'true_values': y_true_levels,
            'model': model,
            'scaler': scaler,
            'feature_stats': feature_stats,
            'importance_scores': importance_scores,
            'significant_features': significant_features,
            'highly_significant_features': highly_significant
        }
    
    def _train_svm_model(self, X_train: np.ndarray, X_test: np.ndarray, 
                        y_train: np.ndarray, y_test: np.ndarray, 
                        prev_log_test: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Train and evaluate SVM model."""
        # Initialize and fit SVM model
        svm_model = SVMModel(
            kernel=self.config.svm_kernel,
            C=self.config.svm_C,
            gamma=self.config.svm_gamma,
            epsilon=self.config.svm_epsilon,
            max_iter=self.config.svm_max_iter
        )
        
        print("Training SVM model...")
        svm_model.fit(X_train, y_train)
        
        # Make predictions
        pred_logret = svm_model.predict(X_test).ravel()
        
        # Reconstruct levels
        y_pred_levels = np.exp(prev_log_test + pred_logret)
        y_true_levels = np.exp(prev_log_test + y_test.ravel())
        
        # Calculate metrics
        mae = np.mean(np.abs(y_pred_levels - y_true_levels))
        rmse = np.sqrt(np.mean((y_pred_levels - y_true_levels)**2))
        mape = np.mean(np.abs((y_true_levels - y_pred_levels) / y_true_levels)) * 100
        
        print(f"\n{model_name} Performance:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        # Calculate feature importance
        print(f"\n📊 Calculating SVM feature importance...")
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[-1])]
        
        feature_stats, importance_scores = svm_model.get_feature_importance(
            X_train, y_train, feature_names
        )

        # Print feature importance summary
        significant_features = [name for name, stats in feature_stats.items() if stats['significant']]
        highly_significant = [name for name, stats in feature_stats.items() if stats['highly_significant']]
        
        print(f"📈 SVM Feature Importance Analysis:")
        print(f"   • Total features: {len(feature_stats)}")
        print(f"   • Important features (top 10%): {len(significant_features)}")
        print(f"   • Highly important features (top 5%): {len(highly_significant)}")
        
        # Show top 5 most important features with their importance scores
        sorted_features = sorted(feature_stats.items(), key=lambda x: x[1]['importance'], reverse=True)
        print(f"\n🔍 TOP 5 MOST IMPORTANT FEATURES:")
        for i, (name, stats) in enumerate(sorted_features[:5]):
            print(f"   {i+1}. {name}: importance={stats['importance']:.4f}, rank={stats['rank']}")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'predictions': y_pred_levels,
            'true_values': y_true_levels,
            'model': svm_model,
            'scaler': svm_model.scaler,
            'feature_stats': feature_stats,
            'importance_scores': importance_scores,
            'significant_features': significant_features,
            'highly_significant_features': highly_significant
        }
    
    def _calculate_dl_feature_importance(self, model: nn.Module, X_train: np.ndarray, 
                                       y_train: np.ndarray, feature_names: List[str], 
                                       method: str = 'permutation') -> Tuple[Dict[str, Any], np.ndarray]:
        """Calculate DL-based feature importance using permutation or gradient-based methods."""
        model.eval()
        
        if method == 'permutation':
            return self._calculate_permutation_importance(model, X_train, y_train, feature_names)
        elif method == 'gradient':
            return self._calculate_gradient_importance(model, X_train, y_train, feature_names)
        else:
            raise ValueError("Method must be 'permutation' or 'gradient'")
    
    def _calculate_permutation_importance(self, model: nn.Module, X_train: np.ndarray, 
                                        y_train: np.ndarray, feature_names: List[str]) -> Tuple[Dict[str, Any], np.ndarray]:
        """Calculate feature importance using permutation method."""
        print("📊 Calculating permutation-based feature importance...")
        
        # Get baseline performance
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
            baseline_pred = model(X_tensor)
            baseline_loss = nn.SmoothL1Loss()(baseline_pred, torch.tensor(y_train, dtype=torch.float32).to(self.device)).item()
        
        feature_importance = {}
        importance_scores = []
        
        # Calculate importance for each feature
        for i in range(X_train.shape[-1]):
            # Create permuted data
            X_permuted = X_train.copy()
            np.random.shuffle(X_permuted[:, :, i])  # Permute feature i across all samples and timesteps
            
            # Calculate performance with permuted feature
            with torch.no_grad():
                X_perm_tensor = torch.tensor(X_permuted, dtype=torch.float32).to(self.device)
                perm_pred = model(X_perm_tensor)
                perm_loss = nn.SmoothL1Loss()(perm_pred, torch.tensor(y_train, dtype=torch.float32).to(self.device)).item()
            
            # Importance = increase in loss when feature is permuted
            importance = perm_loss - baseline_loss
            importance_scores.append(importance)
            
            # Determine significance based on importance magnitude
            # Use relative importance (normalized by max importance)
            max_importance = max(importance_scores) if importance_scores else 1.0
            relative_importance = importance / max_importance if max_importance > 0 else 0
            
            feature_importance[feature_names[i]] = {
                'importance': importance,
                'relative_importance': relative_importance,
                'significant': relative_importance > 0.3,  
                'highly_significant': relative_importance > 0.5, 
                'rank': 0  # Will be updated after all features are processed
            }
        
        # Rank features by importance
        sorted_indices = np.argsort(importance_scores)[::-1]  # Descending order
        for rank, idx in enumerate(sorted_indices):
            feature_importance[feature_names[idx]]['rank'] = rank + 1
        
        return feature_importance, np.array(importance_scores)
    
    def _calculate_gradient_importance(self, model: nn.Module, X_train: np.ndarray, 
                                      y_train: np.ndarray, feature_names: List[str]) -> Tuple[Dict[str, Any], np.ndarray]:
        """Calculate feature importance using gradient-based method."""
        print("📊 Calculating gradient-based feature importance...")
        
        model.eval()
        X_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        # Forward pass
        output = model(X_tensor)
        loss = nn.SmoothL1Loss()(output, y_tensor)
        
        # Calculate gradients
        gradients = torch.autograd.grad(loss, X_tensor, retain_graph=True)[0]
        
        # Calculate importance as mean absolute gradient across all samples and timesteps
        importance_scores = torch.mean(torch.abs(gradients), dim=(0, 1)).detach().cpu().numpy()
        
        feature_importance = {}
        
        # Normalize importance scores
        max_importance = np.max(importance_scores) if np.max(importance_scores) > 0 else 1.0
        normalized_importance = importance_scores / max_importance
        
        for i, (name, importance, norm_importance) in enumerate(zip(feature_names, importance_scores, normalized_importance)):
            feature_importance[name] = {
                'importance': float(importance),
                'relative_importance': float(norm_importance),
                'significant': norm_importance > 0.1,  # Top 10% of features
                'highly_significant': norm_importance > 0.2,  # Top 5% of features
                'rank': 0  # Will be updated after all features are processed
            }
        
        # Rank features by importance
        sorted_indices = np.argsort(importance_scores)[::-1]  # Descending order
        for rank, idx in enumerate(sorted_indices):
            feature_importance[feature_names[idx]]['rank'] = rank + 1
        
        return feature_importance, importance_scores
    
    def compare_models(self, baseline_results: Dict[str, Any], 
                      enhanced_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare baseline and enhanced model results."""
        improvement_metrics = calculate_improvement_metrics(
            baseline_results['mape'], enhanced_results['mape']
        )
        
        comparison = {
            'baseline': baseline_results,
            'enhanced': enhanced_results,
            'improvement': improvement_metrics,
            'feature_improvement': {
                'baseline_features': len(baseline_results.get('significant_features', [])),
                'enhanced_features': len(enhanced_results.get('significant_features', [])),
                'feature_count_baseline': baseline_results.get('feature_count', 0),
                'feature_count_enhanced': enhanced_results.get('feature_count', 0)
            }
        }
        
        return comparison
    
    def save_model(self, model: nn.Module, filepath: str) -> None:
        """Save model to file."""
        torch.save(model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, input_size: int, hidden_size: int, 
                  num_layers: int, output_size: int, dropout: float) -> nn.Module:
        """Load model from file."""
        model = EnhancedLSTMTimeSeries(input_size, hidden_size, num_layers, output_size, dropout)
        model.load_state_dict(torch.load(filepath))
        model.eval()
        print(f"Model loaded from {filepath}")
        return model


class ModelEvaluator:
    """Model evaluation utilities."""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate standard regression metrics."""
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
    
    @staticmethod
    def calculate_feature_importance_summary(feature_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for feature importance."""
        if not feature_stats:
            return {}
        
        importance_values = [stats['importance'] for stats in feature_stats.values()]
        relative_importance_values = [stats['relative_importance'] for stats in feature_stats.values()]
        
        significant_count = sum(1 for stats in feature_stats.values() if stats['significant'])
        highly_significant_count = sum(1 for stats in feature_stats.values() if stats['highly_significant'])
        
        return {
            'total_features': len(feature_stats),
            'significant_features': significant_count,
            'highly_significant_features': highly_significant_count,
            'mean_importance': np.mean(importance_values),
            'std_importance': np.std(importance_values),
            'max_importance': np.max(importance_values),
            'min_importance': np.min(importance_values),
            'mean_relative_importance': np.mean(relative_importance_values),
            'std_relative_importance': np.std(relative_importance_values)
        }
    
    @staticmethod
    def create_performance_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a comprehensive performance summary from multiple results."""
        if not results:
            return {}
        
        mape_values = [r.get('mape', 0) for r in results]
        mae_values = [r.get('mae', 0) for r in results]
        rmse_values = [r.get('rmse', 0) for r in results]
        
        return {
            'count': len(results),
            'mape': {
                'mean': np.mean(mape_values),
                'std': np.std(mape_values),
                'min': np.min(mape_values),
                'max': np.max(mape_values),
                'median': np.median(mape_values)
            },
            'mae': {
                'mean': np.mean(mae_values),
                'std': np.std(mae_values),
                'min': np.min(mae_values),
                'max': np.max(mae_values),
                'median': np.median(mae_values)
            },
            'rmse': {
                'mean': np.mean(rmse_values),
                'std': np.std(rmse_values),
                'min': np.min(rmse_values),
                'max': np.max(rmse_values),
                'median': np.median(rmse_values)
            }
        }
