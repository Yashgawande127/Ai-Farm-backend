"""
Feature Engineering Module for Crop Recommendation System
Implements domain-specific features (SFI, CSI) and statistical features with KPCA
"""

import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
import logging

logger = logging.getLogger(__name__)


def extract_domain_specific_features(X, feature_ranges=None):
    """
    Extract domain-specific features for agriculture:
    - SFI (Soil Fertility Index): Combined measure of soil nutrients (N, P, K, pH)
    - CSI (Climate Suitability Index): Combined measure of climate factors (temp, humidity, rainfall)
    
    Args:
        X: numpy array with shape (n_samples, 7) containing [N, P, K, temp, humidity, ph, rainfall]
        feature_ranges: Optional dictionary or list containing min and max values for each feature.
                       Example: {'min': [0, 5, 5, ...], 'max': [140, 145, 205, ...]}
    
    Returns:
        numpy array with shape (n_samples, 5) containing domain features
    """
    try:
        # Extract individual features
        N = X[:, 0]
        P = X[:, 1]
        K = X[:, 2]
        temperature = X[:, 3]
        humidity = X[:, 4]
        ph = X[:, 5]
        rainfall = X[:, 6]
        
        # Helper for normalization using provided or calculated ranges
        def normalize_feature(data, idx):
            if feature_ranges is not None:
                f_min = feature_ranges['min'][idx]
                f_max = feature_ranges['max'][idx]
            else:
                f_min = np.min(data)
                f_max = np.max(data)
            
            return (data - f_min) / (f_max - f_min + 1e-10)
        
        N_norm = normalize_feature(N, 0)
        P_norm = normalize_feature(P, 1)
        K_norm = normalize_feature(K, 2)
        temp_norm = normalize_feature(temperature, 3)
        humidity_norm = normalize_feature(humidity, 4)
        ph_norm = normalize_feature(ph, 5)
        rainfall_norm = normalize_feature(rainfall, 6)
        
        # SFI (Soil Fertility Index)
        # Weighted combination of N, P, K, and pH
        # Nitrogen, Phosphorus, and Potassium are primary nutrients (30% each)
        # pH affects nutrient availability (10%)
        SFI = (0.3 * N_norm + 0.3 * P_norm + 0.3 * K_norm + 0.1 * ph_norm)
        
        # CSI (Climate Suitability Index)
        # Weighted combination of temperature, humidity, and rainfall
        # All three factors are crucial for crop growth
        CSI = (0.35 * temp_norm + 0.35 * humidity_norm + 0.30 * rainfall_norm)
        
        # NPK Ratio - Important for nutrient balance
        # Indicates the ratio of nitrogen to other primary nutrients
        NPK_ratio = (N + 1) / (P + K + 2)  # Add constants to avoid division by zero
        
        # Nutrient Balance Score
        # Lower values indicate better balance among NPK nutrients
        nutrient_balance = np.std([N_norm, P_norm, K_norm], axis=0)
        
        # Climate Stress Index
        # Measures deviation from ideal climate conditions
        # Assumes ideal humidity is around 75% (0.75 normalized)
        # and temperature should be moderate (0.5 normalized = middle of range)
        climate_stress = np.abs(temp_norm - 0.5) + np.abs(humidity_norm - 0.75)
        
        # Combine domain features
        domain_features = np.column_stack([
            SFI, 
            CSI, 
            NPK_ratio, 
            nutrient_balance,
            climate_stress
        ])
        
        logger.info(f"Domain features extracted - SFI mean: {np.mean(SFI):.4f}, CSI mean: {np.mean(CSI):.4f}")
        
        return domain_features
    
    except Exception as e:
        logger.error(f"Error extracting domain-specific features: {str(e)}")
        raise


def extract_statistical_features(X):
    """
    Extract statistical features from the original features:
    - Mean, Standard Deviation, Skewness, Kurtosis
    - Maximum, Minimum, Median, Range
    
    Args:
        X: numpy array with shape (n_samples, 7)
    
    Returns:
        numpy array with shape (n_samples, 8) containing statistical features
    """
    try:
        # Calculate statistical measures for each sample
        statistical_features = []
        for sample in X:
            stats = [
                np.mean(sample),           # Mean
                np.std(sample),            # Standard deviation
                skew(sample),              # Skewness (asymmetry of distribution)
                kurtosis(sample),          # Kurtosis (tailedness of distribution)
                np.max(sample),            # Maximum value
                np.min(sample),            # Minimum value
                np.median(sample),         # Median (middle value)
                np.ptp(sample)             # Peak-to-peak (range)
            ]
            statistical_features.append(stats)
        
        statistical_features = np.array(statistical_features)
        
        logger.info(f"Statistical features extracted - shape: {statistical_features.shape}")
        
        return statistical_features
    
    except Exception as e:
        logger.error(f"Error extracting statistical features: {str(e)}")
        raise


def apply_kpca(X, n_components=10, kernel='rbf', kpca_model=None):
    """
    Apply Kernel PCA for non-linear dimensionality reduction and correlation-based features
    KPCA captures complex non-linear relationships between features
    
    Args:
        X: numpy array with features
        n_components: number of principal components to extract
        kernel: kernel type ('rbf', 'poly', 'sigmoid', 'cosine')
        kpca_model: pre-fitted KPCA model (for prediction), or None (for training)
    
    Returns:
        Tuple of (transformed features, kpca_model)
    """
    try:
        if kpca_model is None:
            # Training mode - fit new KPCA model
            logger.info(f"Fitting Kernel PCA with {n_components} components and '{kernel}' kernel...")
            kpca_model = KernelPCA(
                n_components=n_components, 
                kernel=kernel,
                gamma=None,  # Auto-compute gamma for RBF kernel
                fit_inverse_transform=True, 
                random_state=42
            )
            X_kpca = kpca_model.fit_transform(X)
            logger.info(f"KPCA fitted and transformed - shape: {X_kpca.shape}")
        else:
            # Prediction mode - use existing KPCA model
            X_kpca = kpca_model.transform(X)
            logger.info(f"KPCA transformed using existing model - shape: {X_kpca.shape}")
        
        return X_kpca, kpca_model
    
    except Exception as e:
        logger.error(f"Error applying KPCA: {str(e)}")
        raise


def create_enhanced_features(X, scaler=None, kpca_model=None, feature_ranges=None, training=True):
    """
    Create enhanced feature set combining:
    1. Original features (7)
    2. Domain-specific features (5): SFI, CSI, NPK_ratio, nutrient_balance, climate_stress
    3. Statistical features (8): mean, std, skew, kurtosis, max, min, median, range
    4. KPCA features (10): correlation-based non-linear features
    
    Total: 30 features
    
    Args:
        X: numpy array with original features [N, P, K, temp, humidity, ph, rainfall]
        scaler: StandardScaler object (None for training, fitted scaler for prediction)
        kpca_model: KernelPCA object (None for training, fitted model for prediction)
        feature_ranges: Optional dict with 'min' and 'max' lists for normalization
        training: True if training, False if prediction
    
    Returns:
        If training: Tuple of (enhanced_features, scaler, kpca_model, calculated_ranges)
        If prediction: enhanced_features only
    """
    try:
        logger.info(f"Creating enhanced features - Original shape: {X.shape}")
        
        # Calculate ranges if in training and not provided
        calculated_ranges = None
        if training and feature_ranges is None:
            calculated_ranges = {
                'min': np.min(X, axis=0).tolist(),
                'max': np.max(X, axis=0).tolist()
            }
            feature_ranges = calculated_ranges
        
        # Extract domain-specific features (SFI, CSI, etc.) using global ranges
        domain_features = extract_domain_specific_features(X, feature_ranges=feature_ranges)
        logger.info(f"Domain features shape: {domain_features.shape}")
        
        # Extract statistical features
        statistical_features = extract_statistical_features(X)
        logger.info(f"Statistical features shape: {statistical_features.shape}")
        
        # Combine original and new features for KPCA
        combined_features = np.column_stack([X, domain_features, statistical_features])
        logger.info(f"Combined features shape before KPCA: {combined_features.shape}")
        
        if training:
            # Training mode - create new scaler and KPCA model
            scaler = StandardScaler()
            combined_features_scaled = scaler.fit_transform(combined_features)
            logger.info("Features scaled using new StandardScaler")
            
            # Apply KPCA for correlation-based feature extraction
            kpca_features, kpca_model = apply_kpca(
                combined_features_scaled, 
                n_components=10, 
                kernel='rbf',
                kpca_model=None
            )
        else:
            # Prediction mode - use existing scaler and KPCA model
            if scaler is None or kpca_model is None:
                raise ValueError("Scaler and KPCA model must be provided for prediction mode")
            
            combined_features_scaled = scaler.transform(combined_features)
            logger.info("Features scaled using existing StandardScaler")
            
            kpca_features, _ = apply_kpca(
                combined_features_scaled,
                kpca_model=kpca_model
            )
        
        # Final enhanced feature set
        enhanced_features = np.column_stack([
            X,                      # Original features (7)
            domain_features,        # Domain-specific features (5)
            statistical_features,   # Statistical features (8)
            kpca_features          # KPCA features (10)
        ])
        
        logger.info(f"Final enhanced features shape: {enhanced_features.shape}")
        logger.info(f"Total features: {enhanced_features.shape[1]}")
        
        if training:
            return enhanced_features, scaler, kpca_model, calculated_ranges
        else:
            return enhanced_features
    
    except Exception as e:
        logger.error(f"Error creating enhanced features: {str(e)}")
        raise


def get_feature_names():
    """
    Get descriptive names for all features in the enhanced feature set
    
    Returns:
        List of feature names
    """
    original_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    domain_features = ['SFI', 'CSI', 'NPK_ratio', 'nutrient_balance', 'climate_stress']
    statistical_features = ['stat_mean', 'stat_std', 'stat_skew', 'stat_kurtosis', 
                           'stat_max', 'stat_min', 'stat_median', 'stat_range']
    kpca_features = [f'KPCA_{i+1}' for i in range(10)]
    
    return original_features + domain_features + statistical_features + kpca_features


def extract_features_for_prediction(features_dict, scaler, kpca_model):
    """
    Convenience function to extract enhanced features from a dictionary of input values
    Used for single predictions in the API
    
    Args:
        features_dict: Dictionary with keys ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        scaler: Fitted StandardScaler
        kpca_model: Fitted KernelPCA model
    
    Returns:
        numpy array with enhanced features ready for prediction
    """
    try:
        # Extract values in correct order
        feature_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        X = np.array([[features_dict[key] for key in feature_order]])
        
        # Create enhanced features
        enhanced_features = create_enhanced_features(
            X, 
            scaler=scaler, 
            kpca_model=kpca_model, 
            training=False
        )
        
        return enhanced_features
    
    except Exception as e:
        logger.error(f"Error extracting features for prediction: {str(e)}")
        raise
