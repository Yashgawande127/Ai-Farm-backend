import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from config import FEATURE_INFO, MODEL_THRESHOLDS

def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def validate_features(features: List[float], strict: bool = False) -> Dict[str, Any]:
    """
    Validate input features against expected ranges
    
    Args:
        features: List of feature values [N, P, K, temperature, humidity, ph, rainfall]
        strict: If True, raise exception for out-of-range values
        
    Returns:
        Dictionary with validation results
    """
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    validation_result = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    if len(features) != 7:
        validation_result['valid'] = False
        validation_result['errors'].append(f"Expected 7 features, got {len(features)}")
        return validation_result
    
    for i, (feature_name, value) in enumerate(zip(feature_names, features)):
        feature_config = FEATURE_INFO[feature_name]
        
        # Check if value is numeric
        try:
            value = float(value)
            features[i] = value
        except (ValueError, TypeError):
            validation_result['valid'] = False
            validation_result['errors'].append(f"{feature_name} must be numeric, got {type(value)}")
            continue
        
        # Check hard limits
        if value < feature_config['min_value'] or value > feature_config['max_value']:
            error_msg = f"{feature_name} value {value} is outside valid range [{feature_config['min_value']}, {feature_config['max_value']}]"
            if strict:
                validation_result['valid'] = False
                validation_result['errors'].append(error_msg)
            else:
                validation_result['warnings'].append(error_msg)
        
        # Check typical ranges
        typical_min, typical_max = feature_config['typical_range']
        if not (typical_min <= value <= typical_max):
            validation_result['warnings'].append(
                f"{feature_name} value {value} is outside typical range [{typical_min}, {typical_max}]"
            )
    
    return validation_result

def format_prediction_response(prediction: str, confidence: float, features: List[float]) -> Dict[str, Any]:
    """Format prediction response with additional information"""
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    return {
        'predicted_crop': prediction,
        'confidence': round(confidence, 4),
        'confidence_level': get_confidence_level(confidence),
        'input_features': {
            name: value for name, value in zip(feature_names, features)
        },
        'timestamp': datetime.now().isoformat(),
        'recommendations': get_crop_recommendations(prediction, confidence)
    }

def get_confidence_level(confidence: float) -> str:
    """Get confidence level description"""
    if confidence >= 0.9:
        return 'Very High'
    elif confidence >= 0.8:
        return 'High'
    elif confidence >= 0.6:
        return 'Medium'
    elif confidence >= 0.4:
        return 'Low'
    else:
        return 'Very Low'

def get_crop_recommendations(crop: str, confidence: float) -> List[str]:
    """Get recommendations based on prediction and confidence"""
    recommendations = []
    
    if confidence < MODEL_THRESHOLDS['min_confidence']:
        recommendations.append("Low confidence prediction. Consider retesting with more accurate measurements.")
    
    # Add crop-specific recommendations
    crop_advice = {
        'rice': [
            "Ensure adequate water supply during growing season",
            "Monitor for blast disease in humid conditions",
            "Consider split application of nitrogen fertilizer"
        ],
        'wheat': [
            "Plant during cooler months for optimal growth",
            "Ensure good drainage to prevent root rot",
            "Monitor for rust diseases in moderate temperatures"
        ],
        'corn': [
            "Plant after soil temperature reaches 10°C",
            "Ensure consistent water supply during tasseling",
            "Apply nitrogen fertilizer in split doses"
        ],
        'cotton': [
            "Requires warm temperatures and long growing season",
            "Ensure adequate potassium for fiber quality",
            "Monitor for bollworm and other pests"
        ]
    }
    
    if crop.lower() in crop_advice:
        recommendations.extend(crop_advice[crop.lower()][:2])  # Add top 2 recommendations
    
    return recommendations

def calculate_model_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    """Calculate comprehensive model performance metrics"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    metrics = {}
    
    # Basic accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, recall, F1 (macro average)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1
    
    # Class-wise metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    # Get unique classes
    unique_classes = np.unique(y_true)
    
    metrics['per_class'] = {}
    for i, cls in enumerate(unique_classes):
        metrics['per_class'][cls] = {
            'precision': precision_per_class[i],
            'recall': recall_per_class[i],
            'f1_score': f1_per_class[i],
            'support': int(support[i])
        }
    
    return metrics

def create_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Create a comprehensive summary of the dataset"""
    feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    summary = {
        'total_samples': len(df),
        'features': len(feature_columns),
        'target_classes': df['label'].nunique(),
        'class_distribution': df['label'].value_counts().to_dict(),
        'feature_statistics': {}
    }
    
    # Feature statistics
    for col in feature_columns:
        if col in df.columns:
            summary['feature_statistics'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'missing_values': int(df[col].isnull().sum())
            }
    
    return summary

def ensure_directory(directory_path: str) -> None:
    """Ensure directory exists, create if not"""
    os.makedirs(directory_path, exist_ok=True)

def save_model_metadata(model_path: str, metadata: Dict[str, Any]) -> None:
    """Save model metadata alongside the model file"""
    import json
    
    metadata_path = model_path.replace('.pkl', '_metadata.json')
    
    # Add timestamp
    metadata['saved_at'] = datetime.now().isoformat()
    metadata['model_path'] = model_path
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def load_model_metadata(model_path: str) -> Optional[Dict[str, Any]]:
    """Load model metadata if it exists"""
    import json
    
    metadata_path = model_path.replace('.pkl', '_metadata.json')
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Could not load model metadata: {e}")
    
    return None

def health_check() -> Dict[str, Any]:
    """Perform application health check"""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'checks': {}
    }
    
    # Check if model directory exists
    model_dir = 'models'
    health_status['checks']['model_directory'] = {
        'status': 'ok' if os.path.exists(model_dir) else 'missing',
        'path': os.path.abspath(model_dir)
    }
    
    # Check if dataset exists
    dataset_paths = ['../Crop_recommendation.csv', 'Crop_recommendation.csv']
    dataset_found = False
    for path in dataset_paths:
        if os.path.exists(path):
            dataset_found = True
            health_status['checks']['dataset'] = {
                'status': 'ok',
                'path': os.path.abspath(path)
            }
            break
    
    if not dataset_found:
        health_status['checks']['dataset'] = {
            'status': 'missing',
            'searched_paths': [os.path.abspath(p) for p in dataset_paths]
        }
    
    # Check if model exists
    model_path = os.path.join(model_dir, 'crop_model.pkl')
    health_status['checks']['trained_model'] = {
        'status': 'ok' if os.path.exists(model_path) else 'missing',
        'path': os.path.abspath(model_path)
    }
    
    # Overall status
    if any(check['status'] == 'missing' for check in health_status['checks'].values()):
        health_status['status'] = 'degraded'
    
    return health_status

# Example usage
if __name__ == "__main__":
    # Test validation function
    test_features = [90, 42, 43, 20.88, 82.00, 6.50, 202.94]
    validation_result = validate_features(test_features)
    print("Validation result:", validation_result)
    
    # Test health check
    health_result = health_check()
    print("Health check:", health_result)