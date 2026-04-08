import pickle
import numpy as np
import os
import logging
from typing import Tuple, Optional
from feature_engineering import create_enhanced_features
import torch
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CropPredictionService:
    """Service class for loading models and making crop predictions with enhanced features"""
    
    def __init__(self, model_path='models/crop_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.rf_model = None
        self.dt_model = None
        self.model_comparison = None
        self.scaler = None
        self.kpca_model = None
        self.feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Deep learning models
        self.dl_models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dl_model_names = ['deepagyieldnet', 'shufflenetv2', 'efficientcapsnet', 'ladnet', 'regnet']
        
    def load_model(self):
        """Load the default trained model from pickle file"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
            raise
    
    def load_all_models(self):
        """Load all available models (Random Forest and Decision Tree) and feature extractors"""
        try:
            # Load Random Forest model
            rf_path = 'models/random_forest_model.pkl'
            if os.path.exists(rf_path):
                with open(rf_path, 'rb') as f:
                    self.rf_model = pickle.load(f)
                logger.info("Random Forest model loaded successfully")
            
            # Load Decision Tree model
            dt_path = 'models/decision_tree_model.pkl'
            if os.path.exists(dt_path):
                with open(dt_path, 'rb') as f:
                    self.dt_model = pickle.load(f)
                logger.info("Decision Tree model loaded successfully")
            
            # Load feature extractors (scaler and KPCA model)
            scaler_path = 'models/feature_scaler.pkl'
            kpca_path = 'models/kpca_model.pkl'
            ranges_path = 'models/feature_ranges.pkl'
            
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Feature scaler loaded successfully")
            
            if os.path.exists(kpca_path):
                with open(kpca_path, 'rb') as f:
                    self.kpca_model = pickle.load(f)
                logger.info("KPCA model loaded successfully")

            if os.path.exists(ranges_path):
                with open(ranges_path, 'rb') as f:
                    self.feature_ranges = pickle.load(f)
                logger.info("Feature ranges loaded successfully")
            else:
                self.feature_ranges = None
                logger.warning("Feature ranges file not found. Domain initialization might be compromised.")
            
            # Load model comparison results
            comparison_path = 'models/model_comparison.json'
            if os.path.exists(comparison_path):
                with open(comparison_path, 'r') as f:
                    self.model_comparison = json.load(f)
                logger.info("Model comparison results loaded successfully")
            
            # Load deep learning models
            self.load_deep_learning_models()
            
            # Load default model as fallback
            self.load_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading all models: {str(e)}")
            raise
    
    def load_deep_learning_models(self):
        """Load deep learning models"""
        try:
            dl_dir = 'models/deep_learning'
            if not os.path.exists(dl_dir):
                logger.info("Deep learning models directory not found, skipping...")
                return
            
            from deep_learning_models import DeepLearningTrainer
            
            for model_name in self.dl_model_names:
                model_path = os.path.join(dl_dir, f'{model_name}_model.pth')
                if os.path.exists(model_path):
                    try:
                        trainer = DeepLearningTrainer(model_name)
                        # We'll need to determine input_size and num_classes dynamically
                        # For now, use standard values - these will be updated when we load real data
                        trainer.load_model(model_path, input_size=30, num_classes=22)  # Enhanced features + crop classes
                        self.dl_models[model_name] = trainer
                        logger.info(f"Deep learning model {model_name} loaded successfully")
                    except Exception as e:
                        logger.warning(f"Failed to load {model_name}: {str(e)}")
            
            logger.info(f"Loaded {len(self.dl_models)} deep learning models")
            
        except Exception as e:
            logger.warning(f"Error loading deep learning models: {str(e)}")

    def load_model_comparison(self):
        """Load model comparison results"""
        try:
            comparison_path = 'models/model_comparison.json'
            if os.path.exists(comparison_path):
                with open(comparison_path, 'r') as f:
                    self.model_comparison = json.load(f)
                return self.model_comparison
            else:
                logger.warning("Model comparison file not found")
                return None
        except Exception as e:
            logger.error(f"Error loading model comparison: {str(e)}")
            return None
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready for predictions"""
        return self.model is not None
    
    def validate_input(self, features: list) -> bool:
        """Validate input features"""
        try:
            # Check if we have the right number of features
            if len(features) != 7:
                raise ValueError(f"Expected 7 features, got {len(features)}")
            
            # Check if all features are numeric
            for i, feature in enumerate(features):
                if not isinstance(feature, (int, float)):
                    try:
                        features[i] = float(feature)
                    except (ValueError, TypeError):
                        raise ValueError(f"Feature {self.feature_names[i]} must be numeric, got {type(feature)}")
            
            # Validate feature ranges (basic sanity checks)
            validation_rules = {
                'N': (0, 200),      # Nitrogen: 0-200
                'P': (0, 150),      # Phosphorus: 0-150  
                'K': (0, 200),      # Potassium: 0-200
                'temperature': (-10, 50),  # Temperature: -10 to 50°C
                'humidity': (0, 100),      # Humidity: 0-100%
                'ph': (0, 14),            # pH: 0-14
                'rainfall': (0, 500)       # Rainfall: 0-500mm
            }
            
            for i, (feature_name, (min_val, max_val)) in enumerate(validation_rules.items()):
                if not (min_val <= features[i] <= max_val):
                    logger.warning(f"{feature_name} value {features[i]} is outside typical range [{min_val}, {max_val}]")
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation failed: {str(e)}")
            raise
    
    def predict(self, features: list) -> Tuple[str, float]:
        """
        Make crop prediction based on input features
        
        Args:
            features: List of 7 numeric values [N, P, K, temperature, humidity, ph, rainfall]
            
        Returns:
            Tuple of (predicted_crop, confidence_score)
        """
        try:
            # Check if model is loaded
            if not self.is_model_loaded():
                raise RuntimeError("Model not loaded. Please load the model first.")
            
            # Validate input
            self.validate_input(features)
            
            # Convert to numpy array
            features_array = np.array(features).reshape(1, -1)
            
            # Apply feature engineering if scaler and KPCA model are available
            if self.scaler is not None and self.kpca_model is not None:
                logger.info("Applying feature engineering transformations...")
                features_array = create_enhanced_features(
                    features_array, 
                    scaler=self.scaler, 
                    kpca_model=self.kpca_model, 
                    feature_ranges=self.feature_ranges,
                    training=False
                )
            else:
                logger.warning("Feature extractors not loaded. Using original features only.")
            
            # Make prediction
            prediction = self.model.predict(features_array)[0]
            
            # Get prediction probabilities for confidence score
            prediction_proba = self.model.predict_proba(features_array)[0]
            confidence = float(np.max(prediction_proba))
            
            logger.info(f"Prediction made: {prediction} with confidence: {confidence:.4f}")
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def get_prediction_with_probabilities(self, features: list) -> dict:
        """
        Get detailed prediction with probabilities for all crops
        
        Args:
            features: List of 7 numeric values [N, P, K, temperature, humidity, ph, rainfall]
            
        Returns:
            Dictionary containing prediction details
        """
        try:
            # Check if model is loaded
            if not self.is_model_loaded():
                raise RuntimeError("Model not loaded. Please load the model first.")
            
            # Validate input
            self.validate_input(features)
            
            # Convert to numpy array
            features_array = np.array(features).reshape(1, -1)
            
            # Apply feature engineering if scaler and KPCA model are available
            if self.scaler is not None and self.kpca_model is not None:
                logger.info("Applying feature engineering transformations...")
                features_array = create_enhanced_features(
                    features_array, 
                    scaler=self.scaler, 
                    kpca_model=self.kpca_model, 
                    feature_ranges=self.feature_ranges,
                    training=False
                )
            else:
                logger.warning("Feature extractors not loaded. Using original features only.")
            
            # Make prediction
            prediction = self.model.predict(features_array)[0]
            
            # Get prediction probabilities
            prediction_proba = self.model.predict_proba(features_array)[0]
            
            # Get class names
            classes = self.model.classes_
            
            # Create probability dictionary
            probabilities = {
                crop: float(prob) 
                for crop, prob in zip(classes, prediction_proba)
            }
            
            # Sort probabilities in descending order
            sorted_probabilities = dict(
                sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            )
            
            result = {
                'predicted_crop': prediction,
                'confidence': float(np.max(prediction_proba)),
                'all_probabilities': sorted_probabilities,
                'top_3_crops': list(sorted_probabilities.keys())[:3]
            }
            
            logger.info(f"Detailed prediction made for crop: {prediction}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during detailed prediction: {str(e)}")
            raise
    
    def predict_with_specific_model(self, features: list, model_type: str) -> Tuple[str, float]:
        """
        Make prediction using a specific model type
        
        Args:
            features: List of 7 numeric values [N, P, K, temperature, humidity, ph, rainfall]
            model_type: Either 'random_forest' or 'decision_tree'
            
        Returns:
            Tuple of (predicted_crop, confidence_score)
        """
        try:
            # Validate input
            self.validate_input(features)
            
            # Convert to numpy array
            features_array = np.array(features).reshape(1, -1)
            
            # Apply feature engineering if scaler and KPCA model are available
            if self.scaler is not None and self.kpca_model is not None:
                logger.info("Applying feature engineering transformations...")
                features_array = create_enhanced_features(
                    features_array, 
                    scaler=self.scaler, 
                    kpca_model=self.kpca_model, 
                    feature_ranges=self.feature_ranges,
                    training=False
                )
            else:
                logger.warning("Feature extractors not loaded. Using original features only.")
            
            # Select the appropriate model
            if model_type == 'random_forest':
                if self.rf_model is None:
                    raise RuntimeError("Random Forest model not loaded. Please load all models first.")
                model = self.rf_model
            elif model_type == 'decision_tree':
                if self.dt_model is None:
                    raise RuntimeError("Decision Tree model not loaded. Please load all models first.")
                model = self.dt_model
            else:
                raise ValueError(f"Unknown model type: {model_type}. Use 'random_forest' or 'decision_tree'")
            
            # Make prediction
            prediction = model.predict(features_array)[0]
            
            # Get prediction probabilities for confidence score
            prediction_proba = model.predict_proba(features_array)[0]
            confidence = float(np.max(prediction_proba))
            
            logger.info(f"Prediction made with {model_type}: {prediction} with confidence: {confidence:.4f}")
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Error during {model_type} prediction: {str(e)}")
            raise
    
    def predict_ensemble(self, features: list) -> dict:
        """
        Make ensemble prediction using both Random Forest and Decision Tree models
        
        Args:
            features: List of 7 numeric values [N, P, K, temperature, humidity, ph, rainfall]
            
        Returns:
            Dictionary containing ensemble prediction results
        """
        try:
            # Validate input
            self.validate_input(features)
            
            # Convert to numpy array
            features_array = np.array(features).reshape(1, -1)
            
            # Apply feature engineering if scaler and KPCA model are available
            if self.scaler is not None and self.kpca_model is not None:
                logger.info("Applying feature engineering transformations...")
                features_array = create_enhanced_features(
                    features_array, 
                    scaler=self.scaler, 
                    kpca_model=self.kpca_model, 
                    feature_ranges=self.feature_ranges,
                    training=False
                )
            else:
                logger.warning("Feature extractors not loaded. Using original features only.")
            
            results = {}
            
            # Get predictions from Random Forest if available
            if self.rf_model is not None:
                rf_pred = self.rf_model.predict(features_array)[0]
                rf_proba = self.rf_model.predict_proba(features_array)[0]
                rf_confidence = float(np.max(rf_proba))
                
                results['random_forest'] = {
                    'prediction': rf_pred,
                    'confidence': rf_confidence,
                    'probabilities': {
                        crop: float(prob) 
                        for crop, prob in zip(self.rf_model.classes_, rf_proba)
                    }
                }
            
            # Get predictions from Decision Tree if available
            if self.dt_model is not None:
                dt_pred = self.dt_model.predict(features_array)[0]
                dt_proba = self.dt_model.predict_proba(features_array)[0]
                dt_confidence = float(np.max(dt_proba))
                
                results['decision_tree'] = {
                    'prediction': dt_pred,
                    'confidence': dt_confidence,
                    'probabilities': {
                        crop: float(prob) 
                        for crop, prob in zip(self.dt_model.classes_, dt_proba)
                    }
                }
            
            # Determine ensemble prediction
            if 'random_forest' in results and 'decision_tree' in results:
                rf_pred = results['random_forest']['prediction']
                dt_pred = results['decision_tree']['prediction']
                rf_conf = results['random_forest']['confidence']
                dt_conf = results['decision_tree']['confidence']
                
                if rf_pred == dt_pred:
                    # Both models agree
                    ensemble_pred = rf_pred
                    ensemble_confidence = (rf_conf + dt_conf) / 2
                    agreement = True
                else:
                    # Models disagree - choose the one with higher confidence
                    if rf_conf > dt_conf:
                        ensemble_pred = rf_pred
                        ensemble_confidence = rf_conf
                    else:
                        ensemble_pred = dt_pred
                        ensemble_confidence = dt_conf
                    agreement = False
                
                results['ensemble'] = {
                    'prediction': ensemble_pred,
                    'confidence': ensemble_confidence,
                    'models_agree': agreement,
                    'confidence_difference': abs(rf_conf - dt_conf)
                }
            
            # Add model comparison info if available
            if self.model_comparison:
                results['model_comparison'] = self.model_comparison
            
            logger.info(f"Ensemble prediction completed: {results.get('ensemble', {}).get('prediction', 'N/A')}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during ensemble prediction: {str(e)}")
            raise
    
    def get_available_models(self) -> dict:
        """Get information about available models"""
        return {
            'default_model': self.model is not None,
            'random_forest': self.rf_model is not None,
            'decision_tree': self.dt_model is not None,
            'model_comparison': self.model_comparison is not None
        }
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        try:
            if not self.is_model_loaded():
                return {"status": "Model not loaded"}
            
            info = {
                "model_type": type(self.model).__name__,
                "n_estimators": getattr(self.model, 'n_estimators', 'N/A'),
                "max_depth": getattr(self.model, 'max_depth', 'N/A'),
                "n_features": getattr(self.model, 'n_features_in_', 'N/A'),
                "n_classes": len(getattr(self.model, 'classes_', [])),
                "classes": list(getattr(self.model, 'classes_', [])),
                "feature_names": self.feature_names
            }
            
            # Get feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = {
                    name: float(importance) 
                    for name, importance in zip(self.feature_names, self.model.feature_importances_)
                }
                info['feature_importance'] = dict(
                    sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                )
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {"error": str(e)}

# Example usage and testing
if __name__ == "__main__":
    # Initialize the prediction service
    service = CropPredictionService()
    
    try:
        # Load all models
        service.load_all_models()
        
        # Example prediction
        # Sample features: [N, P, K, temperature, humidity, ph, rainfall]
        sample_features = [90, 42, 43, 20.88, 82.00, 6.50, 202.94]
        
        print("=== Crop Prediction Comparison ===")
        
        # Make prediction with default model
        crop, confidence = service.predict(sample_features)
        print(f"Default Model - Crop: {crop}, Confidence: {confidence:.4f}")
        
        # Make predictions with specific models
        available_models = service.get_available_models()
        
        if available_models['random_forest']:
            rf_crop, rf_conf = service.predict_with_specific_model(sample_features, 'random_forest')
            print(f"Random Forest - Crop: {rf_crop}, Confidence: {rf_conf:.4f}")
        
        if available_models['decision_tree']:
            dt_crop, dt_conf = service.predict_with_specific_model(sample_features, 'decision_tree')
            print(f"Decision Tree - Crop: {dt_crop}, Confidence: {dt_conf:.4f}")
        
        # Get ensemble prediction
        ensemble_result = service.predict_ensemble(sample_features)
        print(f"\n=== Ensemble Prediction ===")
        if 'ensemble' in ensemble_result:
            ensemble = ensemble_result['ensemble']
            print(f"Ensemble Prediction: {ensemble['prediction']}")
            print(f"Ensemble Confidence: {ensemble['confidence']:.4f}")
            print(f"Models Agree: {ensemble['models_agree']}")
        
        # Get detailed prediction
        detailed_result = service.get_prediction_with_probabilities(sample_features)
        print(f"\n=== Top 3 Predictions ===")
        for i, crop in enumerate(detailed_result['top_3_crops'][:3], 1):
            prob = detailed_result['all_probabilities'][crop]
            print(f"{i}. {crop}: {prob:.4f}")
        
        print(f"\n=== Available Models ===")
        print(f"Available models: {available_models}")
        
    except Exception as e:
        print(f"Error: {e}")