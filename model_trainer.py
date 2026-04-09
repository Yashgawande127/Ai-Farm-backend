import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA
from scipy.stats import skew, kurtosis
import pickle
import os
import json
import logging
from feature_engineering import create_enhanced_features, get_feature_names
from deep_learning_models import train_all_models as train_deep_learning_models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load and preprocess the crop recommendation dataset"""
    try:
        # Load the CSV file
        data_path = 'Crop_recommendation.csv'
        if not os.path.exists(data_path):
            # Try parent directory for local development compatibility
            data_path = '../Crop_recommendation.csv'
        
        df = pd.read_csv(data_path)
        logger.info(f"Dataset loaded successfully with shape: {df.shape}")
        
        # Display basic information about the dataset
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Unique crops: {df['label'].nunique()}")
        logger.info(f"Crop types: {sorted(df['label'].unique())}")
        
        return df
    
    except FileNotFoundError:
        logger.error(f"Dataset file not found at {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def preprocess_data(df):
    """Preprocess the dataset for training"""
    try:
        # Separate features and target
        feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Check if all required columns exist
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")
        
        X = df[feature_columns].values
        y = df['label'].values
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        
        # Check for missing values
        if pd.DataFrame(X).isnull().sum().sum() > 0:
            logger.warning("Missing values found in features")
        
        return X, y, feature_columns
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

# Redundant feature engineering functions removed - using centralized feature_engineering.py


def train_random_forest_model(X, y):
    """Train Random Forest Classifier with enhanced features"""
    try:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Testing set size: {X_test.shape[0]}")
        logger.info(f"Number of features: {X_train.shape[1]}")
        
        # Initialize Random Forest Classifier with optimized parameters
        rf_model = RandomForestClassifier(
            n_estimators=150,          # Increased for better performance with more features
            max_depth=15,              # Increased depth for complex patterns
            min_samples_split=5,       # Minimum samples required to split
            min_samples_leaf=2,        # Minimum samples required at leaf node
            random_state=42,           # For reproducibility
            n_jobs=-1                  # Use all available cores
        )
        
        # Train the model
        logger.info("Training Random Forest Classifier with enhanced features...")
        rf_model.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = rf_model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy with enhanced features: {accuracy:.4f}")
        
        # Generate classification report
        report = classification_report(y_test, y_pred, zero_division=0)
        logger.info(f"Classification Report:\n{report}")
        
        # Feature importance (for all features)
        feature_importance = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(X.shape[1])],
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 10 Feature Importances:\n{feature_importance.head(10)}")
        
        return rf_model, accuracy, X_train, X_test, y_train, y_test
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy: {accuracy:.4f}")
        
        # Generate classification report
        report = classification_report(y_test, y_pred, zero_division=0)
        logger.info(f"Classification Report:\n{report}")
        
        # Feature importance
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Feature Importance:\n{feature_importance}")
        
        return rf_model, accuracy, X_train, X_test, y_train, y_test
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def train_decision_tree_model(X_train, X_test, y_train, y_test):
    """Train Decision Tree Classifier with feature scaling"""
    try:
        logger.info("Training Decision Tree Classifier...")
        
        # Create a pipeline with scaling and Decision Tree
        # Scaling helps with consistent feature interpretation
        dt_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('dt', DecisionTreeClassifier(
                max_depth=15,              # Limit depth to prevent overfitting
                min_samples_split=10,      # Minimum samples required to split
                min_samples_leaf=5,        # Minimum samples required at leaf node
                max_features='sqrt',       # Number of features to consider for best split
                random_state=42            # For reproducibility
            ))
        ])
        
        # Train the model
        dt_pipeline.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = dt_pipeline.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Decision Tree accuracy: {accuracy:.4f}")
        
        # Generate classification report
        report = classification_report(y_test, y_pred, zero_division=0)
        logger.info(f"Decision Tree Classification Report:\n{report}")
        
        # Cross-validation score
        cv_scores = cross_val_score(dt_pipeline, X_train, y_train, cv=5, scoring='accuracy')
        logger.info(f"Decision Tree Cross-validation scores: {cv_scores}")
        logger.info(f"Decision Tree Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Feature importance from the decision tree (for all features)
        n_features = X_train.shape[1]
        feature_names = [f'feature_{i}' for i in range(n_features)]
        dt_feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': dt_pipeline.named_steps['dt'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Decision Tree Top 10 Feature Importances:\n{dt_feature_importance.head(10)}")
        
        return dt_pipeline, accuracy
    
    except Exception as e:
        logger.error(f"Error training Decision Tree model: {str(e)}")
        raise

def compare_models(rf_model, dt_model, X_test, y_test):
    """Compare performance of Random Forest and Decision Tree models"""
    try:
        logger.info("Comparing model performances...")
        
        # Get predictions from both models
        rf_pred = rf_model.predict(X_test)
        dt_pred = dt_model.predict(X_test)
        
        # Calculate accuracies
        rf_accuracy = accuracy_score(y_test, rf_pred)
        dt_accuracy = accuracy_score(y_test, dt_pred)
        
        # Get prediction probabilities for confidence analysis
        rf_proba = rf_model.predict_proba(X_test)
        dt_proba = dt_model.predict_proba(X_test)
        
        # Calculate average confidence (max probability for each prediction)
        rf_confidence = np.mean(np.max(rf_proba, axis=1))
        dt_confidence = np.mean(np.max(dt_proba, axis=1))
        
        # Get feature importance (using generic names for enhanced features)
        n_features = X_test.shape[1]
        feature_names = get_feature_names()
        
        # Random Forest feature importance - limit to top features
        rf_feature_importance = {
            feature: float(importance)
            for feature, importance in zip(feature_names, rf_model.feature_importances_)
        }
        
        # Decision Tree feature importance (from pipeline) - limit to top features
        dt_feature_importance = {
            feature: float(importance)
            for feature, importance in zip(feature_names, dt_model.named_steps['dt'].feature_importances_)
        }
        
        # Create comparison results
        comparison = {
            'random_forest': {
                'accuracy': float(rf_accuracy),
                'avg_confidence': float(rf_confidence),
                'model_type': 'RandomForestClassifier',
                'feature_importance': rf_feature_importance
            },
            'decision_tree': {
                'accuracy': float(dt_accuracy),
                'avg_confidence': float(dt_confidence),
                'model_type': 'DecisionTreeClassifier_Pipeline',
                'feature_importance': dt_feature_importance
            },
            'best_model': 'random_forest' if rf_accuracy > dt_accuracy else 'decision_tree'
        }
        
        # Log comparison results
        logger.info("Model Comparison Results:")
        logger.info(f"Random Forest - Accuracy: {rf_accuracy:.4f}, Avg Confidence: {rf_confidence:.4f}")
        logger.info(f"Decision Tree - Accuracy: {dt_accuracy:.4f}, Avg Confidence: {dt_confidence:.4f}")
        logger.info(f"Best performing model: {comparison['best_model']}")
        
        return comparison
    
    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        raise

def save_model(model, model_path='models/crop_model.pkl'):
    """Save the trained model using pickle"""
    try:
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Model saved successfully at {model_path}")
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def save_feature_extractors(scaler, kpca_model, feature_ranges=None, scaler_path='models/feature_scaler.pkl', kpca_path='models/kpca_model.pkl', ranges_path='models/feature_ranges.pkl'):
    """Save the feature extraction models and ranges"""
    try:
        os.makedirs('models', exist_ok=True)
        
        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Feature scaler saved at {scaler_path}")
        
        # Save KPCA model
        with open(kpca_path, 'wb') as f:
            pickle.dump(kpca_model, f)
        logger.info(f"KPCA model saved at {kpca_path}")

        # Save ranges
        if feature_ranges:
            with open(ranges_path, 'wb') as f:
                pickle.dump(feature_ranges, f)
            logger.info(f"Feature ranges saved at {ranges_path}")
        
    except Exception as e:
        logger.error(f"Error saving feature extractors: {str(e)}")
        raise


def save_comparison_results(comparison_results, results_path='models/model_comparison.json'):
    """Save model comparison results to JSON file"""
    try:
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        # Save the comparison results
        with open(results_path, 'w') as f:
            json.dump(comparison_results, f, indent=4)
        
        logger.info(f"Model comparison results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Error saving comparison results: {str(e)}")
        raise

def train_model():
    """Main function to train and save the crop recommendation models with enhanced features"""
    try:
        # Load the dataset
        df = load_data()
        
        # Preprocess the data (get original features)
        X_original, y, feature_columns = preprocess_data(df)
        
        # Create enhanced features with domain-specific and statistical features + KPCA
        X_enhanced, scaler, kpca_model, feature_ranges = create_enhanced_features(X_original)
        
        # Train the Random Forest model with enhanced features
        rf_model, rf_accuracy, X_train, X_test, y_train, y_test = train_random_forest_model(X_enhanced, y)
        
        # Train the Decision Tree model with enhanced features
        dt_model, dt_accuracy = train_decision_tree_model(X_train, X_test, y_train, y_test)
        
        # Compare both models
        comparison_results = compare_models(rf_model, dt_model, X_test, y_test)
        
        # Save both models
        save_model(rf_model, 'models/random_forest_model.pkl')
        save_model(dt_model, 'models/decision_tree_model.pkl')
        
        # Save feature extractors (scaler, KPCA, and ranges)
        save_feature_extractors(scaler, kpca_model, feature_ranges)
        
        # Save the best performing model as the default
        if comparison_results['best_model'] == 'random_forest':
            save_model(rf_model, 'models/crop_model.pkl')
            best_accuracy = rf_accuracy
        else:
            save_model(dt_model, 'models/crop_model.pkl')
            best_accuracy = dt_accuracy
        
        # Update comparison results with feature information
        comparison_results['feature_engineering'] = {
            'original_features': X_original.shape[1],
            'enhanced_features': X_enhanced.shape[1],
            'domain_features': 5,  # SFI, CSI, NPK_ratio, nutrient_balance, climate_stress
            'statistical_features': 8,
            'kpca_components': 10,
            'feature_extraction': 'SFI, CSI, Statistical Features, KPCA'
        }
        
        # Save comparison results
        save_comparison_results(comparison_results)
        
        logger.info("Model training completed successfully with enhanced features!")
        logger.info(f"Best model: {comparison_results['best_model']} with accuracy: {best_accuracy:.4f}")
        
        return {
            'random_forest_accuracy': rf_accuracy,
            'decision_tree_accuracy': dt_accuracy,
            'best_model': comparison_results['best_model'],
            'best_accuracy': best_accuracy,
            'comparison_results': comparison_results,
            'original_features': X_original.shape[1],
            'enhanced_features': X_enhanced.shape[1]
        }
        
    except Exception as e:
        logger.error(f"Error in model training pipeline: {str(e)}")
        raise

def train_all_models_comprehensive():
    """Train all models including traditional ML and deep learning models"""
    try:
        logger.info("Starting comprehensive model training...")
        
        # Load the dataset
        df = load_data()
        
        # Preprocess the data (get original features)
        X_original, y, feature_columns = preprocess_data(df)
        
        # Create enhanced features with domain-specific and statistical features + KPCA
        X_enhanced, scaler, kpca_model, feature_ranges = create_enhanced_features(X_original)
        
        # Train traditional ML models
        rf_model, rf_accuracy, X_train, X_test, y_train, y_test = train_random_forest_model(X_enhanced, y)
        dt_model, dt_accuracy = train_decision_tree_model(X_train, X_test, y_train, y_test)
        
        # Train deep learning models
        logger.info("Training deep learning models...")
        dl_results = train_deep_learning_models(X_enhanced, y)
        
        # Combine all results
        all_model_results = {
            'random_forest': {
                'accuracy': float(rf_accuracy),
                'model_type': 'RandomForestClassifier'
            },
            'decision_tree': {
                'accuracy': float(dt_accuracy),
                'model_type': 'DecisionTreeClassifier'
            }
        }
        
        # Add deep learning results
        for model_name, results in dl_results.items():
            if 'accuracy' in results:
                all_model_results[model_name] = {
                    'accuracy': float(results['accuracy']),
                    'model_type': 'DeepLearning'
                }
        
        # Save all models
        save_model(rf_model, 'models/random_forest_model.pkl')
        save_model(dt_model, 'models/decision_tree_model.pkl')
        save_feature_extractors(scaler, kpca_model, feature_ranges)
        
        # Find best model
        best_model_name = max(all_model_results.keys(), 
                            key=lambda x: all_model_results[x]['accuracy'])
        best_accuracy = all_model_results[best_model_name]['accuracy']
        
        # Save comprehensive results
        comprehensive_results = {
            'all_models': all_model_results,
            'best_model': best_model_name,
            'best_accuracy': best_accuracy,
            'feature_engineering': {
                'original_features': X_original.shape[1],
                'enhanced_features': X_enhanced.shape[1],
                'domain_features': 5,
                'statistical_features': 8,
                'kpca_components': 10,
                'feature_extraction': 'SFI, CSI, Statistical Features, KPCA'
            },
            'deep_learning_results': dl_results
        }
        
        save_comparison_results(comprehensive_results)
        
        logger.info("Comprehensive model training completed!")
        logger.info(f"Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")
        
        return comprehensive_results
        
    except Exception as e:
        logger.error(f"Error in comprehensive model training: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the training pipeline
    results = train_model()
    print(f"\n{'='*60}")
    print(f"Model Training Completed with Enhanced Features!")
    print(f"{'='*60}")
    print(f"Original Features: {results['original_features']}")
    print(f"Enhanced Features: {results['enhanced_features']}")
    print(f"\nModel Performance:")
    print(f"Random Forest Accuracy: {results['random_forest_accuracy']:.4f}")
    print(f"Decision Tree Accuracy: {results['decision_tree_accuracy']:.4f}")
    print(f"\nBest Model: {results['best_model']}")
    print(f"Best Accuracy: {results['best_accuracy']:.4f}")
    print(f"\nFeature Engineering:")
    print(f"- Domain-specific features: SFI (Soil Fertility Index), CSI (Climate Suitability Index)")
    print(f"- Statistical features: Mean, Std, Skewness, Kurtosis, etc.")
    print(f"- Correlation-based features: KPCA with RBF kernel")
    print(f"{'='*60}")
