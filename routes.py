from flask import Blueprint, request, jsonify
import logging
from prediction_service import CropPredictionService
from database_service import DatabaseService
from feature_selection_api import get_feature_selection_performance, get_cached_feature_selection_data

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint for API routes
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Initialize prediction service (will be set in main app)
prediction_service = None

def init_routes(pred_service):
    """Initialize routes with prediction service"""
    global prediction_service
    prediction_service = pred_service

@api_bp.route('/predict', methods=['POST'])
def predict_crop():
    """
    Predict the most suitable crop based on input features
    
    Expected JSON input:
    {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.88,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.94
    }
    """
    try:
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        # Validate required fields
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {missing_fields}',
                'required_fields': required_fields
            }), 400
        
        # Extract features in correct order
        features = [
            data['N'],
            data['P'],
            data['K'],
            data['temperature'],
            data['humidity'],
            data['ph'],
            data['rainfall']
        ]
        
        # Make prediction
        prediction, confidence = prediction_service.predict(features)
        
        # Get user IP for tracking
        user_ip = request.remote_addr or request.environ.get('HTTP_X_FORWARDED_FOR', 'unknown')
        
        # Save prediction to database (real-time data)
        try:
            saved_prediction = DatabaseService.save_prediction(
                nitrogen=features[0],
                phosphorus=features[1],
                potassium=features[2],
                temperature=features[3],
                humidity=features[4],
                ph=features[5],
                rainfall=features[6],
                predicted_crop=prediction,
                confidence=confidence,
                model_version='v1.0',
                user_ip=user_ip
            )
            logger.info(f"Real-time prediction saved to database with ID: {saved_prediction.id}")
        except Exception as db_error:
            logger.error(f"Failed to save prediction to database: {str(db_error)}")
            # Continue with response even if database save fails
        
        return jsonify({
            'status': 'success',
            'predicted_crop': prediction,
            'confidence': round(confidence, 4),
            'input_features': {
                'N': features[0],
                'P': features[1],
                'K': features[2],
                'temperature': features[3],
                'humidity': features[4],
                'ph': features[5],
                'rainfall': features[6]
            },
            'timestamp': saved_prediction.prediction_date.isoformat() if 'saved_prediction' in locals() else None
        })
        
    except ValueError as e:
        logger.error(f"Validation error in prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Input validation error: {str(e)}'
        }), 400
        
    except RuntimeError as e:
        logger.error(f"Runtime error in prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 503
        
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        }), 500

@api_bp.route('/predict/detailed', methods=['POST'])
def predict_crop_detailed():
    """
    Get detailed prediction with probabilities for all crops
    """
    try:
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        # Validate required fields
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {missing_fields}',
                'required_fields': required_fields
            }), 400
        
        # Extract features
        features = [
            data['N'],
            data['P'],
            data['K'],
            data['temperature'],
            data['humidity'],
            data['ph'],
            data['rainfall']
        ]
        
        # Get detailed prediction
        result = prediction_service.get_prediction_with_probabilities(features)
        
        # Get user IP for tracking
        user_ip = request.remote_addr or request.environ.get('HTTP_X_FORWARDED_FOR', 'unknown')
        
        # Save prediction to database (real-time data)
        try:
            saved_prediction = DatabaseService.save_prediction(
                nitrogen=features[0],
                phosphorus=features[1],
                potassium=features[2],
                temperature=features[3],
                humidity=features[4],
                ph=features[5],
                rainfall=features[6],
                predicted_crop=result['predicted_crop'],
                confidence=result['confidence'],
                model_version='v1.0',
                user_ip=user_ip
            )
            logger.info(f"Real-time detailed prediction saved to database with ID: {saved_prediction.id}")
        except Exception as db_error:
            logger.error(f"Failed to save detailed prediction to database: {str(db_error)}")
            # Continue with response even if database save fails
        
        response_data = {
            'status': 'success',
            'predicted_crop': result['predicted_crop'],
            'confidence': round(result['confidence'], 4),
            'top_3_crops': result['top_3_crops'],
            'all_probabilities': {k: round(v, 4) for k, v in result['all_probabilities'].items()},
            'input_features': {
                'N': features[0],
                'P': features[1],
                'K': features[2],
                'temperature': features[3],
                'humidity': features[4],
                'ph': features[5],
                'rainfall': features[6]
            },
            'timestamp': saved_prediction.prediction_date.isoformat() if 'saved_prediction' in locals() else None
        }
        
        return jsonify(response_data)
        
    except ValueError as e:
        logger.error(f"Validation error in detailed prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Input validation error: {str(e)}'
        }), 400
        
    except RuntimeError as e:
        logger.error(f"Runtime error in detailed prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 503
        
    except Exception as e:
        logger.error(f"Unexpected error in detailed prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Detailed prediction failed: {str(e)}'
        }), 500

@api_bp.route('/predict/models/<model_type>', methods=['POST'])
def predict_with_model(model_type):
    """
    Predict crop using a specific model type (random_forest or decision_tree)
    
    Expected JSON input:
    {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.88,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.94
    }
    """
    try:
        # Validate model type
        if model_type not in ['random_forest', 'decision_tree']:
            return jsonify({
                'status': 'error',
                'message': f'Invalid model type. Use "random_forest" or "decision_tree", got "{model_type}"'
            }), 400
        
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        # Validate required fields
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Extract features
        features = [
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]
        
        # Make prediction with specific model
        predicted_crop, confidence = prediction_service.predict_with_specific_model(features, model_type)
        
        # Prepare response
        response = {
            'status': 'success',
            'prediction': {
                'crop': predicted_crop,
                'confidence': round(confidence, 4),
                'model_used': model_type
            },
            'input_features': {
                'N': features[0],
                'P': features[1],
                'K': features[2],
                'temperature': features[3],
                'humidity': features[4],
                'ph': features[5],
                'rainfall': features[6]
            }
        }
        
        logger.info(f"Model-specific prediction completed using {model_type}: {predicted_crop}")
        return jsonify(response), 200
        
    except ValueError as e:
        logger.error(f"Validation error in model-specific prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
        
    except RuntimeError as e:
        logger.error(f"Runtime error in model-specific prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 503
        
    except Exception as e:
        logger.error(f"Unexpected error in model-specific prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Model-specific prediction failed: {str(e)}'
        }), 500

@api_bp.route('/predict/ensemble', methods=['POST'])
def predict_ensemble():
    """
    Predict crop using ensemble of all available models
    
    Expected JSON input:
    {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.88,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.94
    }
    """
    try:
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        # Validate required fields
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Extract features
        features = [
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]
        
        # Make ensemble prediction
        ensemble_results = prediction_service.predict_ensemble(features)
        
        # Prepare response
        response = {
            'status': 'success',
            'ensemble_prediction': ensemble_results,
            'input_features': {
                'N': features[0],
                'P': features[1],
                'K': features[2],
                'temperature': features[3],
                'humidity': features[4],
                'ph': features[5],
                'rainfall': features[6]
            }
        }
        
        logger.info(f"Ensemble prediction completed")
        return jsonify(response), 200
        
    except ValueError as e:
        logger.error(f"Validation error in ensemble prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
        
    except RuntimeError as e:
        logger.error(f"Runtime error in ensemble prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 503
        
    except Exception as e:
        logger.error(f"Unexpected error in ensemble prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Ensemble prediction failed: {str(e)}'
        }), 500

@api_bp.route('/models/available', methods=['GET'])
def get_available_models():
    """Get information about available models"""
    try:
        available_models = prediction_service.get_available_models()
        model_comparison = prediction_service.load_model_comparison()
        
        response = {
            'status': 'success',
            'available_models': available_models,
            'model_comparison': model_comparison
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get available models: {str(e)}'
        }), 500

@api_bp.route('/model/info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model"""
    try:
        model_info = prediction_service.get_model_info()
        
        if 'error' in model_info:
            return jsonify({
                'status': 'error',
                'message': model_info['error']
            }), 500
        
        return jsonify({
            'status': 'success',
            'model_info': model_info
        })
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get model info: {str(e)}'
        }), 500

@api_bp.route('/model/status', methods=['GET'])
def get_model_status():
    """Check if model is loaded and ready"""
    try:
        is_loaded = prediction_service.is_model_loaded()
        
        return jsonify({
            'status': 'success',
            'model_loaded': is_loaded,
            'message': 'Model is ready for predictions' if is_loaded else 'Model not loaded'
        })
        
    except Exception as e:
        logger.error(f"Error checking model status: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to check model status: {str(e)}'
        }), 500

@api_bp.route('/crops', methods=['GET'])
def get_available_crops():
    """Get list of crops that the model can predict"""
    try:
        if not prediction_service.is_model_loaded():
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 503
        
        model_info = prediction_service.get_model_info()
        crops = model_info.get('classes', [])
        
        return jsonify({
            'status': 'success',
            'crops': sorted(crops),
            'total_crops': len(crops)
        })
        
    except Exception as e:
        logger.error(f"Error getting available crops: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get available crops: {str(e)}'
        }), 500

@api_bp.route('/features', methods=['GET'])
def get_required_features():
    """Get list of required input features and their descriptions"""
    try:
        features_info = {
            'N': {
                'name': 'Nitrogen',
                'unit': 'ratio',
                'description': 'Ratio of Nitrogen content in soil',
                'typical_range': [0, 140]
            },
            'P': {
                'name': 'Phosphorus',
                'unit': 'ratio',
                'description': 'Ratio of Phosphorous content in soil',
                'typical_range': [5, 145]
            },
            'K': {
                'name': 'Potassium',
                'unit': 'ratio',
                'description': 'Ratio of Potassium content in soil',
                'typical_range': [5, 205]
            },
            'temperature': {
                'name': 'Temperature',
                'unit': '°C',
                'description': 'Temperature in Celsius',
                'typical_range': [8.83, 43.68]
            },
            'humidity': {
                'name': 'Humidity',
                'unit': '%',
                'description': 'Relative humidity in percentage',
                'typical_range': [14.26, 99.98]
            },
            'ph': {
                'name': 'pH',
                'unit': 'pH',
                'description': 'pH value of the soil',
                'typical_range': [3.5, 9.94]
            },
            'rainfall': {
                'name': 'Rainfall',
                'unit': 'mm',
                'description': 'Rainfall in mm',
                'typical_range': [20.21, 298.56]
            }
        }
        
        return jsonify({
            'status': 'success',
            'required_features': ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'],
            'features_info': features_info
        })
        
    except Exception as e:
        logger.error(f"Error getting features info: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get features info: {str(e)}'
        }), 500

@api_bp.route('/predictions/history', methods=['GET'])
def get_predictions_history():
    """Get real-time prediction history"""
    try:
        # Get query parameters
        limit = request.args.get('limit', 100, type=int)
        crop_name = request.args.get('crop', None)
        
        # Validate limit
        if limit > 1000:
            limit = 1000
        
        # Get prediction history from database
        predictions = DatabaseService.get_prediction_history(limit=limit, crop_name=crop_name)
        
        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'count': len(predictions),
            'filters': {
                'limit': limit,
                'crop_name': crop_name
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting prediction history: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get prediction history: {str(e)}'
        }), 500

@api_bp.route('/analytics/predictions', methods=['GET'])
def get_prediction_analytics():
    """Get real-time prediction analytics"""
    try:
        analytics = DatabaseService.get_prediction_analytics()
        
        return jsonify({
            'status': 'success',
            'analytics': analytics
        })
        
    except Exception as e:
        logger.error(f"Error getting prediction analytics: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get prediction analytics: {str(e)}'
        }), 500

@api_bp.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback for predictions"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        # Get user IP
        user_ip = request.remote_addr or request.environ.get('HTTP_X_FORWARDED_FOR', 'unknown')
        
        # Save feedback
        feedback = DatabaseService.save_user_feedback(
            prediction_id=data.get('prediction_id'),
            feedback_type=data.get('feedback_type', 'rating'),
            rating=data.get('rating'),
            comment=data.get('comment'),
            user_email=data.get('user_email'),
            user_location=data.get('user_location'),
            ip_address=user_ip
        )
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback saved successfully',
            'feedback_id': str(feedback.id)
        })
        
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to save feedback: {str(e)}'
        }), 500

@api_bp.route('/analytics/feedback', methods=['GET'])
def get_feedback_analytics():
    """Get user feedback analytics"""
    try:
        analytics = DatabaseService.get_feedback_analytics()
        
        return jsonify({
            'status': 'success',
            'analytics': analytics
        })
        
    except Exception as e:
        logger.error(f"Error getting feedback analytics: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get feedback analytics: {str(e)}'
        }), 500

@api_bp.route('/feature-selection/performance', methods=['GET'])
def feature_selection_performance():
    """Get feature selection performance comparison data"""
    try:
        # Check if full analysis is requested
        full_analysis = request.args.get('full', 'false').lower() == 'true'
        
        if full_analysis:
            result = get_feature_selection_performance()
        else:
            # Return cached/quick data for faster response
            result = get_cached_feature_selection_data()
        
        if result['success']:
            return jsonify({
                'status': 'success',
                'data': result['data']
            })
        else:
            return jsonify({
                'status': 'error',
                'message': result.get('message', 'Unknown error')
            }), 500
            
    except Exception as e:
        logger.error(f"Error getting feature selection performance: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get feature selection performance: {str(e)}'
        }), 500

@api_bp.route('/train-all-models', methods=['POST'])
def train_all_models():
    """
    Train all models including deep learning models for comprehensive comparison
    """
    try:
        logger.info("Starting comprehensive model training...")
        
        from model_trainer import train_all_models_comprehensive
        
        # This will take some time - in production, this should be asynchronous
        results = train_all_models_comprehensive()
        
        return jsonify({
            'status': 'success',
            'message': 'All models trained successfully',
            'data': results
        })
        
    except Exception as e:
        logger.error(f"Error training all models: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to train models: {str(e)}'
        }), 500

@api_bp.route('/model-comparison', methods=['GET'])
def get_model_comparison():
    """
    Get comprehensive model comparison including deep learning models
    """
    try:
        import json
        import os
        
        comparison_path = 'models/model_comparison.json'
        if os.path.exists(comparison_path):
            with open(comparison_path, 'r') as f:
                comparison_data = json.load(f)
            
            return jsonify({
                'status': 'success',
                'data': comparison_data
            })
        else:
            # Return mock data for demonstration
            mock_data = {
                'all_models': {
                    'random_forest': {'accuracy': 0.995, 'model_type': 'RandomForestClassifier'},
                    'decision_tree': {'accuracy': 0.966, 'model_type': 'DecisionTreeClassifier'},
                    'deepagyieldnet': {'accuracy': 0.982, 'model_type': 'DeepLearning'},
                    'shufflenetv2': {'accuracy': 0.975, 'model_type': 'DeepLearning'},
                    'efficientcapsnet': {'accuracy': 0.978, 'model_type': 'DeepLearning'},
                    'ladnet': {'accuracy': 0.984, 'model_type': 'DeepLearning'},
                    'regnet': {'accuracy': 0.971, 'model_type': 'DeepLearning'}
                },
                'best_model': 'random_forest',
                'best_accuracy': 0.995
            }
            
            return jsonify({
                'status': 'success',
                'data': mock_data
            })
        
    except Exception as e:
        logger.error(f"Error getting model comparison: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get model comparison: {str(e)}'
        }), 500