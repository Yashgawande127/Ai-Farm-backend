from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
import os
import logging
import mongoengine
from model_trainer import train_model
from prediction_service import CropPredictionService
from routes import api_bp, init_routes
from auth_routes import auth_bp
from config import get_config
from utils import setup_logging, health_check
from database_service import DatabaseService

def create_app(config_name=None):
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Load configuration
    config_obj = get_config(config_name)
    app.config.from_object(config_obj)
    
    # Initialize logging
    setup_logging(app.config.get('LOG_LEVEL', 'INFO'))
    logger = logging.getLogger(__name__)
    
    # Initialize MongoDB connection
    try:
        mongoengine.connect(
            host=app.config.get('MONGODB_URI'),
            alias='default'
        )
        logger.info("Connected to MongoDB successfully")
        
        # Initialize database
        DatabaseService.init_db(app)
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise e
    
    # Enable CORS
    CORS(app, origins=app.config.get('CORS_ORIGINS', ['*']))
    
    # Initialize JWT Manager
    jwt = JWTManager(app)
    
    # Set JWT configuration
    app.config['JWT_SECRET_KEY'] = app.config.get('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = False  # Tokens don't expire by default (we handle it manually)
    
    # Initialize prediction service
    prediction_service = CropPredictionService(app.config.get('MODEL_PATH'))
    
    # Initialize routes with prediction service
    init_routes(prediction_service)
    
    # Register blueprints
    app.register_blueprint(api_bp)
    app.register_blueprint(auth_bp)
    
    # Ensure model directory exists
    os.makedirs('models', exist_ok=True)
    
    # Check if models exist, if not train them
    models_exist = (
        os.path.exists('models/crop_model.pkl') and
        os.path.exists('models/random_forest_model.pkl') and
        os.path.exists('models/decision_tree_model.pkl')
    )
    
    if not models_exist:
        logger.info("Some models are missing. Training new models...")
        try:
            results = train_model()
            logger.info(f"Initial models trained correctly")
        except Exception as e:
            logger.error(f"Failed to train initial models: {str(e)}")
    
    # Load all models into prediction service
    try:
        prediction_service.load_all_models()
        logger.info("All models loaded successfully into prediction service")
    except Exception as e:
        logger.warning(f"Could not load all models: {str(e)}")
        # Fallback to loading just the default model
        try:
            prediction_service.load_model()
        except Exception as fallback_error:
            logger.error(f"Could not load any model: {fallback_error}")

    return app, prediction_service, logger

# Create app instance
app, prediction_service, logger = create_app()

@app.route('/', methods=['GET'])
def health_check_endpoint():
    """Application health check endpoint"""
    try:
        health_data = health_check()
        return jsonify(health_data)
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/train', methods=['POST'])
def train_model_endpoint():
    """Endpoint to train both Random Forest and Decision Tree models"""
    try:
        logger.info("Starting model training...")
        results = train_model()
        logger.info(f"Models trained successfully:")
        logger.info(f"Random Forest accuracy: {results['random_forest_accuracy']:.4f}")
        logger.info(f"Decision Tree accuracy: {results['decision_tree_accuracy']:.4f}")
        logger.info(f"Best model: {results['best_model']}")
        
        # Reload all models in the prediction service
        prediction_service.load_all_models()
        
        return jsonify({
            'status': 'success',
            'message': 'Models trained successfully',
            'results': {
                'random_forest_accuracy': round(results['random_forest_accuracy'], 4),
                'decision_tree_accuracy': round(results['decision_tree_accuracy'], 4),
                'best_model': results['best_model'],
                'best_accuracy': round(results['best_accuracy'], 4)
            },
            'model_paths': {
                'random_forest': 'models/random_forest_model.pkl',
                'decision_tree': 'models/decision_tree_model.pkl',
                'default': app.config.get('MODEL_PATH')
            }
        })
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Training failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)