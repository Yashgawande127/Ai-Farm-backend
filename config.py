import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class"""
    
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'crop-recommendation-secret-key-2024'
    DEBUG = False
    TESTING = False
    
    # JWT configuration
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-secret-key-change-in-production-2024'
    JWT_ACCESS_TOKEN_EXPIRES = os.environ.get('JWT_ACCESS_TOKEN_EXPIRES', 24 * 60 * 60)  # 24 hours in seconds
    JWT_REFRESH_TOKEN_EXPIRES = os.environ.get('JWT_REFRESH_TOKEN_EXPIRES', 30 * 24 * 60 * 60)  # 30 days in seconds
    
    # MongoDB configuration
    MONGODB_URI = os.environ.get('MONGODB_URI')
    MONGODB_DB_NAME = os.environ.get('MONGODB_DB_NAME', 'ai_farm')
    
    # Model configuration
    MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join('models', 'crop_model.pkl'))
    DATASET_PATH = os.environ.get('DATASET_PATH', '../Crop_recommendation.csv')
    
    # API configuration
    API_VERSION = 'v1'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # CORS configuration
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000').split(',')
    
    # Logging configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Model training configuration
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    N_ESTIMATORS = 100
    MAX_DEPTH = 10
    MIN_SAMPLES_SPLIT = 5
    MIN_SAMPLES_LEAF = 2
    
    @staticmethod
    def init_app(app):
        """Initialize application with config"""
        pass

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = os.environ.get('DEBUG', 'True').lower() in ['true', '1', 'yes']
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'DEBUG')

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    
    # Override with environment variables for production
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Log to stderr in production
        import logging
        from logging import StreamHandler
        file_handler = StreamHandler()
        file_handler.setLevel(logging.WARNING)
        app.logger.addHandler(file_handler)

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    MODEL_PATH = os.path.join('tests', 'test_model.pkl')
    DATASET_PATH = os.path.join('tests', 'test_data.csv')

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(config_name: str = None) -> Config:
    """Get configuration based on environment"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    return config.get(config_name, config['default'])

# Feature information for validation and documentation
FEATURE_INFO = {
    'N': {
        'name': 'Nitrogen',
        'unit': 'ratio',
        'description': 'Ratio of Nitrogen content in soil',
        'min_value': 0,
        'max_value': 200,
        'typical_range': [0, 140]
    },
    'P': {
        'name': 'Phosphorus',
        'unit': 'ratio',
        'description': 'Ratio of Phosphorous content in soil',
        'min_value': 0,
        'max_value': 150,
        'typical_range': [5, 145]
    },
    'K': {
        'name': 'Potassium',
        'unit': 'ratio',
        'description': 'Ratio of Potassium content in soil',
        'min_value': 0,
        'max_value': 200,
        'typical_range': [5, 205]
    },
    'temperature': {
        'name': 'Temperature',
        'unit': '°C',
        'description': 'Temperature in Celsius',
        'min_value': -10,
        'max_value': 50,
        'typical_range': [8.83, 43.68]
    },
    'humidity': {
        'name': 'Humidity',
        'unit': '%',
        'description': 'Relative humidity in percentage',
        'min_value': 0,
        'max_value': 100,
        'typical_range': [14.26, 99.98]
    },
    'ph': {
        'name': 'pH',
        'unit': 'pH',
        'description': 'pH value of the soil',
        'min_value': 0,
        'max_value': 14,
        'typical_range': [3.5, 9.94]
    },
    'rainfall': {
        'name': 'Rainfall',
        'unit': 'mm',
        'description': 'Rainfall in mm',
        'min_value': 0,
        'max_value': 500,
        'typical_range': [20.21, 298.56]
    }
}

# Model performance thresholds
MODEL_THRESHOLDS = {
    'min_accuracy': 0.80,  # Minimum acceptable model accuracy
    'min_confidence': 0.50,  # Minimum confidence for predictions
    'max_training_time': 300,  # Maximum training time in seconds
}

# API response messages
API_MESSAGES = {
    'success': 'Operation completed successfully',
    'model_not_loaded': 'Model not loaded. Please train the model first.',
    'invalid_input': 'Invalid input data provided',
    'missing_features': 'Missing required features',
    'training_success': 'Model trained successfully',
    'prediction_success': 'Crop prediction completed',
    'server_error': 'Internal server error occurred'
}