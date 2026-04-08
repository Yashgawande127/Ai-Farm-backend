from mongoengine import Document, StringField, FloatField, DateTimeField, IntField, BooleanField, ReferenceField, DictField, EmailField
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import json

class CropPrediction(Document):
    """Model to store crop prediction requests and results"""
    
    meta = {'collection': 'crop_predictions'}
    
    # Input parameters
    nitrogen = FloatField(required=True)
    phosphorus = FloatField(required=True)
    potassium = FloatField(required=True)
    temperature = FloatField(required=True)
    humidity = FloatField(required=True)
    ph = FloatField(required=True)
    rainfall = FloatField(required=True)
    
    # Prediction results
    predicted_crop = StringField(required=True, max_length=100)
    prediction_confidence = FloatField(default=0.0)
    
    # Additional metadata
    model_version = StringField(default='v1.0', max_length=50)
    prediction_date = DateTimeField(default=datetime.utcnow)
    user_ip = StringField(max_length=45)  # Support IPv6
    
    # Optional user feedback
    user_feedback = StringField()
    feedback_rating = IntField(min_value=1, max_value=5)  # 1-5 rating
    
    def __repr__(self):
        return f'<CropPrediction {str(self.id)}: {self.predicted_crop}>'
    
    def to_dict(self):
        """Convert model to dictionary for JSON serialization"""
        return {
            'id': str(self.id),
            'nitrogen': self.nitrogen,
            'phosphorus': self.phosphorus,
            'potassium': self.potassium,
            'temperature': self.temperature,
            'humidity': self.humidity,
            'ph': self.ph,
            'rainfall': self.rainfall,
            'predicted_crop': self.predicted_crop,
            'prediction_confidence': self.prediction_confidence,
            'model_version': self.model_version,
            'prediction_date': self.prediction_date.isoformat() if self.prediction_date else None,
            'user_ip': self.user_ip,
            'user_feedback': self.user_feedback,
            'feedback_rating': self.feedback_rating
        }

class ModelMetadata(Document):
    """Model to store ML model metadata and performance metrics"""
    
    meta = {'collection': 'model_metadata'}
    
    # Model information
    model_name = StringField(required=True, max_length=100)
    model_version = StringField(required=True, max_length=50)
    model_path = StringField(required=True, max_length=255)
    
    # Performance metrics
    accuracy = FloatField()
    precision = FloatField()
    recall = FloatField()
    f1_score = FloatField()
    
    # Training information
    training_data_size = IntField()
    training_date = DateTimeField(default=datetime.utcnow)
    hyperparameters = DictField()  # Dictionary of hyperparameters
    
    # Model status
    is_active = BooleanField(default=True)
    created_date = DateTimeField(default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ModelMetadata {self.model_name} v{self.model_version}>'
    
    def to_dict(self):
        """Convert model to dictionary for JSON serialization"""
        return {
            'id': str(self.id),
            'model_name': self.model_name,
            'model_version': self.model_version,
            'model_path': self.model_path,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'training_data_size': self.training_data_size,
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'hyperparameters': self.hyperparameters,
            'is_active': self.is_active,
            'created_date': self.created_date.isoformat() if self.created_date else None
        }

class CropData(Document):
    """Model to store crop reference data and characteristics"""
    
    meta = {'collection': 'crop_data'}
    
    # Crop information
    crop_name = StringField(required=True, max_length=100, unique=True)
    crop_type = StringField(max_length=50)  # e.g., cereal, legume, vegetable
    
    # Optimal growing conditions
    optimal_nitrogen_min = FloatField()
    optimal_nitrogen_max = FloatField()
    optimal_phosphorus_min = FloatField()
    optimal_phosphorus_max = FloatField()
    optimal_potassium_min = FloatField()
    optimal_potassium_max = FloatField()
    optimal_temperature_min = FloatField()
    optimal_temperature_max = FloatField()
    optimal_humidity_min = FloatField()
    optimal_humidity_max = FloatField()
    optimal_ph_min = FloatField()
    optimal_ph_max = FloatField()
    optimal_rainfall_min = FloatField()
    optimal_rainfall_max = FloatField()
    
    # Additional crop information
    growing_season = StringField(max_length=100)
    harvest_time = StringField(max_length=100)
    description = StringField()
    
    # Metadata
    created_date = DateTimeField(default=datetime.utcnow)
    updated_date = DateTimeField(default=datetime.utcnow)
    
    def __repr__(self):
        return f'<CropData {self.crop_name}>'
    
    def to_dict(self):
        """Convert model to dictionary for JSON serialization"""
        return {
            'id': str(self.id),
            'crop_name': self.crop_name,
            'crop_type': self.crop_type,
            'optimal_conditions': {
                'nitrogen': {'min': self.optimal_nitrogen_min, 'max': self.optimal_nitrogen_max},
                'phosphorus': {'min': self.optimal_phosphorus_min, 'max': self.optimal_phosphorus_max},
                'potassium': {'min': self.optimal_potassium_min, 'max': self.optimal_potassium_max},
                'temperature': {'min': self.optimal_temperature_min, 'max': self.optimal_temperature_max},
                'humidity': {'min': self.optimal_humidity_min, 'max': self.optimal_humidity_max},
                'ph': {'min': self.optimal_ph_min, 'max': self.optimal_ph_max},
                'rainfall': {'min': self.optimal_rainfall_min, 'max': self.optimal_rainfall_max}
            },
            'growing_season': self.growing_season,
            'harvest_time': self.harvest_time,
            'description': self.description,
            'created_date': self.created_date.isoformat() if self.created_date else None,
            'updated_date': self.updated_date.isoformat() if self.updated_date else None
        }

class UserFeedback(Document):
    """Model to store user feedback about predictions"""
    
    meta = {'collection': 'user_feedback'}
    
    # Link to prediction
    prediction = ReferenceField(CropPrediction)
    
    # Feedback details
    feedback_type = StringField(required=True, max_length=50)  # 'rating', 'comment', 'bug_report'
    rating = IntField(min_value=1, max_value=5)  # 1-5 stars
    comment = StringField()
    
    # User information (optional, for anonymous feedback)
    user_email = StringField(max_length=255)
    user_location = StringField(max_length=100)
    
    # Metadata
    created_date = DateTimeField(default=datetime.utcnow)
    ip_address = StringField(max_length=45)
    
    def __repr__(self):
        return f'<UserFeedback {self.id}: {self.feedback_type}>'
    
    def to_dict(self):
        """Convert model to dictionary for JSON serialization"""
        return {
            'id': str(self.id),
            'prediction_id': str(self.prediction.id) if self.prediction else None,
            'feedback_type': self.feedback_type,
            'rating': self.rating,
            'comment': self.comment,
            'user_email': self.user_email,
            'user_location': self.user_location,
            'created_date': self.created_date.isoformat() if self.created_date else None,
            'ip_address': self.ip_address
        }


class User(Document):
    """Model to store user accounts for authentication"""
    
    meta = {'collection': 'users'}
    
    # Basic user information
    first_name = StringField(required=True, max_length=50)
    last_name = StringField(required=True, max_length=50)
    email = EmailField(required=True, unique=True)
    password_hash = StringField(required=True)
    
    # Account status
    is_active = BooleanField(default=True)
    is_verified = BooleanField(default=False)
    
    # Timestamps
    created_date = DateTimeField(default=datetime.utcnow)
    last_login = DateTimeField()
    
    # Optional profile information
    phone = StringField(max_length=20)
    location = StringField(max_length=100)
    
    # User preferences and settings
    preferences = DictField()
    
    def __repr__(self):
        return f'<User {self.email}>'
    
    def set_password(self, password):
        """Hash and set user password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if provided password matches user's password"""
        return check_password_hash(self.password_hash, password)
    
    @property
    def full_name(self):
        """Get user's full name"""
        return f"{self.first_name} {self.last_name}"
    
    def update_last_login(self):
        """Update the last login timestamp"""
        self.last_login = datetime.utcnow()
        self.save()
    
    def to_dict(self, include_sensitive=False):
        """Convert user model to dictionary for JSON serialization"""
        user_data = {
            'id': str(self.id),
            'first_name': self.first_name,
            'last_name': self.last_name,
            'full_name': self.full_name,
            'email': self.email,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'created_date': self.created_date.isoformat() if self.created_date else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'phone': self.phone,
            'location': self.location,
            'preferences': self.preferences or {}
        }
        
        if include_sensitive:
            user_data['password_hash'] = self.password_hash
            
        return user_data
    
    @classmethod
    def get_by_email(cls, email):
        """Get user by email address"""
        try:
            return cls.objects(email=email).first()
        except Exception:
            return None
    
    @classmethod
    def create_user(cls, first_name, last_name, email, password, **kwargs):
        """Create a new user with hashed password"""
        # Check if user already exists
        existing_user = cls.get_by_email(email)
        if existing_user:
            raise ValueError("User with this email already exists")
        
        # Create new user
        user = cls(
            first_name=first_name,
            last_name=last_name,
            email=email,
            **kwargs
        )
        user.set_password(password)
        user.save()
        return user