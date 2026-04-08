from models import CropPrediction, ModelMetadata, CropData, UserFeedback
from datetime import datetime
import json
import pandas as pd
import logging
import mongoengine
from mongoengine.errors import NotUniqueError

logger = logging.getLogger(__name__)

class DatabaseService:
    """Service class to handle database operations"""
    
    @staticmethod
    def init_db(app):
        """Initialize database with application context"""
        try:
            # MongoDB connection is handled by mongoengine.connect() in app.py
            logger.info("MongoDB connection initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MongoDB: {str(e)}")
            raise e
    
    @staticmethod
    def save_prediction(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, 
                       predicted_crop, confidence=0.0, model_version='v1.0', user_ip=None):
        """Save a crop prediction to database"""
        try:
            prediction = CropPrediction(
                nitrogen=nitrogen,
                phosphorus=phosphorus,
                potassium=potassium,
                temperature=temperature,
                humidity=humidity,
                ph=ph,
                rainfall=rainfall,
                predicted_crop=predicted_crop,
                prediction_confidence=confidence,
                model_version=model_version,
                user_ip=user_ip
            )
            
            prediction.save()
            
            logger.info(f"Prediction saved with ID: {prediction.id}")
            return prediction
            
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
            raise e
    
    @staticmethod
    def get_prediction_history(limit=100, crop_name=None):
        """Get prediction history with optional filtering"""
        try:
            query = CropPrediction.objects()
            
            if crop_name:
                query = query.filter(predicted_crop=crop_name)
            
            predictions = query.order_by('-prediction_date').limit(limit)
            
            return [prediction.to_dict() for prediction in predictions]
            
        except Exception as e:
            logger.error(f"Error fetching prediction history: {str(e)}")
            raise e
    
    @staticmethod
    def get_prediction_analytics():
        """Get analytics data for predictions"""
        try:
            # Total predictions
            total_predictions = CropPrediction.objects.count()
            
            # Most predicted crops - using aggregation
            crop_counts_pipeline = [
                {"$group": {"_id": "$predicted_crop", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            crop_counts = list(CropPrediction.objects.aggregate(crop_counts_pipeline))
            
            # Average confidence by crop
            avg_confidence_pipeline = [
                {"$group": {"_id": "$predicted_crop", "avg_confidence": {"$avg": "$prediction_confidence"}}}
            ]
            avg_confidence = list(CropPrediction.objects.aggregate(avg_confidence_pipeline))
            
            # Predictions by date (last 30 days)
            from datetime import datetime, timedelta
            
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            daily_predictions_pipeline = [
                {"$match": {"prediction_date": {"$gte": thirty_days_ago}}},
                {"$group": {
                    "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$prediction_date"}},
                    "count": {"$sum": 1}
                }},
                {"$sort": {"_id": 1}}
            ]
            daily_predictions = list(CropPrediction.objects.aggregate(daily_predictions_pipeline))
            
            return {
                'total_predictions': total_predictions,
                'crop_distribution': [{'crop': item['_id'], 'count': item['count']} for item in crop_counts],
                'average_confidence': [{'crop': item['_id'], 'confidence': float(item['avg_confidence'])} for item in avg_confidence],
                'daily_predictions': [{'date': item['_id'], 'count': item['count']} for item in daily_predictions]
            }
            
        except Exception as e:
            logger.error(f"Error fetching analytics: {str(e)}")
            raise e
    
    @staticmethod
    def save_model_metadata(model_name, model_version, model_path, metrics=None, hyperparameters=None, training_data_size=None):
        """Save model metadata to database"""
        try:
            # Deactivate previous versions
            ModelMetadata.objects(model_name=model_name, is_active=True).update(set__is_active=False)
            
            # Create new model metadata
            metadata = ModelMetadata(
                model_name=model_name,
                model_version=model_version,
                model_path=model_path,
                training_data_size=training_data_size,
                hyperparameters=hyperparameters or {}
            )
            
            # Add metrics if provided
            if metrics:
                metadata.accuracy = metrics.get('accuracy')
                metadata.precision = metrics.get('precision')
                metadata.recall = metrics.get('recall')
                metadata.f1_score = metrics.get('f1_score')
            
            metadata.save()
            
            logger.info(f"Model metadata saved: {model_name} v{model_version}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error saving model metadata: {str(e)}")
            raise e
    
    @staticmethod
    def get_active_model_metadata(model_name=None):
        """Get active model metadata"""
        try:
            query = ModelMetadata.objects(is_active=True)
            
            if model_name:
                query = query.filter(model_name=model_name)
            
            return query.first()
            
        except Exception as e:
            logger.error(f"Error fetching model metadata: {str(e)}")
            raise e
    
    @staticmethod
    def save_crop_data(crop_info):
        """Save or update crop reference data"""
        try:
            # Check if crop already exists
            existing_crop = CropData.objects(crop_name=crop_info['crop_name']).first()
            
            if existing_crop:
                # Update existing crop
                for key, value in crop_info.items():
                    if hasattr(existing_crop, key) and key != 'id':
                        setattr(existing_crop, key, value)
                existing_crop.updated_date = datetime.utcnow()
                existing_crop.save()
                crop = existing_crop
            else:
                # Create new crop
                crop = CropData(**crop_info)
                crop.save()
            
            logger.info(f"Crop data saved: {crop_info['crop_name']}")
            return crop
            
        except Exception as e:
            logger.error(f"Error saving crop data: {str(e)}")
            raise e
    
    @staticmethod
    def get_crop_data(crop_name=None):
        """Get crop reference data"""
        try:
            if crop_name:
                crop = CropData.objects(crop_name=crop_name).first()
                return crop.to_dict() if crop else None
            else:
                crops = CropData.objects()
                return [crop.to_dict() for crop in crops]
            
        except Exception as e:
            logger.error(f"Error fetching crop data: {str(e)}")
            raise e
    
    @staticmethod
    def save_user_feedback(prediction_id=None, feedback_type='rating', rating=None, comment=None, 
                          user_email=None, user_location=None, ip_address=None):
        """Save user feedback"""
        try:
            # Get prediction reference if prediction_id is provided
            prediction_ref = None
            if prediction_id:
                prediction_ref = CropPrediction.objects(id=prediction_id).first()
            
            feedback = UserFeedback(
                prediction=prediction_ref,
                feedback_type=feedback_type,
                rating=rating,
                comment=comment,
                user_email=user_email,
                user_location=user_location,
                ip_address=ip_address
            )
            
            feedback.save()
            
            logger.info(f"User feedback saved with ID: {feedback.id}")
            return feedback
            
        except Exception as e:
            logger.error(f"Error saving feedback: {str(e)}")
            raise e
    
    @staticmethod
    def get_feedback_analytics():
        """Get feedback analytics"""
        try:
            # Average rating using aggregation
            avg_rating_pipeline = [
                {"$match": {"rating": {"$exists": True, "$ne": None}}},
                {"$group": {"_id": None, "avg_rating": {"$avg": "$rating"}}}
            ]
            avg_rating_result = list(UserFeedback.objects.aggregate(avg_rating_pipeline))
            avg_rating = avg_rating_result[0]['avg_rating'] if avg_rating_result else 0.0
            
            # Rating distribution
            rating_dist_pipeline = [
                {"$match": {"rating": {"$exists": True, "$ne": None}}},
                {"$group": {"_id": "$rating", "count": {"$sum": 1}}},
                {"$sort": {"_id": 1}}
            ]
            rating_dist = list(UserFeedback.objects.aggregate(rating_dist_pipeline))
            
            # Total feedback count
            total_feedback = UserFeedback.objects.count()
            
            # Recent comments
            recent_comments = UserFeedback.objects(comment__exists=True, comment__ne=None).order_by('-created_date').limit(10)
            
            return {
                'average_rating': float(avg_rating) if avg_rating else 0.0,
                'total_feedback': total_feedback,
                'rating_distribution': [{'rating': item['_id'], 'count': item['count']} for item in rating_dist],
                'recent_comments': [feedback.to_dict() for feedback in recent_comments]
            }
            
        except Exception as e:
            logger.error(f"Error fetching feedback analytics: {str(e)}")
            raise e
    
    @staticmethod
    def populate_initial_crop_data():
        """Populate database with initial crop data"""
        try:
            # Check if data already exists
            if CropData.query.count() > 0:
                logger.info("Crop data already exists, skipping initialization")
                return
            
            # Sample crop data (you can expand this)
            crops_data = [
                {
                    'crop_name': 'rice',
                    'crop_type': 'cereal',
                    'optimal_nitrogen_min': 80, 'optimal_nitrogen_max': 120,
                    'optimal_phosphorus_min': 40, 'optimal_phosphorus_max': 60,
                    'optimal_potassium_min': 35, 'optimal_potassium_max': 45,
                    'optimal_temperature_min': 20, 'optimal_temperature_max': 35,
                    'optimal_humidity_min': 80, 'optimal_humidity_max': 90,
                    'optimal_ph_min': 5.5, 'optimal_ph_max': 7.0,
                    'optimal_rainfall_min': 150, 'optimal_rainfall_max': 250,
                    'growing_season': 'Monsoon',
                    'harvest_time': '3-6 months',
                    'description': 'Rice is a staple cereal grain that requires flooded fields and warm climate.'
                },
                {
                    'crop_name': 'wheat',
                    'crop_type': 'cereal',
                    'optimal_nitrogen_min': 100, 'optimal_nitrogen_max': 140,
                    'optimal_phosphorus_min': 45, 'optimal_phosphorus_max': 65,
                    'optimal_potassium_min': 30, 'optimal_potassium_max': 40,
                    'optimal_temperature_min': 15, 'optimal_temperature_max': 25,
                    'optimal_humidity_min': 50, 'optimal_humidity_max': 70,
                    'optimal_ph_min': 6.0, 'optimal_ph_max': 7.5,
                    'optimal_rainfall_min': 50, 'optimal_rainfall_max': 100,
                    'growing_season': 'Rabi (Winter)',
                    'harvest_time': '4-6 months',
                    'description': 'Wheat is a major cereal crop grown in cooler seasons with moderate rainfall.'
                },
                # Add more crops as needed
            ]
            
            for crop_data in crops_data:
                DatabaseService.save_crop_data(crop_data)
            
            logger.info(f"Initialized database with {len(crops_data)} crop records")
            
        except Exception as e:
            logger.error(f"Error populating initial crop data: {str(e)}")
            raise e