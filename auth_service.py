import logging
from datetime import datetime, timedelta
from flask import current_app
from flask_jwt_extended import create_access_token, create_refresh_token
from models import User
from email_validator import validate_email, EmailNotValidError
import re

logger = logging.getLogger(__name__)

class AuthService:
    """Service class for handling authentication operations"""
    
    @staticmethod
    def validate_email_format(email):
        """Validate email format"""
        try:
            validate_email(email)
            return True, None
        except EmailNotValidError as e:
            return False, str(e)
    
    @staticmethod
    def validate_password_strength(password):
        """Validate password strength"""
        if len(password) < 6:
            return False, "Password must be at least 6 characters long"
        
        if len(password) > 128:
            return False, "Password must be less than 128 characters"
        
        # Check for at least one letter
        if not re.search(r'[a-zA-Z]', password):
            return False, "Password must contain at least one letter"
        
        return True, None
    
    @staticmethod
    def validate_name(name, field_name):
        """Validate first name or last name"""
        if not name or not name.strip():
            return False, f"{field_name} is required"
        
        if len(name.strip()) < 1:
            return False, f"{field_name} must be at least 1 character long"
        
        if len(name.strip()) > 50:
            return False, f"{field_name} must be less than 50 characters"
        
        # Check for only letters, spaces, hyphens, and apostrophes
        if not re.match(r"^[a-zA-Z\s\-']+$", name.strip()):
            return False, f"{field_name} can only contain letters, spaces, hyphens, and apostrophes"
        
        return True, None
    
    @classmethod
    def register_user(cls, first_name, last_name, email, password, **kwargs):
        """Register a new user"""
        try:
            # Validate input data
            is_valid, error_msg = cls.validate_name(first_name, "First name")
            if not is_valid:
                return False, error_msg, None
            
            is_valid, error_msg = cls.validate_name(last_name, "Last name")
            if not is_valid:
                return False, error_msg, None
            
            is_valid, error_msg = cls.validate_email_format(email)
            if not is_valid:
                return False, error_msg, None
            
            is_valid, error_msg = cls.validate_password_strength(password)
            if not is_valid:
                return False, error_msg, None
            
            # Clean input data
            first_name = first_name.strip().title()
            last_name = last_name.strip().title()
            email = email.strip().lower()
            
            # Check if user already exists
            existing_user = User.get_by_email(email)
            if existing_user:
                return False, "User with this email already exists", None
            
            # Create new user
            user = User.create_user(
                first_name=first_name,
                last_name=last_name,
                email=email,
                password=password,
                **kwargs
            )
            
            logger.info(f"New user registered: {email}")
            return True, "User registered successfully", user
            
        except Exception as e:
            logger.error(f"Error registering user: {str(e)}")
            return False, "Registration failed. Please try again.", None
    
    @classmethod
    def authenticate_user(cls, email, password):
        """Authenticate user with email and password"""
        try:
            # Validate input
            if not email or not password:
                return False, "Email and password are required", None
            
            is_valid, error_msg = cls.validate_email_format(email)
            if not is_valid:
                return False, "Invalid email format", None
            
            email = email.strip().lower()
            
            # Get user from database
            user = User.get_by_email(email)
            if not user:
                return False, "Invalid email or password", None
            
            # Check if account is active
            if not user.is_active:
                return False, "Account is disabled. Please contact support.", None
            
            # Verify password
            if not user.check_password(password):
                return False, "Invalid email or password", None
            
            # Update last login
            user.update_last_login()
            
            logger.info(f"User authenticated successfully: {email}")
            return True, "Authentication successful", user
            
        except Exception as e:
            logger.error(f"Error authenticating user: {str(e)}")
            return False, "Authentication failed. Please try again.", None
    
    @classmethod
    def generate_tokens(cls, user):
        """Generate JWT access and refresh tokens for user"""
        try:
            # Create token payload
            additional_claims = {
                "user_id": str(user.id),
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "is_verified": user.is_verified
            }
            
            # Generate tokens
            access_token = create_access_token(
                identity=user.email,
                additional_claims=additional_claims,
                expires_delta=timedelta(hours=24)  # Token expires in 24 hours
            )
            
            refresh_token = create_refresh_token(
                identity=user.email,
                expires_delta=timedelta(days=30)  # Refresh token expires in 30 days
            )
            
            return access_token, refresh_token
            
        except Exception as e:
            logger.error(f"Error generating tokens: {str(e)}")
            raise e
    
    @classmethod
    def get_user_by_id(cls, user_id):
        """Get user by ID"""
        try:
            return User.objects(id=user_id).first()
        except Exception as e:
            logger.error(f"Error getting user by ID: {str(e)}")
            return None
    
    @classmethod
    def get_user_profile(cls, user_id):
        """Get user profile information"""
        try:
            user = cls.get_user_by_id(user_id)
            if not user:
                return False, "User not found", None
            
            return True, "Profile retrieved successfully", user.to_dict()
            
        except Exception as e:
            logger.error(f"Error getting user profile: {str(e)}")
            return False, "Failed to retrieve profile", None
    
    @classmethod
    def update_user_profile(cls, user_id, update_data):
        """Update user profile information"""
        try:
            user = cls.get_user_by_id(user_id)
            if not user:
                return False, "User not found", None
            
            # Validate and update allowed fields
            allowed_fields = ['first_name', 'last_name', 'phone', 'location', 'preferences']
            
            for field, value in update_data.items():
                if field in allowed_fields and hasattr(user, field):
                    if field in ['first_name', 'last_name']:
                        is_valid, error_msg = cls.validate_name(value, field.replace('_', ' ').title())
                        if not is_valid:
                            return False, error_msg, None
                        setattr(user, field, value.strip().title())
                    else:
                        setattr(user, field, value)
            
            user.save()
            logger.info(f"User profile updated: {user.email}")
            return True, "Profile updated successfully", user.to_dict()
            
        except Exception as e:
            logger.error(f"Error updating user profile: {str(e)}")
            return False, "Failed to update profile", None