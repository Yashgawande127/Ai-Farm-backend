from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
import logging
from auth_service import AuthService

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint for authentication routes
auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

@auth_bp.route('/register', methods=['POST'])
def register():
    """
    Register a new user account
    
    Expected JSON input:
    {
        "first_name": "John",
        "last_name": "Doe",
        "email": "john@example.com",
        "password": "securepassword"
    }
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        # Extract required fields
        first_name = data.get('first_name', '').strip()
        last_name = data.get('last_name', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        # Validate required fields
        if not all([first_name, last_name, email, password]):
            return jsonify({
                'status': 'error',
                'message': 'All fields are required: first_name, last_name, email, password'
            }), 400
        
        # Optional fields
        phone = data.get('phone', '').strip()
        location = data.get('location', '').strip()
        
        # Attempt to register user
        success, message, user = AuthService.register_user(
            first_name=first_name,
            last_name=last_name,
            email=email,
            password=password,
            phone=phone if phone else None,
            location=location if location else None
        )
        
        if not success:
            return jsonify({
                'status': 'error',
                'message': message
            }), 400
        
        # Generate tokens for the new user
        try:
            access_token, refresh_token = AuthService.generate_tokens(user)
        except Exception as e:
            logger.error(f"Failed to generate tokens for new user: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Registration successful but login failed. Please try logging in.'
            }), 500
        
        logger.info(f"User registered and logged in successfully: {email}")
        
        return jsonify({
            'status': 'success',
            'message': 'Registration successful',
            'user': user.to_dict(),
            'access_token': access_token,
            'refresh_token': refresh_token
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Registration failed. Please try again.'
        }), 500


@auth_bp.route('/login', methods=['POST'])
def login():
    """
    Authenticate user and return JWT tokens
    
    Expected JSON input:
    {
        "email": "john@example.com",
        "password": "securepassword"
    }
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        # Extract credentials
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        # Validate required fields
        if not email or not password:
            return jsonify({
                'status': 'error',
                'message': 'Email and password are required'
            }), 400
        
        # Authenticate user
        success, message, user = AuthService.authenticate_user(email, password)
        
        if not success:
            return jsonify({
                'status': 'error',
                'message': message
            }), 401
        
        # Generate tokens
        try:
            access_token, refresh_token = AuthService.generate_tokens(user)
        except Exception as e:
            logger.error(f"Failed to generate tokens: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Authentication failed. Please try again.'
            }), 500
        
        logger.info(f"User logged in successfully: {email}")
        
        return jsonify({
            'status': 'success',
            'message': 'Login successful',
            'user': user.to_dict(),
            'access_token': access_token,
            'refresh_token': refresh_token
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Login failed. Please try again.'
        }), 500


@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """
    Logout user (client-side token removal)
    Note: In a production system, you might want to blacklist the token
    """
    try:
        current_user_email = get_jwt_identity()
        logger.info(f"User logged out: {current_user_email}")
        
        return jsonify({
            'status': 'success',
            'message': 'Logout successful'
        }), 200
        
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Logout failed'
        }), 500


@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get current user's profile information"""
    try:
        # Get user from JWT token
        jwt_data = get_jwt()
        user_id = jwt_data.get('user_id')
        
        if not user_id:
            return jsonify({
                'status': 'error',
                'message': 'Invalid token'
            }), 401
        
        # Get user profile
        success, message, profile = AuthService.get_user_profile(user_id)
        
        if not success:
            return jsonify({
                'status': 'error',
                'message': message
            }), 404
        
        return jsonify({
            'status': 'success',
            'user': profile
        }), 200
        
    except Exception as e:
        logger.error(f"Get profile error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to retrieve profile'
        }), 500


@auth_bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    """Update current user's profile information"""
    try:
        # Get user from JWT token
        jwt_data = get_jwt()
        user_id = jwt_data.get('user_id')
        
        if not user_id:
            return jsonify({
                'status': 'error',
                'message': 'Invalid token'
            }), 401
        
        # Get update data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        # Update profile
        success, message, profile = AuthService.update_user_profile(user_id, data)
        
        if not success:
            return jsonify({
                'status': 'error',
                'message': message
            }), 400
        
        return jsonify({
            'status': 'success',
            'message': 'Profile updated successfully',
            'user': profile
        }), 200
        
    except Exception as e:
        logger.error(f"Update profile error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to update profile'
        }), 500


@auth_bp.route('/verify-token', methods=['GET'])
@jwt_required()
def verify_token():
    """Verify if the current token is valid"""
    try:
        # Get user from JWT token
        jwt_data = get_jwt()
        user_id = jwt_data.get('user_id')
        current_user_email = get_jwt_identity()
        
        # Get user profile to ensure user still exists
        success, message, profile = AuthService.get_user_profile(user_id)
        
        if not success:
            return jsonify({
                'status': 'error',
                'message': 'User not found'
            }), 404
        
        return jsonify({
            'status': 'success',
            'message': 'Token is valid',
            'user': profile
        }), 200
        
    except Exception as e:
        logger.error(f"Token verification error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Token verification failed'
        }), 401