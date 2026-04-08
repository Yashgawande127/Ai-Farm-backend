# Crop Recommendation Backend

A Flask-based REST API for crop recommendation using machine learning. This backend uses a Random Forest Classifier to predict the most suitable crop based on soil and weather parameters.

## Features

- **Machine Learning Model**: Random Forest Classifier trained on agricultural data
- **RESTful API**: Clean API endpoints for training and prediction
- **Model Persistence**: Trained models are saved using pickle for fast loading
- **Input Validation**: Comprehensive validation of input parameters
- **Detailed Predictions**: Confidence scores and probability distributions
- **Health Monitoring**: Built-in health check and status endpoints

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Navigate to the backend directory**:
   ```bash
   cd backend
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the dataset is available**:
   - Make sure `Crop_recommendation.csv` is in the parent directory
   - Or update the path in `config.py` if located elsewhere

## Running the Application

### Development Mode
```bash
python app.py
```

The API will be available at `http://localhost:5000`

### Production Mode
```bash
# Using Gunicorn (recommended for production)
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## API Endpoints

### Core Endpoints

#### Health Check
```http
GET /
```
Returns application health status and system checks.

#### Train Model
```http
POST /train
```
Trains a new Random Forest model and saves it for future use.

### Prediction Endpoints

#### Basic Prediction
```http
POST /api/predict
```

**Request Body**:
```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.88,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.94
}
```

**Response**:
```json
{
  "status": "success",
  "predicted_crop": "rice",
  "confidence": 0.8542,
  "input_features": {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.88,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.94
  }
}
```

#### Detailed Prediction
```http
POST /api/predict/detailed
```

Returns prediction with probability distribution for all crops.

### Information Endpoints

#### Model Status
```http
GET /api/model/status
```
Check if model is loaded and ready for predictions.

#### Model Information
```http
GET /api/model/info
```
Get detailed information about the loaded model including feature importance.

#### Available Crops
```http
GET /api/crops
```
Get list of crops that the model can predict.

#### Required Features
```http
GET /api/features
```
Get information about required input features and their valid ranges.

## Input Parameters

The model requires 7 input features:

| Parameter | Description | Unit | Typical Range |
|-----------|-------------|------|---------------|
| N | Nitrogen content in soil | ratio | 0-140 |
| P | Phosphorus content in soil | ratio | 5-145 |
| K | Potassium content in soil | ratio | 5-205 |
| temperature | Temperature | °C | 8.83-43.68 |
| humidity | Relative humidity | % | 14.26-99.98 |
| ph | Soil pH level | pH | 3.5-9.94 |
| rainfall | Rainfall | mm | 20.21-298.56 |

## Project Structure

```
backend/
├── app.py                 # Main Flask application
├── model_trainer.py       # Model training logic
├── prediction_service.py  # Prediction service class
├── routes.py             # API route handlers
├── config.py             # Configuration settings
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── models/              # Directory for saved models
│   └── crop_model.pkl   # Trained model (created after training)
└── logs/                # Application logs (created automatically)
```

## Configuration

Configuration is managed in `config.py` with different settings for development, testing, and production environments.

### Environment Variables

- `FLASK_ENV`: Set to 'development', 'production', or 'testing'
- `SECRET_KEY`: Flask secret key (set in production)

## Model Training

The system automatically trains a model on first startup if none exists. You can also trigger training manually:

1. **Via API**: Send a POST request to `/train`
2. **Via Script**: Run `python model_trainer.py` directly

The training process:
- Loads the crop recommendation dataset
- Preprocesses the data (feature extraction, validation)
- Trains a Random Forest Classifier
- Evaluates model performance
- Saves the model using pickle

## Model Performance

The Random Forest Classifier is configured with:
- 100 decision trees (n_estimators=100)
- Maximum depth of 10 levels
- Minimum 5 samples required to split nodes
- Minimum 2 samples required at leaf nodes

Expected performance metrics:
- Accuracy: >90% on test set
- Cross-validation score: >88%
- Feature importance analysis included

## Error Handling

The API includes comprehensive error handling:
- Input validation with detailed error messages
- Model loading error handling
- Graceful degradation when model is unavailable
- Structured error responses with appropriate HTTP status codes

## Logging

Application logs are stored in the `logs/` directory with:
- Daily log rotation
- Structured log format
- Configurable log levels
- Both file and console output

## CORS Support

Cross-Origin Resource Sharing (CORS) is enabled for frontend integration. Configure allowed origins in `config.py`.

## Testing

Run the prediction service directly for testing:

```bash
python prediction_service.py
```

This will load the model and make a sample prediction.

## Troubleshooting

### Common Issues

1. **Model not found**: Train a model first using `/train` endpoint
2. **Import errors**: Ensure all dependencies are installed
3. **Dataset not found**: Check the dataset path in configuration
4. **Permission errors**: Ensure write access to `models/` and `logs/` directories

### Health Check

Visit `http://localhost:5000/` to see system health status and diagnose issues.

## Production Deployment

For production deployment:

1. Set `FLASK_ENV=production`
2. Set a secure `SECRET_KEY`
3. Use a production WSGI server like Gunicorn
4. Configure proper logging and monitoring
5. Set up database for model metadata (optional)
6. Configure load balancing if needed

## Contributing

1. Follow PEP 8 style guidelines
2. Add appropriate error handling
3. Include logging for debugging
4. Update documentation for new features
5. Test with various input scenarios

## License

This project is for educational and research purposes.