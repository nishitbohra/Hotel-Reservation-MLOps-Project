# Hotel Reservation MLOps Project

This project implements a complete MLOps workflow for predicting hotel reservation cancellations. It includes data exploration, model training with multiple algorithms, experiment tracking, hyperparameter tuning, and deployment with FastAPI.

## Project Structure

```
.
├── hotel_mlops_project.py    # Main training script
├── app.py                    # FastAPI deployment application
├── models/                   # Saved models directory
├── visualizations/           # Visualization images directory
├── static/                   # Static files for web app
├── templates/                # HTML templates for web app
└── README.md                 # This file
```

## Features

- **Data Exploration**: Comprehensive EDA with visualizations
- **Model Training**: Multiple models trained and compared
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost
- **Experiment Tracking**: All experiments tracked with MLflow
- **Hyperparameter Tuning**: Grid search for the best model
- **Model Deployment**: Interactive FastAPI web application
- **Model Comparison**: Visual comparison of model performances

## Setup and Installation

1. Clone this repository
2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Place your hotel reservation dataset (CSV file) in the project directory

## Usage

### Training Models

```bash
python hotel_mlops_project.py
```

This will:
- Load and explore the dataset
- Train multiple models
- Track experiments with MLflow
- Tune the best performing model
- Save visualizations and models

### Starting MLflow UI

```bash
mlflow ui --port 5000
```

This will start the MLflow tracking UI at http://localhost:5000

### Deploying the Model

```bash
uvicorn app:app --reload
```

This will start the FastAPI application at http://localhost:8000

## API Documentation

Once the FastAPI application is running, you can access:
- Interactive API docs: http://localhost:8000/docs
- Alternative API docs: http://localhost:8000/redoc
- Web interface: http://localhost:8000

## Project Highlights

### Advanced Features

1. **Multi-model comparison** with comprehensive metrics
2. **Interactive visualizations** in the web application
3. **Feature importance analysis** for individual predictions
4. **Hyperparameter tuning** for optimal model performance
5. **Extensive experiment tracking** with MLflow

### Visualizations

- Correlation heatmaps
- Feature importance plots
- ROC curves for all models
- Confusion matrices
- Model performance comparisons

## Sample API Request

```python
import requests
import json

url = "http://localhost:8000/predict"
data = {
    "lead_time": 45,
    "stays_in_weekend_nights": 1,
    "stays_in_week_nights": 2,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "meal": "BB",
    "country": "PRT",
    "market_segment": "Direct",
    "distribution_channel": "Direct",
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A",
    "assigned_room_type": "A",
    "booking_changes": 0,
    "deposit_type": "No Deposit",
    "days_in_waiting_list": 0,
    "customer_type": "Transient",
    "adr": 120.5,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 1,
    "reservation_status": "Confirmed"
}

response = requests.post(url, json=data)
print(json.dumps(response.json(), indent=4))
```

## MLflow Tracking Screenshot

![MLflow Tracking](visualizations/mlflow_tracking_screenshot.png)

## Requirements

Create a requirements.txt file with the following dependencies:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
mlflow
joblib
fastapi
uvicorn
python-multipart
jinja2
```

## Future Improvements

- Add CI/CD pipeline for model deployment
- Implement model monitoring
- Add data drift detection
- Create more advanced visualizations
- Add user authentication for the web application
