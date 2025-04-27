# app.py
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import List, Dict, Optional
import os
import json

# Create FastAPI app
app = FastAPI(title="Hotel Reservation Prediction API",
              description="API for predicting hotel reservation cancellations",
              version="1.0.0")

# Load model
MODEL_PATH = "models/XGBoost_tuned_model.pkl"  # You might need to update this with your best model
model = joblib.load(MODEL_PATH)

# Create templates directory and mount static files
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Create HTML template file
with open("templates/index.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Hotel Reservation Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; }
        .prediction-result { margin-top: 20px; padding: 15px; border-radius: 5px; }
        .canceled { background-color: #ffcccc; }
        .not-canceled { background-color: #ccffcc; }
        .feature-importance { margin-top: 30px; }
        .chart-container { margin-top: 30px; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">Hotel Reservation Cancellation Prediction</h1>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Make a Prediction</h5>
                    </div>
                    <div class="card-body">
                        <form id="predictionForm" method="post" action="/predict">
                            <div class="mb-3">
                                <label for="lead_time" class="form-label">Lead Time (days)</label>
                                <input type="number" class="form-control" id="lead_time" name="lead_time" required>
                            </div>
                            
                            <div class="mb-3">
                                <label for="stays_in_weekend_nights" class="form-label">Weekend Nights</label>
                                <input type="number" class="form-control" id="stays_in_weekend_nights" name="stays_in_weekend_nights" required>
                            </div>
                            
                            <div class="mb-3">
                                <label for="stays_in_week_nights" class="form-label">Weekday Nights</label>
                                <input type="number" class="form-control" id="stays_in_week_nights" name="stays_in_week_nights" required>
                            </div>
                            
                            <div class="mb-3">
                                <label for="adults" class="form-label">Number of Adults</label>
                                <input type="number" class="form-control" id="adults" name="adults" required>
                            </div>
                            
                            <div class="mb-3">
                                <label for="children" class="form-label">Number of Children</label>
                                <input type="number" class="form-control" id="children" name="children" required>
                            </div>
                            
                            <div class="mb-3">
                                <label for="babies" class="form-label">Number of Babies</label>
                                <input type="number" class="form-control" id="babies" name="babies" required>
                            </div>
                            
                            <div class="mb-3">
                                <label for="meal" class="form-label">Meal Plan</label>
                                <select class="form-control" id="meal" name="meal" required>
                                    <option value="BB">Bed & Breakfast</option>
                                    <option value="HB">Half Board</option>
                                    <option value="FB">Full Board</option>
                                    <option value="SC">Self Catering</option>
                                </select>
                            </div>
                            
                            <div class="mb-3">
                                <label for="country" class="form-label">Country</label>
                                <input type="text" class="form-control" id="country" name="country" required>
                            </div>
                            
                            <div class="mb-3">
                                <label for="market_segment" class="form-label">Market Segment</label>
                                <select class="form-control" id="market_segment" name="market_segment" required>
                                    <option value="Direct">Direct</option>
                                    <option value="Corporate">Corporate</option>
                                    <option value="Online TA">Online Travel Agent</option>
                                    <option value="Offline TA/TO">Offline Travel Agent</option>
                                    <option value="Groups">Groups</option>
                                    <option value="Complementary">Complementary</option>
                                </select>
                            </div>
                            
                            <div class="mb-3">
                                <label for="distribution_channel" class="form-label">Distribution Channel</label>
                                <select class="form-control" id="distribution_channel" name="distribution_channel" required>
                                    <option value="Direct">Direct</option>
                                    <option value="Corporate">Corporate</option>
                                    <option value="TA/TO">Travel Agent/Tour Operator</option>
                                </select>
                            </div>
                            
                            <div class="mb-3">
                                <label for="is_repeated_guest" class="form-label">Repeated Guest?</label>
                                <select class="form-control" id="is_repeated_guest" name="is_repeated_guest" required>
                                    <option value="0">No</option>
                                    <option value="1">Yes</option>
                                </select>
                            </div>
                            
                            <div class="mb-3">
                                <label for="previous_cancellations" class="form-label">Previous Cancellations</label>
                                <input type="number" class="form-control" id="previous_cancellations" name="previous_cancellations" required>
                            </div>
                            
                            <div class="mb-3">
                                <label for="previous_bookings_not_canceled" class="form-label">Previous Bookings Not Canceled</label>
                                <input type="number" class="form-control" id="previous_bookings_not_canceled" name="previous_bookings_not_canceled" required>
                            </div>
                            
                            <div class="mb-3">
                                <label for="reserved_room_type" class="form-label">Reserved Room Type</label>
                                <select class="form-control" id="reserved_room_type" name="reserved_room_type" required>
                                    <option value="A">A</option>
                                    <option value="B">B</option>
                                    <option value="C">C</option>
                                    <option value="D">D</option>
                                    <option value="E">E</option>
                                    <option value="F">F</option>
                                    <option value="G">G</option>
                                    <option value="H">H</option>
                                </select>
                            </div>
                            
                            <div class="mb-3">
                                <label for="assigned_room_type" class="form-label">Assigned Room Type</label>
                                <select class="form-control" id="assigned_room_type" name="assigned_room_type" required>
                                    <option value="A">A</option>
                                    <option value="B">B</option>
                                    <option value="C">C</option>
                                    <option value="D">D</option>
                                    <option value="E">E</option>
                                    <option value="F">F</option>
                                    <option value="G">G</option>
                                    <option value="H">H</option>
                                </select>
                            </div>
                            
                            <div class="mb-3">
                                <label for="booking_changes" class="form-label">Booking Changes</label>
                                <input type="number" class="form-control" id="booking_changes" name="booking_changes" required>
                            </div>
                            
                            <div class="mb-3">
                                <label for="deposit_type" class="form-label">Deposit Type</label>
                                <select class="form-control" id="deposit_type" name="deposit_type" required>
                                    <option value="No Deposit">No Deposit</option>
                                    <option value="Refundable">Refundable</option>
                                    <option value="Non Refund">Non Refundable</option>
                                </select>
                            </div>
                            
                            <div class="mb-3">
                                <label for="days_in_waiting_list" class="form-label">Days in Waiting List</label>
                                <input type="number" class="form-control" id="days_in_waiting_list" name="days_in_waiting_list" required>
                            </div>
                            
                            <div class="mb-3">
                                <label for="customer_type" class="form-label">Customer Type</label>
                                <select class="form-control" id="customer_type" name="customer_type" required>
                                    <option value="Transient">Transient</option>
                                    <option value="Contract">Contract</option>
                                    <option value="Transient-Party">Transient-Party</option>
                                    <option value="Group">Group</option>
                                </select>
                            </div>
                            
                            <div class="mb-3">
                                <label for="adr" class="form-label">Average Daily Rate</label>
                                <input type="number" step="0.01" class="form-control" id="adr" name="adr" required>
                            </div>
                            
                            <div class="mb-3">
                                <label for="required_car_parking_spaces" class="form-label">Required Car Parking Spaces</label>
                                <input type="number" class="form-control" id="required_car_parking_spaces" name="required_car_parking_spaces" required>
                            </div>
                            
                            <div class="mb-3">
                                <label for="total_of_special_requests" class="form-label">Total Special Requests</label>
                                <input type="number" class="form-control" id="total_of_special_requests" name="total_of_special_requests" required>
                            </div>
                            
                            <div class="mb-3">
                                <label for="reservation_status" class="form-label">Reservation Status</label>
                                <select class="form-control" id="reservation_status" name="reservation_status" required>
                                    <option value="Check-Out">Check-Out</option>
                                    <option value="Confirmed">Confirmed</option>
                                    <option value="Canceled">Canceled</option>
                                    <option value="No-Show">No-Show</option>
                                </select>
                            </div>
                            
                            <button type="submit" class="btn btn-primary">Predict Cancellation</button>
                        </form>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div id="predictionResult"></div>
                <div id="featureImportance" class="feature-importance"></div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5>Model Comparison</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <img src="/static/model_comparison_accuracy.png" class="img-fluid" alt="Model Comparison - Accuracy">
                            </div>
                            <div class="col-md-6">
                                <img src="/static/model_comparison_f1.png" class="img-fluid" alt="Model Comparison - F1 Score">
                            </div>
                        </div>
                        <div class="row mt-3">
                            <div class="col-md-12">
                                <img src="/static/roc_curves_comparison.png" class="img-fluid" alt="ROC Curves Comparison">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const formObject = {};
            
            formData.forEach((value, key) => {
                formObject[key] = value;
            });
            
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formObject)
            })
            .then(response => response.json())
            .then(data => {
                const resultDiv = document.getElementById('predictionResult');
                
                // Display prediction result
                const resultClass = data.prediction === 1 ? 'canceled' : 'not-canceled';
                const resultText = data.prediction === 1 ? 'Likely to be canceled' : 'Not likely to be canceled';
                
                resultDiv.innerHTML = `
                    <div class="card">
                        <div class="card-header">
                            <h5>Prediction Result</h5>
                        </div>
                        <div class="card-body prediction-result ${resultClass}">
                            <h3>${resultText}</h3>
                            <p>Probability of cancellation: ${(data.probability * 100).toFixed(2)}%</p>
                        </div>
                    </div>
                `;
                
                // Display feature importance if available
                if (data.feature_importance) {
                    const importanceDiv = document.getElementById('featureImportance');
                    importanceDiv.innerHTML = `
                        <div class="card mt-3">
                            <div class="card-header">
                                <h5>Top Features for this Prediction</h5>
                            </div>
                            <div class="card-body">
                                <img src="data:image/png;base64,${data.feature_importance}" class="img-fluid">
                            </div>
                        </div>
                    `;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('predictionResult').innerHTML = `
                    <div class="alert alert-danger">
                        An error occurred while making the prediction. Please try again.
                    </div>
                `;
            });
        });
    </script>
</body>
</html>
    """)

# Create Pydantic model for request data
class HotelReservation(BaseModel):
    lead_time: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: int
    babies: int
    meal: str
    country: str
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: str
    assigned_room_type: str
    booking_changes: int
    deposit_type: str
    days_in_waiting_list: int
    customer_type: str
    adr: float
    required_car_parking_spaces: int
    total_of_special_requests: int
    reservation_status: str

# Create Pydantic model for prediction response
class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    feature_importance: Optional[str] = None

# Route for home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Function to plot feature importance for a specific input
def get_feature_importance_plot(input_data: dict, model):
    """Generate feature importance plot for input data"""
    try:
        # Extract feature names from model
        if hasattr(model[-1], 'feature_importances_'):
            importances = model[-1].feature_importances_
            feature_names = input_data.keys()
            
            # Match feature names with importances
            # This is a simplified approach - in a real application, you would
            # need to handle the preprocessing transformations appropriately
            sorted_idx = np.argsort(importances)[::-1]
            top_idx = sorted_idx[:10]  # Top 10 features
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(top_idx)), importances[top_idx], align='center')
            plt.yticks(range(len(top_idx)), [feature_names[i] for i in top_idx])
            plt.xlabel('Feature Importance')
            plt.title('Top 10 Important Features')
            plt.tight_layout()
            
            # Save plot to bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Convert to base64 string
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_str
        else:
            return None
    except Exception as e:
        print(f"Error generating feature importance plot: {e}")
        return None

# Route for prediction
@app.post("/predict", response_model=PredictionResponse)
async def predict(reservation: HotelReservation):
    try:
        # Convert input data to DataFrame
        input_dict = reservation.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # Probability of class 1 (cancellation)
        
        # Generate feature importance plot
        feature_importance_img = get_feature_importance_plot(input_dict, model)
        
        # Return prediction
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "feature_importance": feature_importance_img
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Route for model metrics visualization
@app.get("/metrics", response_class=HTMLResponse)
async def metrics(request: Request):
    return templates.TemplateResponse("metrics.html", {"request": request})

# Copy visualization files to static folder
@app.on_event("startup")
async def startup_event():
    # Copy visualization files to static folder
    import shutil
    import glob
    
    # Create static folder if it doesn't exist
    os.makedirs("static", exist_ok=True)
    
    # Copy visualization files
    for file in glob.glob("visualizations/*.png"):
        shutil.copy(file, f"static/{os.path.basename(file)}")
    
    print("Visualization files copied to static folder")

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)