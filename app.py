import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from model.accident_model import AccidentModel
from model.data_processor import DataProcessor

app = Flask(__name__)
model = AccidentModel()

# Initialize or load the model
MODEL_PATH = os.path.join('model', 'accident_model.joblib')

def initialize_model():
    """Initialize the model - either load existing or train new"""
    if os.path.exists(MODEL_PATH):
        model.load_model(MODEL_PATH)
    else:
        print("Training new model...")
        # Generate sample data and train model
        data_processor = DataProcessor()
        sample_data = data_processor.generate_sample_data(n_samples=2000)
        X = sample_data.drop('accident_occurred', axis=1)
        y = sample_data['accident_occurred']
        model.train(X, y)
        model.save_model(MODEL_PATH)
    return model

# Routes
@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions based on user input"""
    try:
        # Get form data
        weather = request.form.get('weather_condition')
        road = request.form.get('road_condition')
        location = request.form.get('location_type')
        day = request.form.get('day_of_week')
        time = int(request.form.get('time_of_day'))
        traffic = int(request.form.get('traffic_density'))
        speed = int(request.form.get('speed_limit'))
        visibility = int(request.form.get('visibility_meters'))
        temperature = float(request.form.get('temperature_celsius'))
        
        # Create input data frame
        input_data = pd.DataFrame({
            'weather_condition': [weather],
            'road_condition': [road],
            'location_type': [location],
            'day_of_week': [day],
            'time_of_day': [time],
            'traffic_density': [traffic],
            'speed_limit': [speed],
            'visibility_meters': [visibility],
            'temperature_celsius': [temperature]
        })
        
        # Make prediction
        _, probability = model.predict(input_data)
        
        # Return prediction
        risk_level = get_risk_level(probability[0])
        
        result = {
            'probability': float(probability[0]),
            'risk_level': risk_level,
            'risk_percentage': f"{probability[0] * 100:.1f}%"
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/dashboard')
def dashboard():
    """Render the dashboard page with model insights"""
    # Generate feature importance plot
    fig, importances = model.plot_feature_importance()
    
    # Convert plot to base64 string for embedding in HTML
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_str = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close(fig)
    
    # Generate sample data for visualization
    data_processor = DataProcessor()
    sample_data = data_processor.generate_sample_data(n_samples=1000)
    
    # Calculate accident rates by weather condition
    weather_counts = sample_data.groupby('weather_condition')['accident_occurred'].mean().reset_index()
    weather_counts = weather_counts.sort_values('accident_occurred', ascending=False)
    
    # Convert to list for JavaScript
    weather_labels = weather_counts['weather_condition'].tolist()
    weather_values = (weather_counts['accident_occurred'] * 100).round(1).tolist()
    
    # Calculate accident rates by time of day
    time_counts = sample_data.groupby('time_of_day')['accident_occurred'].mean().reset_index()
    time_labels = time_counts['time_of_day'].tolist()
    time_values = (time_counts['accident_occurred'] * 100).round(1).tolist()
    
    return render_template(
        'dashboard.html',
        feature_importance_img=img_str,
        weather_labels=weather_labels,
        weather_values=weather_values,
        time_labels=time_labels,
        time_values=time_values
    )

def get_risk_level(probability):
    """Convert probability to risk level"""
    if probability < 0.2:
        return "Very Low"
    elif probability < 0.4:
        return "Low"
    elif probability < 0.6:
        return "Moderate"
    elif probability < 0.8:
        return "High"
    else:
        return "Very High"

if __name__ == '__main__':
    # Initialize the model
    initialize_model()
    
    # Start the app
    app.run(debug=True) 