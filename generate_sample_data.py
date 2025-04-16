"""
Generate sample accident data for training and testing the accident prediction model.
This script creates a CSV file with synthetic data that mimics real accident data.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_accident_data(n_samples=5000, output_file='accident_data.csv'):
    """
    Generate synthetic accident data with various features.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    output_file : str
        Path to save the CSV file
    
    Returns:
    --------
    DataFrame with the generated data
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate random data
    weather_conditions = ['clear', 'rain', 'snow', 'fog']
    road_conditions = ['dry', 'wet', 'icy', 'snowy']
    location_types = ['intersection', 'highway', 'residential', 'rural']
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Generate random timestamps over the last 3 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)
    timestamps = [start_date + timedelta(seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))) for _ in range(n_samples)]
    
    # Extract time of day from timestamps
    hours = [ts.hour for ts in timestamps]
    
    data = {
        'timestamp': timestamps,
        'weather_condition': np.random.choice(weather_conditions, n_samples, p=[0.6, 0.25, 0.1, 0.05]),
        'road_condition': np.random.choice(road_conditions, n_samples, p=[0.7, 0.2, 0.05, 0.05]),
        'location_type': np.random.choice(location_types, n_samples, p=[0.3, 0.3, 0.25, 0.15]),
        'day_of_week': [ts.strftime('%A') for ts in timestamps],
        'time_of_day': hours,
        'traffic_density': np.random.randint(1, 11, n_samples),
        'speed_limit': np.random.choice([25, 30, 35, 40, 45, 55, 65, 70], n_samples),
        'visibility_meters': np.random.randint(50, 10000, n_samples),
        'temperature_celsius': np.random.uniform(-10, 40, n_samples)
    }
    
    # Make data more realistic by creating correlations
    
    # 1. Weather condition affects road condition
    for i in range(n_samples):
        if data['weather_condition'][i] == 'rain':
            if np.random.random() < 0.8:  # 80% chance of wet road when raining
                data['road_condition'][i] = 'wet'
        elif data['weather_condition'][i] == 'snow':
            if np.random.random() < 0.8:  # 80% chance of snowy/icy road when snowing
                data['road_condition'][i] = np.random.choice(['snowy', 'icy'], p=[0.6, 0.4])
    
    # 2. Weather affects visibility
    for i in range(n_samples):
        if data['weather_condition'][i] == 'fog':
            data['visibility_meters'][i] = np.random.randint(50, 500)
        elif data['weather_condition'][i] == 'rain':
            data['visibility_meters'][i] = np.random.randint(200, 2000)
        elif data['weather_condition'][i] == 'snow':
            data['visibility_meters'][i] = np.random.randint(100, 1000)
    
    # 3. Time of day affects traffic density
    for i in range(n_samples):
        hour = data['time_of_day'][i]
        if 7 <= hour <= 9 or 16 <= hour <= 18:  # Rush hours
            data['traffic_density'][i] = np.random.randint(7, 11)
        elif 0 <= hour <= 5:  # Late night
            data['traffic_density'][i] = np.random.randint(1, 4)
    
    # Create accident probability based on features
    accident_prob = np.zeros(n_samples)
    
    # Base factors for accident probability
    for i in range(n_samples):
        # Weather factors
        if data['weather_condition'][i] == 'fog':
            accident_prob[i] += 0.3
        elif data['weather_condition'][i] == 'rain':
            accident_prob[i] += 0.2
        elif data['weather_condition'][i] == 'snow':
            accident_prob[i] += 0.25
        
        # Road condition factors
        if data['road_condition'][i] == 'icy':
            accident_prob[i] += 0.35
        elif data['road_condition'][i] == 'wet':
            accident_prob[i] += 0.15
        elif data['road_condition'][i] == 'snowy':
            accident_prob[i] += 0.25
        
        # Location factors
        if data['location_type'][i] == 'intersection':
            accident_prob[i] += 0.15
        elif data['location_type'][i] == 'highway':
            accident_prob[i] += 0.1
        
        # Time factors
        hour = data['time_of_day'][i]
        if 0 <= hour <= 5:  # Late night
            accident_prob[i] += 0.2
        
        # Day of week factors
        if data['day_of_week'][i] in ['Friday', 'Saturday']:
            accident_prob[i] += 0.05
        
        # Traffic and speed factors
        accident_prob[i] += (data['traffic_density'][i] / 10) * 0.1
        accident_prob[i] += (data['speed_limit'][i] / 70) * 0.1
        
        # Visibility factors (lower visibility = higher risk)
        visibility_factor = 1 - (data['visibility_meters'][i] / 10000)
        accident_prob[i] += visibility_factor * 0.2
    
    # Normalize probabilities to 0-1 range and add randomness
    accident_prob = accident_prob / accident_prob.max()
    accident_prob = 0.7 * accident_prob + 0.3 * np.random.random(n_samples)
    
    # Create binary outcome
    data['accident_occurred'] = (accident_prob > 0.5).astype(int)
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), output_file)
    df.to_csv(output_path, index=False)
    print(f"Sample data saved to {output_path}")
    
    # Print summary statistics
    accident_rate = df['accident_occurred'].mean() * 100
    print(f"Generated {n_samples} samples with accident rate of {accident_rate:.2f}%")
    
    return df

if __name__ == "__main__":
    generate_accident_data() 