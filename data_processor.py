import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class DataProcessor:
    def __init__(self):
        # Define categorical and numerical features
        self.categorical_features = [
            'weather_condition', 
            'road_condition', 
            'location_type',
            'day_of_week'
        ]
        self.numerical_features = [
            'time_of_day', 
            'traffic_density', 
            'speed_limit',
            'visibility_meters',
            'temperature_celsius'
        ]
        self.processor = None
        
    def create_processor(self):
        """Create the column transformer pipeline for preprocessing"""
        # Numerical preprocessing: impute missing values and scale
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing: impute missing values and one-hot encode
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.processor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return self.processor
    
    def preprocess_data(self, X, fit=False):
        """Preprocess the data using the established pipeline"""
        if fit or self.processor is None:
            self.create_processor()
            return self.processor.fit_transform(X)
        else:
            return self.processor.transform(X)
    
    @staticmethod
    def generate_sample_data(n_samples=1000):
        """Generate sample accident data for demonstration"""
        np.random.seed(42)
        
        # Generate random data
        weather_conditions = ['clear', 'rain', 'snow', 'fog']
        road_conditions = ['dry', 'wet', 'icy', 'snowy']
        location_types = ['intersection', 'highway', 'residential', 'rural']
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        data = {
            'weather_condition': np.random.choice(weather_conditions, n_samples),
            'road_condition': np.random.choice(road_conditions, n_samples),
            'location_type': np.random.choice(location_types, n_samples),
            'day_of_week': np.random.choice(days_of_week, n_samples),
            'time_of_day': np.random.randint(0, 24, n_samples),
            'traffic_density': np.random.randint(1, 11, n_samples),
            'speed_limit': np.random.choice([25, 30, 35, 40, 45, 55, 65, 70], n_samples),
            'visibility_meters': np.random.randint(50, 10000, n_samples),
            'temperature_celsius': np.random.uniform(-10, 40, n_samples)
        }
        
        # Create accident probability based on features
        accident_prob = (
            (data['weather_condition'] != 'clear') * 0.2 +
            (data['road_condition'] != 'dry') * 0.2 +
            (data['traffic_density'] > 7) * 0.15 +
            (data['time_of_day'] < 6) * 0.1 +
            (data['time_of_day'] > 20) * 0.1 +
            (data['visibility_meters'] < 500) * 0.25 +
            (data['location_type'] == 'intersection') * 0.1
        )
        
        # Normalize and add some randomness
        accident_prob = 0.7 * accident_prob/accident_prob.max() + 0.3 * np.random.random(n_samples)
        
        # Create binary outcome
        data['accident_occurred'] = (accident_prob > 0.5).astype(int)
        
        return pd.DataFrame(data) 