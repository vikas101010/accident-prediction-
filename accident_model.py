import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
from .data_processor import DataProcessor
import matplotlib.pyplot as plt
import seaborn as sns
import os

class AccidentModel:
    def __init__(self):
        self.model = None
        self.data_processor = DataProcessor()
        self.feature_names = None
        
    def train(self, X, y=None, hyperparameter_tuning=False):
        """Train the accident prediction model"""
        # If no data is provided, generate sample data
        if X is None:
            df = self.data_processor.generate_sample_data()
            X = df.drop('accident_occurred', axis=1)
            y = df['accident_occurred']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Process the data
        X_train_processed = self.data_processor.preprocess_data(X_train, fit=True)
        X_test_processed = self.data_processor.preprocess_data(X_test)
        
        # Create and train the model
        if hyperparameter_tuning:
            # Define parameter grid for tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            
            # Create base model
            base_model = xgb.XGBClassifier(objective='binary:logistic', n_jobs=-1, random_state=42)
            
            # Set up GridSearchCV
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=5,
                scoring='roc_auc',
                verbose=1
            )
            
            # Fit GridSearchCV
            grid_search.fit(X_train_processed, y_train)
            
            # Best model
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            # Create and train model with default parameters
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train_processed, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_processed)
        y_pred_proba = self.model.predict_proba(X_test_processed)[:, 1]
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)
        
        print("\nROC AUC Score:")
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"{roc_auc:.4f}")
        
        return {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': conf_matrix,
            'roc_auc': roc_auc
        }
        
    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Preprocess the input data
        X_processed = self.data_processor.preprocess_data(X)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)[:, 1]
        
        return predictions, probabilities
    
    def save_model(self, filepath='model/accident_model.joblib'):
        """Save the trained model to disk"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and preprocessor
        model_data = {
            'model': self.model,
            'processor': self.data_processor.processor
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='model/accident_model.joblib'):
        """Load a trained model from disk"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.data_processor.processor = model_data['processor']
            print(f"Model loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"Model file not found at {filepath}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def plot_feature_importance(self, figsize=(12, 8)):
        """Plot feature importance"""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Get feature names from the processor
        if hasattr(self.data_processor.processor, 'get_feature_names_out'):
            feature_names = self.data_processor.processor.get_feature_names_out()
        else:
            # Fallback for older scikit-learn versions
            feature_names = [f'feature_{i}' for i in range(self.model.feature_importances_.shape[0])]
        
        # Create DataFrame of feature importances
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        })
        
        # Sort by importance
        importances = importances.sort_values('importance', ascending=False)
        
        # Plot
        plt.figure(figsize=figsize)
        sns.barplot(x='importance', y='feature', data=importances.head(20))
        plt.title('Feature Importance')
        plt.tight_layout()
        
        return plt.gcf(), importances 