import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os


class DynamicPricingModel:
    def __init__(self):
        self.mountain_bike_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.ebike_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.label_encoders = {}
        self.is_trained = False

    def prepare_features(self, df):
        """Prepare features for ML model"""
        # Create a copy to avoid modifying original
        data = df.copy()

        # Convert date to datetime and extract features
        data['date'] = pd.to_datetime(data['date'])
        data['month'] = data['date'].dt.month
        data['day_of_year'] = data['date'].dt.dayofyear
        data['is_weekend'] = (data['date'].dt.dayofweek >= 5).astype(int)

        # Encode categorical variables with error handling
        categorical_columns = ['season', 'weather_condition']
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                data[f'{col}_encoded'] = self.label_encoders[col].fit_transform(data[col])
            else:
                # Handle unknown categories by using the most common one (first in classes_)
                try:
                    data[f'{col}_encoded'] = self.label_encoders[col].transform(data[col])
                except ValueError:
                    # If unknown category, use the first (most common) category
                    data[col] = data[col].apply(
                        lambda x: x if x in self.label_encoders[col].classes_ else self.label_encoders[col].classes_[0])
                    data[f'{col}_encoded'] = self.label_encoders[col].transform(data[col])

        # Select features for training
        feature_columns = [
            'temperature_f', 'month', 'day_of_year', 'is_weekend',
            'season_encoded', 'weather_condition_encoded',
            'mountain_bike_capacity_utilization', 'ebike_capacity_utilization'
        ]

        return data[feature_columns]

    def train(self, csv_file_path):
        """Train the pricing models"""
        # Load data
        df = pd.read_csv(csv_file_path)

        # Prepare features
        X = self.prepare_features(df)

        # Target variables (dynamic prices)
        y_mountain = df['mountain_bike_dynamic_price']
        y_ebike = df['ebike_dynamic_price']

        # Train mountain bike model
        X_train_mb, X_test_mb, y_train_mb, y_test_mb = train_test_split(
            X, y_mountain, test_size=0.2, random_state=42
        )
        self.mountain_bike_model.fit(X_train_mb, y_train_mb)

        # Train e-bike model
        X_train_eb, X_test_eb, y_train_eb, y_test_eb = train_test_split(
            X, y_ebike, test_size=0.2, random_state=42
        )
        self.ebike_model.fit(X_train_eb, y_train_eb)

        # Calculate accuracy
        mb_pred = self.mountain_bike_model.predict(X_test_mb)
        eb_pred = self.ebike_model.predict(X_test_eb)

        mb_mae = mean_absolute_error(y_test_mb, mb_pred)
        eb_mae = mean_absolute_error(y_test_eb, eb_pred)

        self.is_trained = True

        print(f"Model trained successfully!")
        print(f"Mountain Bike MAE: ${mb_mae:.2f}")
        print(f"E-Bike MAE: ${eb_mae:.2f}")

        return {
            'mountain_bike_mae': mb_mae,
            'ebike_mae': eb_mae
        }

    def predict_price(self, temperature, month, day_of_year, is_weekend,
                      season, weather_condition, mb_capacity, eb_capacity):
        """Predict dynamic prices for given conditions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        try:
            # Create input dataframe
            input_data = pd.DataFrame({
                'date': [f'2024-{month:02d}-15'],  # Use a dummy date for the month
                'temperature_f': [temperature],
                'month': [month],
                'day_of_year': [day_of_year],
                'is_weekend': [is_weekend],
                'season': [season],
                'weather_condition': [weather_condition],
                'mountain_bike_capacity_utilization': [mb_capacity],
                'ebike_capacity_utilization': [eb_capacity]
            })

            # Prepare features
            X = self.prepare_features(input_data)

            # Make predictions
            mb_price = self.mountain_bike_model.predict(X)[0]
            eb_price = self.ebike_model.predict(X)[0]

            return {
                'mountain_bike_price': max(20, round(mb_price, 0)),  # Minimum $20
                'ebike_price': max(30, round(eb_price, 0))  # Minimum $30
            }
        except Exception as e:
            raise ValueError(f"Prediction error: {str(e)}")

    def get_feature_importance(self):
        """Get feature importance from trained models"""
        if not self.is_trained:
            return None

        feature_names = [
            'Temperature', 'Month', 'Day of Year', 'Is Weekend',
            'Season', 'Weather Condition', 'Mountain Bike Capacity', 'E-Bike Capacity'
        ]

        mb_importance = self.mountain_bike_model.feature_importances_
        eb_importance = self.ebike_model.feature_importances_

        return {
            'features': feature_names,
            'mountain_bike_importance': mb_importance,
            'ebike_importance': eb_importance
        }

    def save_model(self, filepath='pricing_model.pkl'):
        """Save trained model"""
        if self.is_trained:
            joblib.dump({
                'mountain_bike_model': self.mountain_bike_model,
                'ebike_model': self.ebike_model,
                'label_encoders': self.label_encoders,
                'is_trained': self.is_trained
            }, filepath)
            print(f"Model saved to {filepath}")

    def load_model(self, filepath='pricing_model.pkl'):
        """Load trained model"""
        if os.path.exists(filepath):
            data = joblib.load(filepath)
            self.mountain_bike_model = data['mountain_bike_model']
            self.ebike_model = data['ebike_model']
            self.label_encoders = data['label_encoders']
            self.is_trained = data['is_trained']
            print(f"Model loaded from {filepath}")
            return True
        return False


def calculate_revenue_impact(csv_file_path):
    """Calculate overall revenue impact from historical data"""
    df = pd.read_csv(csv_file_path)

    total_fixed = df['total_fixed_revenue'].sum()
    total_dynamic = df['total_dynamic_revenue'].sum()
    increase = total_dynamic - total_fixed
    increase_percent = (increase / total_fixed) * 100

    return {
        'total_fixed_revenue': total_fixed,
        'total_dynamic_revenue': total_dynamic,
        'additional_revenue': increase,
        'increase_percentage': increase_percent,
        'average_daily_increase': increase / len(df)
    }