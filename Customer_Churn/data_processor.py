import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class DataProcessor:
    """
    This class handles all data cleaning and preprocessing tasks.
    Think of it as your data janitor - it takes messy real-world data
    and makes it clean and ready for machine learning.
    """

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []

    def clean_data(self, df):
        """
        Clean the raw data - fix data types, handle missing values, etc.

        Args:
            df: Raw dataframe from CSV

        Returns:
            Cleaned dataframe
        """
        print("🧹 Starting data cleaning...")

        # Make a copy so we don't modify the original
        df_clean = df.copy()

        # Fix the TotalCharges column - it's stored as text but should be numbers
        # Some values are empty strings, so we need to handle those
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

        # Fill missing TotalCharges with 0 (these are usually new customers)
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0)

        # Convert SeniorCitizen to Yes/No for consistency
        df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

        # Remove customerID - it's just a unique identifier, not useful for prediction
        if 'customerID' in df_clean.columns:
            df_clean = df_clean.drop('customerID', axis=1)

        print(f"✅ Data cleaning complete. Shape: {df_clean.shape}")
        return df_clean

    def prepare_features(self, df):
        """
        Convert categorical variables to numbers that ML algorithms can understand.

        This is called "encoding" - we turn text like "Yes"/"No" into numbers like 1/0.

        Args:
            df: Cleaned dataframe

        Returns:
            DataFrame ready for machine learning
        """
        print("🔄 Preparing features for machine learning...")

        df_processed = df.copy()

        # Separate target variable (what we're predicting)
        target = df_processed['Churn']
        features = df_processed.drop('Churn', axis=1)

        # Convert categorical features to numbers
        categorical_columns = features.select_dtypes(include=['object']).columns

        for column in categorical_columns:
            # Create a label encoder for this column
            le = LabelEncoder()
            features[column] = le.fit_transform(features[column])
            # Save the encoder so we can use it later for new predictions
            self.label_encoders[column] = le

        # Convert target variable (Churn) to numbers: Yes=1, No=0
        target_encoder = LabelEncoder()
        target_encoded = target_encoder.fit_transform(target)
        self.label_encoders['Churn'] = target_encoder

        # Store feature column names for later use
        self.feature_columns = features.columns.tolist()

        print(f"✅ Feature preparation complete. Features: {len(features.columns)}")
        return features, target_encoded

    def scale_features(self, X_train, X_test=None):
        """
        Scale features so they're all on the same scale.

        Why? MonthlyCharges might be 20-100, but tenure might be 1-70.
        We want to make sure no single feature dominates just because
        it has bigger numbers.

        Args:
            X_train: Training features
            X_test: Test features (optional)

        Returns:
            Scaled features
        """
        print("📏 Scaling features...")

        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)

        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            print("✅ Feature scaling complete")
            return X_train_scaled, X_test_scaled

        print("✅ Feature scaling complete")
        return X_train_scaled

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.

        Why? We train the model on some data, then test it on data
        it has never seen before. This tells us how well it will work
        on new customers.

        Args:
            X: Features
            y: Target variable
            test_size: What percentage to use for testing (20% is common)
            random_state: For reproducible results

        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"🔀 Splitting data: {100 - test_size * 100}% train, {test_size * 100}% test")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"✅ Split complete. Training: {len(X_train)}, Testing: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    def process_pipeline(self, df):
        """
        Complete data processing pipeline.

        This runs all the steps in the right order:
        1. Clean data
        2. Prepare features
        3. Split data
        4. Scale features

        Args:
            df: Raw dataframe

        Returns:
            X_train_scaled, X_test_scaled, y_train, y_test
        """
        print("🚀 Starting complete data processing pipeline...")

        # Step 1: Clean data
        df_clean = self.clean_data(df)

        # Step 2: Prepare features
        X, y = self.prepare_features(df_clean)

        # Step 3: Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Step 4: Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        print("🎉 Data processing pipeline complete!")
        return X_train_scaled, X_test_scaled, y_train, y_test

    def get_feature_names(self):
        """
        Get the names of all features after processing.
        Useful for understanding what the model is using.
        """
        return self.feature_columns

    def inverse_transform_target(self, y_encoded):
        """
        Convert predictions back to original labels (1 -> 'Yes', 0 -> 'No').
        """
        return self.label_encoders['Churn'].inverse_transform(y_encoded)