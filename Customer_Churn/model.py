import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


class ChurnPredictor:
    """
    This class handles the machine learning model for churn prediction.

    Think of it as your crystal ball - it learns from past customer behavior
    to predict who might leave in the future.
    """

    def __init__(self, model_type='random_forest'):
        """
        Initialize the model.

        Args:
            model_type: 'random_forest' or 'logistic_regression'
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.feature_importance = None

        # Choose which algorithm to use
        if model_type == 'random_forest':
            # Random Forest: Like asking 100 experts and taking the majority vote
            self.model = RandomForestClassifier(
                n_estimators=100,  # Use 100 decision trees
                max_depth=10,  # Don't make trees too deep (prevents overfitting)
                random_state=42  # For reproducible results
            )
        elif model_type == 'logistic_regression':
            # Logistic Regression: Finds the best line to separate churners from non-churners
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000  # Give it enough time to find the best solution
            )

    def train(self, X_train, y_train, X_test, y_test):
        """
        Train the model on your data.

        This is where the magic happens - the model learns patterns
        from your historical data.

        Args:
            X_train: Training features (customer characteristics)
            y_train: Training labels (did they churn?)
            X_test: Test features (to evaluate performance)
            y_test: Test labels (to check predictions)

        Returns:
            Dictionary with training results
        """
        print(f"🤖 Training {self.model_type} model...")

        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Get feature importance (what factors matter most?)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_

        # Make predictions on test set
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]  # Probability of churn

        # Calculate performance metrics
        results = self._calculate_metrics(y_test, y_pred, y_pred_proba)

        print(f"✅ Model training complete!")
        print(f"   Accuracy: {results['accuracy']:.1%}")
        print(f"   Precision: {results['precision']:.1%}")
        print(f"   Recall: {results['recall']:.1%}")

        return results

    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """
        Calculate various performance metrics.

        These metrics tell us how good our model is:
        - Accuracy: Overall correct predictions
        - Precision: Of predicted churners, how many actually churned?
        - Recall: Of actual churners, how many did we catch?
        - F1: Balance between precision and recall
        - AUC: How well can we separate churners from non-churners?
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred)
        }

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Customer features

        Returns:
            Predictions (0 = won't churn, 1 = will churn)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")

        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Get prediction probabilities.

        Instead of just Yes/No, this gives you the confidence level.
        For example: "85% chance this customer will churn"

        Args:
            X: Customer features

        Returns:
            Array of probabilities [prob_no_churn, prob_churn]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")

        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names):
        """
        Get which features are most important for predictions.

        This tells you things like:
        - "Contract type is the most important factor"
        - "Monthly charges matter more than gender"

        Args:
            feature_names: List of feature names

        Returns:
            DataFrame with features ranked by importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first!")

        if self.feature_importance is None:
            return None

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)

        return importance_df

    def save_model(self, filepath):
        """
        Save the trained model to a file.

        Args:
            filepath: Where to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving!")

        joblib.dump(self.model, filepath)
        print(f"💾 Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load a previously trained model.

        Args:
            filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"📁 Model loaded from {filepath}")


class BusinessMetrics:
    """
    Calculate business metrics from predictions.

    This translates ML predictions into business language:
    - How much revenue is at risk?
    - Which customers should we focus on?
    - What's the ROI of our retention efforts?
    """

    def __init__(self, customer_data):
        """
        Initialize with customer data.

        Args:
            customer_data: DataFrame with customer info including MonthlyCharges
        """
        self.customer_data = customer_data

    def calculate_revenue_at_risk(self, churn_predictions, monthly_charges):
        """
        Calculate total monthly revenue at risk from predicted churners.

        Args:
            churn_predictions: Array of churn predictions (0 or 1)
            monthly_charges: Array of monthly charges for each customer

        Returns:
            Total revenue at risk
        """
        revenue_at_risk = np.sum(monthly_charges[churn_predictions == 1])
        return revenue_at_risk

    def segment_customers(self, churn_probabilities, monthly_charges):
        """
        Segment customers by risk level and value.

        Creates segments like:
        - High Risk, High Value: Immediate attention needed
        - High Risk, Low Value: Automated retention
        - Low Risk, High Value: Keep happy
        - Low Risk, Low Value: Monitor

        Args:
            churn_probabilities: Array of churn probabilities
            monthly_charges: Array of monthly charges

        Returns:
            Dictionary with customer segments
        """
        # Define thresholds
        high_risk_threshold = 0.7  # 70% churn probability
        high_value_threshold = np.percentile(monthly_charges, 75)  # Top 25% of spenders

        # Create segments
        segments = {}

        # High Risk, High Value - Critical customers
        high_risk_high_value = (churn_probabilities >= high_risk_threshold) & (monthly_charges >= high_value_threshold)
        segments['Critical'] = {
            'count': np.sum(high_risk_high_value),
            'revenue_at_risk': np.sum(monthly_charges[high_risk_high_value]),
            'action': 'Personal outreach, special offers'
        }

        # High Risk, Low Value - Automated retention
        high_risk_low_value = (churn_probabilities >= high_risk_threshold) & (monthly_charges < high_value_threshold)
        segments['At Risk'] = {
            'count': np.sum(high_risk_low_value),
            'revenue_at_risk': np.sum(monthly_charges[high_risk_low_value]),
            'action': 'Automated email campaigns'
        }

        # Low Risk, High Value - Keep satisfied
        low_risk_high_value = (churn_probabilities < high_risk_threshold) & (monthly_charges >= high_value_threshold)
        segments['VIP'] = {
            'count': np.sum(low_risk_high_value),
            'revenue_at_risk': np.sum(monthly_charges[low_risk_high_value]),
            'action': 'Regular check-ins, loyalty programs'
        }

        # Low Risk, Low Value - Monitor
        low_risk_low_value = (churn_probabilities < high_risk_threshold) & (monthly_charges < high_value_threshold)
        segments['Stable'] = {
            'count': np.sum(low_risk_low_value),
            'revenue_at_risk': np.sum(monthly_charges[low_risk_low_value]),
            'action': 'Standard service, upsell opportunities'
        }

        return segments

    def calculate_retention_roi(self, revenue_at_risk, retention_cost_per_customer, success_rate=0.3):
        """
        Calculate ROI of retention efforts.

        Args:
            revenue_at_risk: Total revenue that could be lost
            retention_cost_per_customer: Cost of retention campaign per customer
            success_rate: Expected success rate of retention (30% is typical)

        Returns:
            ROI calculation
        """
        customers_at_risk = len(self.customer_data)
        total_retention_cost = customers_at_risk * retention_cost_per_customer
        revenue_saved = revenue_at_risk * success_rate

        roi = (revenue_saved - total_retention_cost) / total_retention_cost

        return {
            'total_retention_cost': total_retention_cost,
            'revenue_saved': revenue_saved,
            'roi': roi,
            'roi_percentage': roi * 100
        }