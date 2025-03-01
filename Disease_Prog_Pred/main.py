import numpy as np
import tensorflow as tf
from Model import create_and_train_model, predict_disease_progression


def generate_synthetic_healthcare_data(num_samples=1000):
    """
    Generate synthetic healthcare dataset for disease progression prediction.

    Features:
    - Age
    - Blood Pressure
    - Cholesterol Level
    - Blood Sugar
    - BMI

    Target: Disease Progression Score (0-10)
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate features
    age = np.random.normal(45, 15, num_samples)
    blood_pressure = np.random.normal(120, 20, num_samples)
    cholesterol = np.random.normal(200, 40, num_samples)
    blood_sugar = np.random.normal(100, 25, num_samples)
    bmi = np.random.normal(25, 5, num_samples)

    # Create feature matrix
    X = np.column_stack([age, blood_pressure, cholesterol, blood_sugar, bmi])

    # Generate target (disease progression score)
    # More complex health indicators lead to higher progression score
    y = (
            0.3 * age / 10 +
            0.2 * (blood_pressure - 120) / 20 +
            0.2 * (cholesterol - 200) / 40 +
            0.1 * (blood_sugar - 100) / 25 +
            0.2 * (bmi - 25) / 5
    )
    # Normalize to 0-10 scale and add some randomness
    y = np.clip(y * 10 + np.random.normal(0, 1, num_samples), 0, 10)

    return X, y


def main():
    # Generate synthetic data
    X, y = generate_synthetic_healthcare_data()

    # Split data into training and testing sets
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    # Create and train the model
    model = create_and_train_model(X_train, y_train)

    # Evaluate the model
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss}")

    # Make some predictions
    sample_patients = X_test[:5]
    predictions = predict_disease_progression(model, sample_patients)

    print("\nSample Patient Predictions:")
    for i, (patient, prediction) in enumerate(zip(sample_patients, predictions), 1):
        print(f"Patient {i} (Age: {patient[0]:.1f}, BP: {patient[1]:.1f}, Chol: {patient[2]:.1f}, "
              f"Sugar: {patient[3]:.1f}, BMI: {patient[4]:.1f}):")
        print(f"Predicted Disease Progression Score: {prediction[0]:.2f}\n")


if __name__ == "__main__":
    main()