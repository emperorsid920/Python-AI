import numpy as np
import torch
import torch.nn.functional as F
from Model import ResourceAllocationModel


def generate_synthetic_resource_data(num_scenarios=1000, num_resources=5):
    """
    Generate synthetic data for resource allocation optimization.

    Parameters represent:
    - Energy sources
    - Water resources
    - Agricultural land
    - Raw materials
    - Labor availability

    Features simulate complex interdependencies and constraints.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    # Simulate resource availability with constraints
    resource_availability = np.random.uniform(50, 500, (num_scenarios, num_resources))

    # Simulate demand and sustainability weights
    demand = np.random.uniform(30, 400, (num_scenarios, num_resources))
    sustainability_weights = np.random.uniform(0.1, 1.0, (num_scenarios, num_resources))

    # Create complex interdependency matrix
    interdependency_matrix = np.random.uniform(-0.5, 0.5, (num_resources, num_resources))
    np.fill_diagonal(interdependency_matrix, 1.0)

    # Repeat interdependency matrix for all scenarios
    interdependency_repeated = np.repeat(
        interdependency_matrix.flatten()[np.newaxis, :],
        num_scenarios,
        axis=0
    )

    # Construct input features
    X = np.concatenate([
        resource_availability,
        demand,
        sustainability_weights,
        interdependency_repeated
    ], axis=1)

    # Generate target: optimal allocation score
    # Combines efficiency, sustainability, and balance
    y = np.sum(
        resource_availability * sustainability_weights /
        (demand + 1e-7),  # Avoid division by zero
        axis=1
    )

    # Normalize target to 0-1 range
    y = (y - y.min()) / (y.max() - y.min())

    return torch.FloatTensor(X), torch.FloatTensor(y)


def main():
    # Generate synthetic resource allocation data
    X, y = generate_synthetic_resource_data()

    # Split data
    split_ratio = 0.8
    split_idx = int(len(X) * split_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Initialize and train the model
    model = ResourceAllocationModel(input_dim=X.shape[1])
    model.train_model(X_train, y_train)

    # Evaluate model performance
    test_loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")

    # Make sample predictions
    print("\nSample Resource Allocation Predictions:")
    sample_scenarios = X_test[:5]
    predictions = model.predict(sample_scenarios)

    for i, (scenario, prediction) in enumerate(zip(sample_scenarios, predictions), 1):
        print(f"\nScenario {i}:")
        print(f"Resource Availability: {scenario[:5].numpy()}")
        print(f"Predicted Optimization Score: {prediction.item():.4f}")


if __name__ == "__main__":
    main()