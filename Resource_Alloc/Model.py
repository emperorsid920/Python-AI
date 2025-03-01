import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ResourceAllocationModel(nn.Module):
    def __init__(self, input_dim, hidden_layers=[64, 32, 16]):
        """
        Neural network for resource allocation optimization.

        Args:
        input_dim (int): Number of input features
        hidden_layers (list): Dimensions of hidden layers
        """
        super().__init__()  # Correct super() call

        # Adaptive feature layers
        self.input_layer = nn.Linear(input_dim, hidden_layers[0])

        # Hidden layers with adaptive regularization
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_layers[i], hidden_layers[i + 1]),
                nn.BatchNorm1d(hidden_layers[i + 1]),
                nn.ReLU(),
                nn.Dropout(0.2)
            ) for i in range(len(hidden_layers) - 1)
        ])

        # Output layer for optimization score
        self.output_layer = nn.Linear(hidden_layers[-1], 1)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights with careful scaling."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass through the network."""
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def train_model(self, X_train, y_train, epochs=100, learning_rate=0.001):
        """
        Train the resource allocation optimization model.

        Args:
        X_train (torch.Tensor): Training input features
        y_train (torch.Tensor): Training target values
        epochs (int): Number of training epochs
        learning_rate (float): Optimizer learning rate
        """
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )

        # Training loop
        for epoch in range(epochs):
            # Forward pass
            outputs = self(X_train)
            loss = criterion(outputs.squeeze(), y_train)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Adjust learning rate
            scheduler.step(loss)

            # Optional: print progress
            if epoch % 20 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    def predict(self, X):
        """
        Make predictions for resource allocation optimization.

        Args:
        X (torch.Tensor): Input scenarios

        Returns:
        torch.Tensor: Predicted optimization scores
        """
        with torch.no_grad():
            return self(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.

        Args:
        X_test (torch.Tensor): Test input features
        y_test (torch.Tensor): Test target values

        Returns:
        float: Mean squared error
        """
        with torch.no_grad():
            predictions = self(X_test)
            loss = F.mse_loss(predictions.squeeze(), y_test)
        return loss.item()