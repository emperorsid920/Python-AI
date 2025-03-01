import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def create_and_train_model(X_train, y_train, epochs=100):
    """
    Create and train a neural network for disease progression prediction.

    Args:
    X_train (numpy.ndarray): Training features
    y_train (numpy.ndarray): Training target values
    epochs (int): Number of training epochs

    Returns:
    tf.keras.Model: Trained neural network model
    """
    # Normalize input features
    normalizer = tf.keras.layers.Normalization()
    normalizer.adapt(X_train)

    # Build the model
    model = Sequential([
        normalizer,
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)  # Output layer for regression
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Train the model
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    return model


def predict_disease_progression(model, X_test):
    """
    Make predictions using the trained model.

    Args:
    model (tf.keras.Model): Trained neural network model
    X_test (numpy.ndarray): Test features

    Returns:
    numpy.ndarray: Predicted disease progression scores
    """
    return model.predict(X_test)