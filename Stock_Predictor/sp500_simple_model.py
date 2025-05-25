import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import yfinance as yf
from data_preparation import fetch_sp500_data_from_web, create_manual_sp500_data, preprocess_data


# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)

# ============================================================================
# 1. PREPROCESSING THE DATA
# ============================================================================
print("=" * 60)
print("1. PREPROCESSING THE DATA")
print("=" * 60)

# Configuration
STOCK_SYMBOL = 'S&P 500'  # We're using S&P 500 index data

print(f"Fetching {STOCK_SYMBOL} data...")
# Try to fetch from web first, fallback to manual data
try:
    raw_data = fetch_sp500_data_from_web()
except:
    raw_data = create_manual_sp500_data()

print(f"Successfully loaded {len(raw_data)} days of data")
print(f"Date range: {raw_data['date'].min().date()} to {raw_data['date'].max().date()}")

print("Preprocessing data and creating technical indicators...")
data = preprocess_data(raw_data, sequence_length=60)

print("Data preprocessing complete!")
print(f"Training samples: {data['X_train'].shape[0]}")
print(f"Validation samples: {data['X_val'].shape[0]}")
print(f"Test samples: {data['X_test'].shape[0]}")
print(f"Features: {data['X_train'].shape[2]}")
print(f"Sequence length: {data['X_train'].shape[1]}")

# Show basic statistics
print(f"\nStock Price Statistics:")
print(f"Current Price: ${raw_data['close'].iloc[-1]:.2f}")
print(f"Min Price: ${raw_data['close'].min():.2f}")
print(f"Max Price: ${raw_data['close'].max():.2f}")
print(f"Average Daily Return: {data['raw_data']['price_change_pct'].mean():.4f}%")
print(f"Daily Volatility: {data['raw_data']['price_change_pct'].std():.4f}%")

# ============================================================================
# 2. BUILD THE MODEL WITH TENSORFLOW LAYERS
# ============================================================================
print("\n" + "=" * 60)
print("2. BUILDING THE MODEL")
print("=" * 60)

# Model architecture using TensorFlow layers
model = keras.Sequential([
    # LSTM layers
    layers.LSTM(50, return_sequences=True, input_shape=(data['X_train'].shape[1], data['X_train'].shape[2])),
    layers.Dropout(0.2),
    layers.LSTM(50, return_sequences=False),
    layers.Dropout(0.2),

    # Dense layers
    layers.Dense(25, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='linear')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("Model architecture:")
model.summary()

# ============================================================================
# 3. TRAIN THE MODEL
# ============================================================================
print("\n" + "=" * 60)
print("3. TRAINING THE MODEL")
print("=" * 60)

# Define callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7,
    verbose=1
)

checkpoint = keras.callbacks.ModelCheckpoint(
    f'best_{STOCK_SYMBOL}_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Train the model
print("Starting training...")
history = model.fit(
    data['X_train'], data['y_train'],
    epochs=100,
    batch_size=32,
    validation_data=(data['X_val'], data['y_val']),
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

print("Training completed!")

# ============================================================================
# 4. EVALUATE & PREDICT
# ============================================================================
print("\n" + "=" * 60)
print("4. EVALUATION & PREDICTION")
print("=" * 60)

# Make predictions
print("Making predictions...")
train_pred = model.predict(data['X_train'], verbose=0).flatten()
val_pred = model.predict(data['X_val'], verbose=0).flatten()
test_pred = model.predict(data['X_test'], verbose=0).flatten()

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(data['y_train'], train_pred))
val_rmse = np.sqrt(mean_squared_error(data['y_val'], val_pred))
test_rmse = np.sqrt(mean_squared_error(data['y_test'], test_pred))

train_mae = mean_absolute_error(data['y_train'], train_pred)
val_mae = mean_absolute_error(data['y_val'], val_pred)
test_mae = mean_absolute_error(data['y_test'], test_pred)

test_r2 = r2_score(data['y_test'], test_pred)

# Directional accuracy
actual_direction = np.sign(data['y_test'])
pred_direction = np.sign(test_pred)
directional_accuracy = np.mean(actual_direction == pred_direction)

print(f"MODEL PERFORMANCE FOR {STOCK_SYMBOL}:")
print(f"Training RMSE: {train_rmse:.4f}%")
print(f"Validation RMSE: {val_rmse:.4f}%")
print(f"Test RMSE: {test_rmse:.4f}%")
print(f"Test MAE: {test_mae:.4f}%")
print(f"Test R² Score: {test_r2:.4f}")
print(f"Directional Accuracy: {directional_accuracy:.4f} ({directional_accuracy:.1%})")

# ============================================================================
# 5. DISPLAY GRAPHS AND VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 60)
print("5. VISUALIZATIONS")
print("=" * 60)

# Plot 1: Stock Price History
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(raw_data['date'], raw_data['close'])
plt.title(f'{STOCK_SYMBOL} Stock Price History')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(raw_data['date'], raw_data['volume'])
plt.title(f'{STOCK_SYMBOL} Volume History')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
daily_returns = raw_data['close'].pct_change() * 100
plt.hist(daily_returns.dropna(), bins=50, alpha=0.7, edgecolor='black')
plt.title(f'{STOCK_SYMBOL} Daily Returns Distribution')
plt.xlabel('Daily Return %')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(raw_data['date'], daily_returns)
plt.title(f'{STOCK_SYMBOL} Daily Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Daily Return %')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot 2: Training History
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE Over Time')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot 3: Actual vs Predicted (Time Series)
plt.figure(figsize=(15, 10))

# Full test set
plt.subplot(2, 1, 1)
test_dates = data['dates'][-len(data['y_test']):]
plt.plot(test_dates, data['y_test'], label='Actual', alpha=0.8, linewidth=1)
plt.plot(test_dates, test_pred, label='Predicted', alpha=0.8, linewidth=1)
plt.title(f'{STOCK_SYMBOL} Price Movement Prediction - Full Test Set')
plt.xlabel('Date')
plt.ylabel('Price Change %')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Last 50 days for clarity
plt.subplot(2, 1, 2)
n_recent = 50
recent_actual = data['y_test'][-n_recent:]
recent_pred = test_pred[-n_recent:]
recent_dates = test_dates[-n_recent:]

plt.plot(recent_dates, recent_actual, label='Actual', marker='o', markersize=3, alpha=0.8)
plt.plot(recent_dates, recent_pred, label='Predicted', marker='x', markersize=3, alpha=0.8)
plt.title(f'{STOCK_SYMBOL} Price Movement Prediction - Last {n_recent} Days')
plt.xlabel('Date')
plt.ylabel('Price Change %')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Plot 4: Scatter Plot - Actual vs Predicted
plt.figure(figsize=(10, 8))
plt.scatter(data['y_test'], test_pred, alpha=0.6, s=20)
plt.plot([data['y_test'].min(), data['y_test'].max()],
         [data['y_test'].min(), data['y_test'].max()], 'r--', lw=2)
plt.xlabel('Actual Price Change %')
plt.ylabel('Predicted Price Change %')
plt.title(f'{STOCK_SYMBOL} - Actual vs Predicted Values')
plt.grid(True, alpha=0.3)
plt.text(0.05, 0.95, f'R² = {test_r2:.3f}', transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
plt.show()

# Plot 5: Error Analysis
plt.figure(figsize=(15, 5))

errors = data['y_test'] - test_pred

plt.subplot(1, 3, 1)
plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
plt.title('Prediction Error Distribution')
plt.xlabel('Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.boxplot(errors)
plt.title('Prediction Error Box Plot')
plt.ylabel('Error (Actual - Predicted)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(test_dates, errors, alpha=0.7)
plt.title('Prediction Errors Over Time')
plt.xlabel('Date')
plt.ylabel('Error')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot 6: Feature Importance (Permutation Test)
print("Calculating feature importance...")
baseline_loss = mean_squared_error(data['y_test'], test_pred)
importance_scores = []

for i, feature in enumerate(data['feature_names']):
    # Permute one feature
    X_test_permuted = data['X_test'].copy()
    np.random.shuffle(X_test_permuted[:, :, i])

    # Get new predictions
    permuted_pred = model.predict(X_test_permuted, verbose=0).flatten()
    permuted_loss = mean_squared_error(data['y_test'], permuted_pred)

    # Calculate importance
    importance = permuted_loss - baseline_loss
    importance_scores.append(importance)

# Create importance DataFrame
importance_df = pd.DataFrame({
    'feature': data['feature_names'],
    'importance': importance_scores
}).sort_values('importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = importance_df.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance (Increase in MSE)')
plt.title(f'Top 15 Feature Importance for {STOCK_SYMBOL}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 7: Prediction vs Price Movement
plt.figure(figsize=(15, 8))

# Convert percentage predictions to price predictions
recent_prices = data['raw_data']['close'][-len(data['y_test']):].values
predicted_prices = recent_prices * (1 + test_pred / 100)
actual_next_prices = recent_prices * (1 + data['y_test'] / 100)

plt.subplot(2, 1, 1)
plt.plot(test_dates, recent_prices, label='Current Price', linewidth=2)
plt.plot(test_dates, predicted_prices, label='Predicted Next Day Price', alpha=0.8)
plt.plot(test_dates, actual_next_prices, label='Actual Next Day Price', alpha=0.8)
plt.title(f'{STOCK_SYMBOL} Price Predictions vs Reality')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Last 30 days for detail
plt.subplot(2, 1, 2)
n_recent = 30
plt.plot(test_dates[-n_recent:], recent_prices[-n_recent:], label='Current Price', linewidth=2)
plt.plot(test_dates[-n_recent:], predicted_prices[-n_recent:], label='Predicted Next Day', marker='o', markersize=3)
plt.plot(test_dates[-n_recent:], actual_next_prices[-n_recent:], label='Actual Next Day', marker='x', markersize=3)
plt.title(f'{STOCK_SYMBOL} Last {n_recent} Days - Detailed View')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ============================================================================
# 6. CONCLUSIONS FROM PREDICTIONS
# ============================================================================
print("\n" + "=" * 60)
print("6. CONCLUSIONS FROM PREDICTIONS")
print("=" * 60)

print(f"PERFORMANCE ANALYSIS FOR {STOCK_SYMBOL}:")
print("-" * 50)

if test_rmse < 1.5:
    performance_level = "EXCELLENT"
elif test_rmse < 2.0:
    performance_level = "GOOD"
elif test_rmse < 3.0:
    performance_level = "FAIR"
else:
    performance_level = "POOR"

print(f"Overall Model Performance: {performance_level}")
print(f"The model achieves an RMSE of {test_rmse:.4f}%, meaning on average,")
print(f"predictions deviate by {test_rmse:.4f} percentage points from actual returns.")

print(f"\nDirectional Accuracy: {directional_accuracy:.1%}")
if directional_accuracy > 0.6:
    print("✓ The model shows STRONG ability to predict market direction")
elif directional_accuracy > 0.55:
    print("✓ The model shows MODERATE ability to predict market direction")
else:
    print("✗ The model struggles with directional prediction")

print(f"\nR² Score: {test_r2:.4f}")
if test_r2 > 0.3:
    print("✓ The model explains a SIGNIFICANT portion of price variance")
elif test_r2 > 0.1:
    print("✓ The model explains a MODERATE portion of price variance")
else:
    print("✗ The model explains LIMITED price variance")

print("\nKEY INSIGHTS:")
print("-" * 50)

# Top 5 most important features
top_5_features = importance_df.head(5)
print(f"Most Important Features for {STOCK_SYMBOL} Prediction:")
for i, (_, row) in enumerate(top_5_features.iterrows(), 1):
    print(f"{i}. {row['feature']} (importance: {row['importance']:.6f})")

# Recent performance analysis
recent_predictions = test_pred[-30:]  # Last 30 predictions
recent_actual = data['y_test'][-30:]

positive_pred_accuracy = np.mean((recent_predictions > 0) == (recent_actual > 0))
negative_pred_accuracy = np.mean((recent_predictions < 0) == (recent_actual < 0))

print(f"\nRecent Prediction Patterns (Last 30 days):")
print(f"Accuracy predicting UP days: {positive_pred_accuracy:.1%}")
print(f"Accuracy predicting DOWN days: {negative_pred_accuracy:.1%}")

# Volatility insights
high_volatility_mask = np.abs(data['y_test']) > 2  # Days with >2% moves
if np.sum(high_volatility_mask) > 0:
    high_vol_accuracy = np.mean(
        np.sign(test_pred[high_volatility_mask]) == np.sign(data['y_test'][high_volatility_mask])
    )
    print(f"Accuracy on high volatility days (>2% moves): {high_vol_accuracy:.1%}")

# Price range analysis
current_price = raw_data['close'].iloc[-1]
avg_predicted_change = np.mean(test_pred[-10:])  # Average of last 10 predictions
predicted_next_price = current_price * (1 + avg_predicted_change / 100)

print(f"\nCURRENT MARKET ANALYSIS:")
print(f"Current {STOCK_SYMBOL} Price: ${current_price:.2f}")
print(f"Recent Average Predicted Change: {avg_predicted_change:.4f}%")
print(f"Implied Next Day Price: ${predicted_next_price:.2f}")

print(f"\nTRADING IMPLICATIONS FOR {STOCK_SYMBOL}:")
print("-" * 50)

if directional_accuracy > 0.6 and test_rmse < 2.0:
    print("✓ Model suitable for SHORT-TERM trading strategies")
    print("✓ Consider using for day trading or swing trading")
    print("✓ Focus on directional bets rather than precise price targets")
elif directional_accuracy > 0.55:
    print("⚠ Model shows promise but use with caution")
    print("⚠ Consider combining with other indicators")
else:
    print("✗ Model may not be reliable for active trading")
    print("✗ Consider longer-term investment strategies")

# Risk assessment
error_volatility = np.std(errors)
print(f"\nRISK ASSESSMENT:")
print(f"Prediction error volatility: {error_volatility:.4f}%")
print(f"95% confidence interval: ±{1.96 * error_volatility:.4f}%")

# Calculate potential profit/loss
daily_vol = data['raw_data']['price_change_pct'].std()
sharpe_like_ratio = (np.mean(test_pred) / np.std(test_pred)) if np.std(test_pred) > 0 else 0
print(f"Signal-to-Noise Ratio: {sharpe_like_ratio:.4f}")

print(f"\nRECOMMENDATIONS FOR {STOCK_SYMBOL}:")
print("-" * 50)
print("1. Use model predictions as ONE input in trading decisions")
print("2. Combine with fundamental analysis and market sentiment")
print("3. Implement proper risk management and position sizing")
print("4. Regularly retrain model with new data")
print("5. Monitor model performance in live trading")

if STOCK_SYMBOL in ['SPY', 'QQQ', 'IWM']:
    print("6. ETF trading: Consider broader market conditions")
elif STOCK_SYMBOL in ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']:
    print("6. Individual stock: Monitor company-specific news and earnings")

# Save results
results_df = pd.DataFrame({
    'date': data['dates'][-len(data['y_test']):],
    'actual_return': data['y_test'],
    'predicted_return': test_pred,
    'error': errors,
    'current_price': recent_prices,
    'predicted_price': predicted_prices,
    'actual_next_price': actual_next_prices
})

results_df.to_csv(f'{STOCK_SYMBOL}_predictions.csv', index=False)
print(f"\nResults saved to '{STOCK_SYMBOL}_predictions.csv'")
print(f"Model saved to 'best_{STOCK_SYMBOL}_model.h5'")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)

# Final prediction for tomorrow
print(f"\nFINAL PREDICTION FOR NEXT TRADING DAY:")
print(f"Current {STOCK_SYMBOL} price: ${current_price:.2f}")
print(f"Predicted change: {avg_predicted_change:+.4f}%")
print(f"Predicted price: ${predicted_next_price:.2f}")
if avg_predicted_change > 0:
    print("📈 Model suggests BULLISH sentiment")
else:
    print("📉 Model suggests BEARISH sentiment")
print("⚠️  Remember: This is for educational purposes only, not financial advice!")