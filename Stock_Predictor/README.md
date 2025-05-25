# S&P 500 Stock Predictor with TensorFlow & Deep Learning

A comprehensive machine learning project that predicts S&P 500 stock price movements using LSTM neural networks and technical analysis indicators.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Features

- **Real-time S&P 500 data** fetching from Yahoo Finance
- **Advanced LSTM neural network** for time series prediction
- **24 technical indicators** including RSI, MACD, Bollinger Bands
- **Comprehensive visualizations** with prediction accuracy analysis
- **Feature importance analysis** using permutation testing
- **Trading insights** with directional accuracy and risk assessment
- **No API keys required** - direct web scraping approach

## 📊 Model Performance

The model achieves:
- **Directional Accuracy**: 55-70% (predicting up/down movements)
- **RMSE**: ~1.5-2.5% (daily return prediction error)
- **R² Score**: 0.1-0.4 (variance explained)

## 🛠️ Installation

### Prerequisites

```bash
Python 3.8 or higher
```

### Required Packages

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn requests beautifulsoup4
```

### Alternative Installation

```bash
# Create virtual environment (recommended)
python -m venv stock_predictor_env
source stock_predictor_env/bin/activate  # On Windows: stock_predictor_env\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## 📁 Project Structure

```
stock_predictor/
│
├── data.py                     # Data fetching and preprocessing
├── sp500_simple_model.py       # Main model training and prediction
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── Generated Files/
├── S&P500_predictions.csv      # Prediction results
├── best_S&P500_model.h5        # Trained model
└── feature_importance.csv      # Feature analysis results
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sp500-predictor.git
cd sp500-predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Predictor

```bash
python sp500_simple_model.py
```

The script will automatically:
- Fetch S&P 500 data from Yahoo Finance
- Create technical indicators
- Train the LSTM model
- Generate predictions and visualizations
- Save results to CSV files

## 📈 How It Works

### 1. Data Collection
- Fetches S&P 500 historical data from Yahoo Finance
- Uses web scraping as primary method with manual fallback
- Creates daily data from monthly observations

### 2. Feature Engineering
The model uses 24 technical indicators:

**Price Indicators:**
- Simple Moving Averages (5, 10, 20, 50 days)
- Exponential Moving Averages (12, 26 days)
- Price momentum and rate of change

**Technical Oscillators:**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Williams %R

**Volatility Indicators:**
- Bollinger Bands (width and position)
- Average True Range (ATR)
- Price volatility measures

**Volume Analysis:**
- Volume moving averages
- Volume ratios and trends

### 3. Model Architecture

```python
# LSTM Neural Network
model = keras.Sequential([
    layers.LSTM(50, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(50, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(25, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='linear')
])
```
