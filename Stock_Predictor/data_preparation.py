import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def fetch_sp500_data_from_web():
    """Fetch S&P 500 data directly from Yahoo Finance website"""

    url = "https://finance.yahoo.com/quote/%5EGSPC/history/?frequency=1mo"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        print("Fetching S&P 500 data from Yahoo Finance...")
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the data table
            table = soup.find('table', {'data-test': 'historical-prices'})

            if table is None:
                # Try alternative selector
                table = soup.find('table')

            if table:
                # Extract table data
                rows = table.find_all('tr')[1:]  # Skip header
                data = []

                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 6:
                        try:
                            date_str = cols[0].text.strip()
                            open_price = float(cols[1].text.strip().replace(',', ''))
                            high_price = float(cols[2].text.strip().replace(',', ''))
                            low_price = float(cols[3].text.strip().replace(',', ''))
                            close_price = float(cols[4].text.strip().replace(',', ''))
                            adj_close = float(cols[5].text.strip().replace(',', ''))
                            volume = cols[6].text.strip().replace(',', '') if len(cols) > 6 else '0'

                            # Clean volume data
                            if volume == '-':
                                volume = 0
                            else:
                                volume = float(volume)

                            data.append({
                                'date': date_str,
                                'open': open_price,
                                'high': high_price,
                                'low': low_price,
                                'close': close_price,
                                'adj_close': adj_close,
                                'volume': volume
                            })
                        except (ValueError, IndexError):
                            continue

                if data:
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date').reset_index(drop=True)
                    print(f"Successfully fetched {len(df)} data points from Yahoo Finance")
                    return df

        print("Failed to fetch data from web, using manual data...")
        return create_manual_sp500_data()

    except Exception as e:
        print(f"Error fetching web data: {e}")
        print("Using manual data instead...")
        return create_manual_sp500_data()


def create_manual_sp500_data():
    """Create S&P 500 data based on the screenshot you provided + extended simulation"""

    print("Creating S&P 500 dataset...")

    # Real data from your screenshot (monthly data)
    real_data = {
        'date': [
            '2024-06-01', '2024-07-01', '2024-08-01', '2024-09-01', '2024-10-01',
            '2024-11-01', '2024-12-01', '2025-01-01', '2025-02-01', '2025-03-01',
            '2025-04-01', '2025-05-01', '2025-05-23'
        ],
        'open': [5297.15, 5471.08, 5537.84, 5628.89, 5757.73, 5728.22, 6040.11, 5903.26, 5969.65, 5968.33, 5597.53,
                 5625.14, 5781.89],
        'high': [5523.64, 5669.67, 5651.62, 5767.37, 5878.46, 6044.17, 6099.97, 6128.18, 6147.43, 5986.09, 5695.31,
                 5968.61, 5829.51],
        'low': [5234.32, 5390.95, 5119.26, 5402.62, 5674.00, 5696.51, 5832.30, 5773.31, 5837.66, 5488.73, 4835.04,
                5578.64, 5767.41],
        'close': [5460.48, 5522.30, 5648.40, 5762.48, 5705.45, 6032.38, 5881.63, 6040.53, 5954.50, 5611.85, 5569.06,
                  5802.82, 5802.82],
        'volume': [76025620000, 80160380000, 81097300000, 79564830000, 82412430000, 84101980000, 86064900000,
                   88639380000, 92317000000, 111387270000, 118936380000, 84366540000, 2650252000]
    }

    # Convert to DataFrame
    df_real = pd.DataFrame(real_data)
    df_real['date'] = pd.to_datetime(df_real['date'])

    # Calculate daily returns from monthly data
    monthly_returns = df_real['close'].pct_change().dropna()
    avg_monthly_return = monthly_returns.mean()
    monthly_volatility = monthly_returns.std()

    # Convert to daily estimates
    avg_daily_return = avg_monthly_return / 21  # ~21 trading days per month
    daily_volatility = monthly_volatility / np.sqrt(21)

    # Generate 2 years of daily data ending at our real data start
    np.random.seed(42)
    start_date = pd.to_datetime('2022-06-01')
    end_date = df_real['date'].min()

    # Create daily date range (weekdays only)
    date_range = pd.bdate_range(start=start_date, end=end_date, freq='B')[:-1]  # Exclude last day to avoid overlap

    # Generate synthetic daily data
    n_days = len(date_range)
    daily_returns = np.random.normal(avg_daily_return, daily_volatility, n_days)

    # Start with a reasonable S&P 500 price from 2022
    initial_price = 4000.0
    prices = [initial_price]

    for i in range(n_days):
        next_price = prices[-1] * (1 + daily_returns[i])
        prices.append(next_price)

    prices = np.array(prices[1:])  # Remove initial price

    # Generate OHLCV data
    opens = prices.copy()

    # Generate realistic intraday movements
    high_multipliers = 1 + np.abs(np.random.normal(0, 0.005, n_days))
    low_multipliers = 1 - np.abs(np.random.normal(0, 0.005, n_days))

    highs = prices * high_multipliers
    lows = prices * low_multipliers

    # Ensure highs >= prices >= lows
    highs = np.maximum(highs, prices)
    lows = np.minimum(lows, prices)

    # Generate volume (based on average from real data)
    avg_volume = df_real['volume'].mean()
    volumes = np.random.normal(avg_volume, avg_volume * 0.3, n_days)
    volumes = np.abs(volumes)  # Ensure positive

    # Create synthetic DataFrame
    df_synthetic = pd.DataFrame({
        'date': date_range,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })

    # Convert real monthly data to daily by interpolation
    df_real_daily = []
    for i in range(len(df_real) - 1):
        start_date = df_real.iloc[i]['date']
        end_date = df_real.iloc[i + 1]['date']
        start_price = df_real.iloc[i]['close']
        end_price = df_real.iloc[i + 1]['open']

        # Create daily dates between monthly points
        daily_dates = pd.bdate_range(start=start_date, end=end_date, freq='B')[:-1]

        if len(daily_dates) > 0:
            # Interpolate prices
            price_steps = (end_price - start_price) / len(daily_dates)

            for j, date in enumerate(daily_dates):
                interpolated_price = start_price + (price_steps * j)

                # Add some realistic daily variation
                daily_change = np.random.normal(0, daily_volatility)
                adjusted_price = interpolated_price * (1 + daily_change)

                # Generate OHLCV
                open_price = interpolated_price if j == 0 else df_real_daily[-1]['close']
                close_price = adjusted_price
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.003)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.003)))
                volume = np.random.normal(avg_volume, avg_volume * 0.2)

                df_real_daily.append({
                    'date': date,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': abs(volume)
                })

    # Add the most recent real data point
    df_real_daily.append({
        'date': df_real.iloc[-1]['date'],
        'open': df_real.iloc[-1]['open'],
        'high': df_real.iloc[-1]['high'],
        'low': df_real.iloc[-1]['low'],
        'close': df_real.iloc[-1]['close'],
        'volume': df_real.iloc[-1]['volume']
    })

    df_real_interpolated = pd.DataFrame(df_real_daily)

    # Combine synthetic and real data
    df_combined = pd.concat([df_synthetic, df_real_interpolated], ignore_index=True)
    df_combined = df_combined.sort_values('date').reset_index(drop=True)

    print(f"Created dataset with {len(df_combined)} daily data points")
    print(f"Date range: {df_combined['date'].min().date()} to {df_combined['date'].max().date()}")

    return df_combined


def preprocess_data(df, sequence_length=60):
    """Add technical indicators and prepare data for ML"""

    print("Adding technical indicators...")

    # Technical Indicators
    # Simple Moving Averages
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()

    # Exponential Moving Averages
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['bb_sma'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_sma'] + (2 * df['bb_std'])
    df['bb_lower'] = df['bb_sma'] - (2 * df['bb_std'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_sma']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Stochastic Oscillator
    df['lowest_low'] = df['low'].rolling(14).min()
    df['highest_high'] = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * ((df['close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low']))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # Price and Volume features
    df['price_change_pct'] = df['close'].pct_change() * 100
    df['volatility'] = df['price_change_pct'].rolling(10).std()
    df['volume_sma'] = df['volume'].rolling(10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # Market sentiment
    df['high_low_pct'] = (df['high'] - df['low']) / df['close'] * 100
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

    # Momentum indicators
    df['momentum'] = df['close'] / df['close'].shift(10) - 1
    df['rate_of_change'] = df['close'].pct_change(10) * 100

    # Average True Range
    high_low = df['high'] - df['low']
    high_close_prev = np.abs(df['high'] - df['close'].shift())
    low_close_prev = np.abs(df['low'] - df['close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    df['atr'] = true_range.rolling(14).mean()

    # Williams %R
    df['williams_r'] = -100 * (df['highest_high'] - df['close']) / (df['highest_high'] - df['lowest_low'])

    # Target variable (next day return)
    df['target'] = df['price_change_pct'].shift(-1)

    # Select features for modeling
    feature_cols = [
        'close', 'volume', 'sma_5', 'sma_10', 'sma_20', 'sma_50',
        'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_histogram',
        'rsi', 'bb_width', 'bb_position', 'stoch_k', 'stoch_d',
        'volatility', 'volume_ratio', 'high_low_pct', 'close_position',
        'momentum', 'rate_of_change', 'atr', 'williams_r'
    ]

    # Clean data
    df_clean = df.dropna().reset_index(drop=True)

    if len(df_clean) == 0:
        raise ValueError("No data remaining after cleaning!")

    # Prepare features and targets
    features = df_clean[feature_cols].values
    targets = df_clean['target'].values
    dates = df_clean['date'].values

    # Normalize features
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0)
    features_normalized = (features - feature_mean) / (
                feature_std + 1e-8)  # Add small epsilon to avoid division by zero

    # Create sequences for LSTM
    X, y = [], []
    for i in range(sequence_length, len(features_normalized)):
        X.append(features_normalized[i - sequence_length:i])
        y.append(targets[i])

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        raise ValueError("Not enough data to create sequences!")

    # Train/Validation/Test split
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'feature_names': feature_cols,
        'scaler_mean': feature_mean,
        'scaler_std': feature_std,
        'dates': dates,
        'raw_data': df_clean
    }