import pandas as pd
import numpy as np

print("Step 2: Engineering Advanced Non-Leaking Financial Features...\n")

input_file = "real_market_data_complete.csv"
df = pd.read_csv(input_file)

# --- 1. RSI (14) Calculation ---
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / (loss + 1e-5)
    return 100 - (100 / (1 + rs))

df['RSI_14'] = df.groupby('Ticker')['Close_Price'].transform(lambda x: calculate_rsi(x))

# --- 2. ATR Volatility Calculation ---
# High/Low available nahi hone par Close price ki percentage volatility se estimate hota hai
df['ATR_Volatility'] = df.groupby('Ticker')['Close_Price'].transform(
    lambda x: x.pct_change().abs().rolling(window=14, min_periods=1).mean() * 100
).fillna(0)

# --- 3. Technical Indicators & Ratios ---
df['SMA_20'] = df.groupby('Ticker')['Close_Price'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
df['SMA_50'] = df.groupby('Ticker')['Close_Price'].transform(lambda x: x.rolling(window=50, min_periods=1).mean())
df['Volume_SMA_20'] = df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())

df['Price_Vs_SMA50'] = df['Close_Price'] / (df['SMA_50'] + 1e-5)
df['Volume_Surge_Ratio'] = df['Volume'] / (df['Volume_SMA_20'] + 1e-5)

output_file = "real_featured_market_data.csv"
df.to_csv(output_file, index=False)
print(f"Step 2 Complete! File Saved: '{output_file}' with {len(df.columns)} features.")