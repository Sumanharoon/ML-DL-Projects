import yfinance as yf
import pandas as pd
import time

tickers = [
    'NVDA', 'AAPL', 'MSFT', 'LMT', 'AMD', 'GOOGL', 'AMZN', 'TSLA', 'META', 'JPM',
    'BAC', 'WMT', 'V', 'MA', 'UNH', 'HD', 'PG', 'XOM', 'CVX', 'JNJ',
    'ORCL', 'COST', 'ABBV', 'NFLX', 'CRM', 'KO', 'PEP', 'AVGO', 'TMO', 'CSCO',
    'ACN', 'MCD', 'LIN', 'ABT', 'DIS', 'PM', 'INTC', 'TXN', 'QCOM', 'IBM',
    'GE', 'CAT', 'BA', 'HON', 'RTX', 'AMGN', 'SPGI', 'LOW', 'GS', 'BLK',
    'MS', 'DE', 'ELV', 'BKNG', 'LRCX', 'MDLZ', 'ADI', 'ADP', 'GILD'
]

all_data = []
print("Step 1: Downloading 100% Complete Real Market Data...\n")

for idx, ticker in enumerate(tickers, start=1):
    print(f"[{idx}/{len(tickers)}] Fetching data for: {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        # 1-hour interval with fallback to daily if hourly blocks
        df = stock.history(period="730d", interval="1h")
        
        if df.empty:
            df = stock.history(period="2y", interval="1d")
            
        if not df.empty:
            df = df.reset_index()
            df['Ticker'] = ticker
            
            time_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
            df['Trade_Timestamp_EST'] = pd.to_datetime(df[time_col]).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            df = df.rename(columns={
                'Open': 'Open_Price', 'High': 'High_Price',
                'Low': 'Low_Price', 'Close': 'Close_Price', 'Volume': 'Volume'
            })
            all_data.append(df[['Ticker', 'Trade_Timestamp_EST', 'Open_Price', 'High_Price', 'Low_Price', 'Close_Price', 'Volume']])
    except Exception as e:
        print(f"   Error fetching {ticker}: {e}")
    
    time.sleep(0.1)

final_df = pd.concat(all_data, ignore_index=True)
output_file = "real_market_data_complete.csv"
final_df.to_csv(output_file, index=False)
print(f"\n Step 1 Complete! Total {len(final_df)} Real Records Saved to '{output_file}'!")