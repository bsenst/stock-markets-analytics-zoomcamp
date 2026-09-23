"""
Question 2: IPO Median Sharpe Ratio for 2025 IPOs (First 8 Months)
What is the median Sharpe ratio (as of 11 September 2026) for companies that went public before 1 September 2025?
"""

import pandas as pd
import numpy as np
import io
import requests
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


def solve_q2():
    # Step 1: Data Loading - Download 2025 IPOs from iposcoop.com
    url = "https://www.iposcoop.com/2025-pricings/"
    response = requests.get(url)
    tables = pd.read_html(io.StringIO(response.text))
    df = tables[0]
    print(f"Total 2025 IPOs: {len(df)}")
    print(df.columns.tolist())
    
    # Step 2: Filtering - Keep only IPOs with Offer Date before 1 September 2025
    df['Offer Date'] = pd.to_datetime(df['Offer Date'], format='%m/%d/%Y', errors='coerce')
    filtered = df[df['Offer Date'] < '2025-09-01'].copy()
    print(f"\nIPOs before Sep 1, 2025: {len(filtered)}")
    
    # Exclude entries with 0% return
    filtered = filtered[filtered['Return'] != 0].copy()
    print(f"After removing 0% return: {len(filtered)}")
    
    # Get tickers
    tickers = filtered['Symbol'].dropna().unique().tolist()
    print(f"\nUnique tickers: {len(tickers)}")
    
    # Step 3: Data Download - Use yfinance to download daily stock data
    print("\nDownloading stock data from yfinance...")
    stocks_data = []
    
    for ticker in tickers:
        try:
            start_date = filtered[filtered['Symbol'] == ticker]['Offer Date'].min()
            if pd.isna(start_date):
                start_date = '2025-01-01'
            else:
                start_date = start_date.strftime('%Y-%m-%d')
            
            data = yf.download(ticker, start=start_date, end='2026-09-12', progress=False, auto_adjust=True)
            
            if not data.empty and len(data) > 30:
                # Flatten MultiIndex columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                data = data.reset_index()
                data['Ticker'] = ticker
                stocks_data.append(data[['Date', 'Close', 'Ticker']])
        except Exception as e:
            continue
    
    print(f"\nSuccessfully downloaded: {len(stocks_data)} stocks")
    
    if len(stocks_data) == 0:
        print("No stock data downloaded!")
        return None
    
    # Combine all stock data
    stocks_df = pd.concat(stocks_data, ignore_index=True)
    print(f"Combined data shape: {stocks_df.shape}")
    print(stocks_df.head())
    
    # Step 4: Feature Engineering
    # growth_252d = Close / Close.shift(252)
    stocks_df['growth_252d'] = stocks_df.groupby('Ticker')['Close'].transform(lambda x: x / x.shift(252))
    
    # Annualized volatility: rolling 30-day std * sqrt(252)
    stocks_df['volatility'] = stocks_df.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(30).std() * np.sqrt(252)
    )
    
    # Step 5: Sharpe Ratio Calculation (risk-free rate = 5%)
    stocks_df['Sharpe'] = (stocks_df['growth_252d'] - 0.05) / stocks_df['volatility']
    
    # Step 6: Final Analysis - Filter for 2026-09-11
    stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
    target_date = pd.Timestamp('2026-09-11')
    
    sep11_data = stocks_df[stocks_df['Date'] == target_date].copy()
    print(f"\nData for 2026-09-11: {len(sep11_data)} stocks")
    
    if len(sep11_data) == 0:
        # Try nearest date
        dates = stocks_df['Date'].unique()
        dates = sorted([d for d in dates if d <= target_date])
        if dates:
            nearest = dates[-1]
            sep11_data = stocks_df[stocks_df['Date'] == nearest].copy()
            print(f"Using nearest date: {nearest}, {len(sep11_data)} stocks")
    
    # Compute descriptive statistics
    print("\nDescriptive statistics:")
    desc = sep11_data[['growth_252d', 'volatility', 'Sharpe']].describe()
    print(desc)
    
    median_sharpe = sep11_data['Sharpe'].median()
    mean_sharpe = sep11_data['Sharpe'].mean()
    count = sep11_data['Sharpe'].count()
    
    print(f"\n{'='*60}")
    print(f"ANSWER Q2: Median Sharpe ratio = {median_sharpe:.4f}")
    print(f"Mean Sharpe ratio = {mean_sharpe:.4f}")
    print(f"Count of stocks = {count}")
    print(f"{'='*60}")
    
    return median_sharpe


if __name__ == "__main__":
    solve_q2()