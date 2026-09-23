"""
Question 3: IPO 'Fixed Months Holding Strategy'
What is the optimal number of months (1 to 12) to hold a newly IPO'd stock in order to maximize the median growth value?
"""

import pandas as pd
import numpy as np
import io
import requests
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


def solve_q3():
    # Use the same data as Question 2
    # Step 1: Data Loading - Download 2025 IPOs from iposcoop.com
    url = "https://www.iposcoop.com/2025-pricings/"
    response = requests.get(url)
    tables = pd.read_html(io.StringIO(response.text))
    df = tables[0]
    print(f"Total 2025 IPOs: {len(df)}")
    
    # Step 2: Filtering
    df['Offer Date'] = pd.to_datetime(df['Offer Date'], format='%m/%d/%Y', errors='coerce')
    filtered = df[df['Offer Date'] < '2025-09-01'].copy()
    filtered = filtered[filtered['Return'] != 0].copy()
    print(f"IPOs before Sep 1, 2025 (non-zero return): {len(filtered)}")
    
    tickers = filtered['Symbol'].dropna().unique().tolist()
    print(f"Unique tickers: {len(tickers)}")
    
    # Step 3: Data Download - Use yfinance
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
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                data = data.reset_index()
                data['Ticker'] = ticker
                stocks_data.append(data[['Date', 'Close', 'Ticker']])
        except Exception as e:
            continue
    
    print(f"Successfully downloaded: {len(stocks_data)} stocks")
    
    if len(stocks_data) == 0:
        print("No stock data downloaded!")
        return None
    
    stocks_df = pd.concat(stocks_data, ignore_index=True)
    print(f"Combined data shape: {stocks_df.shape}")
    
    # Step 4: Identify Entry Points - first available trading day for each ticker
    min_dates = stocks_df.groupby('Ticker')['Date'].min().reset_index()
    min_dates.columns = ['Ticker', 'min_date']
    
    # Merge min_date to main dataframe
    stocks_df = stocks_df.merge(min_dates, on='Ticker')
    
    # Get entry prices (price on min_date)
    entry_prices = stocks_df[stocks_df['Date'] == stocks_df['min_date']][['Ticker', 'Close']].copy()
    entry_prices.columns = ['Ticker', 'entry_price']
    
    print(f"Entry prices found for {len(entry_prices)} tickers")
    
    # Merge entry prices back to main dataframe
    stocks_df = stocks_df.merge(entry_prices, on='Ticker')
    
    # Step 5: Feature Engineering - Calculate future growth for 1-12 months
    # 1 month = 21 trading days
    for month in range(1, 13):
        days = month * 21
        col_name = f'future_growth_{month}_m'
        stocks_df[col_name] = stocks_df.groupby('Ticker')['Close'].transform(
            lambda x: x.shift(-days) / stocks_df.loc[x.index, 'entry_price']
        )
    
    # Step 6: Data Alignment - Get growth values at entry date for each ticker
    entry_growth = stocks_df[stocks_df['Date'] == stocks_df['min_date']].copy()
    
    growth_cols = [f'future_growth_{month}_m' for month in range(1, 13)]
    result = entry_growth[['Ticker'] + growth_cols].copy()
    
    print(f"\nEntry growth data shape: {result.shape}")
    
    # Step 7: Median Analysis
    print("\nDescriptive statistics for future growth columns:")
    desc = result[growth_cols].describe()
    print(desc)
    
    # Find median (50th percentile) for each month
    medians = result[growth_cols].median()
    print("\nMedian growth by month:")
    for month in range(1, 13):
        col = f'future_growth_{month}_m'
        print(f"  Month {month}: {medians[col]:.4f} ({medians[col]*100:.2f}%)")
    
    # Find optimal month (highest median)
    optimal_month = medians.idxmax()
    optimal_median = medians.max()
    optimal_month_num = int(optimal_month.split('_')[2].replace('m', ''))
    
    print(f"\n{'='*60}")
    print(f"ANSWER Q3: Optimal holding period = {optimal_month_num} months")
    print(f"Maximum median growth = {optimal_median:.4f} ({optimal_median*100:.2f}%)")
    print(f"{'='*60}")
    
    return optimal_month_num, optimal_median


if __name__ == "__main__":
    solve_q3()