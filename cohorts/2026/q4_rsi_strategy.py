"""
Question 4: Simple RSI-Based Trading Strategy
What is the total profit (in $ thousands) you would have earned by investing $1000 every time a stock was oversold (RSI < 30)?
"""

import pandas as pd
import numpy as np
import gdown
import warnings
warnings.filterwarnings('ignore')


def solve_q4():
    # Step 1: Data Acquisition
    print("Downloading data from Google Drive...")
    file_id = "1grCTCzMZKY5sJRtdbLVCXg8JXA8VPyg-"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", "data.parquet", quiet=False)
    df = pd.read_parquet("data.parquet", engine="pyarrow")
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head())
    
    # Step 2: Strategy Setup - RSI threshold for oversold signal
    rsi_threshold = 30
    
    # Step 3: Filtering - Between 2000-01-01 and 2025-06-01 where RSI < 30
    # Check date column name
    date_col = 'Date' if 'Date' in df.columns else 'date'
    df[date_col] = pd.to_datetime(df[date_col])
    start_date = pd.Timestamp('2000-01-01')
    end_date = pd.Timestamp('2025-06-01')
    
    filtered = df[
        (df[date_col] >= start_date) & 
        (df[date_col] <= end_date) & 
        (df['rsi'] < rsi_threshold)
    ].copy()
    
    print(f"\nTotal signals (RSI < 30) between 2000-01-01 and 2025-06-01: {len(filtered)}")
    
    # Step 4: Profit Calculation
    # Investment of $1,000 for every signal
    # Use 30-day forward return (growth_future_30d)
    # Net Income = 1000 * (growth_future_30d - 1).sum()
    
    growth_col = 'growth_future_30d'
    
    if growth_col not in filtered.columns:
        # Try to find the correct column name
        growth_cols = [c for c in filtered.columns if 'growth' in c.lower() and '30' in c and 'future' in c.lower()]
        print(f"Growth columns found: {growth_cols}")
        if growth_cols:
            growth_col = growth_cols[0]
        else:
            print("No growth_future_30d column found!")
            return None
    
    print(f"Using growth column: {growth_col}")
    
    # Remove NaN values
    filtered = filtered.dropna(subset=[growth_col])
    print(f"Valid signals after dropping NaN: {len(filtered)}")
    
    # Calculate net income
    # growth_future_30d is the ratio (e.g., 1.0126 for 1.26% return)
    # So (growth_future_30d - 1) is the return
    returns = filtered[growth_col] - 1
    net_income = 1000 * returns.sum()
    
    # In $ thousands
    net_income_k = net_income / 1000
    
    # Additional stats
    avg_return = returns.mean() * 100
    win_rate = (returns > 0).mean() * 100
    
    print(f"\nAverage 30-day return: {avg_return:.2f}%")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Number of trades: {len(filtered)}")
    
    print(f"\n{'='*60}")
    print(f"ANSWER Q4: Net income = ${net_income_k:.2f} thousand")
    print(f"{'='*60}")
    
    return net_income_k


if __name__ == "__main__":
    solve_q4()