"""
Question 1: IPO Withdrawn IPOs by Company Type
What is the total withdrawn IPO value (in $ millions) for the company class with the highest total withdrawal value?
"""

import pandas as pd
import numpy as np
import io
import requests


def classify_company(name):
    """Classify company based on name patterns (order matters - first match wins)."""
    if pd.isna(name):
        return "Other"
    name_str = str(name)
    
    if "Technologies" in name_str:
        return "Technologies"
    elif any(x in name_str for x in ["Acquisition Corp", "Acquisition Corporation", "Corp"]):
        return "Acquisition Corp"
    elif any(x in name_str for x in ["Inc", "Incorporated"]):
        return "Inc."
    elif "Group" in name_str:
        return "Group"
    elif any(x in name_str for x in ["Ltd", "Limited"]):
        return "Limited"
    elif any(x in name_str for x in ["Holdings", "Holding"]):
        return "Holdings"
    else:
        return "Other"


def parse_price(price_str):
    """Extract numeric value from price string like '$8.00'."""
    if pd.isna(price_str) or price_str == '-':
        return np.nan
    cleaned = str(price_str).replace('$', '').replace(',', '').strip()
    try:
        return float(cleaned)
    except:
        return np.nan


def parse_numeric(val):
    """Parse numeric values, handling $ and commas."""
    if pd.isna(val) or val == '-':
        return np.nan
    cleaned = str(val).replace('$', '').replace(',', '').strip()
    try:
        return float(cleaned)
    except:
        return np.nan


def solve_q1():
    # Step 1: Data Loading
    url = "https://www.iposcoop.com/ipos-recently-filed/"
    response = requests.get(url)
    tables = pd.read_html(io.StringIO(response.text))
    df = tables[0]
    
    # Filter for Withdrawn IPOs
    withdrawn_df = df[df['Expected To Trade'] == 'Withdrawn'].copy()
    print(f"Withdrawn IPOs: {len(withdrawn_df)}")
    
    # Step 2: Company Classification
    withdrawn_df['Company Type'] = withdrawn_df['Company'].apply(classify_company)
    print("\nCompany Type counts:")
    print(withdrawn_df['Company Type'].value_counts())
    
    # Step 3: Price Parsing
    withdrawn_df['Price_Low_num'] = withdrawn_df['Price Low'].apply(parse_price)
    withdrawn_df['Price_High_num'] = withdrawn_df['Price High'].apply(parse_price)
    withdrawn_df['Avg_price'] = (withdrawn_df['Price_Low_num'] + withdrawn_df['Price_High_num']) / 2
    
    # Step 4: Numeric Conversion
    withdrawn_df['Shares_millions'] = withdrawn_df['Shares (millions)'].apply(parse_numeric)
    withdrawn_df['Est_Vol_millions'] = withdrawn_df['Est $ Vol (millions)'].apply(parse_numeric)
    
    # Step 5: Value Calculation
    withdrawn_df['Shares_offered_value'] = np.where(
        withdrawn_df['Shares_millions'].notna() & withdrawn_df['Avg_price'].notna(),
        withdrawn_df['Shares_millions'] * withdrawn_df['Avg_price'],
        withdrawn_df['Est_Vol_millions']
    )
    
    # Step 6: Aggregation
    result = withdrawn_df.groupby('Company Type')['Shares_offered_value'].sum().sort_values(ascending=False)
    print("\nTotal withdrawal value by Company Type ($ millions):")
    print(result)
    
    top_type = result.index[0]
    top_value = result.iloc[0]
    
    print(f"\n{'='*60}")
    print(f"ANSWER Q1: {top_type} with ${top_value:.2f} million")
    print(f"{'='*60}")
    
    return top_type, top_value


if __name__ == "__main__":
    solve_q1()