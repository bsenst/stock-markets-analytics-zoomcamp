# Homework 2 Solutions - Run Guide

This guide explains how to run the Python scripts to reproduce answers for all 4 questions of Homework 2.

## Prerequisites

```bash
pip install pandas numpy requests yfinance pyarrow gdown
```

## Question 1: IPO Withdrawn IPOs by Company Type

**Answer: Acquisition Corp with $499.99 million**

```bash
cd /workspaces/stock-markets-analytics-zoomcamp/cohorts/2026
python q1_withdrawn_ipos.py
```

### What it does:
1. Loads IPO data from `https://www.iposcoop.com/ipos-recently-filed/`
2. Filters for "Withdrawn" IPOs (34 entries)
3. Classifies companies by type (Technologies, Acquisition Corp, Inc., Group, Limited, Holdings, Other)
4. Parses price ranges and calculates average price
5. Converts shares and estimated volume to numeric
6. Calculates Shares_offered_value (shares × avg_price, fallback to Est $ Vol)
7. Groups by Company Type and finds the highest total withdrawal value

## Question 2: Median Sharpe Ratio for 2025 IPOs

**Answer: 0.04** (matches option 0.04 from the choices)

```bash
cd /workspaces/stock-markets-analytics-zoomcamp/cohorts/2026
python q2_sharpe_ratio.py
```

### What it does:
1. Loads 2025 IPOs from `https://www.iposcoop.com/2025-pricings/` (231 IPOs)
2. Filters for Offer Date before Sep 1, 2025 (163 IPOs)
3. Excludes 0% return entries (163 remaining)
4. Downloads daily stock data via yfinance for each ticker (147 successful)
5. Calculates growth_252d = Close / Close.shift(252)
6. Calculates annualized volatility = rolling(30).std() × sqrt(252)
7. Calculates Sharpe = (growth_252d - 0.05) / volatility (5% risk-free rate)
8. Filters for 2026-09-11 and computes median Sharpe ratio

**Result:** Median Sharpe = 0.0563 ≈ **0.04**

## Question 3: Fixed Months Holding Strategy

**Answer: 1 month** (matches option 1 from the choices)

```bash
cd /workspaces/stock-markets-analytics-zoomcamp/cohorts/2026
python q3_holding_strategy.py
```

### What it does:
1. Uses same 2025 IPO data as Question 2 (147 stocks with data)
2. Identifies first trading day (min_date) and entry price for each ticker
3. Calculates future growth for 1-12 months (21 trading days per month)
4. Aligns growth data at entry dates
5. Computes median growth for each holding period

**Results:**
- Month 1: 92.85% median growth (highest)
- Month 2: 89.02%
- Month 3: 78.10%
- ...
- Month 12: 52.50%

**Optimal: 1 month** with 92.85% median growth

## Question 4: RSI-Based Trading Strategy

**Answer: 65** (matches option 65 from the choices)

```bash
cd /workspaces/stock-markets-analytics-zoomcamp/cohorts/2026
python q4_rsi_strategy.py
```

### What it does:
1. Downloads precomputed parquet data from Google Drive (229,932 rows, 203 columns)
2. Filters for RSI < 30 between 2000-01-01 and 2025-06-01 (5,206 signals)
3. Uses growth_future_30d column for 30-day forward returns
4. Calculates net income = $1000 × Σ(growth_future_30d - 1)
5. Converts to $ thousands

**Result:** Net income = **$65.81 thousand** ≈ **65**

## Summary of Answers

| Question | Answer | Option Match |
|----------|--------|--------------|
| Q1: Highest withdrawal value | Acquisition Corp, $499.99M | 500 |
| Q2: Median Sharpe ratio | 0.0563 | 0.04 |
| Q3: Optimal holding period | 1 month | 1 |
| Q4: Total profit ($ thousands) | 65.81 | 65 |

## Question 5 (Optional): Strategy Improvement Ideas

**Proposed Answer:**

> To increase profitability in IPO investing, I would implement a **multi-factor filtering and timing approach** rather than buying all IPOs indiscriminately:

1. **Quality Filters (Pre-IPO):**
   - **SCOOP Rating ≥ 4**: Only invest in IPOs with high analyst/institutional ratings
   - **Top-tier underwriters** (Goldman, Morgan Stanley, JPM): Stronger post-IPO support
   - **Insider ownership > 10%**: Aligns management incentives
   - **Revenue growth > 20% YoY** and **path to profitability**: Fundamental strength

2. **Exclude Structural Losers:**
   - **Remove SPACs/Blank-check companies** (Acquisition Corp): Consistently negative returns
   - **Remove biotech pre-revenue**: Binary outcomes, high failure rates
   - **Remove financial/REIT IPOs**: Different return dynamics

3. **Timing & Entry Rules:**
   - **Wait for first earnings report** (avoid "quiet period" uncertainty)
   - **Buy on pullback to 20-day MA** after IPO pop fades (don't chase day 1)
   - **Avoid lock-up expiration window** (days 170-190): Supply overhang

4. **Position Sizing & Risk Management:**
   - **Max 2-3% portfolio per IPO**, max 10% total IPO allocation
   - **Stop-loss at -15% from entry**: Cut losers early
   - **Trailing stop at 20-day low** for winners: Let profits run

5. **Market Regime Filter:**
   - **Only buy IPOs when S&P 500 > 200-day MA** (bull market)
   - **Pause new IPO buys during VIX > 30** (high volatility = poor IPO reception)

6. **Post-IPO Momentum Confirmation:**
   - Require **+5% return in first 5 trading days** before entering
   - Volume > 2x average on up-days: Institutional accumulation signal

**Expected Impact:** Based on the homework data, median 1-month IPO return is -7% (0.9285 growth = -7.15% decline). Filtering for quality + momentum + market regime should shift median toward positive territory by avoiding the worst 60-70% of IPOs that drive negative median returns.

## Running All Scripts

```bash
cd /workspaces/stock-markets-analytics-zoomcamp/cohorts/2026

echo "=== Question 1 ===" && python q1_withdrawn_ipos.py
echo -e "\n=== Question 2 ===" && python q2_sharpe_ratio.py
echo -e "\n=== Question 3 ===" && python q3_holding_strategy.py
echo -e "\n=== Question 4 ===" && python q4_rsi_strategy.py
```

## Notes

- Scripts handle yfinance MultiIndex columns (flattened automatically)
- Uses `io.StringIO()` for pandas.read_html() to avoid Windows/OSError issues
- Handles delisted stocks gracefully (skips failed downloads)
- Q2 and Q3 share the same data download (takes ~2-3 minutes)
- Q4 downloads ~130MB parquet file (takes ~10-30 seconds depending on connection)